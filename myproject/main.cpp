// include aia and ucas utility functions
#include "aiaConfig.h"
#include "ucasConfig.h"
#include "functions.h"

// include opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

int FP = 0;
int TP = 0;
int FN = 0;

int totalSigns = 0;
int test = 0;
const int numOfCriteria = 3;
std::vector<std::pair<float, std::string>> scoreVec;

int foundSigns(std::vector<std::pair<float, cv::Rect2d>> candidateSigns, json metadata);
void drawSignRectangles(std::vector<cv::Mat> imgs, std::vector<json> metadata);


std::string img_path = "E:/UNI/Magistrale/Image/no-entry-recognition/no-entry-recognition_source/subset-IPA-AIA-no-entry/*.jpg";

bool pairSort(const std::pair<float, std::string>& a, const std::pair<float, std::string>& b){

    return (a.first > b.first);
}

int main(){

    // Loading images' paths
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> hsvImages;
    std::vector<json> metadata;
    std::vector<std::string> paths;
    cv::glob(img_path, paths, true);

    for (auto path : paths)
    {
        // Loading images
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty())
            continue;
        images.push_back(img);

        // Loading images' metadata
        path.replace((path.end() - 3), path.end(), "json");
        std::ifstream f(path);
        metadata.push_back(json::parse(f));
    }

    int i = 0;
    int signs = 0;
    for (auto& img : images) {

        cv::Mat img_gray;
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2HSV);
        hsvImages.push_back(img_gray);
        std::vector<cv::Mat> hsv_chans;
        cv::split(img_gray, hsv_chans);
        img_gray = hsv_chans[1];
        //aia::imshow("img", img_gray);
        /* 
        float sigmaCannyX10 = 12.5;
        float sigmaGaussianX10 = 17.5;
        float thresholdCanny = 50;
        float thresholdMultiplier = 3;
        */
        float sigmaCannyX10 = 17.5;
        float sigmaGaussianX10 = 17.5;
        float thresholdCanny = 20;
        float thresholdMultiplier = 2.5;

        // calculate gaussian kernel size so that 99% of data are under the gaussian 
        int n = 6 * (sigmaCannyX10 / 10.0);
        if (n % 2 == 0)
            n++;
        int m = 6 * (sigmaGaussianX10 / 10.0);
        if (m % 2 == 0)
            m++;
        // if 'sigmaGaussianX10' is valid, we apply gaussian smoothing
        if (sigmaGaussianX10 > 0)
            cv::GaussianBlur(img_gray, img_gray, cv::Size(n, n), (sigmaCannyX10 / 10.0), (sigmaCannyX10 / 10.0));

        cv::Mat imgCanny;
        cv::Canny(img_gray, img_gray, thresholdCanny, thresholdMultiplier * thresholdCanny);
        //aia::imshow("img", img_gray);
        cv::morphologyEx(img_gray, imgCanny, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        cv::GaussianBlur(imgCanny, imgCanny, cv::Size(m, m), (sigmaGaussianX10 / 10.0), (sigmaGaussianX10 / 10.0));
        //aia::imshow("img", img_gray);
        cv::morphologyEx(imgCanny, imgCanny, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        cv::GaussianBlur(imgCanny, imgCanny, cv::Size(m, m), (sigmaGaussianX10 / 10.0), (sigmaGaussianX10 / 10.0));
        //aia::imshow("img", img_gray);

        std::vector<cv::Vec3f> circles;

        // Get all the circles found by Hough Transform and place such circles in a vector (circles)
        HoughCircles(imgCanny, circles, cv::HOUGH_GRADIENT, 1,
            100,  // Change this value to detect circles with different distances to each other
            255, 33, 10, 111 // Change the last two parameters (min_radius & max_radius) to detect larger circles
        );

        std::vector<std::pair<float, cv::Rect2d>> candidateSigns;
        for (auto& circle : circles) {

            auto xCenter = circle[0];
            auto yCenter = circle[1];
            auto radius = circle[2];
            cv::Rect2d rect = cv::Rect2d(cv::Point(xCenter - radius, yCenter - radius), cv::Point(xCenter + radius, yCenter + radius));
            candidateSigns.push_back(std::pair<float, cv::Rect2d>(0.0, rect));
        }

        for (auto& pair : candidateSigns){
            
            auto candidateSign = pair.second;

            auto hue = hsv_chans[0].clone();

            cv::Rect2d imageRect = cv::Rect2d(0, 0, hue.size().width, hue.size().height);
            imageRect = imageRect & candidateSign;

            cv::Mat croppedImage = hue(imageRect);
            cv::Mat redPixel = croppedImage.clone();
            cv::Mat redPixel2 = croppedImage.clone();

            cv::inRange(croppedImage, cv::Scalar(0, 0, 0), cv::Scalar(15, 255, 255), redPixel);
            cv::inRange(croppedImage, cv::Scalar(165, 0, 0), cv::Scalar(180, 255, 255), redPixel2);
            redPixel = redPixel | redPixel2;
            int redPixelCount = cv::countNonZero(redPixel);

            float percArea = 0;
            if (redPixelCount > 0)
                percArea = (float)redPixelCount / imageRect.area();

            if(percArea <= CV_PI/4)
                pair.first += (percArea/(CV_PI/4)) * 0.8 / numOfCriteria;
            else
                pair.first += (1 - (percArea-(CV_PI/4))/(CV_PI/4)) * 0.8 / numOfCriteria; 

            float redAvg = 0;
		        
            for(int i = 0; i < croppedImage.rows; i++){
                auto ithRow = croppedImage.ptr(i);
                for(int j = 0; j < croppedImage.cols; j++){
                    if(ithRow[j] <= 15 && ithRow[j] >= 0)
                        redAvg += ithRow[j];
                    else if(ithRow[j] <= 180 && ithRow[j] >= 165)
                        redAvg += ithRow[j] - 180;
                }
            }

            if(redPixelCount > 0)
                redAvg /= redPixelCount;

            redAvg = 1 - abs(redAvg) / 15;

            pair.first += redAvg * 0.2 / numOfCriteria;

            
            cv::Mat CroppedCannyImg = img_gray(imageRect);
            //cv::morphologyEx(CroppedCannyImg, CroppedCannyImg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            //cv::morphologyEx(CroppedCannyImg, CroppedCannyImg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            
            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(CroppedCannyImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

            contours.erase(std::remove_if(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& object)
            {
                double area = cv::contourArea(object);
                cv::Rect bounding_rect = cv::boundingRect(object);
                return area / bounding_rect.area() < 0.7;
            }
            ), contours.end());

            contours.erase(std::remove_if(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& object)
            {
                cv::RotatedRect rot_rect = cv::minAreaRect(object);
                printf("%f\n", rot_rect.size.aspectRatio());
                return std::abs(rot_rect.size.aspectRatio() - 4.45) > 1;
            }
            ), contours.end());

            contours.erase(std::remove_if(contours.begin(), contours.end(),
                [CroppedCannyImg](const std::vector<cv::Point>& object)
            {
                double area = cv::contourArea(object);
                return area < CroppedCannyImg.size().area() * 0.1 || area > CroppedCannyImg.size().area() * 0.9;
            }
            ), contours.end());

            
/* 
            cv::Mat drawing = cv::Mat::zeros(CroppedCannyImg.size(), CV_8UC3);
            for( size_t i = 0; i< contours.size(); i++ ){
                cv::Scalar color = cv::Scalar(255, 0, 0);
                drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
            }
 */

            //aia::imshow("img", CroppedCannyImg);
            //aia::imshow("drawing", drawing);
            if(!contours.empty())
                pair.first += 1 / float(numOfCriteria);

        }
        

        //aia::imshow("img", croppedImage);

        //aia::imshow("img", circles_img);
        //aia::imshow("img", rects_img);

        int previousTotalSigns = totalSigns;

        int tmp = foundSigns(candidateSigns, metadata[i]);

        // if (tmp < (totalSigns - previousTotalSigns))
          //   printf("%s\n", paths[i].c_str());

        signs += tmp;
        i++;

    }

    FN = totalSigns - TP;
    
    printf("TP: %d\nFN: %d\nFP: %d\ntest: %d\n", TP, FN, FP, test);

    // CALCULATE AVERAGE PRECISION
    // Sorting in descending score order
    std::sort(scoreVec.begin(), scoreVec.end(), pairSort);

    int cumulativeTP = 0;
    int cumulativeFP = 0;
    std::vector<std::pair<float, float>> precRecall;

    for(auto& pair : scoreVec){
        if(pair.second == "FP")
            cumulativeFP++;
        else if(pair.second == "TP")
            cumulativeTP++;
        else
            printf("eh no così non va bene\n");

        float precision = (float)cumulativeTP / (cumulativeTP + cumulativeFP);
        float recall = (float)cumulativeTP / (TP + FN);

        precRecall.push_back(std::pair<float, float>(precision, recall));
    }

    float avgPrecision = 0;

    for(int k = 0; k < precRecall.size(); k++){
        avgPrecision += precRecall[k].first * (precRecall[k].second - (k != 0 ? precRecall[k-1].second : 0));
        printf("%f, %f,       %f\n", precRecall[k].first, precRecall[k].second, avgPrecision);
    }

    printf("Average Precision: %f\n", avgPrecision);


    return 0;
}


// FUNCTIONS
void drawSignRectangles(std::vector<cv::Mat> imgs, std::vector<json> metadata) {
    for (int i = 0; i < imgs.size(); i++) {

        for (auto& item : metadata[i]["objects"]) {
            if (item["label"] != "regulatory--no-entry--g1")
                continue;
            cv::Point p1(item["bbox"]["xmin"], item["bbox"]["ymin"]);
            cv::Point p2(item["bbox"]["xmax"], item["bbox"]["ymax"]);

            cv::rectangle(imgs[i], p1, p2, cv::Scalar(255), -1);
        }
    }
}


int foundSigns(std::vector<std::pair<float, cv::Rect2d>> candidateSigns, json metadata) {

    int signs = 0;
    std::vector<cv::Rect2d> trueSigns;

    for (auto& item : metadata["objects"]) {
        if (item["label"] != "regulatory--no-entry--g1")
            continue;

        cv::Point p1(item["bbox"]["xmin"], item["bbox"]["ymin"]);
        cv::Point p2(item["bbox"]["xmax"], item["bbox"]["ymax"]);
        trueSigns.push_back(cv::Rect(p1, p2));
    }

    totalSigns += trueSigns.size();

    int TP_this_img = 0;
    int tmp = 0;

    for (auto& pair : candidateSigns){
        
        auto candidateSign = pair.second;
        std::pair<float, std::string> scorePair(pair.first, "");

        for(auto& trueSign : trueSigns){
            if ((trueSign & candidateSign).area() / (trueSign | candidateSign).area() > 0.5){{
                TP_this_img++;
                scorePair.second = "TP";
                }
                if(pair.first > 0.3)
                    test++;
            }
            else
                tmp++;
        }

        if(tmp == trueSigns.size()){
            FP++;
            scorePair.second = "FP";
        }

        if(scorePair.second != "")
            scoreVec.push_back(scorePair);

    }

    if(TP_this_img > trueSigns.size()){
        printf("pių segnali trovati\n");
        exit(EXIT_FAILURE);
    }

    TP += TP_this_img;

    return TP_this_img;
}