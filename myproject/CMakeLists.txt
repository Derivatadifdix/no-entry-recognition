# include libraries
include_directories (${aia_SOURCE_DIR}/3rdparty/aiacommon)
include_directories (${aia_SOURCE_DIR}/3rdparty/ucascommon)
include(FetchContent)

#Fetch to download a release as dependency 
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)


# find sources
file(GLOB src *.h *.hpp *.cpp)

# fix va_start error in VS 14+
add_definitions(-D_CRT_NO_VA_START_VALIDATION)

# create executable from sources
add_executable(myproject ${src})

# link the executable to other modules / libraries 
target_link_libraries(myproject aiacommon ucascommon ${OpenCV_LIBS} ${Boost_LIBRARIES}  nlohmann_json::nlohmann_json)
