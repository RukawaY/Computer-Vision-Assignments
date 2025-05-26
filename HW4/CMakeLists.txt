cmake_minimum_required(VERSION 3.10)
project(calibration_birdeye)

set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(calibration_birdeye
    calibration_birdeye.cpp
)

target_include_directories(calibration_birdeye PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(calibration_birdeye ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)