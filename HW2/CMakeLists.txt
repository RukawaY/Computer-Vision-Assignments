cmake_minimum_required(VERSION 3.10)
project(ellipse_fit)

set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(ellipse_fit
    ellipse_fit.cpp
)

target_include_directories(ellipse_fit PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(ellipse_fit ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)