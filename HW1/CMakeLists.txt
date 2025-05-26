cmake_minimum_required(VERSION 3.10)
project(video_maker)

set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(video_maker
    video_maker.cpp
)

target_include_directories(video_maker PRIVATE ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

target_link_libraries(video_maker ${OpenCV_LIBS} ${Boost_LIBRARIES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)