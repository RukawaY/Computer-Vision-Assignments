cmake_minimum_required(VERSION 3.10)
project(prepare_data)

set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(prepare_data
    prepare_data.cpp
)

target_include_directories(prepare_data PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(prepare_data PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(prepare_data ${OpenCV_LIBS} ${Boost_LIBRARIES})

message(STATUS "Boost library status:")
message(STATUS "    version: ${Boost_VERSION}")
message(STATUS "    libraries: ${Boost_LIBRARIES}")
message(STATUS "    include path: ${Boost_INCLUDE_DIRS}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)