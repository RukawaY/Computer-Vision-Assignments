cmake_minimum_required(VERSION 3.10)
project(eigenface)

set(OpenCV_DIR "/opt/homebrew/opt/opencv/share/opencv4")

# Add Homebrew's prefix to CMAKE_PREFIX_PATH to help find packages
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
# For Eigen specifically, its CMake files might be in a subdirectory like /opt/homebrew/opt/eigen/share/eigen3/cmake
# or /opt/homebrew/share/eigen3/cmake. CMAKE_PREFIX_PATH should cover the base.
# Alternatively, you could set Eigen3_DIR directly if you know the exact path to Eigen3Config.cmake

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "Boost library status:")
message(STATUS "    version: ${Boost_VERSION}")
message(STATUS "    libraries: ${Boost_LIBRARIES}")
message(STATUS "    include path: ${Boost_INCLUDE_DIRS}")

message(STATUS "Eigen library status:")
message(STATUS "    version: ${Eigen3_VERSION}")
message(STATUS "    libraries: ${Eigen3_LIBRARIES}")
message(STATUS "    include path: ${Eigen3_INCLUDE_DIRS}")

add_executable(train train.cpp)
add_executable(test test.cpp)

target_include_directories(train PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(train 
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES} 
    Eigen3::Eigen
)

target_include_directories(test PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(test 
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES} 
    Eigen3::Eigen
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)