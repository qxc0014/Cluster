cmake_minimum_required( VERSION 2.8 )
project( Spectral_clustreing )
set(CMAKE_BUILD_TYPE "Release")
SET( CMAKE_CXX_STANDARD 14)
SET( CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories( "/usr/include/eigen3" )#加入头文件的路径
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(Spectral_clustreing Spectral_clustreing.cpp)
target_link_libraries(Spectral_clustreing ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
