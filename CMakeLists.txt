cmake_minimum_required(VERSION 3.10)
project(motion_tracking_30fps2_11_11_25)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(motion_tracking_30fps2_11_11_25 motion_tracking_30fps2_11_11_25.cpp)
target_link_libraries(motion_tracking_30fps2_11_11_25 PRIVATE ${OpenCV_LIBS})
