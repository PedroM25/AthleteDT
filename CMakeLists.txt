cmake_minimum_required(VERSION 3.10)
project(AthleteDT)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(utils)
add_subdirectory(detection_methods)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} detection_methods)
