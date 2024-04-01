cmake_minimum_required(VERSION 3.10)
project(AthleteDT)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(AthleteDT main.cpp)

target_link_libraries( AthleteDT ${OpenCV_LIBS} )
