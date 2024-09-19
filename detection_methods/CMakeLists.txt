add_library(detection_methods PersonDetection.h PersonSegmentation.h PersonSegmentation.cpp PersonTracking.h PersonTracking.cpp)

target_include_directories(detection_methods PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(detection_methods utils ${OpenCV_LIBS})
