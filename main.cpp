#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "PersonDetection.h"
#include "PersonTracking.h"
#include "PersonSegmentation.h"

namespace fs = std::filesystem;

const std::string APP_NAME = "AthleteDT";

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Not all mandatory arguments supplied. Usage: " << APP_NAME << " <detection method> <video path> [--visualize]" << std::endl;
        return 1;
    }

    const auto detectionMethodPassed = std::string(argv[1]);
    if (detectionMethodPassed != "tracking" &&
        detectionMethodPassed != "segmentation"){
        std::cerr << "Detection method \'" << detectionMethodPassed << "\' is not supported. Exiting." << std::endl;
        return 1;
    }

    // check if should show live processing of frames
    bool showProcessing = false;
    if (argc > 3 && std::string(argv[3]) == "--visualize"){
        showProcessing = true;
    }

    // create output/ folder
    fs::create_directories("./output");
    
    // Create log file
    const auto videoPath = std::string(argv[2]);
    std::string video_file_name = fs::path(videoPath).stem();
    std::string log_file_name = "output/" + APP_NAME + "_" + video_file_name + "_" + detectionMethodPassed + ".log";
    
    std::ofstream log_file = std::ofstream(log_file_name, std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "Error creating log file. Exiting." << std::endl;
        return 1;
    }

    log_file << "Starting " << APP_NAME << " execution with method \'" << detectionMethodPassed << "\'." << std::endl;

    // video capture
    cv::VideoCapture cap{videoPath};
    if (!cap.isOpened()) {
        log_file << "Error opening video stream or file. Exiting." << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    log_file << "Successfully imported video " << fs::path(videoPath).filename()
        << ", FPS: " << fps 
        << ", Num frames: " << num_frames 
        << ", Resolution: " << frame_width << "x" << frame_height
        << std::endl;

    // video writer
    std::string output_file_name = "output/output_" + video_file_name + "_" + detectionMethodPassed + ".mp4";
    cv::VideoWriter output_video_writer = cv::VideoWriter(output_file_name, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_width,frame_height));
    
    int delay = 1000 / fps; // Calculate delay based on the framerate
    std::unique_ptr<PersonDetection> pd;
    if(detectionMethodPassed == "tracking"){
        pd = std::make_unique<PersonTracking>(cap, output_video_writer, showProcessing, delay, log_file);
    } else if (detectionMethodPassed == "segmentation"){
        pd = std::make_unique<PersonSegmentation>(cap, output_video_writer, showProcessing, delay, log_file);
    }
    
    int64_t start = cv::getTickCount();
    std::cout << "Video processing has started..." << std::endl;
    const auto frameCount = pd->performDetection();
    std::cout << "Processing finished. Check output folder." << std::endl;

    log_file << "No more frames grabbed. Exiting..." << std::endl;
    log_file << "Total number of frames processed: " << frameCount << std::endl;
    int64_t end = cv::getTickCount();
    log_file << "Processing time: " << (end-start)/cv::getTickFrequency() << "s" << std::endl;
    cap.release();
    cv::destroyAllWindows();
    log_file.close();
}
