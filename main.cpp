#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

namespace fs = std::filesystem;

const int IN_W = 300;
const int IN_H = 300;

const float CONF_TRSH = 0.6;
const float MEAN_SUBTRACTION_VAL = 127.5; // Result from doing 255/2
const float SCALING_FACTOR = 0.00784; // Result from doing 2/255
const float SECONDS_BETW_DETECTIONS = 3;

const std::string APP_NAME = "AthleteDT";
const std::string WIN_NAME = "Output";
const std::string PROTO_TXT_PATH = "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.prototxt";
const std::string CAFFE_MODEL_PATH = "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.caffemodel";
const std::string CLASS_NAMES_PATH = "model/MobileNet-SSDCaffe/object_detection_classes_voc0712.txt";

const cv::Scalar COLOR{0,255,0};

std::vector<std::string> class_names{};
std::ofstream log_file;
int frame_count = 0;

/*
 * Read from file supported objects that model can identify
 */
bool readClassNames(){
    std::ifstream file(CLASS_NAMES_PATH);
    
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            class_names.push_back(line);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
        return false;
    }
    return true;
}

/*
 * Returns true if a person is detected; false otherwise
 */
bool person_detected(cv::dnn::Net& net, const cv::Mat& frame, cv::Ptr<cv::Tracker>& tracker){
    bool success = false;
    cv::Mat blob = cv::dnn::blobFromImage(frame, SCALING_FACTOR, cv::Size(IN_W, IN_H), MEAN_SUBTRACTION_VAL);
    net.setInput(blob);
    cv::Mat detections = net.forward();

    cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++){
        float confidence = detectionMat.at<float>(i, 2);
        int class_id = detectionMat.at<float>(i, 1);

        std::string label = class_names[class_id] + " : " + std::to_string(confidence);

        if (class_names[class_id] == "person" && confidence > CONF_TRSH){
            success = true;

            // Object location
            int xLeftTop = detectionMat.at<float>(i, 3) * frame.cols;
            int yLeftTop = detectionMat.at<float>(i, 4) * frame.rows;
            int xRightBottom = detectionMat.at<float>(i, 5) * frame.cols;
            int yRightBottom = detectionMat.at<float>(i, 6) * frame.rows;
            
            log_file << "FRAME " << frame_count << ": " << "Person detected, confidence: " << confidence << ", coordinates: [" 
                << "["<<xLeftTop<<","<<yLeftTop<<"],"
                << "["<<xRightBottom<<","<<yLeftTop<<"],"
                << "["<<xLeftTop<<","<<yRightBottom<<"],"
                << "["<<xRightBottom<<","<<yRightBottom<<"]"
                << "]"
                << std::endl;

            cv::Point leftTopCoords{xLeftTop, yLeftTop};
            cv::Point rightBottomCoords{xRightBottom, yRightBottom};
            cv::Rect rec{leftTopCoords, rightBottomCoords};
            
            tracker->init(frame, rec);
            break; //first one detected, others ignored
        }
    }
    return success;
}

int main(int argc, char **argv)
{
    if (argc == 1) {
        std::cerr << "No arguments passed. Usage: " << APP_NAME << " <video path> [--visualize]" << std::endl;
        return 1;
    }

    // check if should show live processing of frames
    bool showProcessing = false;
    if (argc == 3 && std::string(argv[2]) == "--visualize"){
        showProcessing = true;
    }

    if (!readClassNames()){
        return 1;
    };

    // create output/ folder
    fs::create_directories("./output");
    
    // Create log file
    std::string video_file_name = fs::path(argv[1]).stem();
    std::string log_file_name = "output/" + APP_NAME + "_" + video_file_name + ".log";
    log_file= std::ofstream(log_file_name, std::ios::trunc);
    if (!log_file.is_open()) {
        std::cerr << "Error creating log file. Exiting." << std::endl;
        return 1;
    }

    log_file << "Starting " << APP_NAME << " execution." << std::endl;

    // video capture
    cv::VideoCapture cap{argv[1]};
    if (!cap.isOpened()) {
        log_file << "Error opening video stream or file. Exiting." << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    log_file << "Successfully imported video " << fs::path(argv[1]).filename()
        << ", FPS: " << fps 
        << ", Num frames: " << num_frames 
        << ", Resolution: " << frame_width << "x" << frame_height
        << std::endl;

    // video writer
    std::string output_file_name = "output/output_" + video_file_name + ".mp4";
    cv::VideoWriter output_video_writer = cv::VideoWriter(output_file_name, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame_width,frame_height));
    
    // pre trained network
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(PROTO_TXT_PATH, CAFFE_MODEL_PATH);

    // tracker
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    int delay = 1000 / fps; // Calculate delay based on the framerate
    bool person_detected_first_time = false;
    int64_t start = cv::getTickCount();
    std::cout << "Video processing has started..." << std::endl;
    cv::Mat frame{};
    while (cap.read(frame)){
        frame_count++;
        
        // 1. Detect a person 
        if (!person_detected_first_time){
            person_detected_first_time = person_detected(net, frame, tracker);
            if(!person_detected_first_time){
                log_file << "FRAME " << frame_count << ": " << "No person detected yet" << std::endl;
            }
        }

        // 2. perform object tracking

        if (person_detected_first_time){
            cv::Rect rec{};
            bool tracker_update_ok = tracker->update(frame, rec);
            if (tracker_update_ok){
                // draw rectangle
                cv::rectangle(frame, rec, COLOR, 2);
                cv::putText(frame, "tracker: CSRT", cv::Point(rec.x, rec.y - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1.5);

                log_file << "FRAME " << frame_count << ": " << "Target tracked: [" 
                    << "["<<rec.x<<","<<rec.y<<"],"
                    << "["<<rec.x+rec.width<<","<<rec.y<<"],"
                    << "["<<rec.x<<","<<rec.y+rec.height<<"],"
                    << "["<<rec.x+rec.width<<","<<rec.y+rec.height<<"]"
                    << "]"
                    << std::endl;
            } else {
                log_file << "FRAME " << frame_count << ": " << "Target being tracked lost" << std::endl;
            }
        }

        output_video_writer.write(frame);
        if(showProcessing){
            cv::imshow(WIN_NAME, frame);
        }

        if (cv::waitKey(delay) == 27) { //ESC key
            break;
        }
    }

    std::cout << "Processing finished. Check output folder" << std::endl;

    log_file << "No more frames grabbed. Exiting..." << std::endl;
    log_file << "Total number of frames processed: " << frame_count << std::endl;
    int64_t end = cv::getTickCount();
    log_file << "Processing time: " << (end-start)/cv::getTickFrequency() << "s" << std::endl;
    cap.release();
    cv::destroyAllWindows();
    log_file.close();
}
