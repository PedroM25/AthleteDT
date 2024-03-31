#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

const double CONF_TRSH = 0.5;
const int IN_W = 300;
const int IN_H = 300;
const float MEAN_SUBTRACTION_VAL = 127.5; // Result from doing 255/2
const float SCALING_FACTOR = 0.00784; // Result from doing 2/255
const float SECONDS_BETW_DETECTIONS = 3;
const std::vector<std::string> CLASS_NAMES{"background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                                            "bus", "car", "cat", "chair", "cow", "diningtable",
                                            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                            "sofa", "train", "tvmonitor"};

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "usage: PedroSwissTiming <video path>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap{argv[1]};
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // calculate delay for frame display in real-time
    double fps = cap.get(cv::CAP_PROP_FPS); // Get the framerate of the video
    int delay = 1000 / fps; // Calculate delay based on the framerate

    // pre trained network
    cv::dnn::Net net = cv::dnn::readNetFromCaffe("model/MobileNet-SSDCaffe/MobileNetSSD_deploy.prototxt",
                "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.caffemodel");

    // tracker
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    bool person_detected_first_time = false;
    cv::Mat frame{};
    while (cap.read(frame)){
        
        // 1. Detect a person 
        // If model detects more than one person, first detected will be considered
        if (!person_detected_first_time){
            cv::Mat blob = cv::dnn::blobFromImage(frame, SCALING_FACTOR, cv::Size(IN_W, IN_H), MEAN_SUBTRACTION_VAL);
            net.setInput(blob);
            cv::Mat detections = net.forward();

            cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
            for (int i = 0; i < detectionMat.rows; i++){
                float confidence = detectionMat.at<float>(i, 2);
                int class_id = detectionMat.at<float>(i, 1);

                std::string label = CLASS_NAMES[class_id] + " : " + std::to_string(confidence);
                std::cout << "[DEBUG] " << "Detected " << label << std::endl;

                if (CLASS_NAMES[class_id] == "person" && confidence > CONF_TRSH){
                    person_detected_first_time = true; // CAN NOW PERFORM TRACKING

                    // Object location
                    int xLeftTop = detectionMat.at<float>(i, 3) * frame.cols;
                    int yLeftTop = detectionMat.at<float>(i, 4) * frame.rows;
                    int xRightBottom = detectionMat.at<float>(i, 5) * frame.cols;
                    int yRightBottom = detectionMat.at<float>(i, 6) * frame.rows;


                    cv::Point leftTopCoords{xLeftTop, yLeftTop};
                    cv::Point rightBottomCoords{xRightBottom, yRightBottom};
                    cv::Rect rec{leftTopCoords, rightBottomCoords};
                    // draw rectangle
                    cv::rectangle(frame, rec, cv::Scalar(0, 255, 0), 2);
                    
                    // init tracker
                    tracker->init(frame, rec);

                    // Draw label and confidence of prediction in frame
                    cv::putText(frame, label, cv::Point(xLeftTop, yLeftTop - 15),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0),1.5);
                    
                    cv::imshow("Original video", frame);
                    cv::waitKey(0);
                    break; //first one detected, others ignored
                }
            }
        }

        // 2. perform object tracking

        if (person_detected_first_time){
            cv::Rect rec{};
            bool tracker_update_ok = tracker->update(frame, rec);
            if (tracker_update_ok){
                // draw rectangle
                cv::rectangle(frame, rec, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Original video", frame);

        if (cv::waitKey(delay) == 27) { //ESC key
            break;
        }
    }

    std::cout << "No more frames grabbed. Exiting..." << std::endl;
    cap.release();
    cv::destroyAllWindows();
}