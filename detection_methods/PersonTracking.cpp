#include "PersonTracking.h"

#include <opencv2/tracking.hpp>

namespace  {

const int IN_W = 300;
const int IN_H = 300;
const float CONF_TRSH = 0.6;
const float MEAN_SUBTRACTION_VAL = 127.5; // Result from doing 255/2
const float SCALING_FACTOR = 0.00784; // Result from doing 2/255

}

/*
 * Returns true if a person is detected; false otherwise
 */
bool personDetected(cv::dnn::Net& net, const cv::Mat& frame, cv::Ptr<cv::Tracker>& tracker, const std::vector<std::string>& classNames, std::ofstream& logFile, const int& frameCount){
    bool success = false;
    cv::Mat blob = cv::dnn::blobFromImage(frame, SCALING_FACTOR, cv::Size(IN_W, IN_H), MEAN_SUBTRACTION_VAL);
    net.setInput(blob);
    cv::Mat detections = net.forward();

    cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++){
        float confidence = detectionMat.at<float>(i, 2);
        int class_id = detectionMat.at<float>(i, 1);

        std::string label = classNames[class_id] + " : " + std::to_string(confidence);

        if (classNames[class_id] == "person" && confidence > CONF_TRSH){
            success = true;

            // Object location
            int xLeftTop = detectionMat.at<float>(i, 3) * frame.cols;
            int yLeftTop = detectionMat.at<float>(i, 4) * frame.rows;
            int xRightBottom = detectionMat.at<float>(i, 5) * frame.cols;
            int yRightBottom = detectionMat.at<float>(i, 6) * frame.rows;
            
            logFile << "FRAME " << frameCount << ": " << "Person detected, confidence: " << confidence << ", coordinates: [" 
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

int PersonTracking::performDetection(){

    int retFrameCount = 0;
    // pre trained network
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(PROTO_TXT_PATH, CAFFE_MODEL_PATH);

    // tracker
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    bool person_detected_first_time = false;
    
    cv::Mat frame{};
    while (m_cap.read(frame)){
        retFrameCount++;
        
        // 1. Detect a person 
        if (!person_detected_first_time){
            person_detected_first_time = personDetected(net, frame, tracker, m_classNames, m_logFile, retFrameCount);
            if(!person_detected_first_time){
                m_logFile << "FRAME " << retFrameCount << ": " << "No person detected yet" << std::endl;
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

                m_logFile << "FRAME " << retFrameCount << ": " << "Target tracked: [" 
                    << "["<<rec.x<<","<<rec.y<<"],"
                    << "["<<rec.x+rec.width<<","<<rec.y<<"],"
                    << "["<<rec.x<<","<<rec.y+rec.height<<"],"
                    << "["<<rec.x+rec.width<<","<<rec.y+rec.height<<"]"
                    << "]"
                    << std::endl;
            } else {
                m_logFile << "FRAME " << retFrameCount << ": " << "Target being tracked lost" << std::endl;
            }
        }

        m_outputVideoWriter.write(frame);
        if(m_showProcessing){
            cv::imshow(WIN_NAME, frame);
        }

        if (cv::waitKey(m_delay) == 27) { //ESC key
            break;
        }
    }

    return retFrameCount;
}
