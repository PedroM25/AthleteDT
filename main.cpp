#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

const int IN_W = 300, IN_H = 300;
const float MEAN_SUBTRACTION_VAL = 127.5; // Result from doing 255/2
const float SCALING_FACTOR = 0.00784; // Result from doing 2/255
const float CONF_THRESHOLD = 0.05;
const std::string PERSON_CLASS = "person";
//const float WH_RATIO = IN_W / (float)IN_H;
const std::vector<std::string> CLASS_NAMES= {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

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

    double fps = cap.get(cv::CAP_PROP_FPS); // Get the framerate of the video
    int delay = 1000 / fps; // Calculate delay based on the framerate

    // Load pre-trained SSD model
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow("model/MobileNet-SSDv1/frozen_inference_graph.pb", "model/MobileNet-SSDv1/config.pbtxt");

    cv::Mat frame{};
    while (cap.read(frame)){

        //preprocess

        /*
        cv::Size frame_size = frame.size();
        cv::Size cropSize;
        if (frame_size.width / (float)frame_size.height > WH_RATIO) {
            cropSize = cv::Size(static_cast<int>(frame_size.height * WH_RATIO), frame_size.height);
        } else {
            cropSize = cv::Size(frame_size.width, static_cast<int>(frame_size.width / WH_RATIO));
        }
        cv::Rect crop(cv::Point((frame_size.width - cropSize.width) / 2, (frame_size.height - cropSize.height) / 2), cropSize);
        */

        cv::Mat frame_resized{};
        cv::resize(frame,frame_resized,cv::Size(IN_W, IN_H));
        //std::cout << "[DEBUG] " << "frame size: " << frame.size << std::endl;
        //std::cout << "[DEBUG] " << "frame_resized size: " << frame_resized.size << std::endl;

        cv::Mat blob = cv::dnn::blobFromImage(frame_resized, SCALING_FACTOR, cv::Size(IN_W, IN_H), cv::Scalar(MEAN_SUBTRACTION_VAL,MEAN_SUBTRACTION_VAL,MEAN_SUBTRACTION_VAL), false);

        std::cout << "[DEBUG] " << "blob size: " << blob.size << std::endl;
        net.setInput(blob);
        cv::Mat detections = net.forward();
        //std::cout << "[DEBUG] " << "detections size: " << detections.size << std::endl;
        std::cout << "[DEBUG] " << "detections.size(): " << detections.size() << std::endl;

        cv::Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        //frame = frame(crop);
        for (int i = 0; i < detectionMat.rows; i++){
            float confidence = detectionMat.at<float>(i, 2);
            std::cout << "[DEBUG] " << "confidence: " << confidence << std::endl;

            if (confidence > CONF_THRESHOLD){
                //std::cout << "[DEBUG] " << confidence << " of confidence! Continuing..." << std::endl;

                // class label
                int class_id = detectionMat.at<float>(i, 1);

                // Object location 
                int xLeftBottom = detectionMat.at<float>(i, 3) * frame_resized.cols;
                int yLeftBottom = detectionMat.at<float>(i, 4) * frame_resized.rows;
                int xRightTop   = detectionMat.at<float>(i, 5) * frame_resized.cols;
                int yRightTop   = detectionMat.at<float>(i, 6) * frame_resized.rows;
                
                // Factor for scale to original size of frame
                float heightFactor = frame.size[0]/(float)IN_H;
                float widthFactor = frame.size[1]/(float)IN_W;
                // Scale object detection to frame
                xLeftBottom = widthFactor * xLeftBottom;
                yLeftBottom = heightFactor * yLeftBottom;
                xRightTop   = widthFactor * xRightTop;
                yRightTop   = heightFactor * yRightTop;

                // draw rectangle
                cv::rectangle(frame, cv::Point(xLeftBottom, yLeftBottom), cv::Point(xRightTop, yRightTop), cv::Scalar(0, 255, 0), 2);
                
                std::cout << "[DEBUG] " << "label: " << CLASS_NAMES[class_id] << std::endl;

                // Draw label and confidence of prediction in frame resized
                    std::string label = CLASS_NAMES[class_id] + " : " + std::to_string(confidence);
                    int baseline = 0;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

                    yLeftBottom = std::max(yLeftBottom, labelSize.width);
                    cv::rectangle(frame, cv::Point(xLeftBottom, yLeftBottom - labelSize.width), 
                                            cv::Point(xLeftBottom + labelSize.height, yLeftBottom + baseline), 
                                            cv::Scalar(255, 255, 255), 
                                            cv::FILLED);
                    cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        // TODO

        cv::imshow("Original video", frame);

        if (cv::waitKey(delay) == 27) { //ESC key
            break;
        }
    }

    std::cout << "No more frames grabbed. Exiting..." << std::endl;
    cap.release();
    cv::destroyAllWindows();
}