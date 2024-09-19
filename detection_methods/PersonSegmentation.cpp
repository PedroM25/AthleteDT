#include "PersonSegmentation.h"

namespace {

const float CONF_TRSH = 0.8;
const float MASK_TRSH = 0.3;

}

void drawMaskAndBBox(cv::Mat& frame, int classId, float conf, cv::Rect& bbox, cv::Mat& objectMask, const std::vector<std::string>& classNames, const cv::Scalar& color){
    // bounding box
    rectangle(frame, cv::Point(bbox.x, bbox.y), cv::Point(bbox.x+bbox.width, bbox.y+bbox.height), color, 3);

    // label
    std::string label = classNames[classId] + " : " + std::to_string(conf);
    putText(frame, label, cv::Point(bbox.x, bbox.y - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);

    // preprare mask: resize, threshold, color
    cv::resize(objectMask, objectMask, cv::Size(bbox.width, bbox.height));
    cv::Mat mask = objectMask > MASK_TRSH;
    cv::Mat coloredRoi = 0.3 * color + 0.7 * frame(bbox);
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    // prepare contours
    std::vector<cv::Mat> contours;
    cv::Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(coloredRoi, contours, -1, color, 5, cv::LINE_8, hierarchy, 100);

    // apply
    coloredRoi.copyTo(frame(bbox), mask);
}

void postProcess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std::vector<std::string>& classNames, std::ofstream& logFile, const int& frameCount, const cv::Scalar& color){
    cv::Mat outDetections = outs[0];
    cv::Mat outMasks = outs[1];

    int numDetections = outDetections.size[2];

    outDetections = outDetections.reshape(1, outDetections.total() / 7);
    for (int i = 0; i < numDetections; i++){
        float confidence = outDetections.at<float>(i, 2);
        int classId = outDetections.at<float>(i, 1);

        if (confidence > CONF_TRSH && classNames[classId] == "person"){
            // Extract the bounding box
            int xLeftTop = outDetections.at<float>(i, 3) * frame.cols;
            int yLeftTop =  outDetections.at<float>(i, 4) * frame.rows;
            int xRightBottom = outDetections.at<float>(i, 5) * frame.cols;
            int yRightBottom = outDetections.at<float>(i, 6) * frame.rows;
			
            logFile << "FRAME " << frameCount << ": " << "Person detected, confidence: " << confidence << ", coordinates: [" 
                << "["<<xLeftTop<<","<<yLeftTop<<"],"
                << "["<<xRightBottom<<","<<yLeftTop<<"],"
                << "["<<xLeftTop<<","<<yRightBottom<<"],"
                << "["<<xRightBottom<<","<<yRightBottom<<"]"
                << "]"
                << std::endl;

            // keep boxes inside of bounds
            xLeftTop = std::max(0, std::min(xLeftTop, frame.cols -1));
            yLeftTop = std::max(0, std::min(yLeftTop, frame.rows -1));
            xRightBottom = std::max(0, std::min(xRightBottom, frame.cols -1));
            yRightBottom = std::max(0, std::min(yRightBottom, frame.rows -1));
            
            cv::Rect box{xLeftTop, yLeftTop, xRightBottom - xLeftTop + 1, yRightBottom - yLeftTop + 1};

            // extract mask
            cv::Mat objectMask{outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i,classId)};

            drawMaskAndBBox(frame, classId, confidence, box, objectMask, classNames, color);
            break; //first person detected only; others are ignored
        }
    }
				   
}

int PersonSegmentation::performDetection(){
    int retFrameCount = 0;

    // pre trained network
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH);
    cv::Mat frame{};
    while (m_cap.read(frame)){
        retFrameCount++;
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1., cv::Size(frame.cols, frame.rows), cv::Scalar(), true, false);
        net.setInput(blob);
										 
        std::vector<std::string> outNames{"detection_out_final", "detection_masks"};
        std::vector<cv::Mat> outs{};
        net.forward(outs, outNames);

        postProcess(frame, outs, m_classNames, m_logFile, retFrameCount, COLOR);

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