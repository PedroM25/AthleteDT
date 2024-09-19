#pragma once

#include "PersonDetection.h"

#include "Utils.h"

class PersonTracking final : public PersonDetection
{
public:
    PersonTracking(cv::VideoCapture &cap, cv::VideoWriter &outputVideoWriter, const bool showProcessing, const int delay, std::ofstream &logFile) 
        : PersonDetection(cap, outputVideoWriter, showProcessing, delay, logFile){
            std::ignore = utils::readLinesIntoVector(CLASS_NAMES_PATH, m_classNames);
        };
    virtual ~PersonTracking() = default;

    int performDetection();
private:
    const std::string CLASS_NAMES_PATH = "model/MobileNet-SSDCaffe/object_detection_classes_voc0712.txt";
    const std::string PROTO_TXT_PATH = "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.prototxt";
    const std::string CAFFE_MODEL_PATH = "model/MobileNet-SSDCaffe/MobileNetSSD_deploy.caffemodel";

    const cv::Scalar COLOR{0,255,0};
    std::vector<std::string> m_classNames{};
};

