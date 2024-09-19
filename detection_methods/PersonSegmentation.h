#pragma once

#include "PersonDetection.h"

#include "Utils.h"

class PersonSegmentation final : public PersonDetection
{
public:
    PersonSegmentation(cv::VideoCapture &cap, cv::VideoWriter &outputVideoWriter, bool showProcessing, int delay, std::ofstream &logFile) 
        : PersonDetection(cap, outputVideoWriter, showProcessing, delay, logFile){
            std::ignore = utils::readLinesIntoVector(CLASS_NAMES_PATH, m_classNames);
        };
    virtual ~PersonSegmentation() = default;

    int performDetection();
private:
    const std::string CLASS_NAMES_PATH = "model/mask-rcnn-coco/object_detection_classes_coco.txt";
    const std::string MODEL_WEIGHTS_PATH = "model/mask-rcnn-coco/frozen_inference_graph.pb";
    const std::string MODEL_CONFIG_PATH = "model/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
    
    const cv::Scalar COLOR{0,255,0};
    std::vector<std::string> m_classNames{};
};

