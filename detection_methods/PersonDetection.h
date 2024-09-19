#pragma once

#include <optional>
#include <opencv2/opencv.hpp>

class PersonDetection {

public:
    PersonDetection(cv::VideoCapture& cap, cv::VideoWriter& outputVideoWriter, const bool showProcessing, const int delay, std::ofstream& logFile) : 
        m_cap{cap},
        m_outputVideoWriter{outputVideoWriter},
        m_showProcessing{showProcessing},
        m_delay{delay},
        m_logFile{logFile} {}
    virtual ~PersonDetection() = default;
    
    virtual int performDetection() = 0;

protected:
    cv::VideoCapture& m_cap;
    cv::VideoWriter& m_outputVideoWriter;
    const bool m_showProcessing;
    const int m_delay;
    std::ofstream& m_logFile;

    const std::string WIN_NAME = "Output";
};
