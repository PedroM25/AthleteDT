#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "usage: PedroSwissTiming <image path>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap{argv[1]};
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS); // Get the framerate of the video
    int delay = 1000 / fps; // Calculate delay based on the framerate

    while (true){
        cv::Mat frame{};
        if(!cap.read(frame)){
            std::cout << "No more frames grabbed. Exiting..." << std::endl;
            break;
        }

        cv::Mat gray{};
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::imshow("frame", gray);

        if (cv::waitKey(delay) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}