#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string pipeline =
        "libcamerasrc ! "
        "video/x-raw,width=1920,height=1080,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open Arducam 64MP with GStreamer pipeline\n";
        return -1;
    }

    cv::Mat frame;

    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "ERROR: Cannot read frame\n";
            break;
        }

        cv::imshow("Arducam 64MP", frame);

        if (cv::waitKey(1) == 27) {  // ESC
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;