#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <iostream>

int main() {
    // Open the MJPEG stream from the pipe
    cv::VideoCapture cap("/tmp/camera_pipe.mjpeg");
    if (!cap.isOpened()) {
        std::cerr << "Failed to open MJPEG stream from pipe." << std::endl;
        return 1;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to read frame from pipe." << std::endl;
            break;
        }
        cv::imshow("Arducam Live Stream", frame);
        if (cv::waitKey(1) == 27) break; // ESC to exit
    }

    cap.release();
    return 0;
}
