// arducam64.cpp
// Program to display images from the Arducam 1/1.32" 64MP Auto Focus Camera Module
// Requirements: OpenCV, Arducam SDK/driver installed

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open the default camera (usually /dev/video0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Set resolution (adjust as needed for your camera)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 4624); // Example: 64MP width
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 6944); // Example: 64MP height

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        cv::imshow("Arducam 64MP", frame);
        // Press 'q' to quit
        if (cv::waitKey(1) == 'q') break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
