#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <iostream>

int main() {
    // Start rpicam-vid in the background to stream MJPEG to the pipe
    std::system("pkill rpicam-vid"); // Stop any previous instance
    //std::system("rpicam-vid -t 0 --codec mjpeg -o /tmp/camera_pipe.mjpeg --width 1280 --height 720 --framerate 30 --listen &");
    std::system("rpicam-vid -t 0 --codec mjpeg -o /tmp/camera_pipe.mjpeg --width 1280 --height 720 --framerate 30 &");
    std::this_thread::sleep_for(std::chrono::seconds(1)); // Give time for camera to start

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
    std::system("pkill rpicam-vid");
    return 0;
}
