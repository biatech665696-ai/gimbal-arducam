#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>

int main() {
    // Capture an image using rpicam-still
    int ret = std::system("rpicam-still -t 1000 -o test.jpg");
    if (ret != 0) {
        std::cerr << "Failed to capture image with rpicam-still." << std::endl;
        return 1;
    }

    // Load and display the image
    cv::Mat img = cv::imread("test.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }
    cv::imshow("Camera Frame", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
