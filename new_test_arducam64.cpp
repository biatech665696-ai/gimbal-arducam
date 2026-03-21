#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <cmath>

int main() {
    cv::setUseOptimized(true);
    cv::setNumThreads(0);
    std::string pipeline =
        "libcamerasrc ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1,interlace-mode=progressive ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=1 drop=true";

    // Physical scene dimensions assumption (meters)
    const float SCENE_WIDTH_METERS = 200.0f;
    const float SCENE_HEIGHT_METERS = 200.0f; // adjust if different aspect needed
    const float HFOV_DEG = 70.0f; // horizontal field of view in degrees (adjust per lens)

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open Arducam 64MP with GStreamer pipeline\n";
        return -1;
    }


    cv::Mat frame, fgMask;
    std::vector<std::vector<cv::Point>> contours;
    auto bg = cv::createBackgroundSubtractorMOG2(500, 16, true);
    const double scale = 0.4;  // process at ~40% resolution for more speed
    cv::Mat small, smallBGR, smallMask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    int frameCount = 0;

    struct Track {
        int id;
        cv::Point2f pos;   // in full-resolution coordinates
        int age;            // frames seen
        int missed;         // consecutive missed frames
    };
    std::vector<Track> tracks;
    int nextTrackId = 1;
    const float matchThreshold = 30.0f; // pixels in full-res; small objects proximity
    const int maxMissed = 10;

    while (true) {
        // Increment frame counter for periodic logging and potential time-based calculations
        frameCount++;
        if (!cap.read(frame)) {
            std::cerr << "ERROR: Cannot read frame\n";
            break;
        }

        // Process every frame to minimize latency

        cv::resize(frame, small, cv::Size(), scale, scale, cv::INTER_AREA);
        // Use BGR input for MOG2
        smallBGR = small; // frame already in BGR
        bg->apply(smallBGR, smallMask, 0.015);
        cv::threshold(smallMask, smallMask, 200, 255, cv::THRESH_BINARY); // isolate strong foreground
        cv::morphologyEx(smallMask, smallMask, cv::MORPH_OPEN, kernel);
        cv::dilate(smallMask, smallMask, kernel, cv::Point(-1, -1), 1);
        cv::findContours(smallMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Identify the smallest moving object above a minimal area
        double minArea = 2; // allow smaller objects
        int smallestIdx = -1;
        double smallestArea = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < contours.size(); ++i) {
            double a = cv::contourArea(contours[i]);
            if (a >= minArea && a < smallestArea) {
                smallestArea = a;
                smallestIdx = static_cast<int>(i);
            }
        }

        // Compute centroids (full-res) for all tiny moving objects
        std::vector<cv::Point2f> detections;
        std::vector<cv::Rect> detRects;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area < 10) continue; // reduced noise floor
            cv::Moments m = cv::moments(contours[i]);
            if (m.m00 == 0) continue;
            cv::Point2f cSmall(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
            cv::Point2f cFull(cSmall.x / static_cast<float>(scale), cSmall.y / static_cast<float>(scale));
            detections.push_back(cFull);
            cv::Rect rSmall = cv::boundingRect(contours[i]);
            detRects.emplace_back(int(rSmall.x / scale), int(rSmall.y / scale), int(rSmall.width / scale), int(rSmall.height / scale));
        }

        // Simple nearest-neighbor data association to maintain IDs
        std::vector<int> detAssigned(detections.size(), -1);
        for (size_t t = 0; t < tracks.size(); ++t) {
            int bestIdx = -1;
            float bestDist = matchThreshold;
            for (size_t d = 0; d < detections.size(); ++d) {
                if (detAssigned[d] != -1) continue;
                float dist = cv::norm(detections[d] - tracks[t].pos);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<int>(d);
                }
            }
            if (bestIdx != -1) {
                // Update track
                tracks[t].pos = detections[bestIdx];
                tracks[t].age++;
                tracks[t].missed = 0;
                detAssigned[bestIdx] = static_cast<int>(t);
            } else {
                // Missed this frame
                tracks[t].missed++;
            }
        }
        // Create new tracks for unmatched detections
        for (size_t d = 0; d < detections.size(); ++d) {
            if (detAssigned[d] == -1) {
                tracks.push_back(Track{nextTrackId++, detections[d], 1, 0});
            }
        }
        // Remove stale tracks
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const Track& tr){ return tr.missed > maxMissed; }), tracks.end());

        // Scene center for relative coordinate calculation
        cv::Point2f sceneCenter(static_cast<float>(frame.cols) / 2.0f, static_cast<float>(frame.rows) / 2.0f);

        // Draw detections and IDs; smallest stays red
        for (size_t i = 0, di = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area < 10) continue; // draw smaller objects
            cv::Rect rect = cv::boundingRect(contours[i]);
            cv::Rect fullRect(int(rect.x / scale), int(rect.y / scale), int(rect.width / scale), int(rect.height / scale));
            cv::Scalar color = (static_cast<int>(i) == smallestIdx) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(frame, fullRect, color, 2);
        }
        // Compute pixel->meter scale (center origin, Y positive upward)
        float metersPerPixelX = SCENE_WIDTH_METERS / static_cast<float>(frame.cols);
        float metersPerPixelY = SCENE_HEIGHT_METERS / static_cast<float>(frame.rows);

        // Compute focal lengths in pixels from FOV
        float hfov_rad = HFOV_DEG * static_cast<float>(CV_PI) / 180.0f;
        float vfov_rad = 2.0f * std::atan(std::tan(hfov_rad * 0.5f) * (static_cast<float>(frame.rows) / static_cast<float>(frame.cols)));
        float fx_pix = (static_cast<float>(frame.cols) * 0.5f) / std::tan(hfov_rad * 0.5f);
        float fy_pix = (static_cast<float>(frame.rows) * 0.5f) / std::tan(vfov_rad * 0.5f);

        // Overlay track IDs and physical coordinates (X,Y in meters) and angles
        for (const auto& tr : tracks) {
            float dxPixels = tr.pos.x - sceneCenter.x;
            float dyPixels = tr.pos.y - sceneCenter.y;
            float xMeters = dxPixels * metersPerPixelX;
            float yMeters = -dyPixels * metersPerPixelY; // invert so up is +
            float ax_deg = std::atan2(dxPixels, fx_pix) * 180.0f / static_cast<float>(CV_PI);
            float ay_deg = std::atan2(-dyPixels, fy_pix) * 180.0f / static_cast<float>(CV_PI);
            std::ostringstream oss;
            oss.setf(std::ios::fixed); oss.precision(1);
            oss << tr.id << " (" << xMeters << "m," << yMeters << "m)" << " [" << ax_deg << "deg," << ay_deg << "deg]";
            cv::putText(frame, oss.str(), cv::Point(int(tr.pos.x), int(tr.pos.y)), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 0), 1);
            cv::circle(frame, cv::Point(int(tr.pos.x), int(tr.pos.y)), 2, cv::Scalar(255, 255, 0), -1);
        }

        // Print coordinates to terminal every 15 frames to reduce spam
        if (frameCount % 15 == 0 && !tracks.empty()) {
            std::cout.setf(std::ios::fixed); std::cout.precision(1);
            std::cout << "Frame " << frameCount << " | Tracks: " << tracks.size() << std::endl;
            for (size_t i = 0; i < tracks.size(); ++i) {
                float dxPixels = tracks[i].pos.x - sceneCenter.x;
                float dyPixels = tracks[i].pos.y - sceneCenter.y;
                float xMeters = dxPixels * metersPerPixelX;
                float yMeters = -dyPixels * metersPerPixelY;
                float ax_deg = std::atan2(dxPixels, fx_pix) * 180.0f / static_cast<float>(CV_PI);
                float ay_deg = std::atan2(-dyPixels, fy_pix) * 180.0f / static_cast<float>(CV_PI);
                std::cout << "id=" << tracks[i].id
                          << " x=" << xMeters << "m y=" << yMeters << "m"
                          << " ax=" << ax_deg << "deg ay=" << ay_deg << "deg"
                          << std::endl;
            }
        }

        // Unified periodic logging: id, meters, angles
        if (frameCount % 15 == 0 && !tracks.empty()) {
            std::cout.setf(std::ios::fixed); std::cout.precision(1);
            std::cout << "Frame " << frameCount << " | Tracks: " << tracks.size() << std::endl;
            for (const auto& tr : tracks) {
                float dxPixels = tr.pos.x - sceneCenter.x;
                float dyPixels = tr.pos.y - sceneCenter.y;
                float xMeters = dxPixels * metersPerPixelX;
                float yMeters = -dyPixels * metersPerPixelY;
                float ax_deg = std::atan2(dxPixels, fx_pix) * 180.0f / static_cast<float>(CV_PI);
                float ay_deg = std::atan2(-dyPixels, fy_pix) * 180.0f / static_cast<float>(CV_PI);
                std::cout << "id=" << tr.id
                          << " x=" << xMeters << "m y=" << yMeters << "m"
                          << " ax=" << ax_deg << "deg ay=" << ay_deg << "deg" << std::endl;
            }
        }
        cv::imshow("Arducam 64MP", frame);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {  // ESC or 'q' or 'Q'
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}