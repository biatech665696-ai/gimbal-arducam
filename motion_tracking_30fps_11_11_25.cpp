// C++ translation of motion_tracking_30fps_11_11_25.py
// Features:
// - Fisheye-safe undistort (falls back on error)
// - Save G-channel RAW frames to raw_capture
// - Farneback optical flow on downscaled image to compute avg_speed
// - Dynamic sensitivity: threshold and min contour area depend on avg_speed
// - BackgroundSubtractorMOG2 tuned for sensitivity (history=200, varThreshold=16, no shadows)
// - Simple per-frame contour-index trails with fading and prediction
// - UDP JSON output: id,x,y,w,h,angle_h,angle_v

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <iomanip>
#include <unordered_map>
#include <filesystem>
#include <cmath>
#include <cstring>

#ifdef _WIN32
  #define _WINSOCK_DEPRECATED_NO_WARNINGS
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")
#else
  #include <sys/socket.h>
  #include <arpa/inet.h>
  #include <unistd.h>
#endif

using namespace cv;
using namespace std;

// ---------------- UDP parameters ----------------
static const char* UDP_IP = "192.168.1.100"; // PC IP
static const int UDP_PORT = 5005;

struct UdpSender {
#ifdef _WIN32
    SOCKET sock = INVALID_SOCKET;
#else
    int sock = -1;
#endif
    sockaddr_in addr{};
    bool ok = false;

    UdpSender() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
            cerr << "WSAStartup failed\n";
            return;
        }
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCKET) {
            cerr << "UDP socket creation failed\n";
            return;
        }
#else
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            cerr << "UDP socket creation failed\n";
            return;
        }
#endif
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(UDP_PORT);
        addr.sin_addr.s_addr = inet_addr(UDP_IP);
        ok = true;
    }

    ~UdpSender() {
#ifdef _WIN32
        if (sock != INVALID_SOCKET) closesocket(sock);
        WSACleanup();
#else
        if (sock >= 0) close(sock);
#endif
    }

    bool sendStr(const std::string& s) {
        if (!ok) return false;
        int r = sendto(sock, s.c_str(), (int)s.size(), 0, (sockaddr*)&addr, (socklen_t)sizeof(addr));
        return r >= 0;
    }
};

// ---------------- Camera settings ----------------
static const int CAM_INDEX = 0;
static const int FPS = 30;
static const int FRAME_WIDTH = 9152;  // match Python
static const int FRAME_HEIGHT = 6944; // match Python

// ---------------- Trail settings ----------------
static const int TRAIL_LENGTH = 20; // number of points to keep in trail
static const int TRAIL_FADE = 10;   // fade step

// ---------------- RAW folder ----------------
static const string RAW_DIR = "raw_capture";

static bool ensure_dir(const string& path) {
    try {
        std::filesystem::create_directories(path);
        return true;
    } catch (const std::exception& e) {
        cerr << "Failed to create directory '" << path << "': " << e.what() << "\n";
        return false;
    }
}

static bool save_raw_g_channel(const Mat& frame, const string& dir, int frame_id) {
    vector<Mat> ch;
    split(frame, ch);
    if (ch.size() < 2) return false;
    const Mat& g = ch[1]; // CV_8UC1
    ostringstream oss;
    oss << dir << "/frame_" << setw(6) << setfill('0') << frame_id << ".raw";
    string path = oss.str();
    ofstream ofs(path, ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(g.data), (std::streamsize)(g.total() * g.elemSize()));
    return ofs.good();
}

// ---------------- Fisheye-safe undistort ----------------
static Mat undistort_fisheye_safe(const Mat& frame) {
    try {
        int h = frame.rows, w = frame.cols;
        Mat K = (Mat_<float>(3,3) << w/2.0f, 0, w/2.0f,
                                     0,      w/2.0f, h/2.0f,
                                     0,      0,      1);
        Mat D = Mat::zeros(4, 1, CV_32F);
        Mat R = Mat::eye(3, 3, CV_32F);
        Mat map1, map2, out;
        fisheye::initUndistortRectifyMap(K, D, R, K, Size(w, h), CV_16SC2, map1, map2);
        remap(frame, out, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
        return out;
    } catch (...) {
        return frame.clone();
    }
}

// ---------------- Pixel to angle ----------------
static pair<double,double> pixel_to_angle(double x, double y, double width, double height, double fov_horizontal=180.0, double fov_vertical=180.0) {
    double cx = width / 2.0;
    double cy = height / 2.0;
    double nx = (x - cx) / cx;
    double ny = (y - cy) / cy;
    double angle_x = nx * (fov_horizontal / 2.0);
    double angle_y = ny * (fov_vertical / 2.0);
    return {angle_x, angle_y};
}

int main() {
    ensure_dir(RAW_DIR);

    VideoCapture cap(CAM_INDEX);
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap.set(CAP_PROP_FPS, FPS);
    if (!cap.isOpened()) {
        cerr << "Failed to open camera index " << CAM_INDEX << "\n";
        return 1;
    }

    // Background subtractor tuned for sensitivity
    Ptr<BackgroundSubtractorMOG2> fgbg = createBackgroundSubtractorMOG2(200, 16, false);

    int frame_id = 0;
    Mat prev_gray_for_flow; // G-channel of previous corrected frame
    int flow_scale = 2;     // downscale factor

    UdpSender udp;

    Mat frame;
    while (true) {
        if (!cap.read(frame)) break;
        frame_id++;

        // Save RAW G-channel
        save_raw_g_channel(frame, RAW_DIR, frame_id);

        // Fisheye correction (safe)
        Mat corrected = undistort_fisheye_safe(frame);

        // Take G channel for processing
        vector<Mat> ch;
        split(corrected, ch);
        Mat gray = (ch.size() >= 2) ? ch[1] : corrected;

        // Farneback optical flow on downscaled images to compute avg_speed
        double avg_speed = 0.0; // average magnitude
        try {
            if (!prev_gray_for_flow.empty()) {
                Size smallSize(gray.cols / flow_scale, gray.rows / flow_scale);
                if (smallSize.width > 0 && smallSize.height > 0) {
                    Mat small_prev, small_curr;
                    resize(prev_gray_for_flow, small_prev, smallSize, 0, 0, INTER_AREA);
                    resize(gray,              small_curr, smallSize, 0, 0, INTER_AREA);
                    Mat flow;
                    calcOpticalFlowFarneback(small_prev, small_curr, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
                    Mat flowXY[2];
                    split(flow, flowXY);
                    Mat mag, ang;
                    cartToPolar(flowXY[0], flowXY[1], mag, ang);
                    Scalar meanMag = mean(mag);
                    avg_speed = meanMag[0];
                }
            }
            prev_gray_for_flow = gray.clone();
        } catch (...) {
            avg_speed = 0.0;
        }

        // Dynamic sensitivity
        int dynamic_thresh = (int)std::lround(std::clamp(50.0 - avg_speed * 10.0, 10.0, 100.0));
        int dynamic_min_area = (int)std::lround(std::clamp(800.0 - avg_speed * 50.0, 200.0, 5000.0));

        // Foreground / motion
        Mat fgmask;
        fgbg->apply(gray, fgmask);
        Mat thresh;
        threshold(fgmask, thresh, dynamic_thresh, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        struct Obj { int id; int x; int y; int w; int h; double angle_h; double angle_v; int cx; int cy; };
        vector<Obj> objects;

        static unordered_map<int, deque<Point>> trails; // per-frame index based

        for (int i = 0; i < (int)contours.size(); ++i) {
            const auto& cnt = contours[i];
            double area = contourArea(cnt);
            if (area < dynamic_min_area) continue;

            Rect r = boundingRect(cnt);
            int cx = r.x + r.width / 2;
            int cy = r.y + r.height / 2;
            auto ang = pixel_to_angle((double)cx, (double)cy, (double)corrected.cols, (double)corrected.rows);

            objects.push_back(Obj{i, r.x, r.y, r.width, r.height, ang.first, ang.second, cx, cy});

            if (!trails.count(i)) trails[i] = deque<Point>();
            auto& dq = trails[i];
            dq.push_back(Point(cx, cy));
            while ((int)dq.size() > TRAIL_LENGTH) dq.pop_front();

            // Draw rectangle and center
            rectangle(corrected, r, Scalar(0, 255, 0), 2);
            circle(corrected, Point(cx, cy), 4, Scalar(0, 0, 255), FILLED);

            // Draw trail with fading
            for (size_t j = 1; j < dq.size(); ++j) {
                int alpha = max(0, 255 - (int)(dq.size() - j) * TRAIL_FADE);
                line(corrected, dq[j-1], dq[j], Scalar(255, alpha, alpha), 2);
            }

            // Prediction from last motion vector
            if (dq.size() >= 2) {
                Point p1 = dq[dq.size()-1];
                Point p0 = dq[dq.size()-2];
                Point pred(p1.x + (p1.x - p0.x) * 5, p1.y + (p1.y - p0.y) * 5);
                circle(corrected, pred, 6, Scalar(255, 255, 0), 2);
            }
        }

        // UDP send JSON if objects exist
        if (!objects.empty()) {
            ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << "[";
            for (size_t i = 0; i < objects.size(); ++i) {
                const auto& o = objects[i];
                if (i) oss << ",";
                oss << "{"
                    << "\"id\":" << o.id << ","
                    << "\"x\":" << o.x << ","
                    << "\"y\":" << o.y << ","
                    << "\"w\":" << o.w << ","
                    << "\"h\":" << o.h << ","
                    << "\"angle_h\":" << setprecision(6) << o.angle_h << ","
                    << "\"angle_v\":" << setprecision(6) << o.angle_v
                    << "}";
            }
            oss << "]";
            string payload = oss.str();
            udp.sendStr(payload);
            try {
                cout << "[Frame " << frame_id << "] Sent " << objects.size() << " objects to " << UDP_IP << ":" << UDP_PORT << "\n";
                cout << payload << "\n";
            } catch (...) {
                cerr << "Failed to print payload\n";
            }
        }

        imshow("Motion Tracking Fisheye", corrected);
        int key = waitKey(1) & 0xFF;
        if (key == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
