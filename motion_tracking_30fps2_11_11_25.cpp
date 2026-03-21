// Translated from Python script #motion_tracking_30fps2_11_11_25.py
// Assumption: Using C++17 and OpenCV C++ API (no stable pure C API for modern OpenCV).
// Functionality: motion detection with fisheye correction, simple tracking, speed calc,
// UDP JSON payloads, trails, and on-demand 2-point calibration.

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <chrono>


#ifdef _WIN32
  #define _WINSOCK_DEPRECATED_NO_WARNINGS
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")
#else
  #include <sys/socket.h>
  #include <arpa/inet.h>
  #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
#endif

using namespace cv;
using namespace std;

// ---------------- UDP parameters ----------------
static const char* UDP_IP = "127.0.0.1"; // Default loopback; change as needed
static const int UDP_PORT = 5005;        // Default UDP port

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
static const int CAM_INDEX = 2;
static const int FPS = 30;
static const int FRAME_WIDTH = 1920;
static const int FRAME_HEIGHT = 1080;

// ---------------- Trail settings ----------------
static const int TRAIL_LENGTH = 20; // number of points to keep in trail
static const int TRAIL_FADE = 10;   // fade step

// ---------------- RAW folder ----------------
static const string RAW_DIR = "raw_capture";

// ---------------- Speed and detection parameters ----------------
static double METERS_PER_PIXEL = 0.02; // default, refined by calibration
static const double SPEED_THRESHOLD_MPS = 0.5; // show/send if >= threshold
static const int MATCH_MAX_DISTANCE_PX = 200; // not used in IoU tracker
static const double MAX_SPEED_MPS = 100.0; // upper cap to filter artifacts
static const double MAX_BBOX_SCALE_CHANGE = 3.0; // area change filter

// ---------------- Calibration state ----------------
static bool calibrating = false;
static vector<Point> cal_points;
static const string WINDOW_NAME = "Motion Tracking Fisheye";

static void mouse_callback(int event, int x, int y, int, void*) {
    if (calibrating && event == EVENT_LBUTTONDOWN) {
        cal_points.emplace_back(x, y);
        cout << "Calibration click: (" << x << "," << y << ")\n";
    }
}

static double iouRect(const Rect& a, const Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) return 0.0;
    double inter = double(x2 - x1) * double(y2 - y1);
    double uni = double(a.width) * a.height + double(b.width) * b.height - inter;
    return (uni > 0.0) ? (inter / uni) : 0.0;
}

struct Track {
    Rect bbox{};
    Rect last_bbox{};
    double time = 0.0;
    double speed_mps = 0.0;
};

struct SimpleTracker {
    unordered_map<int, Track> tracks;
    int next_id = 0;
    double iou_threshold;

    explicit SimpleTracker(double iou_thr = 0.3) : iou_threshold(iou_thr) {}

    vector<pair<int, Track>> update(const vector<Rect>& detections, double ts) {
        unordered_map<int, Track> updated;
        vector<char> used_det(detections.size(), 0);

        // greedy match by IoU
        for (auto& kv : tracks) {
            int tid = kv.first;
            Track tr = kv.second;
            double best_iou = 0.0;
            int best_j = -1;
            for (int j = 0; j < (int)detections.size(); ++j) {
                if (used_det[j]) continue;
                double iou = iouRect(tr.bbox, detections[j]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_j = j;
                }
            }
            if (best_j >= 0 && best_iou >= iou_threshold) {
                Rect det = detections[best_j];
                used_det[best_j] = 1;
                // compute speed
                double prev_cx = tr.bbox.x + tr.bbox.width / 2.0;
                double prev_cy = tr.bbox.y + tr.bbox.height / 2.0;
                double cx = det.x + det.width / 2.0;
                double cy = det.y + det.height / 2.0;
                double dt = ts - tr.time;
                if (dt <= 0) dt = 1.0 / FPS;
                double dx_m = (cx - prev_cx) * METERS_PER_PIXEL;
                double dy_m = (cy - prev_cy) * METERS_PER_PIXEL;
                double dist_m = hypot(dx_m, dy_m);
                double speed = (dt > 0.0) ? (dist_m / dt) : 0.0;
                // outlier protections
                try {
                    double prev_area = double(tr.bbox.width) * double(tr.bbox.height);
                    double new_area = double(det.width) * double(det.height);
                    if (prev_area > 0.0 && (new_area / prev_area > MAX_BBOX_SCALE_CHANGE || prev_area / new_area > MAX_BBOX_SCALE_CHANGE)) {
                        speed = 0.0;
                        det = tr.bbox;
                    }
                } catch (...) {
                    // ignore
                }
                if (speed > MAX_SPEED_MPS) speed = MAX_SPEED_MPS;
                Track nt{det, tr.bbox, ts, speed};
                updated[tid] = nt;
            } else {
                // unmatched old track: drop by not inserting
            }
        }

        // create new tracks for unmatched detections
        for (int j = 0; j < (int)detections.size(); ++j) {
            if (used_det[j]) continue;
            int tid = next_id++;
            Rect det = detections[j];
            Track nt{det, det, ts, 0.0};
            updated[tid] = nt;
        }

        tracks = std::move(updated);
        // prepare output
        vector<pair<int, Track>> out;
        out.reserve(tracks.size());
        for (auto& kv : tracks) out.emplace_back(kv.first, kv.second);
        return out;
    }
};

static Mat undistort_fisheye_cached(const Mat& frame) {
    int h = frame.rows, w = frame.cols;
    static Size cacheSize(0,0);
    static Mat map1, map2;
    if (cacheSize.width != w || cacheSize.height != h) {
        Mat K = (Mat_<float>(3,3) << w/2.0f, 0, w/2.0f,
                                     0,      w/2.0f, h/2.0f,
                                     0,      0,      1);
        Mat D = Mat::zeros(4, 1, CV_32F);
        Mat R = Mat::eye(3, 3, CV_32F);
        fisheye::initUndistortRectifyMap(K, D, R, K, Size(w, h), CV_16SC2, map1, map2);
        cacheSize = Size(w,h);
    }
    Mat out;
    remap(frame, out, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
    return out;
}

static pair<double,double> pixel_to_angle(double x, double y, double width, double height, double fov_horizontal=180.0, double fov_vertical=180.0) {
    double cx = width / 2.0;
    double cy = height / 2.0;
    double nx = (x - cx) / cx;
    double ny = (y - cy) / cy;
    double angle_x = nx * (fov_horizontal / 2.0);
    double angle_y = ny * (fov_vertical / 2.0);
    return {angle_x, angle_y};
}

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

static double now_seconds() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    auto t = steady_clock::now();
    return duration<double>(t - t0).count();
}

static bool parse_distance_input_to_meters(const string& s, double& outMeters) {
    // Accepts: digits with optional decimal (.,), optional unit (mm, cm, m)
    // Examples: "0,45", "0.45", "45cm", "450 mm", "1.2m"
    try {
        std::regex re(R"(([+-]?\d+[.,]?\d*)\s*(mm|cm|m)?)", std::regex::icase);
        std::smatch m;
        if (!std::regex_search(s, m, re)) return false;
        string num = m[1].str();
        for (auto& c : num) if (c == ',') c = '.';
        string unit = m[2].matched ? m[2].str() : string();
        for (auto& c : unit) c = (char)tolower((unsigned char)c);
        double val = stod(num);
        if (unit == "mm") val /= 1000.0;
        else if (unit == "cm") val /= 100.0;
        // else unit == "m" or empty
        outMeters = val;
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char** argv) {
    // Prepare IO
    ensure_dir(RAW_DIR);
    namedWindow(WINDOW_NAME, WINDOW_NORMAL);
    setMouseCallback(WINDOW_NAME, mouse_callback);

    // Camera init for Arducam 64MP (V4L2 only)
    int frame_width = 0;
    int frame_height = 0;
    int device_index = 0;
    int requested_w = 4624; // Default for Arducam 64MP 16MP binning
    int requested_h = 3472;
    // Allow override via command-line
    for (int i=1;i<argc;++i) {
        std::string arg = argv[i];
        if (arg.rfind("--device=",0)==0) {
            try { device_index = std::stoi(arg.substr(9)); } catch(...) {}
        } else if (arg.rfind("--mode=",0)==0) {
            std::string val = arg.substr(7);
            auto xPos = val.find('x');
            if (xPos != std::string::npos) {
                try { requested_w = std::stoi(val.substr(0,xPos)); requested_h = std::stoi(val.substr(xPos+1)); } catch(...) {}
            }
        }
    }
    std::cerr << "Using device index: " << device_index << std::endl;
    std::cerr << "Requested mode: " << requested_w << "x" << requested_h << std::endl;

    cv::VideoCapture cap;
    std::cerr << "Trying V4L2 backend..." << std::endl;
    cap.open(device_index, cv::CAP_V4L2);
    if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, requested_w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, requested_h);
        frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        std::cerr << "Camera opened. Using resolution: " << frame_width << "x" << frame_height << std::endl;
        std::cerr << "Backend: " << cap.getBackendName() << std::endl;
    } else {
        std::cerr << "ERROR: Unable to open camera using V4L2 backend." << std::endl;
        std::cerr << "Check camera connection, permissions, and that no other process is using the camera." << std::endl;
        return -1;
    }
    // For further customization, you can set additional camera properties here
    // Example: cap.set(cv::CAP_PROP_FPS, 30);
    // Example: cap.set(cv::CAP_PROP_BRIGHTNESS, ...);
    // Load calibration data
    cv::Mat K, D;
    cv::FileStorage fs("calibration_data.yml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["camera_matrix"] >> K;
        fs["dist_coeffs"] >> D;
        fs.release();
    } else {
        std::cerr << "WARNING: Calibration data not found, using default intrinsics." << std::endl;
        K = (cv::Mat_<double>(3,3) << 1, 0, frame_width/2.0,
                                       0, 1, frame_height/2.0,
                                       0, 0, 1);
        D = cv::Mat::zeros(4, 1, CV_64F);
    }

    // Set up fisheye undistort map
    cv::Mat map1, map2;
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::fisheye::initUndistortRectifyMap(K, D, R, K, cv::Size(frame_width, frame_height), CV_16SC2, map1, map2);

    Ptr<BackgroundSubtractorMOG2> fgbg = createBackgroundSubtractorMOG2(500, 50, true);

    SimpleTracker tracker(0.15);
    unordered_map<int, deque<Point>> trails; // id -> trail

    UdpSender udp;

    int frame_id = 0;

    Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "ERROR: Failed to grab frame from camera. Exiting." << std::endl;
            break;
        }
        frame_id++;

        // Save raw G-channel
        save_raw_g_channel(frame, RAW_DIR, frame_id);

        // Fisheye correction
        Mat corrected = undistort_fisheye_cached(frame);

        // Use G channel for processing
        vector<Mat> ch;
        split(corrected, ch);
        Mat gray = (ch.size() >= 2) ? ch[1] : corrected;

        // Background / motion
        Mat fgmask;
        fgbg->apply(gray, fgmask);
        Mat thresh;
        threshold(fgmask, thresh, 200, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Rect> detections;
        detections.reserve(contours.size());
        for (const auto& cnt : contours) {
            double area = contourArea(cnt);
            if (area < 5000.0) continue;
            Rect r = boundingRect(cnt);
            detections.push_back(r);
        }

        double ts = now_seconds();
        auto tracks = tracker.update(detections, ts);

        // Collect objects for UDP
        struct Obj { int id; int x; int y; int w; int h; double angle_h; double angle_v; double speed_mps; };
        vector<Obj> objects;

        for (auto& it : tracks) {
            int tid = it.first;
            const Track& tr = it.second;
            const Rect& r = tr.bbox;
            double speed = tr.speed_mps;
            if (speed <= 0.0) continue;
            if (speed > MAX_SPEED_MPS) continue;

            int cx = r.x + r.width / 2;
            int cy = r.y + r.height / 2;
            auto ang = pixel_to_angle((double)cx, (double)cy, (double)corrected.cols, (double)corrected.rows);

            if (speed >= SPEED_THRESHOLD_MPS && speed <= MAX_SPEED_MPS) {
                objects.push_back(Obj{tid, r.x, r.y, r.width, r.height, ang.first, ang.second, speed});
                // draw
                Scalar color(0, 0, 255);
                stringstream lab; lab.setf(std::ios::fixed); lab<<setprecision(1)<<speed<<" m/s";
                rectangle(corrected, r, color, 3);
                putText(corrected, lab.str(), Point(r.x, max(0, r.y - 10)), FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
                // trail
                auto& dq = trails[tid];
                dq.push_back(Point(cx, cy));
                while ((int)dq.size() > TRAIL_LENGTH) dq.pop_front();
                for (size_t j = 1; j < dq.size(); ++j) {
                    int alpha = max(0, 255 - (int)(dq.size() - j) * TRAIL_FADE);
                    line(corrected, dq[j-1], dq[j], Scalar(255, alpha, alpha), 2);
                }
                if (trails[tid].size() >= 2) {
                    auto& pts = trails[tid];
                    int dx = pts.back().x - pts[pts.size()-2].x;
                    int dy = pts.back().y - pts[pts.size()-2].y;
                    Point pred(pts.back().x + dx * 5, pts.back().y + dy * 5);
                    circle(corrected, pred, 6, Scalar(255, 255, 0), 2);
                }
            }
        }

        // Display the resulting frame
        cv::imshow("Frame", corrected);

        // Wait for a key press (1ms) and handle window events
        if (cv::waitKey(1) >= 0) {
            break;
        }

        // UDP send
        if (!objects.empty()) {
            // Build JSON manually
            ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < objects.size(); ++i) {
                const auto& o = objects[i];
                if (i) oss << ",";
                oss.setf(std::ios::fixed);
                oss << "{"
                    << "\"id\":" << o.id << ","
                    << "\"x\":" << o.x << ","
                    << "\"y\":" << o.y << ","
                    << "\"w\":" << o.w << ","
                    << "\"h\":" << o.h << ","
                    << "\"angle_h\":" << setprecision(6) << o.angle_h << ","
                    << "\"angle_v\":" << setprecision(6) << o.angle_v << ","
                    << "\"speed_mps\":" << setprecision(6) << o.speed_mps
                    << "}";
            }
            oss << "]";
            string payload = oss.str();
            udp.sendStr(payload);
            try {
                cout << "[Frame " << frame_id << "] Sent " << objects.size() << " objects to " << UDP_IP << ":" << UDP_PORT << "\n";
                cout << payload << "\n";
            } catch (...) {
                cerr << "Failed printing payload\n";
            }
        }

        imshow(WINDOW_NAME, corrected);
        int key = waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == 'c') {
            // calibration mode: collect 2 clicks
            calibrating = true;
            cal_points.clear();
            cout << "Calibration mode: click two points on the image (left mouse button), then enter real distance in meters in the console." << endl;
            // wait for two clicks or q
            for (;;) {
                imshow(WINDOW_NAME, corrected);
                int k = waitKey(10) & 0xFF;
                if (k == 'q') break;
                if ((int)cal_points.size() >= 2) break;
            }
            calibrating = false;
            if (cal_points.size() >= 2) {
                Point p0 = cal_points[0], p1 = cal_points[1];
                double px_dist = hypot(double(p1.x - p0.x), double(p1.y - p0.y));
                int attempts = 0;
                bool ok = false;
                double real_m = 0.0;
                string line;
                while (attempts < 3) {
                    cout << "Enter real distance between points in meters (e.g. 0.45 or 45cm). Press Enter to cancel: ";
                    std::getline(cin, line);
                    if (line.empty()) break;
                    double v;
                    if (parse_distance_input_to_meters(line, v)) { real_m = v; ok = true; break; }
                    attempts++;
                    cout << "Invalid input. Attempts left: " << (3 - attempts) << "\n";
                }
                if (!ok) {
                    cout << "Calibration aborted or invalid input." << endl;
                } else {
                    if (px_dist > 0.0) {
                        METERS_PER_PIXEL = real_m / px_dist;
                        cout.setf(std::ios::fixed);
                        cout << "Calibration done. METERS_PER_PIXEL = " << setprecision(6) << METERS_PER_PIXEL << " m/px" << endl;
                    } else {
                        cout << "Zero pixel distance, calibration aborted." << endl;
                    }
                }
            }
        }
    }

    // Properly cleanup GStreamer pipeline
    // First stop reading frames and close OpenCV windows
    destroyAllWindows();
    
    // Release the capture object - this will set pipeline to NULL state
    if (cap.isOpened()) {
        cap.release();
    }
    
    // Give GStreamer time to cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    return 0;
}
