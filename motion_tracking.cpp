#include <opencv2/opencv.hpp>
#include <opencv2/fisheye.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <deque>
#include <map>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <regex>
#include <chrono>
#include <fstream>
#include <sstream>

#pragma comment(lib, "ws2_32.lib")

// ---------------- UDP параметры ----------------
const char* UDP_IP = "192.168.1.100";
const int UDP_PORT = 5005;

// ---------------- Настройки камеры ----------------
const int CAM_INDEX = 0;
const int FPS = 30;
const int FRAME_WIDTH = 2400;
const int FRAME_HEIGHT = 1200;

// ---------------- Настройки trail ----------------
const int TRAIL_LENGTH = 20;
const int TRAIL_FADE = 10;

// ---------------- Папка для RAW ----------------
const std::string RAW_DIR = "raw_capture";

// Очереди для хранения траекторий объектов
std::map<int, std::deque<cv::Point>> trails;

// Параметры скорости
double METERS_PER_PIXEL = 0.02;
const double SPEED_THRESHOLD_MPS = 0.5;
const double MAX_SPEED_MPS = 100.0;
const double MAX_BBOX_SCALE_CHANGE = 3.0;

// Калибровка
bool calibrating = false;
std::vector<cv::Point> cal_points;

// Структура для трека
struct Track {
    cv::Rect bbox;
    cv::Rect last_bbox;
    double time;
    double speed_mps;
};

// Класс SimpleTracker
class SimpleTracker {
public:
    std::map<int, Track> tracks;
    int next_id = 0;
    double iou_threshold = 0.15;

    double compute_iou(const cv::Rect& a, const cv::Rect& b) {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        if (x2 <= x1 || y2 <= y1) return 0.0;
        double inter = (x2 - x1) * (y2 - y1);
        double uni = a.width * a.height + b.width * b.height - inter;
        return inter / uni;
    }

    std::vector<std::map<std::string, double>> update(const std::vector<cv::Rect>& detections, double ts) {
        std::map<int, Track> updated;
        std::vector<bool> used_det(detections.size(), false);

        for (auto& [tid, tr] : tracks) {
            double best_iou = 0.0;
            int best_j = -1;
            for (size_t j = 0; j < detections.size(); ++j) {
                if (used_det[j]) continue;
                double iou = compute_iou(tr.bbox, detections[j]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_j = j;
                }
            }
            if (best_j != -1 && best_iou >= iou_threshold) {
                cv::Rect det = detections[best_j];
                used_det[best_j] = true;
                // compute speed
                cv::Point prev_c(tr.bbox.x + tr.bbox.width / 2, tr.bbox.y + tr.bbox.height / 2);
                cv::Point c(det.x + det.width / 2, det.y + det.height / 2);
                double dt = ts - tr.time;
                if (dt <= 0) dt = 1.0 / FPS;
                double dx_m = (c.x - prev_c.x) * METERS_PER_PIXEL;
                double dy_m = (c.y - prev_c.y) * METERS_PER_PIXEL;
                double dist_m = std::hypot(dx_m, dy_m);
                double speed = dist_m / dt;
                // outlier protections
                double prev_area = tr.bbox.width * tr.bbox.height;
                double new_area = det.width * det.height;
                if (prev_area > 0 && (new_area / prev_area > MAX_BBOX_SCALE_CHANGE || prev_area / new_area > MAX_BBOX_SCALE_CHANGE)) {
                    speed = 0.0;
                    det = tr.bbox;
                }
                speed = std::min(speed, MAX_SPEED_MPS);
                updated[tid] = {det, tr.bbox, ts, speed};
            }
        }

        // new tracks
        for (size_t j = 0; j < detections.size(); ++j) {
            if (used_det[j]) continue;
            int tid = next_id++;
            updated[tid] = {detections[j], detections[j], ts, 0.0};
        }

        tracks = updated;
        std::vector<std::map<std::string, double>> out;
        for (auto& [tid, tr] : tracks) {
            out.push_back({{"id", (double)tid}, {"bbox", 0}, {"speed_mps", tr.speed_mps}});
            // Note: bbox as Rect, but for simplicity, we'll handle separately
        }
        return out;
    }
};

SimpleTracker tracker;

// Функция коррекции fisheye
cv::Mat undistort_fisheye(const cv::Mat& frame) {
    int h = frame.rows, w = frame.cols;
    cv::Mat K = (cv::Mat_<double>(3, 3) << w / 2.0, 0, w / 2.0, 0, w / 2.0, h / 2.0, 0, 0, 1);
    cv::Mat D = cv::Mat::zeros(1, 4, CV_64F);
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_64F), K, cv::Size(w, h), CV_16SC2, map1, map2);
    cv::Mat corrected;
    cv::remap(frame, corrected, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return corrected;
}

// Перевод пикселей в углы
std::pair<double, double> pixel_to_angle(int x, int y, int width, int height, double fov_h = 180, double fov_v = 180) {
    double cx = width / 2.0, cy = height / 2.0;
    double nx = (x - cx) / cx;
    double ny = (y - cy) / cy;
    double angle_x = nx * (fov_h / 2.0);
    double angle_y = ny * (fov_v / 2.0);
    return {angle_x, angle_y};
}

// Mouse callback
void mouse_callback(int event, int x, int y, int flags, void* userdata) {
    if (calibrating && event == cv::EVENT_LBUTTONDOWN) {
        cal_points.push_back(cv::Point(x, y));
        std::cout << "Calibration click: " << cv::Point(x, y) << std::endl;
    }
}

// Парсер расстояния
double parse_distance(const std::string& s) {
    std::regex re(R"((\d+[.,]?\d*)\s*(mm|cm|m)?)", std::regex_constants::icase);
    std::smatch match;
    if (std::regex_search(s, match, re)) {
        std::string num_str = match[1].str();
        std::replace(num_str.begin(), num_str.end(), ',', '.');
        double val = std::stod(num_str);
        std::string unit = match[2].str();
        std::transform(unit.begin(), unit.end(), unit.begin(), ::tolower);
        if (unit == "mm") val /= 1000.0;
        else if (unit == "cm") val /= 100.0;
        return val;
    }
    throw std::invalid_argument("Invalid input");
}

int main() {
    // Создать папку
    std::filesystem::create_directories(RAW_DIR);

    // Инициализация Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return 1;
    }

    // Создать сокет
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Socket creation failed" << std::endl;
        WSACleanup();
        return 1;
    }

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(UDP_PORT);
    inet_pton(AF_INET, UDP_IP, &server_addr.sin_addr);

    // Камера
    cv::VideoCapture cap(CAM_INDEX);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        closesocket(sock);
        WSACleanup();
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap.set(cv::CAP_PROP_FPS, FPS);

    cv::Ptr<cv::BackgroundSubtractorMOG2> fgbg = cv::createBackgroundSubtractorMOG2(500, 50);

    cv::namedWindow("Motion Tracking Fisheye");
    cv::setMouseCallback("Motion Tracking Fisheye", mouse_callback);

    int frame_id = 0;
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        frame_id++;

        // Сохранить RAW G-канал
        cv::Mat g_channel = frame.col(1); // G channel
        std::string raw_path = RAW_DIR + "/frame_" + std::to_string(frame_id).insert(0, 6 - std::to_string(frame_id).length(), '0') + ".raw";
        std::ofstream raw_file(raw_path, std::ios::binary);
        raw_file.write((char*)g_channel.data, g_channel.total() * g_channel.elemSize());
        raw_file.close();

        // Коррекция fisheye
        cv::Mat corrected = undistort_fisheye(frame);

        // G-канал
        cv::Mat gray = corrected.col(1);

        // Фон
        cv::Mat fgmask;
        fgbg->apply(gray, fgmask);
        cv::Mat thresh;
        cv::threshold(fgmask, thresh, 200, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> detections;
        for (const auto& cnt : contours) {
            if (cv::contourArea(cnt) < 5000) continue;
            cv::Rect bbox = cv::boundingRect(cnt);
            detections.push_back(bbox);
        }

        double ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        auto tracks = tracker.update(detections, ts);

        std::vector<std::map<std::string, double>> objects;
        for (auto& t : tracks) {
            int tid = (int)t["id"];
            // Need to get bbox from tracks
            if (tracker.tracks.find(tid) == tracker.tracks.end()) continue;
            Track tr = tracker.tracks[tid];
            int x = tr.bbox.x, y = tr.bbox.y, w = tr.bbox.width, h = tr.bbox.height;
            double speed = tr.speed_mps;
            if (speed <= 0.0) continue;
            if (speed > MAX_SPEED_MPS) continue;

            cv::Point cx_cy(x + w / 2, y + h / 2);
            auto [angle_h, angle_v] = pixel_to_angle(cx_cy.x, cx_cy.y, corrected.cols, corrected.rows);
            std::map<std::string, double> obj = {
                {"id", (double)tid},
                {"x", (double)x},
                {"y", (double)y},
                {"w", (double)w},
                {"h", (double)h},
                {"angle_h", angle_h},
                {"angle_v", angle_v},
                {"speed_mps", speed}
            };
            if (speed >= SPEED_THRESHOLD_MPS) {
                objects.push_back(obj);
                cv::rectangle(corrected, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 3);
                std::string label = std::to_string(speed).substr(0, 4) + " m/s";
                cv::putText(corrected, label, cv::Point(x, std::max(0, y - 10)), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                // trail
                if (trails.find(tid) == trails.end()) trails[tid] = std::deque<cv::Point>();
                trails[tid].push_back(cx_cy);
                if (trails[tid].size() > TRAIL_LENGTH) trails[tid].pop_front();
                auto pts = trails[tid];
                for (size_t j = 1; j < pts.size(); ++j) {
                    int alpha = std::max(0, 255 - (int)(pts.size() - j) * TRAIL_FADE);
                    cv::line(corrected, pts[j - 1], pts[j], cv::Scalar(255, alpha, alpha), 2);
                }
                // prediction
                if (pts.size() >= 2) {
                    cv::Point dx = pts.back() - pts[pts.size() - 2];
                    cv::Point pred = pts.back() + dx * 5;
                    cv::circle(corrected, pred, 6, cv::Scalar(255, 255, 0), 2);
                }
            }
        }

        // Отправка
        if (!objects.empty()) {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < objects.size(); ++i) {
                if (i > 0) ss << ",";
                ss << "{";
                for (auto it = objects[i].begin(); it != objects[i].end(); ++it) {
                    if (it != objects[i].begin()) ss << ",";
                    ss << "\"" << it->first << "\":" << it->second;
                }
                ss << "}";
            }
            ss << "]";
            std::string data = ss.str();
            sendto(sock, data.c_str(), data.size(), 0, (sockaddr*)&server_addr, sizeof(server_addr));
            std::cout << "[Frame " << frame_id << "] Sent " << objects.size() << " objects to " << UDP_IP << ":" << UDP_PORT << std::endl;
            std::cout << data << std::endl;
        }

        cv::imshow("Motion Tracking Fisheye", corrected);
        int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'c') {
            calibrating = true;
            cal_points.clear();
            std::cout << "Calibration mode: click two points on the image (left mouse button), then enter real distance in meters in the console." << std::endl;
            while (cal_points.size() < 2) {
                cv::imshow("Motion Tracking Fisheye", corrected);
                if (cv::waitKey(10) == 'q') break;
            }
            calibrating = false;
            if (cal_points.size() >= 2) {
                cv::Point p0 = cal_points[0], p1 = cal_points[1];
                double px_dist = cv::norm(p1 - p0);
                std::cout << "Enter real distance between points in meters (e.g. 0.45 or 45cm): ";
                std::string input;
                std::getline(std::cin, input);
                try {
                    double real = parse_distance(input);
                    if (px_dist > 0) {
                        METERS_PER_PIXEL = real / px_dist;
                        std::cout << "Calibration done. METERS_PER_PIXEL = " << METERS_PER_PIXEL << " m/px" << std::endl;
                    } else {
                        std::cout << "Zero pixel distance, calibration aborted." << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cout << "Invalid input: " << e.what() << std::endl;
                }
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    closesocket(sock);
    WSACleanup();
    return 0;
}