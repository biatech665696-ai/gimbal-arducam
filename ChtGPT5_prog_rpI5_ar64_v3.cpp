
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <map>
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

// Прототип функции для очистки мешающих процессов
void killPreviousInstances();
/*
 * ===========================================================================================
 * СИСТЕМА ПРЕДИКТИВНОГО УПРАВЛЕНИЯ ПОДВЕСОМ С ARDUCAM 64MP
 * ===========================================================================================
 * 
 * Интеграция алгоритма ChatGPT5 с реальным оборудованием:
 * - Камера: Arducam 64MP (1920x1080 @ 45 fps)
 * - Сервоприводы: GPIO18 (горизонталь), GPIO19 (вертикаль)
 * - Управление: PWM через sysfs
 * - Предиктивный контроль с компенсацией системной задержки
 * 
 * ===========================================================================================
 */

using namespace std;

// Global for display
std::mutex displayMutex;
cv::Mat displayFrame;
std::atomic<bool> hasNewFrame(false);

#define GPIO_HORIZONTAL 18
#define GPIO_VERTICAL 19

#define PWM_CHIP 0
#define PWM_CHANNEL_HORIZONTAL 2  // GPIO18
#define PWM_CHANNEL_VERTICAL 3    // GPIO19

// Servo parameters
constexpr long period_ns = 20000000;   // 20 ms period in nanoseconds
constexpr long min_ns = 1000000;       // 1 ms  -> 0°
constexpr long max_ns = 2000000;       // 2 ms  -> 180°
constexpr float SERVO_CENTER_ANGLE = 90.0f;  // Центральное положение серво (нейтраль)
constexpr bool kVerboseServoLogs = true;  // ENABLED for debugging
constexpr bool kVerboseTrackingLogs = true;
constexpr bool kVerboseFrameLoopLogs = true;

// Predictive control parameters (integrated from ChatGPT5 algorithm)
constexpr double SYSTEM_DELAY = 0.045;  // Системная задержка 45мс
constexpr bool USE_PREDICTIVE_CONTROL = true;  // Включить предиктивное управление

// Camera parameters (Arducam 64MP @ 1920x1080)
const double CX = 960.0;   // Optical center X (half of 1920)
const double CY = 540.0;   // Optical center Y (half of 1080)
const double F  = 1200.0;  // Focal length in pixels (approximate for wide field of view)

// Global tracking mode control
std::atomic<bool> trackingEnabled(true);  // Start with tracking ENABLED by default

/* =============== PWM CONTROL FUNCTIONS =============== */

static std::string getPwmPath(int channel) {
    return "/sys/class/pwm/pwmchip" + std::to_string(PWM_CHIP) + "/pwm" + std::to_string(channel);
}

static bool writeToFile(const std::string& path, const std::string& value) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return false;
    }
    file << value;
    file.close();
    
    if (file.fail()) {
        std::cerr << "ERROR: Failed to write to " << path << std::endl;
        return false;
    }
    return true;
}

static bool exportPwmChannel(int channel) {
    std::string pwmPath = getPwmPath(channel);
    std::ifstream check(pwmPath);
    if (check.good()) {
        check.close();
        return true; // Already exported
    }
    
    std::string exportPath = "/sys/class/pwm/pwmchip" + std::to_string(PWM_CHIP) + "/export";
    return writeToFile(exportPath, std::to_string(channel));
}

static bool initSysfsPwm() {
    std::cerr << "Initializing PWM channels..." << std::endl;
    
    // Set GPIO alternate functions for PWM
    std::cerr << "Setting GPIO18 to PWM mode (alt3)..." << std::endl;
    system("pinctrl set 18 a3");
    std::cerr << "Setting GPIO19 to PWM mode (alt3)..." << std::endl;
    system("pinctrl set 19 a3");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Export PWM channels
    std::cerr << "Exporting PWM channel " << PWM_CHANNEL_HORIZONTAL << "..." << std::endl;
    if (!exportPwmChannel(PWM_CHANNEL_HORIZONTAL)) {
        std::cerr << "ERROR: Failed to export PWM channel " << PWM_CHANNEL_HORIZONTAL << std::endl;
        return false;
    }
    std::cerr << "Exporting PWM channel " << PWM_CHANNEL_VERTICAL << "..." << std::endl;
    if (!exportPwmChannel(PWM_CHANNEL_VERTICAL)) {
        std::cerr << "ERROR: Failed to export PWM channel " << PWM_CHANNEL_VERTICAL << std::endl;
        return false;
    }
    
    // Small delay to allow sysfs to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Set period for both channels
    std::cerr << "Setting period to " << period_ns << "ns..." << std::endl;
    if (!writeToFile(getPwmPath(PWM_CHANNEL_HORIZONTAL) + "/period", std::to_string(period_ns))) {
        std::cerr << "ERROR: Failed to set period for PWM channel " << PWM_CHANNEL_HORIZONTAL << std::endl;
        return false;
    }
    if (!writeToFile(getPwmPath(PWM_CHANNEL_VERTICAL) + "/period", std::to_string(period_ns))) {
        std::cerr << "ERROR: Failed to set period for PWM channel " << PWM_CHANNEL_VERTICAL << std::endl;
        return false;
    }
    
    // Set initial duty cycle (90 degrees center) BEFORE enabling
    constexpr float SERVO_CENTER_ANGLE_INIT = 90.0f;  // Центральное положение при инициализации
    long center_duty = min_ns + static_cast<long>((SERVO_CENTER_ANGLE_INIT / 180.0f) * (max_ns - min_ns));
    std::cerr << "Setting initial duty cycle to " << center_duty << "ns (90°)..." << std::endl;
    if (!writeToFile(getPwmPath(PWM_CHANNEL_HORIZONTAL) + "/duty_cycle", std::to_string(center_duty))) {
        std::cerr << "ERROR: Failed to set duty_cycle for PWM channel " << PWM_CHANNEL_HORIZONTAL << std::endl;
        return false;
    }
    if (!writeToFile(getPwmPath(PWM_CHANNEL_VERTICAL) + "/duty_cycle", std::to_string(center_duty))) {
        std::cerr << "ERROR: Failed to set duty_cycle for PWM channel " << PWM_CHANNEL_VERTICAL << std::endl;
        return false;
    }
    
    // Enable both channels
    std::cerr << "Enabling PWM channels..." << std::endl;
    if (!writeToFile(getPwmPath(PWM_CHANNEL_HORIZONTAL) + "/enable", "1")) {
        std::cerr << "ERROR: Failed to enable PWM channel " << PWM_CHANNEL_HORIZONTAL << std::endl;
        return false;
    }
    if (!writeToFile(getPwmPath(PWM_CHANNEL_VERTICAL) + "/enable", "1")) {
        std::cerr << "ERROR: Failed to enable PWM channel " << PWM_CHANNEL_VERTICAL << std::endl;
        return false;
    }
    
    std::cerr << "PWM initialization complete!" << std::endl;
    return true;
}

static void shutdownSysfsPwm() {
    // Disable PWM channels
    writeToFile(getPwmPath(PWM_CHANNEL_HORIZONTAL) + "/enable", "0");
    writeToFile(getPwmPath(PWM_CHANNEL_VERTICAL) + "/enable", "0");
    
    // Set duty cycle to 0
    writeToFile(getPwmPath(PWM_CHANNEL_HORIZONTAL) + "/duty_cycle", "0");
    writeToFile(getPwmPath(PWM_CHANNEL_VERTICAL) + "/duty_cycle", "0");
}

void setServoAngle(int channel, float angle_deg)
{
    std::cout << "[SERVO CMD] channel=" << channel << " angle=" << angle_deg << "°";
    
    if(angle_deg < 0.0f) angle_deg = 0.0f;
    if(angle_deg > 180.0f) angle_deg = 180.0f;

    const long duty_cycle_ns = min_ns + static_cast<long>((angle_deg / 180.0f) * (max_ns - min_ns));
    std::string path = getPwmPath(channel) + "/duty_cycle";
    
    std::cout << " duty=" << duty_cycle_ns << "ns path=" << path << std::endl;
    
    bool success = writeToFile(path, std::to_string(duty_cycle_ns));
    if (!success) {
        std::cerr << "  ✗✗✗ SERVO WRITE FAILED! Check permissions on " << path << std::endl;
    } else {
        std::cout << "  ✓ Servo command sent successfully" << std::endl;
    }
}

/* =============== DATA STRUCTURES =============== */

struct FrameData
{
    cv::Mat frame;
    double timestamp;
};



#include <vector>
struct Detection
{
    double x;
    double y;
    int box_x = 0;
    int box_y = 0;
    int box_w = 0;
    int box_h = 0;
    bool valid = false;
    std::vector<cv::Rect> all_boxes; // bounding boxes for all moving objects
};

struct AngleState
{
    double theta;
    double phi;
    double wtheta;
    double wphi;
};

/* =============== THREAD-SAFE QUEUE =============== */

template<typename T>
class SafeQueue
{
    queue<T> q;
    mutex m;
    condition_variable cv;

public:

    void push(const T& v)
    {
        unique_lock<mutex> lock(m);
        // Дропаем старые кадры — всегда оставляем только самый свежий
        while (!q.empty()) q.pop();
        q.push(v);
        cv.notify_one();
    }

    T pop()
    {
        unique_lock<mutex> lock(m);
        cv.wait(lock,[&]{return !q.empty();});

        T v=q.front();
        q.pop();

        return v;
    }
};

/* =============== SUBPIXEL CENTROID =============== */

Detection centroid(cv::Mat &roi, int ox, int oy, double learningRate = 0.01)
{
    // === BACKGROUND SUBTRACTION MOG2 (ADAPTIVE BACKGROUND MODEL) ===
    // This method adapts to camera movement and filters moving background
    static cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = 
        cv::createBackgroundSubtractorMOG2(50, 52, false);  // history=50 (~1.1s @ 45fps) — быстрее учит новый фон
    
    // Прогрев: первые 20 кадров обучаем с высоким learningRate чтобы модель
    // фона сформировалась за ~0.5 сек вместо 11 секунд.
    static int warmupFrames = 0;
    
    static cv::Point2f lastValidCenter(-1, -1);
    static int consecutiveDetections = 0;
    static int consecutiveMisses = 0;
    static std::vector<cv::Rect> prevFrameBoxes;   // для временно́й фильтрации шумов
    const int MIN_DETECTIONS = 2;
    const int MAX_MISSES = 60;
    
    Detection d;
    d.valid = false;

    // Convert to grayscale
    cv::Mat gray;
    if (roi.channels() > 1) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi.clone();
    }
    
    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);

    // Apply background subtraction (adapts to camera movement)
    cv::Mat fgMask;
    // Прогрев первых 20 кадров: высокий learningRate чтобы быстро построить модель фона
    double effectiveLR = (warmupFrames < 20) ? 0.5 : learningRate;
    if (warmupFrames < 20) warmupFrames++;
    bgSubtractor->apply(gray, fgMask, effectiveLR);
    // Во время прогрева маску обнуляем — детекция ещё ненадёжна
    if (warmupFrames < 20) fgMask = cv::Mat::zeros(fgMask.size(), fgMask.type());

    // Morphology to clean up noise and connect nearby regions
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 3);   // Remove noise + shadows (3 iterations)
    cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel);  // Fill gaps

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Collect and filter significant moving objects
    std::vector<std::pair<double, int>> validObjects;
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        cv::Rect bbox = cv::boundingRect(contours[i]);
        double bboxArea = bbox.width * bbox.height;
        double solidity = (bboxArea > 0) ? (area / bboxArea) : 0;  // Density: contour_area / bbox_area
        double aspectRatio = (double)bbox.width / (double)bbox.height;
        
        // STRICT FILTERING to reject background noise and shadows:
        // 1. Size: 20-2000 pixels (user requirement)
        // 2. Solidity > 0.45 (reject sparse/fragmented contours and shadows)
        // 3. Reasonable bounding box size
        // 4. Reasonable aspect ratio (not extremely elongated)
        if (area >= 20.0 && area <= 2000.0 &&
            solidity > 0.45 &&  // Strict: reject shadows and noise
            bbox.width >= 5 && bbox.height >= 5 &&
            bbox.width <= 200 && bbox.height <= 200 &&
            aspectRatio > 0.2 && aspectRatio < 6.0) {
            
            d.all_boxes.push_back(cv::Rect(bbox.x + ox, bbox.y + oy, bbox.width, bbox.height));
            validObjects.push_back(std::make_pair(area, i));
        }
    }

    // Check if we have a candidate object
    cv::Point2f currentCenter(-1, -1);
    bool foundCandidate = false;
    
    if (!validObjects.empty()) {
        // Sort by area
        std::sort(validObjects.begin(), validObjects.end(),
                 [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                     return a.first > b.first;
                 });
        
        // === SPATIAL GATE: если уже есть захваченная цель — выбираем ===
        // кандидата БЛИЖАЙШЕГО к ней, а не просто самого большого.
        // Это не даёт камере перепрыгнуть на фоновый шум после движения.
        int bestIdx = validObjects[0].second;
        if (lastValidCenter.x > 0) {
            float bestDist = 1e9f;
            for (const auto& vo : validObjects) {
                cv::Moments m = cv::moments(contours[vo.second]);
                if (m.m00 <= 0) continue;
                cv::Point2f c((m.m10 / m.m00) + ox, (m.m01 / m.m00) + oy);
                float dist = cv::norm(c - lastValidCenter);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = vo.second;
                }
            }
        }
        
        int idx = bestIdx;
        cv::Moments m = cv::moments(contours[idx]);
        
        if (m.m00 > 0) {
            currentCenter.x = (m.m10 / m.m00) + ox;
            currentCenter.y = (m.m01 / m.m00) + oy;
            
            // If we had a previous valid center, check if new center is close (not jumping randomly)
            if (lastValidCenter.x > 0) {
                float distance = cv::norm(currentCenter - lastValidCenter);
                // Allow larger movement for tracking fast objects (150 pixels between frames)
                if (distance < 150.0) {  // Increased from 100 to 150 for faster motion
                    foundCandidate = true;
                    consecutiveDetections++;
                    consecutiveMisses = 0;
                } else {
                    // Object jumped too far - probably noise/background
                    consecutiveDetections = 0;
                    lastValidCenter = currentCenter; // reset anchor to new position
                }
            } else {
                // First detection: set lastValidCenter NOW so next frame enters
                // the distance-check branch and can increment consecutiveDetections
                // to reach MIN_DETECTIONS. Without this, lastValidCenter stays -1
                // forever and the counter is reset to 1 every frame — d.valid
                // can never become true.
                foundCandidate = true;
                consecutiveDetections = 1;
                lastValidCenter = currentCenter; // ← critical fix
            }
            
            // Only validate object after MIN_DETECTIONS consecutive frames
            if (foundCandidate && consecutiveDetections >= MIN_DETECTIONS) {
                d.x = currentCenter.x;
                d.y = currentCenter.y;
                cv::Rect bbox = cv::boundingRect(contours[idx]);
                d.box_x = bbox.x + ox;
                d.box_y = bbox.y + oy;
                d.box_w = bbox.width;
                d.box_h = bbox.height;
                d.valid = true;
                lastValidCenter = currentCenter;
            }
        }
    }
    
    if (!foundCandidate) {
        consecutiveMisses++;
        consecutiveDetections = 0;
        
        // After MAX_MISSES frames without detection, reset tracking
        if (consecutiveMisses > MAX_MISSES) {
            lastValidCenter = cv::Point2f(-1, -1);
        }
    }

    // === TEMPORAL PERSISTENCE FILTER для all_boxes ===
    // Показываем бокс только если он перекрывается с боксом из предыдущего кадра.
    // Одиночные фоновые вспышки (1 кадр) отфильтровываются, реальные объекты
    // подтверждаются уже со 2-го кадра — чувствительность не снижается.
    {
        std::vector<cv::Rect> confirmedBoxes;
        for (const auto& box : d.all_boxes) {
            for (const auto& prev : prevFrameBoxes) {
                if ((box & prev).area() > 0) {
                    confirmedBoxes.push_back(box);
                    break;
                }
            }
        }
        prevFrameBoxes = d.all_boxes;
        d.all_boxes    = confirmedBoxes;
    }

    // Background model updates automatically in MOG2

    return d;
}

/* =============== ROI COMPUTATION =============== */

cv::Rect computeROI(double x, double y, int w, int h, int objW = 0, int objH = 0)
{
    // === ADAPTIVE ROI SIZE (position follows object exactly, only size is smoothed) ===
    static double smoothSize = 400.0;
    const double SIZE_SMOOTHING = 0.2;  // плавное изменение размера

    // === ADAPTIVE ROI SIZE ===
    // Searching: large ROI (400px) to find object
    // Captured:  shrink to 4x object size (min 120px, max 400px) для точной слежки
    double targetSize;
    if (objW > 0 && objH > 0) {
        double objMaxDim = std::max(objW, objH);
        targetSize = std::max(120.0, std::min(400.0, objMaxDim * 4.0));
    } else {
        targetSize = 400.0;  // Full size when searching
    }
    // Smooth only the size — центр ROI == позиция объекта (без лага)
    smoothSize = SIZE_SMOOTHING * targetSize + (1.0 - SIZE_SMOOTHING) * smoothSize;
    int size = static_cast<int>(smoothSize);

    int rx = static_cast<int>(x) - size/2;
    int ry = static_cast<int>(y) - size/2;

    rx = max(0, rx);
    ry = max(0, ry);

    if(rx + size > w) rx = w - size;
    if(ry + size > h) ry = h - size;

    return cv::Rect(rx, ry, size, size);
}

/* =============== PIXEL TO ANGLES =============== */

void pixelToAngles(double x, double y, double cx, double cy, double &theta, double &phi)
{
    double vx = x - cx;
    double vy = y - cy;
    double vz = F;

    theta = atan2(vx, vz);
    phi = atan2(vy, sqrt(vx*vx + vz*vz));
}

/* =============== KALMAN FILTER =============== */

class AngleKalman
{
public:
    // State vector: [theta, phi, wtheta, wphi]
    double x[4];
    
    // Covariance matrix (4x4) - simplified diagonal representation
    double P[4];

    double lastTime;

    AngleKalman()
    {
        // Initialize state to zero
        for (int i = 0; i < 4; i++) {
            x[i] = 0.0;
            P[i] = 1.0;  // Initial uncertainty
        }
        lastTime = timeNow();
    }

    double timeNow()
    {
        auto t = chrono::high_resolution_clock::now().time_since_epoch();
        return chrono::duration<double>(t).count();
    }

    AngleState update(double theta, double phi, bool valid)
    {
        double t = timeNow();
        double dt = t - lastTime;
        lastTime = t;

        if (dt < 0.001) dt = 0.001;  // Prevent division by zero

        // Prediction step (F * x)
        // x = F * x where F is the state transition matrix
        // New position = old position + velocity * dt
        x[0] = x[0] + x[2] * dt;  // theta_new = theta_old + wtheta * dt
        x[1] = x[1] + x[3] * dt;  // phi_new = phi_old + wphi * dt
        // velocities stay the same (x[2] and x[3])

        // Prediction covariance: P = F * P * F^T + Q (simplified)
        // Just increase uncertainty with time
        P[0] += dt * dt * 0.01;  // Position uncertainty increases
        P[1] += dt * dt * 0.01;
        P[2] += 0.001;  // Velocity uncertainty
        P[3] += 0.001;

        // Update step (only if valid measurement)
        if (valid)
        {
            // Measurement model: z = H * x where H = [1 0 0 0; 0 1 0 0]
            // Innovation: y = z - H * x
            double y0 = theta - x[0];  // Innovation for theta
            double y1 = phi - x[1];     // Innovation for phi

            // Measurement noise covariance R
            double R = 1e-5;

            // Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
            // Simplified for diagonal P and simple H
            double K0 = P[0] / (P[0] + R);
            double K1 = P[1] / (P[1] + R);

            // State update: x = x + K * y
            x[0] = x[0] + K0 * y0;
            x[1] = x[1] + K1 * y1;

            // Update velocities based on innovation (simple derivative)
            if (dt > 0) {
                x[2] = 0.9 * x[2] + 0.1 * (y0 / dt);  // Exponential smoothing
                x[3] = 0.9 * x[3] + 0.1 * (y1 / dt);
            }

            // Covariance update: P = (I - K * H) * P
            P[0] = (1.0 - K0) * P[0];
            P[1] = (1.0 - K1) * P[1];
        }
        else
        {
            // When no valid detection, dampen the filter response
            // Slowly decay velocities to prevent oscillation
            x[2] = x[2] * 0.95;  // Decay velocity estimates
            x[3] = x[3] * 0.95;
            
            // Slowly move back to center
            x[0] = x[0] * 0.98;  // Decay theta toward 0
            x[1] = x[1] * 0.98;  // Decay phi toward 0
            
            // Increase uncertainty when no measurement
            P[0] += 0.01;
            P[1] += 0.01;
        }

        return {x[0], x[1], x[2], x[3]};
    }
};

/* =============== PREDICTIVE CONTROL =============== */

void predictFuture(AngleState &s,double dt)
{
    s.theta+=s.wtheta*dt;
    s.phi  +=s.wphi*dt;
}

/* =============== PD CONTROLLER =============== */

class PD
{
public:

    double kp=0.3;  // Reduced from 2.0 for stability
    double kd=0.1;  // Reduced from 0.5 for stability

    double prev=0;

    double update(double err,double dt)
    {
        if (dt < 0.001) dt = 0.001;  // Prevent division by zero
        
        double d=(err-prev)/dt;
        prev=err;

        return kp*err + kd*d;
    }
};

/* =============== GIMBAL CONTROL =============== */

class Gimbal
{
public:

    PD yawPID;
    PD pitchPID;

    double yaw=90.0;    // Start at center
    double pitch=90.0;   // Start at center

    double lastTime;

    Gimbal()
    {
        lastTime=timeNow();
    }

    double timeNow()
    {
        auto t=chrono::high_resolution_clock::now()
               .time_since_epoch();

        return chrono::duration<double>(t).count();
    }

    void update(double targetYaw,double targetPitch)
    {
        double t=timeNow();
        double dt=t-lastTime;
        lastTime=t;

        if (dt < 0.001) return;  // Skip if dt too small
        if (dt > 0.5) dt = 0.5;  // Limit max dt to prevent jumps

        double ey=targetYaw-yaw;
        double ep=targetPitch-pitch;

        double uy=yawPID.update(ey,dt);
        double up=pitchPID.update(ep,dt);

        // Apply limits to control output (max 5 degrees per update)
        if (uy > 5.0) uy = 5.0;
        if (uy < -5.0) uy = -5.0;
        if (up > 5.0) up = 5.0;
        if (up < -5.0) up = -5.0;

        yaw+=uy;
        pitch+=up;

        // Clamp angles to servo limits (with safe margin)
        if(yaw < 30.0) yaw = 30.0;
        if(yaw > 150.0) yaw = 150.0;
        if(pitch < 30.0) pitch = 30.0;
        if(pitch > 150.0) pitch = 150.0;

        sendServo(yaw,pitch);
    }

    void sendServo(double y,double p)
    {
        if (kVerboseTrackingLogs) {
            cout << "Yaw: " << y << "° Pitch: " << p << "°" << endl;
        }
        
        // Send commands to real servos
        setServoAngle(PWM_CHANNEL_HORIZONTAL, static_cast<float>(y));
        setServoAngle(PWM_CHANNEL_VERTICAL, static_cast<float>(p));
    }
};

/* =============== CAMERA THREAD =============== */

void cameraThread(SafeQueue<FrameData>&queue,atomic<bool>&run)
{
    cv::VideoCapture cap;
    bool usingRealCamera = false;
    int failureCount = 0;
    
    std::cout << "\n=== Camera Initialization ===" << std::endl;
    
    // Try method 1: GStreamer with libcamerasrc (Arducam 64MP)
    std::cout << "Attempting GStreamer libcamerasrc..." << std::endl;
    try {
        std::string cameraPipeline = 
            "libcamerasrc sharpness=16.0 exposure-time=50000 analogue-gain=8.0 ! "
            "video/x-raw,format=NV12,width=1920,height=1080,framerate=45/1,interlace-mode=progressive ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink sync=false max-buffers=1 drop=true";
        
        cap.open(cameraPipeline, cv::CAP_GSTREAMER);
        if (cap.isOpened()) {
            cv::Mat testFrame;
            if (cap.read(testFrame) && !testFrame.empty()) {
                std::cout << "✓ Arducam 64MP via GStreamer: SUCCESS (1920x1080 @ 45fps)" << std::endl;
                usingRealCamera = true;
            } else {
                std::cerr << "✗ GStreamer opened but cannot read frames" << std::endl;
                cap.release();
            }
        } else {
            std::cerr << "✗ GStreamer libcamerasrc: FAILED" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "✗ GStreamer exception: " << e.what() << std::endl;
        if (cap.isOpened()) cap.release();
    }
    
    // Try method 2: V4L2 camera (index 0)
    if (!usingRealCamera) {
        std::cout << "Attempting default camera (index 0)..." << std::endl;
        try {
            cap.open(0);
            if (cap.isOpened()) {
                cv::Mat testFrame;
                if (cap.read(testFrame) && !testFrame.empty()) {
                    std::cout << "✓ Default camera: SUCCESS" << std::endl;
                    usingRealCamera = true;
                } else {
                    std::cerr << "✗ Default camera opened but cannot read frames" << std::endl;
                    cap.release();
                }
            } else {
                std::cerr << "✗ Default camera: FAILED" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "✗ Default camera exception: " << e.what() << std::endl;
            if (cap.isOpened()) cap.release();
        }
    }
    
    // Final fallback: Synthetic test mode
    if (!usingRealCamera) {
        std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
        std::cout << "║   RUNNING IN TEST MODE                 ║" << std::endl;
        std::cout << "║   Generating synthetic frames @ 45fps  ║" << std::endl;
        std::cout << "║   With moving target for testing       ║" << std::endl;
        std::cout << "╚════════════════════════════════════════╝" << std::endl;
    }
    std::cout << "\n" << std::endl;

    int frameCount = 0;
    
    while(run)
    {
        FrameData f;

        if (usingRealCamera) {
            if (!cap.read(f.frame)) {
                failureCount++;
                // If we get 50 consecutive read failures, switch to test mode
                if (failureCount > 50) {
                    std::cerr << "WARNING: Camera read failing consistently. Switching to TEST MODE (synthetic frames)" << std::endl;
                    usingRealCamera = false;
                    cap.release();
                    failureCount = 0;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
            } else {
                failureCount = 0;  // Reset failure counter on successful read
            }
        }
        
        if (!usingRealCamera) {
            // Generate synthetic test frame with moving targets
            f.frame = cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(40, 45, 50));
            
            // Add gradient background for variety
            for (int y = 0; y < 1080; y += 20) {
                cv::line(f.frame, cv::Point(0, y), cv::Point(1920, y), 
                        cv::Scalar(50 + y/30, 40 + y/40, 55), 1);
            }
            
            // Create multiple moving objects
            // Object 1: Large circular target
            int obj1X = 960 + (int)(300 * sin(frameCount * 0.03));
            int obj1Y = 540 + (int)(200 * cos(frameCount * 0.02));
            cv::circle(f.frame, cv::Point(obj1X, obj1Y), 40, cv::Scalar(0, 0, 255), -1);
            cv::circle(f.frame, cv::Point(obj1X, obj1Y), 45, cv::Scalar(255, 255, 255), 3);
            
            // Object 2: Smaller fast-moving target
            int obj2X = 600 + (int)(250 * cos(frameCount * 0.05));
            int obj2Y = 400 + (int)(180 * sin(frameCount * 0.04));
            cv::rectangle(f.frame, cv::Point(obj2X - 25, obj2Y - 25), 
                         cv::Point(obj2X + 25, obj2Y + 25), cv::Scalar(255, 128, 0), -1);
            
            // Object 3: Another moving circle
            int obj3X = 1200 + (int)(200 * sin(frameCount * 0.025));
            int obj3Y = 700 + (int)(150 * cos(frameCount * 0.035));
            cv::circle(f.frame, cv::Point(obj3X, obj3Y), 30, cv::Scalar(255, 255, 0), -1);
            
            // Test mode label
            cv::putText(f.frame, "TEST MODE - Synthetic Frames", cv::Point(50, 50),
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
            
            frameCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(22)); // ~45 fps
        }

        f.timestamp=
        chrono::duration<double>(
        chrono::high_resolution_clock::now()
        .time_since_epoch()).count();

        queue.push(f);
    }
    
    if (usingRealCamera) {
        cap.release();
    }
}

/* =============== TRACKING THREAD =============== */

void trackingThread(SafeQueue<FrameData>&queue,atomic<bool>&run)
{
    AngleKalman kalman;
    Gimbal gimbal;

    double cx = CX;  // Will be updated to actual frame center
    double cy = CY;
    
    // Smoothing filters for servo command stability
    double lastYawDeg = 90.0;
    double lastPitchDeg = 90.0;
    double lastSentYawDeg = 90.0;
    double lastSentPitchDeg = 90.0;
    const double SMOOTHING_FACTOR = 0.35;
    const double SERVO_DEADBAND = 0.2;
    // Счётчик кадров стабилизации после движения серво
    // Пока > 0: подавляем детекцию + быстрое переобучение фона
    int servoSettleFrames = 0;

    while(run)
    {
        FrameData f=queue.pop();
        
        // Update center coordinates from actual frame size
        cx = f.frame.cols / 2.0;
        cy = f.frame.rows / 2.0;

        // === CHECK MODE FIRST - BEFORE HEAVY PROCESSING ===
        // Track previous mode state to detect transitions
        static bool prevTrackingEnabled = true;
        bool currentTrackingEnabled = trackingEnabled.load();
        bool modeJustChanged = (prevTrackingEnabled != currentTrackingEnabled);
        prevTrackingEnabled = currentTrackingEnabled;
        
        // === FIXED MODE: center servos, but still run detection below ===
        static int framesSinceFixedMode = 0;
        // learningRate for BGS: 1.0 on mode change (instant reset), 0.05 for
        // first 60 frames (fast warm-up at new static position), 0.01 normal
        double bgsLearningRate = 0.01;
        // Если серво недавно двигалось — ускоряем переобучение фона
        bool cameraSettling = (servoSettleFrames > 0);
        if (cameraSettling) {
            bgsLearningRate = 0.7;
            servoSettleFrames--;
        }
        if (!currentTrackingEnabled) {
            if (modeJustChanged) {
                framesSinceFixedMode = 0;
                bgsLearningRate = 1.0;  // instant BGS reset to new background
                std::cout << ">>> FIXED MODE ACTIVATED - centering servos, resetting BGS <<<" << std::endl;
                for (int i = 0; i < 10; i++) {
                    setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
                    setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
                }
            } else {
                framesSinceFixedMode++;
                // Fast warm-up for first 30 frames so BGS learns static background quickly
                bgsLearningRate = (framesSinceFixedMode <= 30) ? 0.1 : 0.01;
                if (framesSinceFixedMode <= 60) {
                    setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
                    setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
                } else if (framesSinceFixedMode % 10 == 0) {
                    setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
                    setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
                }
            }
            lastYawDeg   = 90.0;
            lastPitchDeg = 90.0;
        }

        // === DETECTION - runs in BOTH FIXED and TRACKING modes ===
        cv::Mat display = f.frame.clone();  // For visualization

        // Resize for faster processing (half resolution)
        cv::Mat resized;
        cv::resize(f.frame, resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        // Detect motion on downscaled frame
        Detection d = centroid(gray, 0, 0, bgsLearningRate);
        
        // Пока камера стабилизируется после движения серво — игнорируем детекцию.
        // Это разрывает петлю: фон сместился → детектор видит фон → серво двигается → фон снова смещается.
        if (cameraSettling) {
            d.valid = false;
            d.all_boxes.clear();
        }
        
        // Scale detection coordinates back to original resolution
        if (d.valid) {
            d.x *= 2.0;
            d.y *= 2.0;
            d.box_x *= 2;
            d.box_y *= 2;
            d.box_w *= 2;
            d.box_h *= 2;
        }
        
        // Scale all detected boxes back to original resolution
        for (auto& box : d.all_boxes) {
            box.x *= 2;
            box.y *= 2;
            box.width *= 2;
            box.height *= 2;
        }

        double theta=0;
        double phi=0;

        if(d.valid)
            pixelToAngles(d.x, d.y, cx, cy, theta, phi);

        AngleState s=
            kalman.update(theta,phi,d.valid);

        if (USE_PREDICTIVE_CONTROL) {
            predictFuture(s,SYSTEM_DELAY);
        }

        // === SERVO CONTROL (only in TRACKING mode) ===
        
        // Debug: Show tracking state
        static int frameCounter = 0;
        frameCounter++;
        if (frameCounter % 30 == 0) {  // Print every 30 frames (~1 second at 30fps)
            std::cout << "\n=== TRACKING STATUS ===" << std::endl;
            std::cout << "Mode: " << (currentTrackingEnabled ? "TRACKING" : "FIXED") << std::endl;
            std::cout << "Objects detected: " << d.all_boxes.size() << std::endl;
            std::cout << "Valid target: " << (d.valid ? "YES" : "NO") << std::endl;
            if (d.valid) {
                std::cout << "Target position: (" << d.x << ", " << d.y << ")" << std::endl;
            }
            std::cout << "Current servo: Yaw=" << lastYawDeg << "° Pitch=" << lastPitchDeg << "°" << std::endl;
            std::cout << "===================\n" << std::endl;
        }
        
        double yawDeg = lastYawDeg;
        double pitchDeg = lastPitchDeg;
        
        // TRACKING MODE: Follow detected objects
        if (d.valid) {
            // Convert angles to servo degrees (0-180)
            double thetaDeg = (s.theta * 180.0 / M_PI);
            double phiDeg = (s.phi * 180.0 / M_PI);
            
            // Adaptive gain: высокий когда объект далеко от центра, низкий когда близко
            // Это позволяет быстро догонять и не перелетать когда уже рядом
            double errorMag = std::sqrt(thetaDeg * thetaDeg + phiDeg * phiDeg);
            const double GAIN_MIN = 2.0;   // gain когда объект в центре
            const double GAIN_MAX = 3.5;   // gain когда объект далеко
            const double ERROR_MAX = 20.0; // угол (°) при котором достигается max gain
            double adaptiveGain = GAIN_MIN + (GAIN_MAX - GAIN_MIN) * std::min(errorMag / ERROR_MAX, 1.0);
            
            yawDeg = 90.0 - (thetaDeg * adaptiveGain);
            pitchDeg = 90.0 - (phiDeg * adaptiveGain);
            
            // Debug output
            std::cout << "theta=" << thetaDeg << "° phi=" << phiDeg 
                     << "° err=" << errorMag << "° gain=" << adaptiveGain
                     << " -> Yaw=" << yawDeg << "° Pitch=" << pitchDeg << "°" << std::endl;
            
            // Safety limits
            if (yawDeg < 30.0) yawDeg = 30.0;
            if (yawDeg > 150.0) yawDeg = 150.0;
            if (pitchDeg < 30.0) pitchDeg = 30.0;
            if (pitchDeg > 150.0) pitchDeg = 150.0;
            
            // Apply exponential smoothing
            yawDeg = SMOOTHING_FACTOR * yawDeg + (1.0 - SMOOTHING_FACTOR) * lastYawDeg;
            pitchDeg = SMOOTHING_FACTOR * pitchDeg + (1.0 - SMOOTHING_FACTOR) * lastPitchDeg;
            
            lastYawDeg = yawDeg;
            lastPitchDeg = pitchDeg;
            
            // Move servos only in TRACKING mode
            if (currentTrackingEnabled) {
                // Движение серво: запускаем счётчик стабилизации если ۖугол > 0.8°
                double moveAmount = std::abs(yawDeg - lastSentYawDeg)
                                  + std::abs(pitchDeg - lastSentPitchDeg);
                if (moveAmount > 0.8) {
                    servoSettleFrames = 9;  // 9 кадров ~0.20с
                }
                setServoAngle(PWM_CHANNEL_HORIZONTAL, static_cast<float>(yawDeg));
                setServoAngle(PWM_CHANNEL_VERTICAL, static_cast<float>(pitchDeg));
                lastSentYawDeg = yawDeg;
                lastSentPitchDeg = pitchDeg;
            }
        } else {
            // No target detected - hold last position
            yawDeg = lastYawDeg;
            pitchDeg = lastPitchDeg;
        }

        // NOTE: cx and cy MUST remain fixed at the optical center of the camera
        // (CX, CY constants). Moving them to d.x/d.y "zeroes out" the angular
        // error every frame and completely disables tracking — the servo would
        // always compute theta≈0, phi≈0 and stay at 90°.
        // (old buggy code: if(d.valid){ cx=d.x; cy=d.y; } ← REMOVED)
        


        // === VISUALIZATION ===
        
        // Draw ROI window if object is detected
        if (d.valid) {
            cv::Rect roi = computeROI(d.x, d.y, display.cols, display.rows, d.box_w, d.box_h);
            // Draw ROI rectangle with cyan color
            cv::rectangle(display, roi, cv::Scalar(255, 255, 0), 3);  // Cyan border, thickness=3
            // Add label for ROI
            cv::putText(display, "ROI", cv::Point(roi.x + 5, roi.y + 25), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        }
        
        // Draw bright green rectangles around all detected moving objects with labels
        int objCounter = 0;
        for (const auto& box : d.all_boxes) {
            cv::rectangle(display, box, cv::Scalar(0,255,0), 3);  // Thicker border
            // Add semi-transparent fill
            cv::Mat overlay = display.clone();
            cv::rectangle(overlay, box, cv::Scalar(0,255,0), -1);
            cv::addWeighted(overlay, 0.15, display, 0.85, 0, display);
            
            // Label each object
            objCounter++;
            std::string label = "OBJ" + std::to_string(objCounter);
            cv::putText(display, label, cv::Point(box.x, box.y - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        
        // Draw centroid if valid - make it more visible
        if (d.valid) {
            // Red target marker
            cv::circle(display, cv::Point(static_cast<int>(d.x), static_cast<int>(d.y)), 
                      10, cv::Scalar(0, 0, 255), -1);
            cv::circle(display, cv::Point(static_cast<int>(d.x), static_cast<int>(d.y)), 
                      15, cv::Scalar(255, 255, 255), 3);
            // Crosshair on target
            int tx = static_cast<int>(d.x);
            int ty = static_cast<int>(d.y);
            cv::line(display, cv::Point(tx - 25, ty), cv::Point(tx + 25, ty), 
                    cv::Scalar(255, 0, 0), 2);
            cv::line(display, cv::Point(tx, ty - 25), cv::Point(tx, ty + 25), 
                    cv::Scalar(255, 0, 0), 2);
        }
        
        // Draw center crosshair (use actual frame dimensions, not constants)
        int cx_int = display.cols / 2;
        int cy_int = display.rows / 2;
        cv::line(display, cv::Point(cx_int - 20, cy_int), cv::Point(cx_int + 20, cy_int), 
                cv::Scalar(255, 255, 0), 2);
        cv::line(display, cv::Point(cx_int, cy_int - 20), cv::Point(cx_int, cy_int + 20), 
                cv::Scalar(255, 255, 0), 2);
        
        // Draw info text with background for better visibility
        std::ostringstream info;
        info << std::fixed << std::setprecision(1);
        info << "Yaw: " << yawDeg << "deg  Pitch: " << pitchDeg << "deg";
        
        // Tracking mode status
        std::string modeStr = trackingEnabled.load() ? "TRACKING ON" : "FIXED 90deg";
        cv::Scalar modeColor = trackingEnabled.load() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 128, 255);
        
        // Background rectangle for text
        cv::rectangle(display, cv::Point(5, 5), cv::Point(650, 120), 
                     cv::Scalar(0, 0, 0), -1);
        cv::rectangle(display, cv::Point(5, 5), cv::Point(650, 120), 
                     cv::Scalar(0, 255, 255), 2);
        
        cv::putText(display, info.str(), cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Show tracking mode
        cv::putText(display, "Mode: " + modeStr, cv::Point(10, 55), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, modeColor, 2);
        
        // Show number of detected objects
        std::ostringstream objInfo;
        objInfo << "Objects detected: " << d.all_boxes.size();
        cv::putText(display, objInfo.str(), cv::Point(10, 80), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        if (d.valid) {
            std::ostringstream pos;
            pos << "Target: (" << static_cast<int>(d.x) << ", " << static_cast<int>(d.y) << ")";
            cv::putText(display, pos.str(), cv::Point(10, 105), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(display, "Status: Searching for target...", cv::Point(10, 105), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 128, 0), 2);
        }
        
        // Update display frame
        {
            std::lock_guard<std::mutex> lock(displayMutex);
            displayFrame = display;
            hasNewFrame = true;
        }
        
    }
}

/* =============== MAIN =============== */

int main()
{
    killPreviousInstances();
    std::cout << "============================================================================================" << std::endl;
    std::cout << "PREDICTIVE GIMBAL CONTROL SYSTEM WITH ARDUCAM 64MP" << std::endl;
    std::cout << "============================================================================================" << std::endl;
    std::cout << "Camera: Arducam 64MP (1920x1080 @ 45fps)" << std::endl;
    std::cout << "Servos: GPIO18 (Horizontal) / GPIO19 (Vertical)" << std::endl;
    std::cout << "Algorithm: Kalman Filter + Predictive Control (45ms compensation)" << std::endl;
    std::cout << "============================================================================================" << std::endl;
    
    // Initialize PWM for servo control
    std::cout << "\nInitializing servo control..." << std::endl;
    if (!initSysfsPwm()) {
        std::cerr << "FATAL ERROR: Failed to initialize PWM" << std::endl;
        return 1;
    }
    
    SafeQueue<FrameData> queue;

    atomic<bool> run(true);

    std::cout << "\nStarting camera thread..." << std::endl;
    thread cam(cameraThread, ref(queue), ref(run));

    std::cout << "Starting tracking thread..." << std::endl;
    thread track(trackingThread, ref(queue), ref(run));

    std::cout << "\n============================================================================================" << std::endl;
    std::cout << "System running!" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  F   - Toggle tracking mode (FIXED 90deg <-> TRACKING ON)" << std::endl;
    std::cout << "  Q   - Quit program" << std::endl;
    std::cout << "  ESC - Quit program" << std::endl;
    std::cout << "\n*** TRACKING MODE ENABLED BY DEFAULT ***" << std::endl;
    std::cout << "Servos will follow detected objects (5.0x sensitivity)" << std::endl;
    std::cout << "Press F to lock servos at 90deg" << std::endl;
    std::cout << "============================================================================================" << std::endl;
    
    // Create display window
    cv::namedWindow("Predictive Gimbal Control", cv::WINDOW_NORMAL);
    cv::resizeWindow("Predictive Gimbal Control", 1280, 720);
    // Keep window on top and focused for keyboard input
    cv::setWindowProperty("Predictive Gimbal Control", cv::WND_PROP_TOPMOST, 1);
    std::cout << "\n*** CLICK ON THE WINDOW TO ACTIVATE KEYBOARD CONTROL ***\n" << std::endl;
    
    // Display loop
    while (run) {
        if (hasNewFrame.load()) {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(displayMutex);
                if (!displayFrame.empty()) {
                    frame = displayFrame.clone();
                    hasNewFrame = false;
                }
            }
            
            if (!frame.empty()) {
                cv::imshow("Predictive Gimbal Control", frame);
            }
        }
        
        // Check for key press (minimal 1ms delay for fastest response)
        int key = cv::waitKey(1);
        if (key != -1) {  // If any key was pressed
            std::cout << "\n[KEY DETECTED] Code: " << key << " (char: '" << (char)key << "')" << std::endl;
        }
        
        if (key == 'q' || key == 'Q' || key == 27) {  // Q or ESC - quit
            std::cout << "\n=== QUIT KEY PRESSED ===" << std::endl;
            run = false;
            break;
        } else if (key == 'f' || key == 'F') {  // F - toggle tracking mode
            std::cout << "\n=== F KEY DETECTED - TOGGLING MODE ===" << std::endl;
            trackingEnabled.store(!trackingEnabled.load());
            if (trackingEnabled.load()) {
                std::cout << ">>> TRACKING MODE ENABLED - Servos will follow detected objects <<<" << std::endl;
            } else {
                std::cout << ">>> FIXED MODE - Servos locked at 90°, 90° <<<" << std::endl;
                std::cout << ">>> IMMEDIATE CENTERING (no delays)... <<<" << std::endl;
                // Send commands WITHOUT delays for instant response
                for (int i = 0; i < 10; i++) {
                    setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
                    setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
                    // NO SLEEP - instant burst!
                }
                std::cout << ">>> 10x instant commands sent! <<<" << std::endl;
            }
        }
    }

    std::cout << "\nShutting down..." << std::endl;
    run=false;

    cam.join();
    track.join();
    
    // Shutdown PWM and return servos to center
    std::cout << "Returning servos to center position..." << std::endl;
    setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
    setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "Disabling PWM..." << std::endl;
    shutdownSysfsPwm();
    
    std::cout << "System stopped." << std::endl;

    return 0;
}

// Завершить все процессы с именем ChtGPT5_prog_rpI5_ar64_v1 кроме текущего
void killPreviousInstances() {

    // 1. Kill all processes using /dev/video* or libcamera (lsof/ps based)
    system("lsof /dev/video* 2>/dev/null | awk 'NR>1 {print $2}' | xargs -r kill -9");
    system("pkill -9 libcamera");
    system("pkill -9 VLC");
    system("pkill -9 ffmpeg");
    system("pkill -9 gst-launch");
    system("pkill -9 python");
    system("pkill -9 test_camera");
    system("ps aux | grep libcamera | grep -v grep | awk '{print $2}' | xargs -r kill -9");

    // 2. Kill all previous instances of this program (by name, except self)
    DIR *dir = opendir("/proc");
    if (!dir) return;
    pid_t self = getpid();
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (!isdigit(entry->d_name[0])) continue;
        pid_t pid = atoi(entry->d_name);
        if (pid == self) continue;
        char cmdline[256] = {0};
        char path[256];
        snprintf(path, sizeof(path), "/proc/%d/cmdline", pid);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        size_t len = fread(cmdline, 1, sizeof(cmdline)-1, f);
        fclose(f);
        if (len > 0 && strstr(cmdline, "ChtGPT5_prog_rpI5_ar64_v1") && !strstr(cmdline, "sh -c")) {
            kill(pid, SIGKILL);
        }
    }
    closedir(dir);
}

//https://github.com/biatech665696-ai/gimbal-arducam.git