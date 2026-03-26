
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
#include <deque>
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
 * ЖУРНАЛ ИЗМЕНЕНИЙ — версия GPT5_v.0  (26 марта 2026)
 * ===========================================================================================
 *
 * [1] ИСПРАВЛЕНИЕ ЗНАКОВ УПРАВЛЕНИЯ СЕРВОПРИВОДАМИ
 *     Было:   yawDeg = 90 + theta*1.2 / pitchDeg = 90 - phi*1.2
 *     Стало:  yawDeg = 90 - theta*1.2 / pitchDeg = 90 - phi*1.2
 *     Причина: горизонтальная ось уходила в сторону ОТ объекта.
 *     Вертикальная ось исправлена отдельно после проверки на железе.
 *
 * [2] ПРОПОРЦИОНАЛЬНОЕ УПРАВЛЕНИЕ СЕРВОПРИВОДАМИ (П-регулятор)
 *     Шаг серво теперь прямо пропорционален пиксельному расстоянию
 *     объекта от центра кадра:
 *         stepYaw   = (err_x / (ширина/2))  * MAX_STEP_DEG   (MAX=6°)
 *         stepPitch = (err_y / (высота/2))  * MAX_STEP_DEG
 *     Объект у края → шаг 6°; объект у центра → шаг ~0°.
 *     Убрано экспоненциальное сглаживание позиции серво
 *     (оно мешало реакции и добавляло задержку).
 *
 * [3] УСТРАНЕНИЕ ПЕТЛИ ОБРАТНОЙ СВЯЗИ «СЕРВО → ФОН → СЕРВО»
 *     При движении серво камера сдвигается, MOG2 видит весь фон как
 *     движение и сразу двигает серво снова. Исправления:
 *     а) При старте settle — ПОЛНЫЙ ПЕРЕСОЗДАНИЯ объекта MOG2
 *        (cv::createBackgroundSubtractorMOG2 заново), стирая всю
 *        200-кадровую историю старого фона.
 *     б) Добавлен параметр reinitBGS в centroid() и флаг needsBGSReinit
 *        в tracking-треде.
 *     в) Добавлен сброс внутренних счётчиков centroid() (resetCounters)
 *        во время settle, чтобы накопленные ложные срабатывания не
 *        прорвались сразу после паузы.
 *
 * [4] ЗАМЕНА СЧЁТЧИКОВ КАДРОВ НА ТАЙМЕРЫ РЕАЛЬНОГО ВРЕМЕНИ
 *     Все задержки (warmup, settle, postSettle) переведены с кадров
 *     на std::chrono::steady_clock, что делает их независимыми от FPS:
 *         warmup   = 250 мс  (прогрев MOG2 после reinit)
 *         settle   = 200 мс  (подавление детекции после движения серво)
 *         postSettle = 300 мс (ускоренное обучение фона LR=0.3)
 *     До исправления при ~6 fps реальные задержки достигали 4+ секунд.
 *     После — стабильно ~0.7 с независимо от загрузки CPU.
 *
 * [5] УСКОРЕНИЕ ОБРАБОТКИ КАДРОВ (16 → 20 fps)
 *     а) Масштаб обработки: 0.5x → 0.25x (960×540 → 480×270).
 *        Скорость MOG2 + морфологии выросла в ~4 раза.
 *     б) Ядро морфологии: ELLIPSE 5×5 (3 итерации OPEN) →
 *        RECT 3×3 (1 итерация OPEN + 1 CLOSE).
 *     в) Gaussian blur: Size(5,5) → Size(3,3).
 *     г) Ядро морфологии объявлено static (не пересоздаётся каждый кадр).
 *     д) Фильтры контуров пересчитаны под масштаб 0.25x
 *        (площадь делится на 16, размеры bbox на 4).
 *     е) Координаты детекции масштабируются обратно ×4 в оригинал.
 *
 * [6] ПОВЫШЕНИЕ ЧУВСТВИТЕЛЬНОСТИ ДЕТЕКЦИИ (без ухудшения подавления фона)
 *     а) varThreshold MOG2: 75 → 35 (детектирует более медленное движение).
 *     б) learningRate по умолчанию: 0.01 → 0.005 (MOG2 медленнее
 *        «запоминает» объект как фон — дольше держит его как цель).
 *     в) Минимальная площадь контура: 2 → 1 px² (на масштабе 0.25x).
 *     г) Максимальная площадь:  130 → 250 px².
 *     д) solidity (плотность контура): >0.50 → >0.42.
 *     е) Тени по-прежнему удаляются threshold(fgMask, 200, 255) ДО
 *        анализа контуров; CONFIRM_FRAMES=4 — финальный фильтр шума.
 *
 * [7] АДАПТИВНЫЙ ROI
 *     Размер ROI больше не фиксированный (400×400 px), а зависит от
 *     размера bounding box обнаруженного объекта:
 *         roiW = clamp(objW * 4.0,  MIN=150, MAX=кадр/2)
 *         roiH = clamp(objH * 4.0,  MIN=150, MAX=кадр/2)
 *     Позиция ROI следует за объектом с адаптивной скоростью:
 *     при расстоянии > 50 px — мгновенный прыжок (α=1.0),
 *     иначе — плавное сглаживание (α = 0.5 + dist/100).
 *
 * [8] ОТОБРАЖЕНИЕ FPS В СТАТУСНОЙ СТРОКЕ
 *     Каждые 30 кадров в консоль выводится реальный FPS, измеренный
 *     через std::chrono (а не расчётный).
 *
 * ===========================================================================================
 * ЖУРНАЛ ИЗМЕНЕНИЙ — версия GPT5_v.0  (26 марта 2026)
 * ===========================================================================================
 *
 * [9]  Состояние centroid() инкапсулировано в класс MotionDetector
 *      (устраняет static-переменные, делает функцию потокобезопасной).
 *
 * [10] SafeQueue::pop() принимает atomic<bool>& run, чтобы не зависать
 *      навсегда при завершении программы.
 *
 * [11] Атомарная операция toggle trackingEnabled через XOR
 *      (устраняет race condition read-modify-write).
 *
 * [12] Удалён мёртвый код: классы PD, Gimbal; неиспользуемые константы
 *      BACKLASH_*, SMOOTHING_FACTOR, SERVO_DEADBAND.
 *
 * [13] Kalman: при потере цели — сохраняем позицию, только увеличиваем
 *      неопределённость (без decay к нулю).
 *
 * [14] killPreviousInstances(): заменён system("pkill -9 python") и
 *      другие опасные вызовы на точечное завершение только собственных
 *      процессов через /proc scan.
 *
 * [15] Адаптивный ROI фактически интегрирован в конвейер детекции
 *      (centroid обрабатывает вырезку, а не весь кадр).
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
constexpr bool kVerboseServoLogs = false;  // OFF — каждый cout+endl блокирует поток на ~5ms
constexpr bool kVerboseTrackingLogs = false;
constexpr bool kVerboseFrameLoopLogs = false;

// Predictive control parameters (integrated from ChatGPT5 algorithm)
constexpr double SYSTEM_DELAY = 0.05;   // Реальная задержка: ~50ms при 20fps (1 кадр capture + 1 кадр processing)
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

// Кэшируем последнее записанное duty_cycle чтобы не писать в sysfs повторно
static long lastDuty[4] = {0, 0, 0, 0};

void setServoAngle(int channel, float angle_deg)
{
    if(angle_deg < 0.0f) angle_deg = 0.0f;
    if(angle_deg > 180.0f) angle_deg = 180.0f;

    const long duty_cycle_ns = min_ns + static_cast<long>((angle_deg / 180.0f) * (max_ns - min_ns));

    // Пропускаем запись если значение не изменилось (экономим sysfs I/O)
    if (channel >= 0 && channel < 4 && lastDuty[channel] == duty_cycle_ns)
        return;

    std::string path = getPwmPath(channel) + "/duty_cycle";
    bool success = writeToFile(path, std::to_string(duty_cycle_ns));

    if (success) {
        if (channel >= 0 && channel < 4)
            lastDuty[channel] = duty_cycle_ns;
        if (kVerboseServoLogs)
            std::cout << "[SERVO] ch" << channel << " " << angle_deg << "° duty=" << duty_cycle_ns << "ns" << std::endl;
    } else {
        std::cerr << "SERVO WRITE FAILED: " << path << std::endl;
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

    // Возвращает false если run стал false (программа завершается)
    bool pop(T& out, atomic<bool>& run)
    {
        unique_lock<mutex> lock(m);
        cv.wait(lock,[&]{return !q.empty() || !run.load();});
        if (!run.load() && q.empty()) return false;

        out = q.front();
        q.pop();
        return true;
    }

    // Разбудить ожидающий поток при завершении
    void notifyAll() { cv.notify_all(); }
};

/* =============== MOTION DETECTOR (encapsulated centroid state) =============== */

class MotionDetector
{
public:
    MotionDetector()
        : bgSubtractor_(cv::createBackgroundSubtractorMOG2(200, 20, true))
        , clahe_(cv::createCLAHE(2.0, cv::Size(8, 8)))
        , kernel_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)))
        , warmupUntil_(std::chrono::steady_clock::now())
        , lastValidCenter_(-1, -1)
        , consecutiveDetections_(0)
        , consecutiveMisses_(0)
    {}

    void reinitBGS()
    {
        bgSubtractor_ = cv::createBackgroundSubtractorMOG2(200, 20, true);
        warmupUntil_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(80);
    }

    void resetCounters()
    {
        consecutiveDetections_ = 0;
        consecutiveMisses_ = 0;
        lastValidCenter_ = cv::Point2f(-1, -1);
        boxHistory_.clear();
    }

    Detection detect(cv::Mat &roi, int ox, int oy, double learningRate = 0.005)
    {
        const int MIN_DETECTIONS = 2;
        const int MAX_MISSES = 60;

        Detection d;
        d.valid = false;

        cv::Mat gray;
        if (roi.channels() > 1)
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        else
            gray = roi.clone();

        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.0);
        clahe_->apply(gray, gray);

        cv::Mat fgMask;
        bool inWarmup = (std::chrono::steady_clock::now() < warmupUntil_);
        double effectiveLR = inWarmup ? 0.5 : learningRate;
        bgSubtractor_->apply(gray, fgMask, effectiveLR);
        cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);
        if (inWarmup) fgMask = cv::Mat::zeros(fgMask.size(), fgMask.type());

        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN,  kernel_);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel_);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::pair<double, int>> validObjects;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            cv::Rect bbox = cv::boundingRect(contours[i]);
            double bboxArea = bbox.width * bbox.height;
            double solidity = (bboxArea > 0) ? (area / bboxArea) : 0;
            double aspectRatio = (double)bbox.width / (double)bbox.height;

            if (area >= 1.0 && area <= 250.0 &&
                solidity > 0.42 &&
                bbox.width >= 1 && bbox.height >= 1 &&
                bbox.width <= 60 && bbox.height <= 60 &&
                aspectRatio > 0.15 && aspectRatio < 8.0) {

                d.all_boxes.push_back(cv::Rect(bbox.x + ox, bbox.y + oy, bbox.width, bbox.height));
                validObjects.push_back(std::make_pair(area, (int)i));
            }
        }

        cv::Point2f currentCenter(-1, -1);
        bool foundCandidate = false;

        if (!validObjects.empty()) {
            std::sort(validObjects.begin(), validObjects.end(),
                     [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                         return a.first > b.first;
                     });

            int bestIdx = validObjects[0].second;
            if (lastValidCenter_.x > 0) {
                float bestDist = 1e9f;
                for (const auto& vo : validObjects) {
                    cv::Moments m = cv::moments(contours[vo.second]);
                    if (m.m00 <= 0) continue;
                    cv::Point2f c((m.m10 / m.m00) + ox, (m.m01 / m.m00) + oy);
                    float dist = cv::norm(c - lastValidCenter_);
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

                if (lastValidCenter_.x > 0) {
                    float distance = cv::norm(currentCenter - lastValidCenter_);
                    if (distance < 150.0) {
                        foundCandidate = true;
                        consecutiveDetections_++;
                        consecutiveMisses_ = 0;
                    } else {
                        consecutiveDetections_ = 0;
                        lastValidCenter_ = currentCenter;
                    }
                } else {
                    foundCandidate = true;
                    consecutiveDetections_ = 1;
                    lastValidCenter_ = currentCenter;
                }

                if (foundCandidate && consecutiveDetections_ >= MIN_DETECTIONS) {
                    d.x = currentCenter.x;
                    d.y = currentCenter.y;
                    cv::Rect bbox = cv::boundingRect(contours[idx]);
                    d.box_x = bbox.x + ox;
                    d.box_y = bbox.y + oy;
                    d.box_w = bbox.width;
                    d.box_h = bbox.height;
                    d.valid = true;
                    lastValidCenter_ = currentCenter;
                }
            }
        }

        if (!foundCandidate) {
            consecutiveMisses_++;
            consecutiveDetections_ = 0;
            if (consecutiveMisses_ > MAX_MISSES)
                lastValidCenter_ = cv::Point2f(-1, -1);
        }

        // Temporal persistence filter (CONFIRM_FRAMES кадров)
        {
            const int CONFIRM_FRAMES = 1;  // =1 отключает temporal persistence; MIN_DETECTIONS=2 уже фильтрует шум
            boxHistory_.push_back(d.all_boxes);
            if ((int)boxHistory_.size() > CONFIRM_FRAMES)
                boxHistory_.pop_front();

            std::vector<cv::Rect> confirmedBoxes;
            if ((int)boxHistory_.size() == CONFIRM_FRAMES) {
                const int margin = 20;
                for (const auto& box : boxHistory_.back()) {
                    bool confirmedInAll = true;
                    for (int fi = 0; fi < CONFIRM_FRAMES - 1 && confirmedInAll; fi++) {
                        bool foundInFrame = false;
                        for (const auto& prevBox : boxHistory_[fi]) {
                            cv::Rect b1(box.x - margin,     box.y - margin,
                                        box.width  + 2*margin, box.height + 2*margin);
                            cv::Rect b2(prevBox.x - margin, prevBox.y - margin,
                                        prevBox.width + 2*margin, prevBox.height + 2*margin);
                            if ((b1 & b2).area() > 0) { foundInFrame = true; break; }
                        }
                        if (!foundInFrame) confirmedInAll = false;
                    }
                    if (confirmedInAll) confirmedBoxes.push_back(box);
                }
            }
            d.all_boxes = confirmedBoxes;
        }

        return d;
    }

private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Mat kernel_;
    std::chrono::steady_clock::time_point warmupUntil_;
    cv::Point2f lastValidCenter_;
    int consecutiveDetections_;
    int consecutiveMisses_;
    std::deque<std::vector<cv::Rect>> boxHistory_;
};

/* =============== ROI COMPUTATION =============== */

// objW, objH — размер bounding box объекта в пикселях оригинального разрешения.
// ROI = bbox × PADDING_FACTOR, но не меньше MIN_SIZE и не больше половины кадра.
cv::Rect computeROI(double x, double y, int w, int h, int objW = 0, int objH = 0)
{
    // === ADAPTIVE SIZE ===
    const double PADDING_FACTOR = 4.0;   // ROI в 4 раза больше объекта
    const int    MIN_SIZE       = 150;   // минимальный размер ROI
    const int    MAX_SIZE       = std::min(w, h) / 2;  // не больше половины кадра

    int roiW, roiH;
    if (objW > 0 && objH > 0) {
        roiW = std::max(MIN_SIZE, (int)(objW * PADDING_FACTOR));
        roiH = std::max(MIN_SIZE, (int)(objH * PADDING_FACTOR));
    } else {
        roiW = roiH = MIN_SIZE;
    }
    roiW = std::min(roiW, MAX_SIZE);
    roiH = std::min(roiH, MAX_SIZE);

    // === SMOOTH POSITION (центр ROI плавно следует за объектом) ===
    static double smoothX = x;
    static double smoothY = y;
    // Если объект далеко — догоняем мгновенно, иначе плавно
    double dist = std::sqrt((x - smoothX)*(x - smoothX) + (y - smoothY)*(y - smoothY));
    double alpha = (dist > 50.0) ? 1.0 : (0.5 + dist / 100.0);
    alpha = std::min(1.0, alpha);
    smoothX = alpha * x + (1.0 - alpha) * smoothX;
    smoothY = alpha * y + (1.0 - alpha) * smoothY;

    int rx = (int)(smoothX - roiW / 2.0);
    int ry = (int)(smoothY - roiH / 2.0);

    rx = std::max(0, rx);
    ry = std::max(0, ry);
    if (rx + roiW > w) rx = w - roiW;
    if (ry + roiH > h) ry = h - roiH;

    return cv::Rect(rx, ry, roiW, roiH);
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
        // Лимит на prediction drift: при dt > 0.15s (3+ кадров без измерения)
        // velocity уже ненадёжна → обнуляем чтобы не уводить позицию.
        if (dt > 0.15) {
            x[2] = 0;
            x[3] = 0;
        }
        x[0] = x[0] + x[2] * dt;  // theta_new = theta_old + wtheta * dt
        x[1] = x[1] + x[3] * dt;  // phi_new = phi_old + wphi * dt

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
                x[2] = 0.88 * x[2] + 0.12 * (y0 / dt);  // Exponential smoothing
                x[3] = 0.88 * x[3] + 0.12 * (y1 / dt);
            }

            // Covariance update: P = (I - K * H) * P
            P[0] = (1.0 - K0) * P[0];
            P[1] = (1.0 - K1) * P[1];
        }
        else
        {
            // Нет измерения: гасим скорость, но ПОЗИЦИЮ ДЕРЖИМ.
            // Decay позиции вызывал drift к центру за 9 settle-кадров
            // (0.9^9=0.39×) → серво "забывал" куда смотреть → осцилляция.
            x[2] *= 0.85;
            x[3] *= 0.85;
            // x[0], x[1] НЕ гасим — серво держит последнюю позицию

            P[0] += 0.01;
            P[1] += 0.01;
        }

        // Hard clamp: theta/phi не могут быть больше ±π (±180°)
        // Если всё равно разошлось — обрезаем и обнуляем скорость
        const double MAX_ANGLE = M_PI;
        for (int i = 0; i < 2; i++) {
            if (x[i] > MAX_ANGLE || x[i] < -MAX_ANGLE) {
                x[i] = 0.0;
                x[i+2] = 0.0;  // velocity
                P[i] = 1.0;
            }
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
    MotionDetector detector;

    double cx = CX;
    double cy = CY;

    double lastYawDeg = 90.0;
    double lastPitchDeg = 90.0;
    double lastSentYawDeg = 90.0;
    double lastSentPitchDeg = 90.0;

    // Счётчик кадров стабилизации после движения серво
    int servoSettleFrames = 0;
    bool needsBGSReinit = false;

    // Последний известный ROI (для адаптивной вырезки)
    cv::Rect lastROI;

    while(run)
    {
        FrameData f;
        if (!queue.pop(f, run)) break;
        
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
        // Frame-based settle: 9 кадров с LR=0.7 после каждого шага серво.
        // Быстро учит новый фон за 9 кадров (~200ms при 45fps), потом LR=0.01.
        // Нет postSettle с LR=0.3 — объект не поглощается в фон.
        bool cameraSettling = (servoSettleFrames > 0);
        if (cameraSettling) {
            // LR=0: НЕ обновлять BGS во время дрожания камеры.
            // Старый LR=0.7 поглощал объект в фон → 2с слепота после settle.
            // Сдвиг камеры на 2-5° = 5-10px при 0.25x → MOG2 справится сам.
            bgsLearningRate = 0;
            servoSettleFrames--;
        }
        bool doReinitNow = needsBGSReinit;
        needsBGSReinit = false;
        if (!currentTrackingEnabled) {
            if (modeJustChanged) {
                framesSinceFixedMode = 0;
                bgsLearningRate = 1.0;
                std::cout << ">>> FIXED MODE ACTIVATED <<<" << std::endl;
                setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
                setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
                // Сброс Калмана при входе в FIXED: убирает накопленные ошибки
                kalman = AngleKalman();
            } else {
                framesSinceFixedMode++;
                bgsLearningRate = (framesSinceFixedMode <= 30) ? 0.1 : 0.01;
                // setServoAngle кэширует duty — повторная запись 90° бесплатна
                setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
                setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
            }
            lastYawDeg   = 90.0;
            lastPitchDeg = 90.0;
        }

        // === DETECTION - runs in BOTH FIXED and TRACKING modes ===
        cv::Mat display = f.frame.clone();  // For visualization

        // Resize to 0.25x for fast processing (480x270 instead of 960x540)
        cv::Mat resized;
        cv::resize(f.frame, resized, cv::Size(), 0.25, 0.25, cv::INTER_LINEAR);
        
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        // Reinit BGS если нужен полный сброс
        if (doReinitNow)
            detector.reinitBGS();

        // Детекция всегда на полном кадре 0.25x — ROI ломает MOG2
        // (сдвиг ROI между кадрами делает per-pixel модель фона невалидной)
        Detection d = detector.detect(gray, 0, 0, bgsLearningRate);
        
        // Noise gate: >10 объектов = шум от сдвига камеры, не реальные цели.
        // MOG2 при сдвиге видит ВСЕ пиксели как "движение" → 70-100+ контуров.
        if ((int)d.all_boxes.size() > 10) {
            d.valid = false;
            d.all_boxes.clear();
        }

        // Подавляем детекцию при settle (камера дрожит → BGS шумит).
        // Короткое окно (4 кадра ~200ms) — Kalman ДЕРЖИТ позицию (без decay),
        // серво не drift'ит к центру.
        if (cameraSettling) {
            d.valid = false;
            d.all_boxes.clear();
        }

        // ROI для визуализации (не для детекции!)
        if (d.valid) {
            lastROI = computeROI(d.x, d.y, gray.cols, gray.rows, d.box_w, d.box_h);
        }

        // Scale detection coordinates back to original resolution (0.25x -> 1x = *4)
        if (d.valid) {
            d.x *= 4.0;
            d.y *= 4.0;
            d.box_x *= 4;
            d.box_y *= 4;
            d.box_w *= 4;
            d.box_h *= 4;
        }
        
        // Scale all detected boxes back to original resolution
        for (auto& box : d.all_boxes) {
            box.x *= 4;
            box.y *= 4;
            box.width *= 4;
            box.height *= 4;
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
        
        // Debug: Show tracking state + FPS
        static int frameCounter = 0;
        static auto fpsTimer = std::chrono::steady_clock::now();
        frameCounter++;
        if (frameCounter % 30 == 0) {
            auto fpsNow = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(fpsNow - fpsTimer).count();
            double fps = 30.0 / elapsed;
            fpsTimer = fpsNow;
            std::cout << "\n=== TRACKING STATUS ===" << std::endl;
            std::cout << "FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;
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
            double thetaDeg = (s.theta * 180.0 / M_PI);
            double phiDeg   = (s.phi   * 180.0 / M_PI);

            // Kalman-сглаженная ошибка в пикселях (убирает дрожание детекции)
            // SYSTEM_DELAY=0.045: минимальное предсказание вперёд
            double ex = s.theta * F;  // >0 объект справа
            double ey = s.phi   * F;  // >0 объект снизу

            // Нормируем на cx (960) для ОБЕИХ осей — симметричный отклик.
            // Раньше cy=540 давал вертикальный шаг в 960/540=1.78x больше
            // горизонтального → серво описывало эллипс вместо круга.
            double norm_ex = ex / cx;
            double norm_ey = ey / cx;  // cx, не cy!

            // Шаг прямо пропорционален расстоянию: MAX_STEP_DEG при объекте у края
            const double MAX_STEP_DEG = 15.0;
            double stepYaw   = norm_ex * MAX_STEP_DEG;
            double stepPitch = norm_ey * MAX_STEP_DEG;

            // Применяем шаг к текущей позиции серво
            // Знаки: объект справа (ex>0) → серво вправо (yaw убывает по нашей конвенции)
            yawDeg   = lastYawDeg   - stepYaw;
            pitchDeg = lastPitchDeg - stepPitch;

            if (kVerboseTrackingLogs) {
                std::cout << "theta=" << thetaDeg << "° phi=" << phiDeg
                         << "° err=(" << (int)ex << "," << (int)ey << "px)"
                         << " step=(" << stepYaw << "°," << stepPitch << "°)"
                         << " -> Yaw=" << yawDeg << "° Pitch=" << pitchDeg << "°" << std::endl;
            }
            
            // Safety limits
            if (yawDeg < 5.0) yawDeg = 5.0;
            if (yawDeg > 175.0) yawDeg = 175.0;
            if (pitchDeg < 5.0) pitchDeg = 5.0;
            if (pitchDeg > 175.0) pitchDeg = 175.0;

            lastYawDeg = yawDeg;
            lastPitchDeg = pitchDeg;
            
            // Move servos only in TRACKING mode
            if (currentTrackingEnabled) {
                // Два порога:
                // > 0.8°: settle 200ms (подавляет детекцию пока камера дрожит).
                //          БЕЗ reinit: postSettle LR=0.3 быстро адаптирует MOG2
                //          к небольшому сдвигу без warmup-слепоты.
                // > 3.0°: settle + полный reinit (фон меняется кардинально).
                double moveAmount = std::abs(yawDeg - lastSentYawDeg)
                                  + std::abs(pitchDeg - lastSentPitchDeg);
                if (moveAmount > 2.0) {
                    servoSettleFrames = 4;  // 4 кадра ~200ms при 20fps
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

        // === TRAJECTORY: последние 30 валидных детекций ===
        static std::deque<cv::Point2f> trajHistory;

        if (d.valid) {
            trajHistory.push_back(cv::Point2f(d.x, d.y));
            if (trajHistory.size() > 30) trajHistory.pop_front();
        }

        for (int i = 0; i < (int)trajHistory.size(); i++) {
            float t = (float)i / std::max((int)trajHistory.size() - 1, 1);
            cv::Scalar col(255 * (1 - t), 0, 255 * t);  // синий→красный
            if (i > 0)
                cv::line(display, trajHistory[i-1], trajHistory[i], col, 2);
            cv::circle(display, trajHistory[i], 4, col, -1);
        }

        // === TRAJECTORY цента кадра (куда серво целится) — cyan ===
        static std::deque<cv::Point2f> aimHistory;
        if (d.valid) {
            // Центр кадра в пикселях минус остаточная ошибка Kalman после predictFuture
            // Это куда серво фактически нацелилось на последнем шаге
            aimHistory.push_back(cv::Point2f((float)(cx + s.theta * F),
                                             (float)(cy + s.phi   * F)));
            if (aimHistory.size() > 30) aimHistory.pop_front();
        }
        for (int i = 0; i < (int)aimHistory.size(); i++) {
            float t = (float)i / std::max((int)aimHistory.size() - 1, 1);
            cv::Scalar col(200 * t, 255, 200 * t);
            if (i > 0)
                cv::line(display, aimHistory[i-1], aimHistory[i], col, 2);
            cv::circle(display, aimHistory[i], 4, col, -1);
        }

        // Рисуем куда целится серво (предсказанная позиция после predictFuture)
        // ex, ey уже вычислены выше, рисуем крест в точке предсказания
        if (d.valid) {
            double predX = cx + s.theta * F;
            double predY = cy + s.phi   * F;
            int px = static_cast<int>(predX);
            int py = static_cast<int>(predY);
            if (px > 0 && px < display.cols && py > 0 && py < display.rows) {
                cv::drawMarker(display, cv::Point(px, py),
                              cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 20, 2);
                cv::putText(display, "AIM", cv::Point(px + 12, py - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
            }
        }

        // Draw ROI window if object is detected
        if (d.valid) {
            cv::Rect roi = computeROI(d.x, d.y, display.cols, display.rows, d.box_w, d.box_h);
            cv::rectangle(display, roi, cv::Scalar(255, 255, 0), 3);
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
    
    // Initialize PWM for servo control (non-fatal: GUI runs without hardware)
    std::cout << "\nInitializing servo control..." << std::endl;
    if (!initSysfsPwm()) {
        std::cerr << "WARNING: PWM init failed - running without servo control (GUI only)" << std::endl;
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
            // Атомарный toggle: XOR с 1 (true)
            bool wasEnabled = trackingEnabled.exchange(!trackingEnabled.load());
            bool nowEnabled = !wasEnabled;
            if (nowEnabled) {
                // Сброс Калмана при переключении в TRACKING чтобы не тащить
                // накопленную ошибку из FIXED-режима
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
    queue.notifyAll();

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

// Завершить предыдущие экземпляры этой программы (по имени, кроме себя)
// НЕ убиваем python/ffmpeg/VLC — они могут быть не нашими.
void killPreviousInstances() {
    const char* ownNames[] = {
        "ChtGPT5_prog_rpI5_ar64_v1",
        "GPT5_v",
        nullptr
    };

    DIR *dir = opendir("/proc");
    if (!dir) return;
    pid_t self = getpid();
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (!isdigit(entry->d_name[0])) continue;
        pid_t pid = atoi(entry->d_name);
        if (pid == self || pid <= 1) continue;
        char cmdline[512] = {0};
        char path[64];
        snprintf(path, sizeof(path), "/proc/%d/cmdline", pid);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        size_t len = fread(cmdline, 1, sizeof(cmdline)-1, f);
        fclose(f);
        if (len == 0) continue;
        bool isOurs = false;
        for (int i = 0; ownNames[i]; i++) {
            if (strstr(cmdline, ownNames[i]) && !strstr(cmdline, "sh -c")) {
                isOurs = true;
                break;
            }
        }
        if (isOurs) {
            std::cerr << "Killing previous instance PID " << pid << std::endl;
            kill(pid, SIGTERM);
        }
    }
    closedir(dir);
}

//https://github.com/biatech665696-ai/gimbal-arducam.git