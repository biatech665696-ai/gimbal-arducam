
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
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>

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

// HTTP MJPEG streaming globals
std::atomic<bool> streamRunning(false);
std::atomic<bool> remoteQuit(false);
std::atomic<bool> remoteToggle(false);
std::atomic<bool> remoteScanToggle(false);
std::atomic<bool> remoteTrajToggle(false);
cv::Mat streamFrame;
std::mutex streamMutex;
const int STREAM_PORT = 8080;
pthread_t streamThread;

// Per-connection MJPEG client handler
static void* handleHttpClient(void* arg) {
    int clientSocket = *(int*)arg;
    delete (int*)arg;

    char buffer[1024] = {0};
    recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
    std::string request(buffer);

    bool isStreamRequest = (request.rfind("GET /stream.mjpg", 0) == 0 ||
                           request.rfind("GET /video", 0) == 0);
    bool isRootRequest = (request.rfind("GET / ", 0) == 0 ||
                         request.rfind("GET /\r", 0) == 0);
    bool isCmdRequest = (request.rfind("GET /cmd/", 0) == 0);

    if (isCmdRequest) {
        std::string ok = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\nOK";
        // Extract command: /cmd/X
        size_t cmdStart = 9; // length of "GET /cmd/"
        size_t cmdEnd = request.find(' ', cmdStart);
        std::string cmd = request.substr(cmdStart, cmdEnd - cmdStart);
        if (cmd == "q") {
            std::cout << "\n[REMOTE] Quit command received" << std::endl;
            remoteQuit = true;
        } else if (cmd == "f") {
            std::cout << "\n[REMOTE] Toggle mode command received" << std::endl;
            remoteToggle = true;
        } else if (cmd == "s") {
            std::cout << "\n[REMOTE] Toggle scan command received" << std::endl;
            remoteScanToggle = true;
        } else if (cmd == "t") {
            std::cout << "\n[REMOTE] Toggle trajectory command received" << std::endl;
            remoteTrajToggle = true;
        }
        send(clientSocket, ok.c_str(), ok.length(), 0);
        close(clientSocket);
        return nullptr;
    }

    if (isRootRequest) {
        std::string html =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n\r\n"
            "<html><head><meta charset='utf-8'><title>Gimbal Control</title></head>"
            "<body style='margin:0;background:#000;display:flex;flex-direction:column;"
            "justify-content:center;align-items:center;height:100vh;outline:none' tabindex='0'>"
            "<img src='/stream.mjpg' style='max-width:100%;max-height:75vh'>"
            "<div id='status' style='color:#0f0;font:18px monospace;margin-top:8px'>"
            "Click page first, then use keyboard or buttons</div>"
            "<div style='margin-top:8px;display:flex;gap:12px;flex-wrap:wrap;justify-content:center'>"
            "<button onclick=\"sendCmd('f')\" style='font:bold 22px monospace;padding:12px 24px;"
            "background:#004;color:#0ff;border:2px solid #0ff;border-radius:8px;cursor:pointer'>"
            "F - Track/Fixed</button>"
            "<button onclick=\"sendCmd('s')\" style='font:bold 22px monospace;padding:12px 24px;"
            "background:#004;color:#0ff;border:2px solid #0ff;border-radius:8px;cursor:pointer'>"
            "S - Scan</button>"
            "<button onclick=\"sendCmd('t')\" style='font:bold 22px monospace;padding:12px 24px;"
            "background:#004;color:#0ff;border:2px solid #0ff;border-radius:8px;cursor:pointer'>"
            "T - Trajectory</button>"
            "<button onclick=\"sendCmd('q')\" style='font:bold 22px monospace;padding:12px 24px;"
            "background:#400;color:#f88;border:2px solid #f00;border-radius:8px;cursor:pointer'>"
            "Q - Quit</button>"
            "</div>"
            "<script>"
            "function sendCmd(cmd){"
            "  var st=document.getElementById('status');"
            "  st.textContent='Sending: '+cmd.toUpperCase()+'...';"
            "  st.style.color='#ff0';"
            "  var x=new XMLHttpRequest();"
            "  x.open('GET','/cmd/'+cmd,true);"
            "  x.timeout=3000;"
            "  x.onload=function(){st.textContent='Sent: '+cmd.toUpperCase()+' OK';st.style.color='#0f0';};"
            "  x.onerror=function(){st.textContent='Error sending '+cmd.toUpperCase();st.style.color='#f00';};"
            "  x.ontimeout=function(){st.textContent='Timeout sending '+cmd.toUpperCase();st.style.color='#f00';};"
            "  x.send();"
            "}"
            "var body=document.body;"
            "body.addEventListener('click',function(){body.focus();});"
            "body.focus();"
            "document.addEventListener('keydown',function(e){"
            "  var k=e.key.toLowerCase();"
            "  if(k=='q'||k=='escape'||k=='f'||k=='s'||k=='t'){"
            "    e.preventDefault();"
            "    sendCmd(k=='escape'?'q':k);"
            "  }"
            "});"
            "</script>"
            "</body></html>";
        send(clientSocket, html.c_str(), html.length(), 0);
        close(clientSocket);
        return nullptr;
    }

    if (!isStreamRequest) {
        std::string notFound = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n404";
        send(clientSocket, notFound.c_str(), notFound.length(), 0);
        close(clientSocket);
        return nullptr;
    }

    // Send MJPEG stream
    std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    send(clientSocket, header.c_str(), header.length(), 0);

    while (streamRunning) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(streamMutex);
            if (streamFrame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            frame = streamFrame.clone();
        }

        std::vector<uchar> jpg;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
        if (!cv::imencode(".jpg", frame, jpg, params) || jpg.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::ostringstream frameHeader;
        frameHeader << "--frame\r\n"
                   << "Content-Type: image/jpeg\r\n"
                   << "Content-Length: " << jpg.size() << "\r\n\r\n";

        std::string hdr = frameHeader.str();
        ssize_t sent = send(clientSocket, hdr.c_str(), hdr.length(), MSG_NOSIGNAL);
        if (sent <= 0) break;
        sent = send(clientSocket, jpg.data(), jpg.size(), MSG_NOSIGNAL);
        if (sent <= 0) break;
        sent = send(clientSocket, "\r\n", 2, MSG_NOSIGNAL);
        if (sent <= 0) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    close(clientSocket);
    return nullptr;
}

void* httpStreamServer(void* arg) {
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        std::cerr << "[HTTP] Failed to create socket" << std::endl;
        return nullptr;
    }

    int opt = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(STREAM_PORT);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "[HTTP] Failed to bind port " << STREAM_PORT << ": " << strerror(errno) << std::endl;
        close(serverSocket);
        return nullptr;
    }

    if (listen(serverSocket, 5) < 0) {
        std::cerr << "[HTTP] Failed to listen" << std::endl;
        close(serverSocket);
        return nullptr;
    }

    std::cout << "\u2713 HTTP MJPEG stream: http://169.254.67.80:" << STREAM_PORT << "/" << std::endl;
    std::cout << "  Direct: http://169.254.67.80:" << STREAM_PORT << "/stream.mjpg" << std::endl;

    while (streamRunning) {
        sockaddr_in clientAddr{};
        socklen_t clientLen = sizeof(clientAddr);
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientLen);
        if (clientSocket < 0) continue;

        pthread_t clientThread;
        int* argSock = new int(clientSocket);
        if (pthread_create(&clientThread, nullptr, handleHttpClient, argSock) == 0) {
            pthread_detach(clientThread);
        } else {
            handleHttpClient(argSock);
        }
    }

    close(serverSocket);
    return nullptr;
}

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
std::atomic<bool> scanEnabled(false);     // Scan mode OFF by default
std::atomic<bool> scanActiveNow(false);   // True when scan is currently driving servos
std::atomic<bool> trajectoryEnabled(true); // Trajectory drawing ON by default

// === DUAL-LOOP SERVO: 100Hz interpolation + inter-frame velocity prediction ===
// Outer loop (tracking ~17Hz): detection → PD → publishes target + velocity
// Inner loop (servo 100Hz): extrapolates target along velocity between detections
std::atomic<double> servoTargetYaw(90.0);
std::atomic<double> servoTargetPitch(90.0);
std::atomic<bool>   servoInstantSnap(false);
std::atomic<double> servoActualYaw(90.0);
std::atomic<double> servoActualPitch(90.0);
std::atomic<double> servoVelYaw(0.0);     // object velocity in servo deg/s
std::atomic<double> servoVelPitch(0.0);
std::atomic<double> servoUpdateTime(0.0); // steady_clock seconds when target was last set

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

/* =============== MOG2 + OPTICAL FLOW HYBRID =============== */
// MOG2 для per-pixel адаптивного детектирования переднего плана
// (проверенная детекция маленьких объектов) + sparse LK optical flow
// для определения движения камеры → адаптивный learning rate MOG2:
//   - камера движется (flowMag > 2px): LR=0.5 → поглощение нового фона за 1-2 кадра
//   - переходный период (3 кадра после остановки): LR=0.2
//   - стабильно: LR=0.005 → медленное обучение, объект остаётся foreground
// Устраняет 5-кадровый слепой зазор v0 при сохранении превосходной детекции MOG2.

class MotionDetector
{
public:
    MotionDetector()
        : mog2_(cv::createBackgroundSubtractorMOG2(500, 16.0, false))
        , kernel2_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)))
        , kernel3_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)))
        , lastValidCenter_(-1, -1)
        , prevValidCenter_(-1, -1)
        , consecutiveDetections_(0)
        , consecutiveMisses_(0)
        , lastFGPixels_(0)
        , lastRawContours_(0)
        , lastGlobalFlow_(0, 0)
        , framesSinceMotion_(999)
        , prevRawFG_(0)
        , cooldownFrames_(0)
        , postCooldownLR_(0)
    {
        mog2_->setNMixtures(5);
        mog2_->setComplexityReductionThreshold(0.05);
        mog2_->setBackgroundRatio(0.9);
    }

    int lastFGPixels() const { return lastFGPixels_; }
    int lastRawContours() const { return lastRawContours_; }
    cv::Point2f lastGlobalFlow() const { return lastGlobalFlow_; }

    void reinitBGS() {
        mog2_ = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
        mog2_->setNMixtures(5);
        mog2_->setComplexityReductionThreshold(0.05);
        mog2_->setBackgroundRatio(0.9);
        prevGray_ = cv::Mat();
        prevRawFG_ = 0;
        cooldownFrames_ = 0;
        postCooldownLR_ = 0;
    }
    void resetCounters()
    {
        consecutiveDetections_ = 0;
        consecutiveMisses_ = 0;
        lastValidCenter_ = cv::Point2f(-1, -1);
        reinitBGS();
    }
    void resetConsecutive() { consecutiveDetections_ = 0; }

    Detection detect(cv::Mat &roi, int ox, int oy, double /*learningRate*/ = 0.0)
    {
        const int MIN_DETECTIONS = 1;
        const int MAX_MISSES = 60;

        Detection d;
        d.valid = false;

        cv::Mat gray;
        if (roi.channels() > 1)
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        else
            gray = roi.clone();

        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0.8);

        // === CAMERA MOTION ESTIMATION (sparse LK + affine) ===
        lastGlobalFlow_ = cv::Point2f(0, 0);

        if (!prevGray_.empty() && prevGray_.size() == gray.size()) {
            std::vector<cv::Point2f> prevPts;
            cv::goodFeaturesToTrack(prevGray_, prevPts, 100, 0.01, 10);

            if (prevPts.size() >= 4) {
                std::vector<cv::Point2f> currPts;
                std::vector<uchar> status;
                std::vector<float> err;
                cv::calcOpticalFlowPyrLK(prevGray_, gray, prevPts, currPts, status, err,
                    cv::Size(21, 21), 3);

                std::vector<cv::Point2f> goodPrev, goodCurr;
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i] && err[i] < 12.0f) {
                        goodPrev.push_back(prevPts[i]);
                        goodCurr.push_back(currPts[i]);
                    }
                }

                if (goodPrev.size() >= 4) {
                    cv::Mat inlierMask;
                    lastAffine_ = cv::estimateAffinePartial2D(
                        goodPrev, goodCurr, inlierMask, cv::RANSAC, 3.0);
                    if (!lastAffine_.empty()) {
                        lastGlobalFlow_ = cv::Point2f(
                            (float)lastAffine_.at<double>(0, 2),
                            (float)lastAffine_.at<double>(1, 2));
                    }
                }
            }
        }

        float flowMag = std::sqrt(lastGlobalFlow_.x * lastGlobalFlow_.x +
                                   lastGlobalFlow_.y * lastGlobalFlow_.y);

        // === DUAL-MODE DETECTION (select best, not OR) ===
        // Mode A: MOG2           — when camera stable, background model adapted
        // Mode B: Compensated frame-diff — during/after camera motion (affine warp)
        cv::Mat fgMask;

        // --- Always feed MOG2 to keep background model updated ---
        cv::Mat maskA;
        double lr;
        if (flowMag >= 2.0f) {
            lr = 0.5;
        } else if (prevRawFG_ > 5000 && flowMag < 1.0f) {
            lr = 0.3;
        } else if (framesSinceMotion_ < 5) {
            lr = 0.15;
        } else {
            lr = 0.008;
            if (postCooldownLR_ > 0) { lr = 0.08; postCooldownLR_--; }
        }
        mog2_->apply(gray, maskA, lr);
        cv::threshold(maskA, maskA, 200, 255, cv::THRESH_BINARY);
        int rawFG = cv::countNonZero(maskA);
        prevRawFG_ = rawFG;

        // --- Mode B: Compensated frame-diff ---
        cv::Mat maskB;
        bool haveMaskB = false;
        if (!lastAffine_.empty() && !prevGray_.empty() && prevGray_.size() == gray.size()) {
            cv::Mat warpedPrev;
            cv::warpAffine(prevGray_, warpedPrev, lastAffine_, gray.size(),
                           cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
            cv::absdiff(warpedPrev, gray, maskB);
            cv::threshold(maskB, maskB, 25, 255, cv::THRESH_BINARY);

            const int E = std::max(5, (int)std::ceil(flowMag) + 3);
            if (E < maskB.rows / 2 && E < maskB.cols / 2) {
                maskB.rowRange(0, E).setTo(0);
                maskB.rowRange(maskB.rows - E, maskB.rows).setTo(0);
                maskB.colRange(0, E).setTo(0);
                maskB.colRange(maskB.cols - E, maskB.cols).setTo(0);
            }
            haveMaskB = true;
        }

        // --- Select best mode ---
        if (rawFG < 1500 && flowMag < 2.0f) {
            // Mode A: MOG2 is clean and camera mostly stable
            fgMask = maskA;
        } else if (haveMaskB && flowMag >= 1.0f) {
            // Mode B: camera actually moving → use compensated frame-diff
            fgMask = maskB;
        } else if (rawFG < 3000) {
            // Transition: camera barely moving, MOG2 recovering — use MOG2 anyway
            fgMask = maskA;
        } else if (haveMaskB) {
            // MOG2 completely broken, no choice — use frame-diff
            fgMask = maskB;
        } else {
            fgMask = maskA;
        }

        lastFGPixels_ = rawFG;

        // Update motion tracking
        if (flowMag >= 2.0f)
            framesSinceMotion_ = 0;
        else if (framesSinceMotion_ < 999)
            framesSinceMotion_++;

        prevGray_ = gray.clone();

        // Morphology: skip OPEN (would erode tiny object), CLOSE connects nearby
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel3_);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        lastFGPixels_ = cv::countNonZero(fgMask);
        lastRawContours_ = (int)contours.size();

        // === ФИЛЬТРАЦИЯ КОНТУРОВ ===
        std::vector<std::pair<double, int>> validObjects;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            cv::Rect bbox = cv::boundingRect(contours[i]);
            double bboxArea = bbox.width * bbox.height;
            double solidity = (bboxArea > 0) ? (area / bboxArea) : 0;
            double aspectRatio = (double)bbox.width / (double)bbox.height;

            const int EDGE_MARGIN = 5;
            bool atEdge = (bbox.x <= EDGE_MARGIN || bbox.y <= EDGE_MARGIN ||
                          bbox.x + bbox.width >= roi.cols - EDGE_MARGIN ||
                          bbox.y + bbox.height >= roi.rows - EDGE_MARGIN);

            if (!atEdge &&
                area >= 3.0 && area <= 1500.0 &&
                solidity > 0.2 &&
                bbox.width >= 2 && bbox.height >= 2 &&
                bbox.width <= 100 && bbox.height <= 100 &&
                aspectRatio > 0.12 && aspectRatio < 8.0) {

                d.all_boxes.push_back(cv::Rect(bbox.x + ox, bbox.y + oy, bbox.width, bbox.height));
                validObjects.push_back(std::make_pair(area, (int)i));
            }
        }

        // === ВЫБОР ЛУЧШЕГО ОБЪЕКТА ===
        cv::Point2f currentCenter(-1, -1);
        bool foundCandidate = false;

        // Adaptive distance threshold: larger when camera moving fast
        // After 3+ misses: accept any closest contour (re-acquire mode)
        float maxDist = (consecutiveMisses_ >= 3) ? 9999.0f : (80.0f + flowMag * 15.0f);

        if (!validObjects.empty()) {
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

            cv::Moments m = cv::moments(contours[bestIdx]);
            if (m.m00 > 0) {
                currentCenter.x = (m.m10 / m.m00) + ox;
                currentCenter.y = (m.m01 / m.m00) + oy;

                if (lastValidCenter_.x > 0) {
                    float distance = cv::norm(currentCenter - lastValidCenter_);
                    if (distance < maxDist) {
                        // Direction check: reject if moving backward (ghost detection)
                        bool directionOK = true;
                        if (prevValidCenter_.x > 0 && distance > 3.0f) {
                            cv::Point2f prevDir = lastValidCenter_ - prevValidCenter_;
                            cv::Point2f newDir = currentCenter - lastValidCenter_;
                            float dot = prevDir.x * newDir.x + prevDir.y * newDir.y;
                            float prevMag = cv::norm(prevDir);
                            if (prevMag > 3.0f && dot < -0.5f * prevMag * distance) {
                                directionOK = false;
                            }
                        }
                        if (directionOK) {
                            foundCandidate = true;
                            consecutiveDetections_++;
                            consecutiveMisses_ = 0;
                        } else {
                            consecutiveDetections_ = 0;
                            // Update anchor so we don't get stuck on stale reference points
                            prevValidCenter_ = lastValidCenter_;
                            lastValidCenter_ = currentCenter;
                        }
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
                    cv::Rect bbox = cv::boundingRect(contours[bestIdx]);
                    d.box_x = bbox.x + ox;
                    d.box_y = bbox.y + oy;
                    d.box_w = bbox.width;
                    d.box_h = bbox.height;
                    d.valid = true;
                    prevValidCenter_ = lastValidCenter_;
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

        // DIAG: why detection failed
        if (!d.valid) {
            const char* reason = "unknown";
            if (validObjects.empty()) reason = "NO_VALID_OBJ";
            else if (lastValidCenter_.x < 0) reason = "NO_ANCHOR";
            else {
                // find actual closest distance
                float closestDist = 1e9f;
                for (const auto& vo : validObjects) {
                    cv::Moments mm = cv::moments(contours[vo.second]);
                    if (mm.m00 <= 0) continue;
                    cv::Point2f cc((mm.m10/mm.m00)+ox, (mm.m01/mm.m00)+oy);
                    float dd = cv::norm(cc - lastValidCenter_);
                    if (dd < closestDist) closestDist = dd;
                }
                if (closestDist >= maxDist) reason = "DIST_REJECT";
                else reason = "DIR_REJECT";
            }
            fprintf(stderr, "MISS[%s]: validObj=%d flow=%.1f rawFG=%d ctr=%d maxD=%.0f miss=%d",
                    reason, (int)validObjects.size(), flowMag, rawFG, lastRawContours_,
                    maxDist, consecutiveMisses_);
            if (!validObjects.empty() && lastValidCenter_.x > 0) {
                float closestDist = 1e9f;
                for (const auto& vo : validObjects) {
                    cv::Moments mm = cv::moments(contours[vo.second]);
                    if (mm.m00 <= 0) continue;
                    cv::Point2f cc((mm.m10/mm.m00)+ox, (mm.m01/mm.m00)+oy);
                    float dd = cv::norm(cc - lastValidCenter_);
                    if (dd < closestDist) closestDist = dd;
                }
                fprintf(stderr, " closestDist=%.1f lastValid=(%.0f,%.0f)",
                        closestDist, lastValidCenter_.x, lastValidCenter_.y);
            }
            fprintf(stderr, "\n");
        }

        return d;
    }

private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_;
    cv::Mat kernel2_;
    cv::Mat kernel3_;
    cv::Mat prevGray_;    // uint8, for LK tracking
    cv::Mat lastAffine_;  // prev→curr affine for compensated frame diff
    cv::Point2f lastValidCenter_;
    cv::Point2f prevValidCenter_;  // one before lastValidCenter_ for direction check
    int consecutiveDetections_;
    int consecutiveMisses_;
    int lastFGPixels_;
    int lastRawContours_;
    cv::Point2f lastGlobalFlow_;
    int framesSinceMotion_;
    int prevRawFG_;
    int cooldownFrames_;  // frames since last FG gate/noise gate
    int postCooldownLR_;  // frames of elevated LR after cooldown ends
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
    if (!cap.isOpened()) {
        std::cout << "Attempting default camera (index 0)..." << std::endl;
        try {
            cap.open(0);
            if (cap.isOpened()) {
                cv::Mat testFrame;
                if (cap.read(testFrame) && !testFrame.empty()) {
                    std::cout << "✓ Default camera: SUCCESS" << std::endl;
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
    
    if (!cap.isOpened()) {
        std::cerr << "FATAL: No camera available. Exiting camera thread." << std::endl;
        run = false;
        return;
    }
    std::cout << std::endl;
    
    while(run)
    {
        FrameData f;

        if (!cap.read(f.frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        f.timestamp=
        chrono::duration<double>(
        chrono::high_resolution_clock::now()
        .time_since_epoch()).count();

        queue.push(f);
    }
    
    cap.release();
}

/* =============== DUAL-LOOP SERVO THREAD (100Hz) =============== */
// Inner loop: between detections (~60ms gap at 17fps), extrapolates
// servo target along object velocity — servo tracks continuously,
// not just when a new frame arrives.

void servoThread(std::atomic<bool>& run)
{
    double currentYaw   = servoTargetYaw.load();
    double currentPitch = servoTargetPitch.load();
    const double ALPHA = 0.8;  // faster convergence to predicted target

    while (run.load()) {
        double baseYaw   = servoTargetYaw.load();
        double basePitch = servoTargetPitch.load();

        if (servoInstantSnap.exchange(false)) {
            currentYaw   = baseYaw;
            currentPitch = basePitch;
        } else {
            // Inter-frame velocity prediction:
            // Between detections, advance target along object velocity vector.
            // This is the "inner loop" — fills the 60ms gap between frames.
            double tUpdate = servoUpdateTime.load();
            double now = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            double dt = now - tUpdate;

            double targetYaw   = baseYaw;
            double targetPitch = basePitch;

            // Predict for 0..0.15s after last detection (~2.5 frames)
            if (dt > 0.005 && dt < 0.15 && tUpdate > 0.0) {
                targetYaw   += servoVelYaw.load()   * dt;
                targetPitch += servoVelPitch.load()  * dt;
                targetYaw   = std::clamp(targetYaw,   5.0, 175.0);
                targetPitch = std::clamp(targetPitch, 5.0, 175.0);
            }

            currentYaw   += ALPHA * (targetYaw   - currentYaw);
            currentPitch += ALPHA * (targetPitch - currentPitch);
        }

        setServoAngle(PWM_CHANNEL_HORIZONTAL, static_cast<float>(currentYaw));
        setServoAngle(PWM_CHANNEL_VERTICAL,   static_cast<float>(currentPitch));

        servoActualYaw.store(currentYaw);
        servoActualPitch.store(currentPitch);

        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 100Hz
    }
}

/* =============== TRACKING THREAD =============== */

void trackingThread(SafeQueue<FrameData>&queue,atomic<bool>&run)
{
    MotionDetector detector;

    double cx = CX;
    double cy = CY;

    double lastYawDeg = 90.0;
    double lastPitchDeg = 90.0;
    double lastSentYawDeg = 90.0;
    double lastSentPitchDeg = 90.0;

    // === SCAN MODE STATE ===
    double scanYawDeg = 0.0;              // Current scan position
    int scanDirection = 1;                // 1 = forward (0→180), -1 = backward (180→0)
    const double SCAN_STEP = 30.0;        // Degrees per step
    const double SCAN_DWELL_SEC = 1.0;    // Seconds to dwell at each position
    auto scanStepTime = std::chrono::steady_clock::now();
    bool scanDwelling = false;            // True while waiting at a scan position
    auto noDetectionSince = std::chrono::steady_clock::now();
    const double SCAN_START_DELAY = 3.0;  // Seconds without detection before scan starts
    bool wasScanning = false;

    // === OPTICAL FLOW + P-CONTROLLER ===
    // OF детектирует каждый кадр без settle.
    // Глобальное движение камеры компенсируется медианой flow.

    // === PREDICTIVE COAST STATE ===
    double coastVx = 0, coastVy = 0;     // last known velocity (px/s in full-res)
    double coastLastX = 0, coastLastY = 0; // last detected position
    auto coastLastDetectTime = std::chrono::steady_clock::now();

    // Для визуализации
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
        
        // === FIXED MODE ===
        if (!currentTrackingEnabled) {
            if (modeJustChanged) {
                std::cout << ">>> FIXED MODE ACTIVATED <<<" << std::endl;
            }
            servoTargetYaw.store(90.0);
            servoTargetPitch.store(90.0);
            servoInstantSnap.store(true);
            lastYawDeg   = 90.0;
            lastPitchDeg = 90.0;
        }

        // При переключении FIXED → TRACKING: reset detector
        if (modeJustChanged && currentTrackingEnabled) {
            detector.reinitBGS();
        }

        // === FRAME PREPARATION ===
        cv::Mat display = f.frame.clone();
        cv::Mat resized;
        cv::resize(f.frame, resized, cv::Size(), 0.25, 0.25, cv::INTER_LINEAR);
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        // === DETECTION (каждый кадр, без settle) ===
        Detection d = detector.detect(gray, 0, 0);

        // Noise gate: >5 объектов после фильтрации = шум MOG2
        if ((int)d.all_boxes.size() > 5) {
            d.valid = false;
            d.all_boxes.clear();
            detector.resetConsecutive();
        }

        // Scale → full-res
        if (d.valid) {
            d.x *= 4; d.y *= 4;
            d.box_x *= 4; d.box_y *= 4; d.box_w *= 4; d.box_h *= 4;
        }
        for (auto& box : d.all_boxes) {
            box.x *= 4; box.y *= 4; box.width *= 4; box.height *= 4;
        }

        // === ПРЯМОЕ НАВЕДЕНИЕ ===
        // Объект на ex пикселей от центра = ex * (FOV/width) градусов от серво
        // Коэффициент 1.5x компенсирует задержку камера→детект→серво (~50-100мс)
        // === VELOCITY ESTIMATION (for lead correction) ===
        static std::deque<std::tuple<double, double, double>> velHistory; // time, worldX_deg, worldY_deg
        if (modeJustChanged) velHistory.clear();
        if (d.valid) {
            double now = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            // World position = servo angle + object offset from center (in degrees)
            double worldX = lastYawDeg   + (d.x - cx) * (72.0 / 1920.0);
            double worldY = lastPitchDeg + (d.y - cy) * (72.0 / 1920.0);
            velHistory.push_back({now, worldX, worldY});
            while (velHistory.size() > 8) velHistory.pop_front();
        }

        double yawDeg = lastYawDeg;
        double pitchDeg = lastPitchDeg;
        const double DEG_PER_PX = 72.0 / 1920.0 * 1.0;  // 0.03938 deg/px (1.05x, never overshoots center)
        // const double DEG_PER_PX = 72.0 / 1920.0 * 1.05;  // 0.03938 deg/px (1.05x, never overshoots center)

        // === SCAN MODE LOGIC ===
        bool scanActive = false;
        if (scanEnabled.load() && currentTrackingEnabled) {
            if (d.valid) {
                // Object detected — stop scanning, reset timer
                noDetectionSince = std::chrono::steady_clock::now();
                if (wasScanning) {
                    std::cout << "[SCAN] Object detected! Switching to tracking." << std::endl;
                    wasScanning = false;
                    scanDwelling = false;
                }
            } else {
                // No detection — check if we should start/continue scanning
                auto now = std::chrono::steady_clock::now();
                double noDetSec = std::chrono::duration<double>(now - noDetectionSince).count();

                if (noDetSec >= SCAN_START_DELAY) {
                    scanActive = true;
                    scanActiveNow = true;

                    if (!wasScanning) {
                        // Just entered scan mode — start from current position
                        wasScanning = true;
                        scanYawDeg = std::round(lastYawDeg / SCAN_STEP) * SCAN_STEP;
                        if (scanYawDeg < 0) scanYawDeg = 0;
                        if (scanYawDeg > 180) scanYawDeg = 180;
                        // If >= 90° go toward 180 first; if < 90° go toward 0 first
                        scanDirection = (scanYawDeg >= 90.0) ? 1 : -1;
                        scanDwelling = true;
                        scanStepTime = now;
                        detector.reinitBGS();
                        std::cout << "[SCAN] Starting scan at " << scanYawDeg
                                  << " deg (dir=" << scanDirection << ")" << std::endl;
                    }

                    // Dwell timeout — move to next step
                    double dwellElapsed = std::chrono::duration<double>(now - scanStepTime).count();
                    if (scanDwelling && dwellElapsed >= SCAN_DWELL_SEC) {
                        // Advance to next scan position
                        scanYawDeg += scanDirection * SCAN_STEP;

                        // Reverse direction at limits
                        if (scanYawDeg > 180.0) {
                            scanYawDeg = 180.0;
                            scanDirection = -1;
                        } else if (scanYawDeg < 0.0) {
                            scanYawDeg = 0.0;
                            scanDirection = 1;
                        }

                        scanStepTime = now;
                        detector.reinitBGS();  // Reset BGS after servo moves
                        std::cout << "[SCAN] Step -> " << scanYawDeg << " deg (dir=" << scanDirection << ")" << std::endl;
                    }

                    // Apply scan servo position
                    servoTargetYaw.store(scanYawDeg);
                    servoTargetPitch.store(90.0);
                    servoInstantSnap.store(true);
                    lastYawDeg = scanYawDeg;
                    lastPitchDeg = 90.0;
                    yawDeg = scanYawDeg;
                    pitchDeg = 90.0;
                }
            }
        } else {
            // Scan disabled or tracking disabled — reset scan state
            if (wasScanning) {
                std::cout << "[SCAN] Scan mode deactivated." << std::endl;
                wasScanning = false;
                scanDwelling = false;
            }
            scanActiveNow = false;
            noDetectionSince = std::chrono::steady_clock::now();
        }

        if (d.valid && currentTrackingEnabled && !scanActive) {
            double ex = d.x - cx;
            double ey = d.y - cy;

            // Lead correction: predict where object will be after detection delay
            // Require 3+ points for reliable velocity (2 points too noisy)
            // Velocity from last 2 points (instantaneous, not averaged)
            // Acceleration from comparing recent vs older velocity
            double vx = 0, vy = 0;  // world velocity in deg/s
            double ax = 0, ay = 0;  // world acceleration in deg/s²
            bool hasVelocity = false;
            int n = velHistory.size();
            if (n >= 2) {
                auto& [t0, x0, y0] = velHistory[n-2];
                auto& [t1, x1, y1] = velHistory[n-1];
                double dt = t1 - t0;
                if (dt > 0.01 && dt < 1.0) {
                    vx = (x1 - x0) / dt;
                    vy = (y1 - y0) / dt;
                    hasVelocity = true;
                }
                // Acceleration: compare current velocity to older velocity
                if (n >= 4) {
                    auto& [ta, xa, ya] = velHistory[n-4];
                    auto& [tb, xb, yb] = velHistory[n-3];
                    double dta = tb - ta;
                    double dtv = t1 - tb;
                    if (dta > 0.01 && dtv > 0.02) {
                        double vx_old = (xb - xa) / dta;
                        double vy_old = (yb - ya) / dta;
                        ax = (vx - vx_old) / dtv;
                        ay = (vy - vy_old) / dtv;
                    }
                }
            }

            // Save velocity for coast mode
            if (hasVelocity) {
                coastVx = vx;
                coastVy = vy;
            }
            coastLastDetectTime = std::chrono::steady_clock::now();
            coastLastX = d.x;
            coastLastY = d.y;

            // Clamp max error: >400px from center = likely false detection
            const double MAX_ERR = 400.0;
            if (std::abs(ex) > MAX_ERR) ex = (ex > 0 ? MAX_ERR : -MAX_ERR);
            if (std::abs(ey) > MAX_ERR) ey = (ey > 0 ? MAX_ERR : -MAX_ERR);

            // PD controller — reduced Kp to keep in proportional zone
            // (with Kp=2.5 everything was clamped → bang-bang behavior)
            // Higher Kd acts as velocity matcher for moving targets
            static double prevEx = 0.0, prevEy = 0.0;
            static auto prevTime = std::chrono::steady_clock::now();
            auto nowTime = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(nowTime - prevTime).count();
            if (dt < 0.001) dt = 0.001;

            double dist = std::sqrt(ex * ex + ey * ey);

            // Moderate P: proportional zone extends to ~187px before clamp
            double Kp = DEG_PER_PX * 1.0;
            if (dist < 15.0) Kp *= (0.3 + 0.7 * dist / 15.0);

            // Strong D-term: velocity matching + overshoot damping
            double Kd = 0.008;
            double dex = (ex - prevEx) / dt;
            double dey = (ey - prevEy) / dt;

            // Feedforward: velocity + acceleration (compensate for lag on accel/decel)
            double FF = 1.0;
            double FA = 0.3;  // acceleration feedforward gain
            double ffYaw   = hasVelocity ? FF * vx * dt + FA * ax * dt * dt : 0.0;
            double ffPitch = hasVelocity ? FF * vy * dt + FA * ay * dt * dt : 0.0;

            double stepYaw   = Kp * ex + Kd * dex + ffYaw;
            double stepPitch = Kp * ey + Kd * dey + ffPitch;

            prevEx = ex;
            prevEy = ey;
            prevTime = nowTime;

            // Adaptive slew limit: base 4° (68°/s), up to 8° (136°/s) when
            // FF velocity demands it. High slew only when tracking fast object,
            // not on noise/static — so MOG2 stays stable.
            double worldSpeed = std::sqrt(vx * vx + vy * vy);  // deg/s
            double MAX_STEP_DEG = 4.0 + std::min(4.0, worldSpeed * 0.04);  // scales up with speed
            if (stepYaw >  MAX_STEP_DEG) stepYaw =  MAX_STEP_DEG;
            if (stepYaw < -MAX_STEP_DEG) stepYaw = -MAX_STEP_DEG;
            if (stepPitch >  MAX_STEP_DEG) stepPitch =  MAX_STEP_DEG;
            if (stepPitch < -MAX_STEP_DEG) stepPitch = -MAX_STEP_DEG;

            fprintf(stderr, "CTRL: err=(%.0f,%.0f) dist=%.0f P=(%.2f,%.2f) D=(%.2f,%.2f) FF=(%.2f,%.2f) step=(%.2f,%.2f) slew=%.1f spd=%.0f\n",
                    ex, ey, dist, Kp*ex, Kp*ey, Kd*dex, Kd*dey, ffYaw, ffPitch, stepYaw, stepPitch, MAX_STEP_DEG, worldSpeed);

            yawDeg   = lastYawDeg   - stepYaw;
            pitchDeg = lastPitchDeg - stepPitch;

            if (yawDeg < 5.0) yawDeg = 5.0;
            if (yawDeg > 175.0) yawDeg = 175.0;
            if (pitchDeg < 5.0) pitchDeg = 5.0;
            if (pitchDeg > 175.0) pitchDeg = 175.0;

            lastYawDeg = yawDeg;
            lastPitchDeg = pitchDeg;
            servoTargetYaw.store(yawDeg);
            servoTargetPitch.store(pitchDeg);
            lastSentYawDeg = yawDeg;
            lastSentPitchDeg = pitchDeg;

            // Publish velocity for 100Hz inter-frame prediction (dual-loop inner loop)
            if (hasVelocity) {
                servoVelYaw.store(-vx);   // vx now in deg/s (world), negative: track direction
                servoVelPitch.store(-vy);
            } else {
                servoVelYaw.store(0.0);
                servoVelPitch.store(0.0);
            }
            servoUpdateTime.store(std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

            // Без settle: OF компенсирует движение камеры
        }
        // === PREDICTIVE COAST: keep tracking by extrapolation when detection is lost ===
        // SHORT coast with LOW speed to prevent runaway
        else if (!d.valid && currentTrackingEnabled && !scanActive) {
            auto now = std::chrono::steady_clock::now();
            double sinceLastDetect = std::chrono::duration<double>(now - coastLastDetectTime).count();
            const double COAST_MAX_SEC = 0.15; // very short coast — linear extrap hurts circular motion

            if (sinceLastDetect < COAST_MAX_SEC && sinceLastDetect > 0.02 &&
                (std::abs(coastVx) > 5.0 || std::abs(coastVy) > 5.0)) {
                double decay = 1.0 - sinceLastDetect / COAST_MAX_SEC;

                double predX = coastLastX + coastVx * sinceLastDetect;
                double predY = coastLastY + coastVy * sinceLastDetect;

                double ex = predX - cx;
                double ey = predY - cy;

                const double MAX_ERR = 200.0;
                if (std::abs(ex) > MAX_ERR) ex = (ex > 0 ? MAX_ERR : -MAX_ERR);
                if (std::abs(ey) > MAX_ERR) ey = (ey > 0 ? MAX_ERR : -MAX_ERR);

                double Kp = DEG_PER_PX * 0.4 * decay;
                double stepYaw   = Kp * ex;
                double stepPitch = Kp * ey;

                const double MAX_STEP_DEG = 0.5;
                if (stepYaw >  MAX_STEP_DEG) stepYaw =  MAX_STEP_DEG;
                if (stepYaw < -MAX_STEP_DEG) stepYaw = -MAX_STEP_DEG;
                if (stepPitch >  MAX_STEP_DEG) stepPitch =  MAX_STEP_DEG;
                if (stepPitch < -MAX_STEP_DEG) stepPitch = -MAX_STEP_DEG;

                yawDeg   = lastYawDeg   - stepYaw;
                pitchDeg = lastPitchDeg - stepPitch;

                if (yawDeg < 5.0) yawDeg = 5.0;
                if (yawDeg > 175.0) yawDeg = 175.0;
                if (pitchDeg < 5.0) pitchDeg = 5.0;
                if (pitchDeg > 175.0) pitchDeg = 175.0;

                lastYawDeg = yawDeg;
                lastPitchDeg = pitchDeg;
                servoTargetYaw.store(yawDeg);
                servoTargetPitch.store(pitchDeg);
            }
        }

        // ROI для визуализации
        if (d.valid) {
            lastROI = computeROI(d.x, d.y, display.cols, display.rows, d.box_w, d.box_h);
        }

        // Debug status
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
            std::cout << "Objects: " << d.all_boxes.size()
                      << " Valid: " << (d.valid ? "YES" : "NO") << std::endl;
            if (d.valid) {
                double dbg_ex = d.x - cx, dbg_ey = d.y - cy;
                std::cout << "Target: (" << (int)d.x << ", " << (int)d.y
                          << ") err=(" << (int)dbg_ex << "," << (int)dbg_ey << "px)"
                          << " [SERVO MOVE]" << std::endl;
            }
            std::cout << "Servo: Yaw=" << lastYawDeg << "\u00b0 Pitch=" << lastPitchDeg << "\u00b0" << std::endl;
            std::cout << "OF: fg=" << detector.lastFGPixels() << "px contours=" << detector.lastRawContours()
                      << " globalFlow=(" << std::setprecision(2) 
                      << detector.lastGlobalFlow().x << "," << detector.lastGlobalFlow().y << ")" 
                      << std::setprecision(1) << std::endl;
            std::cout << "===================\n" << std::endl;
        }
        


        // === VISUALIZATION ===

        // === TRAJECTORY: последние 30 валидных детекций (красный) ===
        static std::deque<cv::Point2f> trajHistory;

        if (trajectoryEnabled.load()) {
            if (d.valid) {
                trajHistory.push_back(cv::Point2f(d.x, d.y));
                if (trajHistory.size() > 30) trajHistory.pop_front();
            }

            for (int i = 0; i < (int)trajHistory.size(); i++) {
                int alpha = 80 + 175 * i / std::max((int)trajHistory.size() - 1, 1);
                cv::Scalar col(0, 0, alpha);  // красный (BGR), от тёмного к яркому
                if (i > 0)
                    cv::line(display, trajHistory[i-1], trajHistory[i], col, 2);
                cv::circle(display, trajHistory[i], 4, col, -1);
            }
        } else {
            trajHistory.clear();
        }

        // Рисуем куда целится серво (позиция детекции)
        if (d.valid) {
            int px = static_cast<int>(d.x);
            int py = static_cast<int>(d.y);
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
        std::string modeStr;
        cv::Scalar modeColor;
        if (!trackingEnabled.load()) {
            modeStr = "FIXED 90deg";
            modeColor = cv::Scalar(0, 128, 255);
        } else if (scanActiveNow.load()) {
            modeStr = "SCANNING " + std::to_string((int)scanYawDeg) + "deg";
            modeColor = cv::Scalar(0, 255, 255);  // yellow
        } else if (scanEnabled.load()) {
            modeStr = "TRACKING (scan standby)";
            modeColor = cv::Scalar(0, 255, 0);
        } else {
            modeStr = "TRACKING ON";
            modeColor = cv::Scalar(0, 255, 0);
        }
        
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
            // === KEY HELP OVERLAY (bottom-right corner) ===
            const char* helpLines[] = {
                "F - Track / Fixed",
                "S - Scan on/off",
                "T - Trajectory on/off",
                "Q - Quit"
            };
            const int nLines = 4;
            const double fontScale = 0.65;
            const int thickness = 2;
            const int lineH = 28;
            const int padX = 12, padY = 8;
            int boxW = 280;
            int boxH = nLines * lineH + padY * 2;
            int bx = display.cols - boxW - 10;
            int by = display.rows - boxH - 10;

            cv::Mat overlay = display.clone();
            cv::rectangle(overlay, cv::Point(bx, by), cv::Point(bx + boxW, by + boxH),
                         cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(overlay, 0.6, display, 0.4, 0, display);
            cv::rectangle(display, cv::Point(bx, by), cv::Point(bx + boxW, by + boxH),
                         cv::Scalar(0, 255, 255), 2);

            for (int i = 0; i < nLines; i++) {
                cv::putText(display, helpLines[i],
                           cv::Point(bx + padX, by + padY + (i + 1) * lineH - 5),
                           cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255), thickness);
            }

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

    std::cout << "Starting servo thread (100Hz interpolation)..." << std::endl;
    thread servo(servoThread, ref(run));

    std::cout << "Starting tracking thread..." << std::endl;
    thread track(trackingThread, ref(queue), ref(run));

    std::cout << "\n============================================================================================" << std::endl;
    std::cout << "System running!" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  F   - Toggle tracking mode (FIXED 90deg <-> TRACKING ON)" << std::endl;
    std::cout << "  S   - Toggle scan mode (scan 0-180 deg when no object)" << std::endl;
    std::cout << "  T   - Toggle trajectory drawing" << std::endl;
    std::cout << "  Q   - Quit program" << std::endl;
    std::cout << "  ESC - Quit program" << std::endl;
    std::cout << "\n*** TRACKING MODE ENABLED BY DEFAULT ***" << std::endl;
    std::cout << "Servos will follow detected objects (5.0x sensitivity)" << std::endl;
    std::cout << "Press F to lock servos at 90deg" << std::endl;
    std::cout << "Press S to enable/disable scan mode" << std::endl;
    std::cout << "============================================================================================" << std::endl;
    
    // Create display window
    cv::namedWindow("Predictive Gimbal Control", cv::WINDOW_NORMAL);
    cv::resizeWindow("Predictive Gimbal Control", 1280, 720);
    // Keep window on top and focused for keyboard input
    cv::setWindowProperty("Predictive Gimbal Control", cv::WND_PROP_TOPMOST, 1);
    std::cout << "\n*** CLICK ON THE WINDOW TO ACTIVATE KEYBOARD CONTROL ***\n" << std::endl;

    // Start HTTP MJPEG stream server
    streamRunning = true;
    if (pthread_create(&streamThread, nullptr, httpStreamServer, nullptr) != 0) {
        std::cerr << "✗ Failed to start HTTP stream server" << std::endl;
        streamRunning = false;
    }
    
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
                // Update HTTP MJPEG stream frame
                if (streamRunning) {
                    std::lock_guard<std::mutex> lock(streamMutex);
                    cv::resize(frame, streamFrame, cv::Size(1600, 900));
                }
            }
        }
        
        // Check for key press (minimal 1ms delay for fastest response)
        int key = cv::waitKey(1);
        if (key != -1) {  // If any key was pressed
            std::cout << "\n[KEY DETECTED] Code: " << key << " (char: '" << (char)key << "')" << std::endl;
        }
        
        if (key == 'q' || key == 'Q' || key == 27 || remoteQuit.load()) {  // Q or ESC - quit (local or remote)
            std::cout << "\n=== QUIT " << (remoteQuit.load() ? "(REMOTE)" : "(LOCAL)") << " ===" << std::endl;
            run = false;
            break;
        } else if (key == 't' || key == 'T' || remoteTrajToggle.exchange(false)) {  // T - toggle trajectory
            bool wasT = trajectoryEnabled.exchange(!trajectoryEnabled.load());
            std::cout << ">>> TRAJECTORY " << (!wasT ? "ON" : "OFF") << " <<<" << std::endl;
        } else if (key == 's' || key == 'S' || remoteScanToggle.exchange(false)) {  // S - toggle scan mode
            bool wasScan = scanEnabled.exchange(!scanEnabled.load());
            bool nowScan = !wasScan;
            if (nowScan) {
                std::cout << ">>> SCAN MODE ENABLED - Camera will scan 0-180 deg when no object <<<" << std::endl;
            } else {
                std::cout << ">>> SCAN MODE DISABLED <<<" << std::endl;
            }
        } else if (key == 'f' || key == 'F' || remoteToggle.exchange(false)) {  // F - toggle tracking mode (local or remote)
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
                servoTargetYaw.store(90.0);
                servoTargetPitch.store(90.0);
                servoInstantSnap.store(true);
                std::cout << ">>> Servo snap to 90° <<<" << std::endl;
            }
        }
    }

    // === SOFT COOLDOWN EXIT: smoothly return servos to center ===
    std::cout << "\nSoft cooldown: returning servos to center..." << std::endl;
    servoTargetYaw.store(90.0);
    servoTargetPitch.store(90.0);
    // Don't snap — let the servo thread's exponential interpolation glide smoothly
    {
        auto t0 = std::chrono::steady_clock::now();
        const double THRESHOLD = 0.5; // degrees from center to consider "arrived"
        const int TIMEOUT_MS = 2000;  // safety timeout
        while (true) {
            double dy = std::abs(servoActualYaw.load() - 90.0);
            double dp = std::abs(servoActualPitch.load() - 90.0);
            if (dy < THRESHOLD && dp < THRESHOLD) {
                std::cout << "Servos centered (yaw err=" << dy << "° pitch err=" << dp << "°)" << std::endl;
                break;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed > TIMEOUT_MS) {
                std::cout << "Cooldown timeout (" << TIMEOUT_MS << "ms), forcing center." << std::endl;
                setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
                setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    std::cout << "Shutting down threads..." << std::endl;
    run=false;
    queue.notifyAll();

    cam.join();
    track.join();
    servo.join();
    
    if (streamRunning) {
        streamRunning = false;
        pthread_join(streamThread, nullptr);
        std::cout << "HTTP stream server stopped." << std::endl;
    }

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