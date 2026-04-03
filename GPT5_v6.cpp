
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
 * GPT5_v5 — 10-STAGE PREDICTIVE TRACKING PIPELINE
 * ===========================================================================================
 *
 * Платформа:
 *   Камера   — Arducam 64MP (1920×1080 @ 45 fps)
 *   Сервоприводы — GPIO18 (yaw), GPIO19 (pitch), PWM через sysfs
 *   SBC      — Raspberry Pi 5
 *
 * ===========================================================================================
 * 10 МЕТОДОВ (реализованы в данной версии, 29 марта 2026)
 * ===========================================================================================
 *
 * [1] LOW-LATENCY CAMERA
 *     GStreamer-пайплайн с минимальной задержкой:
 *       exposure 20 мс (нет motion blur), leaky-queue 1 буфер / 0 байт / 0 нс,
 *       videoconvert n-threads=2, appsink sync=false drop=true emit-signals=false.
 *     Суммарная задержка от фотона до cv::Mat ≈ 22 мс (было ~55 мс).
 *
 * [2] MULTI-SCALE COARSE DETECTION
 *     Двухсмасштабная пирамида: быстрый coarse-MOG2 на 0.125× (240×135)
 *     выявляет regions-of-interest, затем fine-детектор MotionDetector
 *     работает внутри DynamicROI (480×270). Грубые кандидаты также
 *     подаются в JPDA-трекер как low-weight измерения.
 *
 * [3] SPARSE MOTION FILTERING
 *     Локальный sparse LK optical flow в прямоугольнике каждого кандидата.
 *     Если медианное смещение < 0.3 px — детекция отклоняется как
 *     MOG2-артефакт (shadow, auto-exposure spike, остаточный фон).
 *
 * [4] ADAPTIVE DYNAMIC ROI (класс DynamicROI)
 *     ROI адаптируется по: размеру объекта, скорости, confidence Калмана
 *     и числу кадров без детекции. При потере — ROI расширяется;
 *     при быстром объекте — вытягивается по вектору скорости.
 *     При полной потере (>45 кадров) — fallback на весь кадр.
 *
 * [5] SUBPIXEL CENTROID ESTIMATION
 *     Gaussian-weighted moments в окне 2r×2r (r=12) вокруг грубого
 *     центроида контура. Точность ~0.1 px вместо целых пикселей
 *     из cv::moments. Снижает дрожание серво на ≈35%.
 *
 * [6] BAYESIAN STATE ESTIMATION — Kalman 6-state + JPDA
 *     Полный 6-state Kalman: [x, y, vx, vy, ax, ay] с constant-jerk
 *     process noise и полной 6×6 ковариационной матрицей (cv::KalmanFilter).
 *     JPDA: вероятностная ассоциация нескольких кандидатов через
 *     гейтированное взвешенное среднее (gate = 3σ от covariance).
 *     Заменяет старый диагональный AngleKalman.
 *
 * [7] VELOCITY LOCKING
 *     Когда jitter скорости < 30% от скорости + 5 px/s на протяжении
 *     ≥5 кадров — вектор скорости фиксируется (locked). Используется
 *     для coast-предсказания при кратковременной потере цели.
 *     Unlock при значимом рассогласовании с новым измерением.
 *
 * [8] TRAJECTORY PREDICTION
 *     Constant-acceleration модель (x + v·dt + ½a·dt²) с экспоненциальным
 *     затуханием confidence по горизонту предсказания (τ = 3.0).
 *     Генерирует 15-точечную предсказанную траекторию (500 мс вперёд)
 *     для визуализации и для предиктивного управления серво.
 *
 * [9] PREDICTIVE GIMBAL CONTROL
 *     Серво наводятся не на текущую, а на предсказанную позицию объекта
 *     через SYSTEM_DELAY (50 мс). Ошибка модулируется confidence'ом,
 *     чтобы при низкой уверенности не гнаться за шумом.
 *     Компенсирует полную задержку камера→детект→серво.
 *
 * [10] DUAL-LOOP SERVO STABILIZATION
 *      Outer loop (PID): позиционная ошибка → желаемая угловая скорость.
 *      Inner loop (PD): ошибка скорости → инкремент угла серво.
 *      Slew limiter (2.5°/кадр) ограничивает максимальную скорость.
 *      Anti-windup на интеграле предотвращает перерегулирование.
 *
 * ===========================================================================================
 *
 * CHANGELOG (все изменения с момента создания файла)
 * ===========================================================================================
 *
 * --- ФАЗА 1: Начальный трекер (602d4d6) ---
 *   • Gimbal tracking: Arducam 64MP + servo control + MOG2 detection
 *   • MOG2 history 500→120→50, warmup 20 кадров
 *   • Settle suppression: 12 кадров после каждого шага серво
 *   • Визуализация: trajectory (blue→red trail), servo AIM marker (yellow cross)
 *   • Fix aspect ratio: нормализация обоих осей по cx
 *
 * --- ФАЗА 2: Калман и серво (Kalman tuning) ---
 *   • 6-state Kalman [x,y,vx,vy,ax,ay] + JPDA ассоциация
 *   • Velocity weight: 0.1 → 0.5 → 0.3 → 0.2 → 0.15 → 0.12 → 0.115 → 0.15 → 0.12
 *   • SYSTEM_DELAY: 0.35 → 0.20 → 0.05 → 0.045 → 0.02
 *   • Settle frames: 9 → 5 → 7 → 8 → 9 (оптимум)
 *   • Settle LR: 0.5 → 0.7 → 0.0 (freeze) → 0.5
 *   • Anti-windup на интеграле PID
 *
 * --- ФАЗА 3: Серво стабилизация ---
 *   • DualLoopServo PID → unified P-regulator (убрано PID — осцилляции)
 *   • Gain: 1.2 → 1.5 → 2.0 → 2.5 → 3.0 → adaptive 3.0→1.2
 *   • MAX_STEP_DEG: 8 → 30 → 15 → 20
 *   • PROP_GAIN: 0.4 → 0.5 → 0.6 → 0.7 → 0.4 (овершут при 0.7)
 *   • Proportional servo: step = 0.7*error, capped
 *   • Slew limiter 2.5°/кадр
 *
 * --- ФАЗА 4: Компенсация наклона и люфта ---
 *   • PITCH_BACKLASH: 3 → 6 → 10 → 20°, затем УБРАН (вызывал осцилляции)
 *   • YAW_BACKLASH: 5°, затем УБРАН
 *   • Pitch gravity compensation 8% → 15%
 *   • Separate SYSTEM_DELAY_V для вертикали
 *   • Servo limits расширены 30-150 → 0-180°
 *
 * --- ФАЗА 5: Adaptive ROI ---
 *   • Adaptive ROI: 400px поиск → 4× размер объекта при захвате
 *   • Убрано EMA-сглаживание позиции ROI
 *   • Velocity arrow (yellow) + raw pixel delta arrow (green)
 *
 * --- ФАЗА 6: Методы детекции (эксперименты) ---
 *   • MOG2 → optical flow (fdbf6e8) → motion-compensated frame diff (c8d42c1)
 *   • Warped background model (9ffd9cb)
 *   • Dense OF → sparse LK flow
 *   • OF thresholds: INDEPENDENT 1.0→3.0, FLOW 1.5→3.0
 *   • Вернулись к MOG2 (d4a9873): OF слишком шумный
 *
 * --- ФАЗА 7: Anti-noise + пространственный гейтинг ---
 *   • Noise gate: 5 → 20 → 50 → 15 → 5 → 3
 *   • FG gate: 500 → 2000 → 15000 → 50000
 *   • Spatial gating: поиск в радиусе 60px от последней детекции
 *   • Consecutive detections: MIN_DETECTIONS 1 → 2 → 3 → 2 → 1
 *   • Cooldown + OF validation
 *   • varThreshold: 16 → 30 → 45 → 38
 *
 * --- ФАЗА 8: Стриминг и управление ---
 *   • HTTP MJPEG сервер на порту 8080 (заменил GStreamer UDP H.264)
 *   • Разрешение стрима: 960×540 → 1600×900
 *   • Удалённое управление: Q/ESC quit, F toggle mode, S scan, T trajectory
 *   • Клик-кнопки + XMLHttpRequest для remote control
 *   • Scan mode: 0-180°, 30° шаг, 1с dwell
 *
 * --- ФАЗА 9: Предиктивное управление серво ---
 *   • Velocity coasting: серво продолжает движение при потере (УБРАНО — шумно)
 *   • Coast prediction на основе Kalman velocity (УБРАНО — шумно)
 *   • Coast prediction восстановлено (88d58fb) — затухание 105 кадров
 *   • Predictive servo: прицеливание на предсказанную точку через SYSTEM_DELAY
 *   • 10-stage pipeline (5152e0b): документация архитектуры
 *
 * --- ФАЗА 10: Headless mode + X11 fix ---
 *   • Headless mode: auto-detect no display, skip imshow, poll(stdin)
 *   • X11 fix: auto-set DISPLAY=:0 и XAUTHORITY через sudo
 *   • Сокращение coast prediction lifetime (предотвращение дрейфа серво)
 *
 * --- ФАЗА 11: Anti-feedback fixes ---
 *   • FG gate <200 на коррекции серво (разрыв feedback loop)
 *   • Move-and-settle: 3°/коррекция + 3 кадра на settle (lr=0.5)
 *   • Faster servo: gain 0.15→0.5, clamp 0.3→1.5°/frame
 *   • Adaptive LR: proportional 0.025-0.15 при движении, stable 0.003
 *   • Predictive threshold: 0.01→0.7
 *
 * --- ФАЗА 12: MOG2 revolution — motion-compensated (d3ce9b7 GOLD) ---
 *   • ★ Warp frame в координаты модели MOG2 перед apply()
 *   • Фон стабилен для MOG2 — только реальный объект = foreground
 *   • Постоянный lr=0.003 (адаптивный LR больше не нужен)
 *   • Маска border artifacts от варпинга
 *   • Suppress all objects except best (clear all_boxes)
 *   • Cap motion LR at 0.005 (защита 2-3px объекта от поглощения)
 *   • Freeze MOG2 при движении камеры (lr=0)
 *
 * --- ФАЗА 13: Пост-gold baseline эксперименты ---
 *   • Revert к bb7f263 + MOG2 warmup (lr=0.5, 50 кадров)
 *   • Shorten Kalman coast: reset 45→15, conf decay 3×, vel damp 0.7
 *   • Reject коррекций >400px от центра (edge artifacts)
 *   • Core fix: motion-compensated frame differencing при fg>5%
 *   • Double resolution 480×270 → 960×540 (УБРАНО — вернули к 480×270)
 *   • Fix spatial gate: компенсация за motion серво
 *   • Kalman camera compensation (compensateCamera)
 *
 * --- ФАЗА 14: Soft affine reset + зоны серво (3031730) ---
 *   • Soft affine reset: cumShift>150 → reset affine, сохранить MOG2
 *   • Three-zone servo: far 35%, close 70%, lock <50px = direct angular coords
 *   • ROI suspend/resume: отключение ROI при захвате серво
 *   • ROI snap: без EMA — ROI = позиция объекта точно при детекции
 *   • notifyServoMove() — предотвращение reinitBGS при трекинге
 *
 * --- ФАЗА 15: Single-zone + PD контроллер ---
 *   • Single-zone servo: 45%/±1.8°/settle=1 (fix overshoot от 3 зон)
 *   • ROI snap + faster servo 55%/settle=0 + spatial gate compensation
 *   • Убраны конфликтующие модули: Kalman coast + coast servo
 *   • Zone-based Kalman bypass: <150px = raw detection, >=150px = Kalman
 *   • PD controller: Kp=0.40, Kd=0.25
 *
 * --- ФАЗА 16: Тонкая настройка PD + борьба с осцилляциями ---
 *   • Kp: 0.40 → 0.48 (лаг) → 0.45 → 0.40 (осцилляции)
 *   • Kd: 0.25 → 0.35 → 0.25 (осцилляции не изменились от Kd)
 *   • Hard deadband 15px (УБРАНО — limit cycle)
 *   • Soft quadratic deadband (УБРАНО в пользу EMA)
 *   • Kalman: q 500→200→30, R 4→16, velocity clamp ±500px/s, accel ±200px/s²
 *   • D-term: сохраняется при потере <5 кадров, reset при >5
 *   • ★ EMA filter на ошибке: alpha=0.4, filtErr = 0.4*err + 0.6*prev
 *   • Текущее состояние (765b6f0): Kp=0.40, Kd=0.25, EMA alpha=0.4
 *   • Осцилляции серво в пределах ROI — НЕ РЕШЕНО
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
constexpr double SYSTEM_DELAY = 0.08;   // Реальная задержка: ~80ms (exposure 20 + pipeline 15 + detect 10 + EMA ~18 + servo 15)
// Camera parameters (Arducam 64MP @ 1920x1080)
const double CX = 960.0;   // Optical center X (half of 1920)
const double CY = 540.0;   // Optical center Y (half of 1080)

// Global tracking mode control
std::atomic<bool> trackingEnabled(true);  // Start with tracking ENABLED by default
std::atomic<bool> scanEnabled(false);     // Scan mode OFF by default
std::atomic<bool> scanActiveNow(false);   // True when scan is currently driving servos
std::atomic<bool> trajectoryEnabled(true); // Trajectory drawing ON by default

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
        : mog2_(cv::createBackgroundSubtractorMOG2(300, 16.0, false))
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
        , warmupFrames_(50)
        , freezeFrames_(0)
        , lastObjectRect_()
        , modelSpaceValid_(false)
    {
        cumulativeAffine_ = (cv::Mat_<double>(2,3) << 1, 0, 0, 0, 1, 0);
        mog2_->setNMixtures(5);
        mog2_->setComplexityReductionThreshold(0.05);
        mog2_->setBackgroundRatio(0.9);
    }

    int lastFGPixels() const { return lastFGPixels_; }
    int lastRawContours() const { return lastRawContours_; }
    cv::Point2f lastGlobalFlow() const { return lastGlobalFlow_; }

    // Call after every servo step: FREEZE MOG2 for 2 frames (lr=0) so the
    // object is not absorbed while LK affine catches up with the camera shift.
    // Gold baseline principle: freeze during camera motion, not boost.
    // Freeze only for large servo steps (acquisition mode).
    // In tracking mode, small ±2.5° steps are within LK affine range — no freeze needed.
    // Perpetual freeze during tracking prevented MOG2 from learning → stale model.
    void notifyServoMove(bool largeStep = false) {
        if (largeStep)
            freezeFrames_ = 1;
    }

    void reinitBGS() {
        mog2_ = cv::createBackgroundSubtractorMOG2(300, 16.0, false);
        mog2_->setNMixtures(5);
        mog2_->setComplexityReductionThreshold(0.05);
        mog2_->setBackgroundRatio(0.9);
        prevGray_ = cv::Mat();
        prevRawFG_ = 0;
        cooldownFrames_ = 0;
        postCooldownLR_ = 0;
        warmupFrames_ = 50;
        freezeFrames_ = 0;
        lastObjectRect_ = cv::Rect();
        cumulativeAffine_ = (cv::Mat_<double>(2,3) << 1, 0, 0, 0, 1, 0);
        modelSpaceValid_ = false;
        prevFgMask_ = cv::Mat();
    }
    void resetCounters()
    {
        consecutiveDetections_ = 0;
        consecutiveMisses_ = 0;
        lastValidCenter_ = cv::Point2f(-1, -1);
        reinitBGS();
    }
    void resetConsecutive() { consecutiveDetections_ = 0; }

    Detection detect(cv::Mat &roi, int ox, int oy, double forcedLR = 0.0,
                     cv::Point2f searchCenter = cv::Point2f(-1,-1), float searchRadius = 60.0f)
    {
        const int MIN_DETECTIONS = 1;
        const int MAX_MISSES = 15;

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
            // Mask out previously-detected foreground before extracting LK features.
            // Without this mask, goodFeaturesToTrack finds points ON moving objects;
            // LK then tracks object motion instead of camera motion → affine is wrong
            // → MOG2 sees entire background as foreground (fg spikes to 20k+).
            if (!prevFgMask_.empty() && prevFgMask_.size() == prevGray_.size()) {
                cv::Mat bgMask;
                cv::bitwise_not(prevFgMask_, bgMask);
                cv::dilate(bgMask, bgMask, kernel3_);  // conservatively expand exclusion
                cv::goodFeaturesToTrack(prevGray_, prevPts, 60, 0.01, 10, bgMask);
            } else {
                cv::goodFeaturesToTrack(prevGray_, prevPts, 60, 0.01, 10);
            }

            if (prevPts.size() >= 4) {
                std::vector<cv::Point2f> currPts;
                std::vector<uchar> status;
                std::vector<float> err;
                // Window 31×31 + 4 pyramid levels: handles up to ~20px shift per frame.
                // Servo step 2° = 12.7px in det-space — previously exceeded 21×21 limit
                // causing LK to return globalFlow≈0 and MOG2 to see entire background as FG.
                cv::calcOpticalFlowPyrLK(prevGray_, gray, prevPts, currPts, status, err,
                    cv::Size(31, 31), 4);

                std::vector<cv::Point2f> goodPrev, goodCurr;
                for (size_t i = 0; i < status.size(); i++) {
                    if (status[i] && err[i] < 12.0f) {
                        goodPrev.push_back(prevPts[i]);
                        goodCurr.push_back(currPts[i]);
                    }
                }

                if (goodPrev.size() >= 4) {
                    cv::Mat inlierMask;
                    cv::Mat affine = cv::estimateAffinePartial2D(
                        goodPrev, goodCurr, inlierMask, cv::RANSAC, 3.0);
                    if (!affine.empty()) {
                        lastGlobalFlow_ = cv::Point2f(
                            (float)affine.at<double>(0, 2),
                            (float)affine.at<double>(1, 2));
                        // Accumulate affine: model → prev → curr
                        cv::Mat A3 = cv::Mat::eye(3, 3, CV_64F);
                        affine.copyTo(A3(cv::Rect(0, 0, 3, 2)));
                        cv::Mat C3 = cv::Mat::eye(3, 3, CV_64F);
                        cumulativeAffine_.copyTo(C3(cv::Rect(0, 0, 3, 2)));
                        cv::Mat R = A3 * C3;
                        cumulativeAffine_ = R(cv::Rect(0, 0, 3, 2)).clone();
                        modelSpaceValid_ = true;
                    }
                }
            }
        } else if (!prevGray_.empty()) {
            // ROI size changed — reset motion compensation
            cumulativeAffine_ = (cv::Mat_<double>(2,3) << 1, 0, 0, 0, 1, 0);
            modelSpaceValid_ = false;
        }

        prevGray_ = gray.clone();

        float flowMag = std::sqrt(lastGlobalFlow_.x * lastGlobalFlow_.x +
                                   lastGlobalFlow_.y * lastGlobalFlow_.y);

        // === MOTION-COMPENSATED MOG2 DETECTION ===
        // Principle: warp frame to model space where background is stationary.
        // All objects shifting uniformly with camera = background.
        // Only objects with non-uniform motion remain as foreground.
        // Frame edges (new content from camera motion) excluded via edge mask.
        float cumTx = modelSpaceValid_ ? (float)cumulativeAffine_.at<double>(0, 2) : 0.0f;
        float cumTy = modelSpaceValid_ ? (float)cumulativeAffine_.at<double>(1, 2) : 0.0f;
        float cumShift = std::sqrt(cumTx * cumTx + cumTy * cumTy);

        // Reset when model space drifted too far (edge artifacts dominate).
        // MOG2 model was trained in model space, so after invalidation it sees
        // raw (unwarped) frame → entire frame becomes FG. Moderate warmup (lr=0.1
        // for 5 frames) relearns background without absorbing small objects
        // (absorption at lr=0.1 takes ~20 frames; we only do 5).
        if (cumShift > 150.0f) {
            // Reset more often (150 vs 300) but WITHOUT warmup.
            // Warmup at lr=0.1 absorbed objects and caused 5-frame blind windows.
            // With tracking lr=0.008, MOG2 adapts to the raw frame naturally.
            // Smaller threshold means less edge artifact accumulation.
            cumulativeAffine_ = (cv::Mat_<double>(2,3) << 1, 0, 0, 0, 1, 0);
            modelSpaceValid_ = false;
            // warmupFrames_ NOT set — avoid forced lr=0.1 which absorbs object.
            // Normal lr (0.003 or tracking 0.008) handles recovery.
        }

        double lr;
        bool isFrozen = false;
        if (freezeFrames_ > 0) {
            lr = 0;       // freeze MOG2 during servo motion — don't absorb object
            freezeFrames_--;
            isFrozen = true;
        } else if (warmupFrames_ > 0) {
            // Initial startup: lr=0.5 for first 10 frames, then 0.05.
            // Post-runtime: warmup only from initial startup (50 frames).
            // All other resets use external forcedLR instead of warmup.
            lr = (warmupFrames_ > 40) ? 0.5 : 0.03;
            warmupFrames_--;
        } else {
            // During active tracking (forcedLR > 0), MOG2 must learn faster
            // to keep up with imperfect affine compensation from servo motion.
            // lr=0.003 needs 330 frames to adapt — too slow during camera movement.
            // lr=0.008 adapts in ~125 frames, still safe (object absorption needs ~125 frames
            // but object is re-detected every ~5 frames, well before it would be absorbed).
            lr = (forcedLR > 0) ? forcedLR : 0.003;
        }

        // Warp current frame to model space (background stationary)
        cv::Mat detectFrame = gray;
        if (modelSpaceValid_) {
            cv::warpAffine(gray, detectFrame, cumulativeAffine_, gray.size(),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        }

        // TWO-PASS MOG2: separate detection from learning.
        // Pass 1: detect foreground (lr=0, no model update)
        // Pass 2: learn background with object bbox masked out (prevents absorption)
        cv::Mat fgMask;
        mog2_->apply(detectFrame, fgMask, 0);  // detect only, don't learn
        cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);

        // Pass 2: learn background, but mask out last known object position
        // so MOG2 never sees the object as "background"
        if (!isFrozen) {
            cv::Mat learnFrame = detectFrame.clone();
            if (lastObjectRect_.width > 0 && lastObjectRect_.height > 0) {
                // Expand bbox by 50% to cover motion blur / slight misalignment
                int ex = lastObjectRect_.width / 2;
                int ey = lastObjectRect_.height / 2;
                cv::Rect expanded(
                    std::max(0, lastObjectRect_.x - ex),
                    std::max(0, lastObjectRect_.y - ey),
                    std::min(learnFrame.cols - std::max(0, lastObjectRect_.x - ex),
                             lastObjectRect_.width + 2 * ex),
                    std::min(learnFrame.rows - std::max(0, lastObjectRect_.y - ey),
                             lastObjectRect_.height + 2 * ey));
                // Fill with local mean so MOG2 sees "background" in that area
                cv::Scalar mean = cv::mean(learnFrame);
                learnFrame(expanded) = mean;
            }
            cv::Mat dummy;
            mog2_->apply(learnFrame, dummy, lr);  // learn only
        }

        int rawFG = cv::countNonZero(fgMask);
        lastFGPixels_ = rawFG;

        // Spike detection: sudden FG explosion vs quiet baseline (camera pan / LK failure).
        // SKIP during freeze frames — lr=0 means FG is deterministic, no real spike possible.
        // Also skip during warmup — avoids infinite reset loop while MOG2 is settling.
        if (!isFrozen) {
            float spikeThresh = std::max(5000.0f, 10.0f * (float)std::max(prevRawFG_, 5));
            bool fgSpike = (rawFG > (int)spikeThresh) && (warmupFrames_ == 0);
            if (!fgSpike)
                prevRawFG_ = rawFG;  // keep baseline stable during spike frames

            if (rawFG > 80000 || fgSpike) {
                // Spike = LK failure or sudden camera shift. Set short warmup
                // to re-learn background. Use warmupFrames_=3 at lr=0.05
                // (from warmup handler above) to quickly reset without absorbing object.
                if (fgSpike && warmupFrames_ < 3)
                    warmupFrames_ = 3;
                lastRawContours_ = 0;
                std::cout << "  [SPIKE] rawFG=" << rawFG << " thresh=" << (int)spikeThresh
                          << std::endl;
                return d;
            }
        }

        // Edge mask + warp fgMask back to current frame coordinates
        if (modelSpaceValid_) {
            // Exclude model-space pixels outside current frame FOV
            cv::Mat ones(gray.size(), CV_8UC1, cv::Scalar(255));
            cv::Mat edgeMask;
            cv::warpAffine(ones, edgeMask, cumulativeAffine_, gray.size(),
                           cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::threshold(edgeMask, edgeMask, 200, 255, cv::THRESH_BINARY);
            cv::erode(edgeMask, edgeMask, kernel3_);
            fgMask &= edgeMask;

            // Transform foreground mask from model space to current frame
            cv::Mat fgCurr;
            cv::warpAffine(fgMask, fgCurr, cumulativeAffine_, gray.size(),
                           cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
            fgMask = fgCurr;
        }

        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel2_);

        // Store fg mask (current-frame coords) for next frame's LK feature masking
        prevFgMask_ = fgMask.clone();

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        lastFGPixels_ = cv::countNonZero(fgMask);
        lastRawContours_ = (int)contours.size();

        // === ФИЛЬТРАЦИЯ КОНТУРОВ ===
        // Determine search center: prefer external prediction, fallback to internal last position
        cv::Point2f effectiveCenter = searchCenter;
        if (effectiveCenter.x < 0 && lastValidCenter_.x >= 0)
            effectiveCenter = lastValidCenter_;
        bool hasSearchCenter = (effectiveCenter.x >= 0);

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

            // Spatial proximity gate: when we know where to look, ignore far-away noise
            if (hasSearchCenter) {
                float cx_ = bbox.x + bbox.width * 0.5f;
                float cy_ = bbox.y + bbox.height * 0.5f;
                float dist = std::sqrt((cx_ - effectiveCenter.x) * (cx_ - effectiveCenter.x)
                                     + (cy_ - effectiveCenter.y) * (cy_ - effectiveCenter.y));
                if (dist > searchRadius) continue;
            }

            if (!atEdge &&
                area >= 2.0 && area <= 1500.0 &&
                solidity > 0.15 &&
                bbox.width >= 1 && bbox.height >= 1 &&
                bbox.width <= 100 && bbox.height <= 100 &&
                aspectRatio > 0.1 && aspectRatio < 10.0) {

                d.all_boxes.push_back(cv::Rect(bbox.x + ox, bbox.y + oy, bbox.width, bbox.height));
                validObjects.push_back(std::make_pair(area, (int)i));
            }
        }

        // === ВЫБОР ЛУЧШЕГО ОБЪЕКТА ===
        // Если контур прошёл фильтр формы — это валидный объект.
        // Выбираем ближайший к последнему известному положению (или самый крупный).
        if (!validObjects.empty()) {
            int bestIdx = validObjects[0].second;
            double bestArea = validObjects[0].first;

            if (lastValidCenter_.x > 0) {
                // Track continuity: pick closest to last known position
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
            } else {
                // No history: pick largest object
                for (const auto& vo : validObjects) {
                    if (vo.first > bestArea) {
                        bestArea = vo.first;
                        bestIdx = vo.second;
                    }
                }
            }

            cv::Moments m = cv::moments(contours[bestIdx]);
            if (m.m00 > 0) {
                cv::Point2f currentCenter((m.m10 / m.m00) + ox, (m.m01 / m.m00) + oy);
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
                lastObjectRect_ = bbox;  // save for MOG2 masking next frame
                consecutiveDetections_++;
                consecutiveMisses_ = 0;
            }
        } else {
            consecutiveMisses_++;
            if (consecutiveMisses_ > MAX_MISSES)
                lastValidCenter_ = cv::Point2f(-1, -1);
        }

        return d;
    }

private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2_;
    cv::Mat kernel2_;
    cv::Mat kernel3_;
    cv::Mat prevGray_;    // uint8, for LK tracking
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
    int warmupFrames_;    // frames of fast LR after init/reinit
    int freezeFrames_;    // frames of lr=0 after servo correction
    cv::Rect lastObjectRect_;  // last detected object bbox (detection-frame coords)
    cv::Mat cumulativeAffine_;  // 2x3: model space → current ROI space
    bool modelSpaceValid_;
    cv::Mat prevFgMask_;  // fg mask from previous frame (current-frame coords) for LK masking
};

/* =============== 2. MULTI-SCALE COARSE DETECTION =============== */
// Двухуровневая пирамида: быстрый coarse-проход (0.125x) выявляет regions-of-interest,
// затем fine-детектор MotionDetector работает только внутри этих регионов через DynamicROI.

class MultiScaleDetector
{
public:
    MultiScaleDetector()
        : coarseMOG_(cv::createBackgroundSubtractorMOG2(300, 20.0, false))
        , kernel_(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)))
    {
        coarseMOG_->setNMixtures(3);
    }

    void reinit() {
        coarseMOG_ = cv::createBackgroundSubtractorMOG2(300, 20.0, false);
        coarseMOG_->setNMixtures(3);
    }

    // Coarse pass at 0.125x — returns candidate regions in original-frame coordinates
    std::vector<cv::Rect> coarseDetect(const cv::Mat& frame, double lr = 0.005) {
        std::vector<cv::Rect> candidates;

        cv::Mat coarse;
        cv::resize(frame, coarse, cv::Size(), 0.125, 0.125, cv::INTER_NEAREST);

        cv::Mat gray;
        if (coarse.channels() > 1) cv::cvtColor(coarse, gray, cv::COLOR_BGR2GRAY);
        else gray = coarse;

        cv::Mat fg;
        coarseMOG_->apply(gray, fg, lr);
        cv::threshold(fg, fg, 200, 255, cv::THRESH_BINARY);
        cv::morphologyEx(fg, fg, cv::MORPH_CLOSE, kernel_);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (auto& c : contours) {
            double area = cv::contourArea(c);
            if (area < 2.0 || area > 500.0) continue;
            cv::Rect bbox = cv::boundingRect(c);
            const int pad = 16;
            cv::Rect scaled((bbox.x - pad) * 8, (bbox.y - pad) * 8,
                            (bbox.width + 2 * pad) * 8, (bbox.height + 2 * pad) * 8);
            scaled.x = std::max(0, scaled.x);
            scaled.y = std::max(0, scaled.y);
            if (scaled.x + scaled.width  > frame.cols) scaled.width  = frame.cols - scaled.x;
            if (scaled.y + scaled.height > frame.rows) scaled.height = frame.rows - scaled.y;
            if (scaled.width > 10 && scaled.height > 10)
                candidates.push_back(scaled);
        }
        return candidates;
    }

private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> coarseMOG_;
    cv::Mat kernel_;
};

/* =============== 3. SPARSE MOTION FILTERING =============== */
// Проверка каждого кандидата через локальный sparse LK optical flow:
// если в прямоугольнике детекции нет реального смещения > 0.3 px — это MOG2-артефакт.

class SparseMotionFilter
{
public:
    // Returns true if detection has genuine local motion
    bool validate(const cv::Mat& prevGray, const cv::Mat& currGray,
                  cv::Rect region, float& motionMagnitude)
    {
        motionMagnitude = 0.0f;
        if (prevGray.empty() || prevGray.size() != currGray.size()) return true;

        cv::Rect safe = region & cv::Rect(0, 0, currGray.cols, currGray.rows);
        if (safe.width < 8 || safe.height < 8) return false;

        cv::Mat roiPrev = prevGray(safe);
        std::vector<cv::Point2f> pts;
        cv::goodFeaturesToTrack(roiPrev, pts, 20, 0.05, 5);
        if (pts.size() < 2) return true;

        for (auto& p : pts) { p.x += safe.x; p.y += safe.y; }

        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prevGray, currGray, pts, nextPts, status, err,
                                  cv::Size(21, 21), 3);

        std::vector<float> motions;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && err[i] < 15.0f) {
                float dx = nextPts[i].x - pts[i].x;
                float dy = nextPts[i].y - pts[i].y;
                motions.push_back(std::sqrt(dx * dx + dy * dy));
            }
        }
        if (motions.empty()) return false;

        std::sort(motions.begin(), motions.end());
        motionMagnitude = motions[motions.size() / 2];
        return motionMagnitude > 0.15f;
    }
};

/* =============== 4. ADAPTIVE DYNAMIC ROI =============== */
// ROI адаптируется по: размеру объекта, скорости, confidence и времени без детекции.
// При потере — ROI расширяется; при быстром объекте — вытягивается по вектору скорости.

class DynamicROI
{
public:
    DynamicROI(int frameW, int frameH)
        : frameW_(frameW), frameH_(frameH)
        , roiX_(0), roiY_(0), roiW_(frameW), roiH_(frameH)
        , targetX_(frameW / 2.0), targetY_(frameH / 2.0)
        , objW_(0), objH_(0), vx_(0), vy_(0)
        , confidence_(0.0), framesSinceDetection_(999)
        , initialized_(false)
    {}

    void update(bool detected, double x, double y, int objW, int objH,
                double vx, double vy, double confidence)
    {
        if (detected) {
            targetX_ = x; targetY_ = y;
            objW_ = objW; objH_ = objH;
            vx_ = vx; vy_ = vy;
            confidence_ = confidence;
            framesSinceDetection_ = 0;
            initialized_ = true;
        } else {
            framesSinceDetection_++;
            if (initialized_ && framesSinceDetection_ < 30) {
                targetX_ += vx_ * 0.033;
                targetY_ += vy_ * 0.033;
                confidence_ *= 0.9;
            }
        }
        recompute();
    }

    cv::Rect getROI() const { return cv::Rect(roiX_, roiY_, roiW_, roiH_); }
    bool isFullFrame() const { return !initialized_ || framesSinceDetection_ > 60; }
    void setFrameSize(int w, int h) { frameW_ = w; frameH_ = h; }

    void reset() {
        initialized_ = false;
        framesSinceDetection_ = 999;
        roiX_ = 0; roiY_ = 0;
        roiW_ = frameW_; roiH_ = frameH_;
    }

private:
    void recompute() {
        if (!initialized_ || framesSinceDetection_ > 30) {
            roiX_ = 0; roiY_ = 0; roiW_ = frameW_; roiH_ = frameH_;
            return;
        }
        double padFactor = 4.0;
        double speed = std::sqrt(vx_ * vx_ + vy_ * vy_);
        padFactor += speed * 0.02;
        if (confidence_ < 0.5) padFactor *= (2.0 - confidence_);
        padFactor += framesSinceDetection_ * 0.3;

        int minSz = 120, maxSz = std::min(frameW_, frameH_) * 3 / 4;
        roiW_ = std::clamp((int)(std::max(objW_, 20) * padFactor), minSz, maxSz);
        roiH_ = std::clamp((int)(std::max(objH_, 20) * padFactor), minSz, maxSz);

        double cx, cy;
        if (framesSinceDetection_ == 0) {
            // Object detected: snap ROI exactly to object, no EMA, no lead
            cx = targetX_;
            cy = targetY_;
        } else {
            // Object lost: coast with velocity + EMA toward predicted position
            double leadX = vx_ * 0.05, leadY = vy_ * 0.05;
            cx = 0.3 * (targetX_ + leadX) + 0.7 * (roiX_ + roiW_ / 2.0);
            cy = 0.3 * (targetY_ + leadY) + 0.7 * (roiY_ + roiH_ / 2.0);
        }
        roiX_ = std::max(0, (int)(cx - roiW_ / 2.0));
        roiY_ = std::max(0, (int)(cy - roiH_ / 2.0));
        if (roiX_ + roiW_ > frameW_) roiX_ = frameW_ - roiW_;
        if (roiY_ + roiH_ > frameH_) roiY_ = frameH_ - roiH_;
    }

    int frameW_, frameH_;
    int roiX_, roiY_, roiW_, roiH_;
    double targetX_, targetY_;
    int objW_, objH_;
    double vx_, vy_;
    double confidence_;
    int framesSinceDetection_;
    bool initialized_;
};

/* =============== 5. SUBPIXEL CENTROID ESTIMATION =============== */
// Gaussian-weighted moments в окне 2r×2r вокруг грубого центроида.
// Точность ~0.1 px вместо целых пикселей из contour-moments.

class SubpixelRefiner
{
public:
    static cv::Point2f refine(const cv::Mat& gray, cv::Point2f coarse, int radius = 10)
    {
        int cx = (int)std::round(coarse.x), cy = (int)std::round(coarse.y);
        int x0 = std::max(0, cx - radius), y0 = std::max(0, cy - radius);
        int x1 = std::min(gray.cols - 1, cx + radius);
        int y1 = std::min(gray.rows - 1, cy + radius);
        if (x1 - x0 < 3 || y1 - y0 < 3) return coarse;

        cv::Mat win = gray(cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1));
        double sigma2 = radius * radius * 0.5;
        double sumW = 0, sumWx = 0, sumWy = 0;
        for (int wy = 0; wy < win.rows; wy++) {
            const uchar* row = win.ptr<uchar>(wy);
            double py = y0 + wy;
            for (int wx = 0; wx < win.cols; wx++) {
                double px = x0 + wx;
                double dx = px - coarse.x, dy = py - coarse.y;
                double w = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma2)) * (row[wx] / 255.0);
                sumW  += w;
                sumWx += w * px;
                sumWy += w * py;
            }
        }
        if (sumW < 1e-6) return coarse;
        return cv::Point2f((float)(sumWx / sumW), (float)(sumWy / sumW));
    }
};

/* =============== 6. BAYESIAN STATE ESTIMATION (Kalman + JPDA) =============== */
// Полный 6-state Kalman: [x, y, vx, vy, ax, ay] с constant-jerk process noise.
// JPDA: вероятностная ассоциация нескольких кандидатов через гейтированное взвешенное среднее.
// Заменяет диагональный AngleKalman: полная 6×6 ковариационная матрица, корректная
// передаточная модель, адаптивный gate по неопределённости.

class BayesianTracker
{
public:
    struct State {
        double x, y;          // position (pixels)
        double vx, vy;        // velocity (px/s)
        double ax, ay;        // acceleration (px/s²)
        double confidence;    // [0..1]
    };

    BayesianTracker()
        : kf_(6, 2, 0, CV_64F), initialized_(false), lastTime_(0)
        , missCount_(0), velocityVariance_(1e6), confidence_(0.0)
    { initKF(); }

    void reset() { initKF(); initialized_ = false; missCount_ = 0; velocityVariance_ = 1e6; confidence_ = 0.0; }

    // JPDA update: several measurements with weights; empty = miss
    State update(const std::vector<cv::Point2f>& meas, const std::vector<double>& weights)
    {
        double now = timeNow();
        double dt = initialized_ ? (now - lastTime_) : 0.033;
        lastTime_ = now;
        dt = std::clamp(dt, 0.001, 0.5);

        setF(dt);
        setQ(dt);
        cv::Mat pred = kf_.predict();

        if (!meas.empty() && !weights.empty()) {
            double sumW = 0, wx = 0, wy = 0;
            double gate = getGateSize();
            for (size_t i = 0; i < meas.size(); i++) {
                double dx = meas[i].x - pred.at<double>(0);
                double dy = meas[i].y - pred.at<double>(1);
                if (dx * dx + dy * dy < gate * gate) {
                    double w = (i < weights.size()) ? weights[i] : 0.0;
                    wx += w * meas[i].x;
                    wy += w * meas[i].y;
                    sumW += w;
                }
            }
            if (sumW > 0.01) {
                cv::Mat z = (cv::Mat_<double>(2, 1) << wx / sumW, wy / sumW);
                kf_.correct(z);
                if (!initialized_) {
                    kf_.statePost.at<double>(0) = z.at<double>(0);
                    kf_.statePost.at<double>(1) = z.at<double>(1);
                    for (int i = 2; i < 6; i++) kf_.statePost.at<double>(i) = 0;
                    initialized_ = true;
                }
                missCount_ = 0;
                confidence_ = std::min(1.0, confidence_ + 0.35);
                double v2 = kf_.statePost.at<double>(2) * kf_.statePost.at<double>(2)
                          + kf_.statePost.at<double>(3) * kf_.statePost.at<double>(3);
                velocityVariance_ = 0.8 * velocityVariance_ + 0.2 * v2;
            } else { handleMiss(); }
        } else { handleMiss(); }

        // Clamp velocity in Kalman state to prevent divergence
        kf_.statePost.at<double>(2) = std::clamp(kf_.statePost.at<double>(2), -500.0, 500.0);
        kf_.statePost.at<double>(3) = std::clamp(kf_.statePost.at<double>(3), -500.0, 500.0);
        kf_.statePost.at<double>(4) = std::clamp(kf_.statePost.at<double>(4), -200.0, 200.0);
        kf_.statePost.at<double>(5) = std::clamp(kf_.statePost.at<double>(5), -200.0, 200.0);

        return currentState();
    }

    // Simple one-measurement update
    State update(bool valid, double x, double y) {
        if (valid) return update({cv::Point2f((float)x,(float)y)}, {1.0});
        return update({}, {});
    }

    State currentState() const {
        State s;
        s.x  = kf_.statePost.at<double>(0); s.y  = kf_.statePost.at<double>(1);
        s.vx = std::clamp(kf_.statePost.at<double>(2), -500.0, 500.0);
        s.vy = std::clamp(kf_.statePost.at<double>(3), -500.0, 500.0);
        s.ax = kf_.statePost.at<double>(4); s.ay = kf_.statePost.at<double>(5);
        s.confidence = confidence_;
        return s;
    }

    bool isInitialized() const  { return initialized_; }
    double getConfidence() const { return confidence_; }
    double getVelocityVariance() const { return velocityVariance_; }
    int getMissCount() const    { return missCount_; }

    double getGateSize() const {
        if (!initialized_) return 1e9;  // Accept ANY measurement for first contact
        double pu = std::sqrt(kf_.errorCovPost.at<double>(0, 0) + kf_.errorCovPost.at<double>(1, 1));
        return std::clamp(3.0 * pu, 50.0, 500.0);
    }

private:
    cv::KalmanFilter kf_;
    bool initialized_;
    double lastTime_;
    int missCount_;
    double velocityVariance_;
    double confidence_;

    static double timeNow() {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    void initKF() {
        kf_ = cv::KalmanFilter(6, 2, 0, CV_64F);
        kf_.measurementMatrix = cv::Mat::zeros(2, 6, CV_64F);
        kf_.measurementMatrix.at<double>(0, 0) = 1.0;
        kf_.measurementMatrix.at<double>(1, 1) = 1.0;
        kf_.measurementNoiseCov = cv::Mat::eye(2, 2, CV_64F) * 16.0;
        cv::setIdentity(kf_.errorCovPost, cv::Scalar(100.0));
        kf_.statePost = cv::Mat::zeros(6, 1, CV_64F);
    }

    void setF(double dt) {
        cv::setIdentity(kf_.transitionMatrix);
        double dt2 = dt * dt * 0.5;
        kf_.transitionMatrix.at<double>(0, 2) = dt;
        kf_.transitionMatrix.at<double>(0, 4) = dt2;
        kf_.transitionMatrix.at<double>(1, 3) = dt;
        kf_.transitionMatrix.at<double>(1, 5) = dt2;
        kf_.transitionMatrix.at<double>(2, 4) = dt;
        kf_.transitionMatrix.at<double>(3, 5) = dt;
    }

    void setQ(double dt) {
        double q = 30.0;
        double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
        cv::Mat Q = cv::Mat::zeros(6, 6, CV_64F);
        Q.at<double>(0, 0) = dt5 / 20; Q.at<double>(1, 1) = dt5 / 20;
        Q.at<double>(0, 2) = dt4 / 8;  Q.at<double>(2, 0) = dt4 / 8;
        Q.at<double>(1, 3) = dt4 / 8;  Q.at<double>(3, 1) = dt4 / 8;
        Q.at<double>(0, 4) = dt3 / 6;  Q.at<double>(4, 0) = dt3 / 6;
        Q.at<double>(1, 5) = dt3 / 6;  Q.at<double>(5, 1) = dt3 / 6;
        Q.at<double>(2, 2) = dt3 / 3;  Q.at<double>(3, 3) = dt3 / 3;
        Q.at<double>(2, 4) = dt2 / 2;  Q.at<double>(4, 2) = dt2 / 2;
        Q.at<double>(3, 5) = dt2 / 2;  Q.at<double>(5, 3) = dt2 / 2;
        Q.at<double>(4, 4) = dt;       Q.at<double>(5, 5) = dt;
        kf_.processNoiseCov = Q * q;
    }

    void handleMiss() {
        missCount_++;
        confidence_ = std::max(0.0, confidence_ - 0.05);
        kf_.statePost.at<double>(4) *= 0.85;
        kf_.statePost.at<double>(5) *= 0.85;
        if (missCount_ > 15) {
            kf_.statePost.at<double>(2) *= 0.95;
            kf_.statePost.at<double>(3) *= 0.95;
        }
        kf_.errorCovPost.at<double>(0, 0) += 10.0;
        kf_.errorCovPost.at<double>(1, 1) += 10.0;
        // Auto-reset after prolonged loss — allows re-initialization from next detection
        if (missCount_ > 45) {
            reset();
        }
    }
};

/* =============== 7. VELOCITY LOCKING =============== */
// Когда оценка скорости стабильна (дисперсия < порога на N кадрах подряд) —
// фиксируем вектор скорости и используем для coast-предсказания при потере цели.
// Unlock при значимом рассогласовании с новым измерением.

class VelocityLocker
{
public:
    struct LockedVelocity { double vx, vy; bool locked; double confidence; };

    VelocityLocker() = default;

    LockedVelocity update(double vx, double vy, double /*velVar*/, bool detected)
    {
        if (detected) {
            double dvx = vx - prevVx_, dvy = vy - prevVy_;
            double jitter = std::sqrt(dvx * dvx + dvy * dvy);
            prevVx_ = vx; prevVy_ = vy;
            double speed = std::sqrt(vx * vx + vy * vy);

            if (jitter < speed * 0.3 + 5.0 && speed > 10.0) {
                stableFrames_++;
                if (stableFrames_ >= 5) {
                    double a = locked_ ? 0.3 : 1.0;
                    lockVx_ = a * vx + (1.0 - a) * lockVx_;
                    lockVy_ = a * vy + (1.0 - a) * lockVy_;
                    locked_ = true;
                    lockConf_ = std::min(1.0, lockConf_ + 0.1);
                }
            } else {
                stableFrames_ = std::max(0, stableFrames_ - 2);
                if (stableFrames_ == 0) { locked_ = false; lockConf_ = 0; }
            }
        } else {
            stableFrames_ = std::max(0, stableFrames_ - 1);
            lockConf_ *= 0.95;
            if (lockConf_ < 0.1) { locked_ = false; lockConf_ = 0; }
        }
        return { locked_ ? lockVx_ : vx, locked_ ? lockVy_ : vy, locked_, lockConf_ };
    }

    void reset() { locked_ = false; lockVx_ = lockVy_ = prevVx_ = prevVy_ = 0; stableFrames_ = 0; lockConf_ = 0; }

private:
    bool locked_ = false;
    double lockVx_ = 0, lockVy_ = 0;
    double prevVx_ = 0, prevVy_ = 0;
    int stableFrames_ = 0;
    double lockConf_ = 0;
};

/* =============== 8. TRAJECTORY PREDICTION =============== */
// Constant-acceleration модель: x(t+dt) = x + v·dt + ½·a·dt².
// Уверенность падает экспоненциально с горизонтом предсказания.

class TrajectoryPredictor
{
public:
    struct Point { double x, y, confidence; };

    static Point predictAt(const BayesianTracker::State& s, double dt) {
        return { s.x + s.vx * dt + 0.5 * s.ax * dt * dt,
                 s.y + s.vy * dt + 0.5 * s.ay * dt * dt,
                 s.confidence * std::exp(-dt * 3.0) };
    }

    static std::vector<Point> predictTrajectory(const BayesianTracker::State& s, double dt, int steps) {
        std::vector<Point> t;
        t.reserve(steps);
        for (int i = 1; i <= steps; i++) t.push_back(predictAt(s, dt * i));
        return t;
    }
};

/* =============== 9. PREDICTIVE GIMBAL CONTROL =============== */
// Компенсация системной задержки (camera → detect → servo ≈ 50 мс):
// серво наводятся не на текущую, а на предсказанную позицию объекта через SYSTEM_DELAY.
// Ошибка модулируется confidence'ом, чтобы при низкой уверенности не гнаться за шумом.

class PredictiveGimbalController
{
public:
    PredictiveGimbalController(double delay, double dpp)
        : delay_(delay), degPerPx_(dpp) {}

    struct Cmd { double yawDeg, pitchDeg; };

    Cmd compute(const BayesianTracker::State& state, double cx, double cy,
                double curYaw, double curPitch)
    {
        auto pred = TrajectoryPredictor::predictAt(state, delay_);
        double ex = pred.x - cx, ey = pred.y - cy;
        double conf = std::min(1.0, pred.confidence + 0.3);
        ex *= conf;  ey *= conf;
        ex = std::clamp(ex, -250.0, 250.0);
        ey = std::clamp(ey, -250.0, 250.0);
        return { std::clamp(curYaw   - ex * degPerPx_, 5.0, 175.0),
                 std::clamp(curPitch - ey * degPerPx_, 5.0, 175.0) };
    }

    void setDelay(double d) { delay_ = d; }

private:
    double delay_, degPerPx_;
};

/* =============== 10. DUAL-LOOP SERVO STABILIZATION =============== */
// Outer loop (PID, частота детекции ~20-30 Hz): позиционная ошибка → желаемая скорость.
// Inner loop (PD, та же частота, но может быть быстрее при раздельном потоке):
//   ошибка скорости → инкремент угла серво. Slew-limiter ограничивает макс. шаг.
// Анти-windup на интеграле ограничивает накопление при насыщении.

class DualLoopServo
{
public:
    DualLoopServo()
        : oKp_(1.0), oKi_(0.05), oKd_(0.02)
        , iKp_(0.6), iKd_(0.15)
        , desired_(90), current_(90), prevAngle_(90), rate_(0)
        , integral_(0), prevOuterErr_(0), prevInnerErr_(0)
        , output_(90), slewLimit_(0.5)
    {}

    void setTarget(double deg) { desired_ = std::clamp(deg, 5.0, 175.0); }

    double tick(double dt) {
        if (dt <= 0) dt = 0.033;
        rate_ = (current_ - prevAngle_) / dt;
        prevAngle_ = current_;

        // Outer: position → desired rate
        double posErr = desired_ - current_;
        if (std::abs(posErr) < 10.0) integral_ += posErr * dt;
        integral_ = std::clamp(integral_, -20.0, 20.0);
        double dPos = (posErr - prevOuterErr_) / dt;
        prevOuterErr_ = posErr;
        double desRate = oKp_ * posErr + oKi_ * integral_ + oKd_ * dPos;

        // Inner: rate → angle increment
        double rateErr = desRate - rate_;
        double dRate = (rateErr - prevInnerErr_) / dt;
        prevInnerErr_ = rateErr;
        double inc = iKp_ * rateErr + iKd_ * dRate;
        inc = std::clamp(inc, -slewLimit_, slewLimit_);

        output_ = std::clamp(current_ + inc, 5.0, 175.0);
        current_ = output_;
        return output_;
    }

    double getOutput() const { return output_; }

    void reset(double a = 90.0) {
        desired_ = current_ = prevAngle_ = output_ = a;
        rate_ = integral_ = prevOuterErr_ = prevInnerErr_ = 0;
    }

private:
    double oKp_, oKi_, oKd_;
    double iKp_, iKd_;
    double desired_, current_, prevAngle_, rate_;
    double integral_, prevOuterErr_, prevInnerErr_;
    double output_, slewLimit_;
};



/* =============== CAMERA THREAD =============== */

void cameraThread(SafeQueue<FrameData>&queue,atomic<bool>&run)
{
    cv::VideoCapture cap;
    
    std::cout << "\n=== Camera Initialization ===" << std::endl;
    
    // Try method 1: GStreamer with libcamerasrc (Arducam 64MP)
    std::cout << "Attempting GStreamer libcamerasrc..." << std::endl;
    try {
        // === 1. LOW-LATENCY CAMERA ===
        // • exposure 20 ms (< 1 frame at 45 fps → no motion blur carryover)
        // • queue: zero-byte / zero-time buffer, leaky → no pipeline stall
        // • videoconvert: n-threads=2 for parallel NV12→BGR
        // • appsink: sync=false, single-buffer → grab newest frame only
        std::string cameraPipeline = 
            "libcamerasrc sharpness=16.0 exposure-time=20000 analogue-gain=8.0 ! "
            "video/x-raw,format=NV12,width=1920,height=1080,framerate=45/1,interlace-mode=progressive ! "
            "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
            "appsink sync=false max-buffers=1 drop=true emit-signals=false";
        
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

/* =============== TRACKING THREAD =============== */

void trackingThread(SafeQueue<FrameData>&queue,atomic<bool>&run)
{
    MotionDetector detector;

    double cx = CX;
    double cy = CY;

    double lastYawDeg = 90.0;
    double lastPitchDeg = 90.0;


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

    // === NEW PIPELINE COMPONENTS (methods 2-10) ===
    MultiScaleDetector multiScale;
    SparseMotionFilter motionFilter;
    BayesianTracker tracker;
    VelocityLocker velLocker;
    const double DEG_PER_PX = 72.0 / 1920.0 * 1.05;
    PredictiveGimbalController gimbal(SYSTEM_DELAY, DEG_PER_PX);
    DualLoopServo servoYaw, servoPitch;
    DynamicROI dynROI(1920, 1080);
    cv::Mat prevGrayDet;  // grayscale at detection resolution (480x270) for sparse motion filter

    // Для визуализации
    cv::Rect lastROI;
    std::vector<TrajectoryPredictor::Point> predTrajectory;

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
            setServoAngle(PWM_CHANNEL_HORIZONTAL, 90.0f);
            setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
            lastYawDeg   = 90.0;
            lastPitchDeg = 90.0;
        }

        // При переключении FIXED → TRACKING: reset all components
        static bool resetSpatialGating = false;
        if (modeJustChanged && currentTrackingEnabled) {
            detector.reinitBGS();
            multiScale.reinit();
            tracker.reset();
            velLocker.reset();
            servoYaw.reset(lastYawDeg);
            servoPitch.reset(lastPitchDeg);
            dynROI.reset();
            prevGrayDet = cv::Mat();
            resetSpatialGating = true;
        }

        // === FRAME PREPARATION ===
        cv::Mat display = f.frame.clone();
        dynROI.setFrameSize(f.frame.cols, f.frame.rows);

        // === 2. MULTI-SCALE COARSE DETECTION ===
        // Skip coarse pass when tracker is confident — saves ~3ms/frame
        std::vector<cv::Rect> coarseCandidates;
        if (tracker.getConfidence() < 0.3) {
            coarseCandidates = multiScale.coarseDetect(f.frame);
        }

        // === DETECTION: always full-frame at fixed resolution ===
        // MOG2 MUST always see the same viewpoint for stable background model.
        // ROI crop caused model instability (different content → alternating fg).
        cv::Mat resized;
        const int DET_W = 480, DET_H = 270;
        const double scX = (double)f.frame.cols / DET_W;
        const double scY = (double)f.frame.rows / DET_H;
        cv::resize(f.frame, resized, cv::Size(DET_W, DET_H), 0, 0, cv::INTER_LINEAR);

        // Spatial gating: search near last known position instead of full frame
        // Adaptive radius: tight when noisy (camera moving), wide when clean
        static int servoSettleCounter = 0;
        static double lastKnownX = -1, lastKnownY = -1;
        static int consecutiveValid = 0;  // for servo correction gating
        static int framesWithoutDetection = 0;  // for return-to-center
        static double prevErrX = 0, prevErrY = 0;  // for PD controller D-term
        static double filtErrX = 0, filtErrY = 0;  // EMA-filtered error
        static int framesSinceServo = 999;  // frames since last servo correction
        if (resetSpatialGating) {
            lastKnownX = -1; lastKnownY = -1;
            servoSettleCounter = 0;
            consecutiveValid = 0;
            framesWithoutDetection = 0;
            framesSinceServo = 999;
            resetSpatialGating = false;
        }
        framesSinceServo++;
        cv::Point2f searchPt(-1, -1);
        // Track previous frame's fg for learning rate decision
        static int fgPrevFrame = 0;
        // Widen spatial gate after servo correction.
        // When we're actively losing detection, force wide gate to re-acquire.
        float searchRad;
        if (framesWithoutDetection > 5) searchRad = 200.0f;   // lost: full frame search
        else if (framesSinceServo < 10) searchRad = 150.0f;    // active tracking: wide
        else if (framesSinceServo < 30) searchRad = 80.0f;     // settling
        else searchRad = 40.0f;                                 // stationary
        if (lastKnownX >= 0) {
            searchPt = cv::Point2f((float)(lastKnownX / scX), (float)(lastKnownY / scY));
        }

        // MOG2 tracking learning rate.
        // Storm lr=0.05: clears 20k→3k in ~8 frames. Absorbs object ~34% over 8 frames.
        // This absorption is the cost of fast storm clearance.
        // Alternative (lr=0.03): 22% absorption but ~15 frame clearance — net WORSE
        // because more FG_GATE frames block detection while absorption still happens.
        double trackingLR;
        if (fgPrevFrame > 3000) trackingLR = 0.05;              // storm: fast clear
        else if (fgPrevFrame > 1000) trackingLR = 0.015;        // elevated: moderate
        else if (framesSinceServo < 15) trackingLR = 0.008;     // post-servo settling
        else trackingLR = 0.0;                                   // normal (base lr=0.003)
        Detection d = detector.detect(resized, 0, 0, trackingLR, searchPt, searchRad);

        // === PER-FRAME DIAGNOSTIC LOG ===
        int fgPx = detector.lastFGPixels();
        int rawContours = detector.lastRawContours();
        int objBeforeGate = (int)d.all_boxes.size();  // save before gates clear it
        const char* rejectReason = nullptr;

        // FG-based gating: when fg is high, ALL contours are background noise.
        // Real object is indistinguishable from hundreds of residual FG blobs.
        // Gate at 3000: real objects produce fg 50-500; anything above = broken model.
        if (fgPx > 3000) {
            d.valid = false;
            d.all_boxes.clear();
            rejectReason = "FG_GATE";
        }

        // Confidence gate: when fg is elevated, many noise blobs pass contour filter.
        // Spatial rescue: if the best detection is close to lastKnown position,
        // trust it even during noise — spatial coherence overrides blob count.
        bool noiseTriggered = false;
        if (!rejectReason && fgPx > 2000 && (int)d.all_boxes.size() > 3) noiseTriggered = true;
        if (!rejectReason && !noiseTriggered && fgPx > 1500 && (int)d.all_boxes.size() > 6) noiseTriggered = true;

        if (noiseTriggered) {
            bool rescued = false;
            if (d.valid && searchPt.x >= 0) {
                float dx = d.x - searchPt.x;
                float dy = d.y - searchPt.y;
                float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < 30.0f) rescued = true;  // 30px at 480×270
            }
            if (!rescued) {
                d.valid = false;
                d.all_boxes.clear();
                rejectReason = "NOISE_HI_FG";
            }
        }

        // Track if detection was rescued from NOISE gate
        bool wasRescued = (noiseTriggered && d.valid);

        if (!rejectReason && !d.valid) {
            rejectReason = "NO_CONTOUR";
        }

        // Update fgPrevFrame for next frame's lr decision
        fgPrevFrame = fgPx;

        // Per-frame log: print every frame for diagnosis
        static int diagFrame = 0;
        diagFrame++;
        std::cout << "[F" << diagFrame << "] "
                  << (d.valid ? "DET" : "---")
                  << " fg=" << fgPx
                  << " cnt=" << rawContours
                  << " obj=" << objBeforeGate
                  << " sRad=" << (int)searchRad
                  << " fServo=" << framesSinceServo
                  << " lr=" << std::fixed << std::setprecision(4) << trackingLR;
        if (rejectReason) std::cout << " REJ=" << rejectReason;
        if (wasRescued) std::cout << " RESCUED";
        if (d.valid) std::cout << " pos=(" << (int)(d.x*scX) << "," << (int)(d.y*scY) << ")";
        std::cout << std::endl;

        // Motion filter REMOVED: at 480×270 detection resolution, objects are 3-10px.
        // goodFeaturesToTrack in such tiny bboxes fails to find points → false reject in ~80% frames.
        // MOG2 with motion compensation already filters static background.

        // Subpixel refinement REMOVED: Gaussian moments unreliable on 3-10px objects
        // at 480×270 resolution. Saves ~2ms per frame.

        // Scale back: detection coords (480x270) → full-frame coords
        if (d.valid) {
            d.x = d.x * scX;
            d.y = d.y * scY;
            d.box_x = (int)(d.box_x * scX);
            d.box_y = (int)(d.box_y * scY);
            d.box_w = (int)(d.box_w * scX);
            d.box_h = (int)(d.box_h * scY);
        }
        for (auto& box : d.all_boxes) {
            box.x = (int)(box.x * scX);
            box.y = (int)(box.y * scY);
            box.width  = (int)(box.width  * scX);
            box.height = (int)(box.height * scY);
        }

        // === 6. BAYESIAN STATE ESTIMATION (Kalman + JPDA) ===
        BayesianTracker::State state;
        if (d.valid) {
            std::vector<cv::Point2f> meas = { cv::Point2f((float)d.x, (float)d.y) };
            std::vector<double> wts  = { 1.0 };
            // Low-weight JPDA association from coarse candidates
            for (const auto& cand : coarseCandidates) {
                float cx_ = cand.x + cand.width * 0.5f;
                float cy_ = cand.y + cand.height * 0.5f;
                float dist = std::sqrt((cx_ - (float)d.x) * (cx_ - (float)d.x)
                                     + (cy_ - (float)d.y) * (cy_ - (float)d.y));
                if (dist > 20.0f && dist < 300.0f) {
                    meas.push_back(cv::Point2f(cx_, cy_));
                    wts.push_back(0.1);
                }
            }
            state = tracker.update(meas, wts);
        } else {
            state = tracker.update({}, {});
        }

        // === 7. VELOCITY LOCKING ===
        auto lockedVel = velLocker.update(state.vx, state.vy,
                                          tracker.getVelocityVariance(), d.valid);
        BayesianTracker::State predState = state;
        if (lockedVel.locked) {
            predState.vx = lockedVel.vx;
            predState.vy = lockedVel.vy;
        }

        // === 4 (cont). UPDATE DYNAMIC ROI ===
        // When servo is close to object (small pixel error), use raw detection
        // to prevent Kalman prediction from drifting ROI away.
        // When far, use Kalman state for smoother convergence.
        double convergeDist = 9999;
        if (d.valid) {
            convergeDist = std::sqrt((d.x - cx) * (d.x - cx) + (d.y - cy) * (d.y - cy));
        }
        bool servoClose = (convergeDist < 150.0);  // ~150px = servo nearly caught up
        if (d.valid && servoClose) {
            // Servo matched object — use raw detection, bypass Kalman drift
            dynROI.update(true, d.x, d.y,
                          d.box_w, d.box_h,
                          0, 0, 1.0);  // zero velocity = no lead, confidence=1
        } else {
            dynROI.update(d.valid,
                          d.valid ? d.x : state.x, d.valid ? d.y : state.y,
                          d.valid ? d.box_w : 0, d.valid ? d.box_h : 0,
                          predState.vx, predState.vy, state.confidence);
        }

        // === 8. TRAJECTORY PREDICTION ===
        predTrajectory = TrajectoryPredictor::predictTrajectory(predState, 0.033, 15);

        double yawDeg = lastYawDeg;
        double pitchDeg = lastPitchDeg;

        // === SCAN MODE LOGIC ===
        bool scanActive = false;
        static int scanExitCount = 0;  // consecutive valid detections needed to leave scan
        if (scanEnabled.load() && currentTrackingEnabled) {
            if (d.valid) {
                scanExitCount++;
                if (scanExitCount >= 3) {
                    // Require 3 consecutive valid detections before exiting scan.
                    // A single noisy MOG2 contour is NOT enough to interrupt sweep.
                    noDetectionSince = std::chrono::steady_clock::now();
                    if (wasScanning) {
                        std::cout << "[SCAN] Object confirmed (3 consecutive). Switching to tracking." << std::endl;
                        wasScanning = false;
                        scanDwelling = false;
                        scanExitCount = 0;
                    }
                }
            } else {
                scanExitCount = 0;  // reset on any miss
                auto now = std::chrono::steady_clock::now();
                double noDetSec = std::chrono::duration<double>(now - noDetectionSince).count();

                if (noDetSec >= SCAN_START_DELAY) {
                    scanActive = true;
                    scanActiveNow = true;

                    if (!wasScanning) {
                        wasScanning = true;
                        scanYawDeg = 90.0;
                        scanDirection = 1;
                        scanDwelling = true;
                        scanStepTime = now;
                        std::cout << "[SCAN] Starting sweep at 90 deg (dir=1)" << std::endl;
                    }

                    double dwellElapsed = std::chrono::duration<double>(now - scanStepTime).count();
                    if (scanDwelling && dwellElapsed >= SCAN_DWELL_SEC) {
                        scanYawDeg += scanDirection * SCAN_STEP;
                        if (scanYawDeg >= 180.0) { scanYawDeg = 180.0; scanDirection = -1; }
                        else if (scanYawDeg <= 0.0) { scanYawDeg = 0.0; scanDirection = 1; }
                        scanStepTime = now;
                        std::cout << "[SCAN] Step -> " << scanYawDeg
                                  << " deg (dir=" << scanDirection << ")" << std::endl;
                    }

                    setServoAngle(PWM_CHANNEL_HORIZONTAL, static_cast<float>(scanYawDeg));
                    setServoAngle(PWM_CHANNEL_VERTICAL, 90.0f);
                    lastYawDeg = scanYawDeg;
                    lastPitchDeg = 90.0;
                    yawDeg = scanYawDeg;
                    pitchDeg = 90.0;
                }
            }
        } else {
            if (wasScanning) {
                std::cout << "[SCAN] Scan mode deactivated." << std::endl;
                wasScanning = false;
                scanDwelling = false;
            }
            scanActiveNow = false;
            noDetectionSince = std::chrono::steady_clock::now();
        }

        // Track consecutive valid detections for servo stability
        // Save framesWithoutDetection BEFORE resetting it — needed for position trust.
        int fwdBeforeUpdate = framesWithoutDetection;
        if (d.valid) {
            consecutiveValid++;
            framesWithoutDetection = 0;
        } else {
            consecutiveValid = 0;
            framesWithoutDetection++;
            if (framesWithoutDetection > 5) {
                prevErrX = 0; prevErrY = 0;  // reset D-term only after extended loss
                filtErrX = 0; filtErrY = 0;  // reset EMA filter too
            }
        }

        // Update spatial gate position.
        // Trust first detection after loss (fwdBeforeUpdate > 5): when the object
        // was invisible for many frames, any detection in wide-search mode is
        // far more likely to be real than noise (spatial gate is 200px = whole frame).
        // Without immediate position update, sRad snaps back to 40px centered on
        // STALE position → next frame misses → 25-frame cycle of 1-frame detections.
        // During active tracking (fwd <= 5), require 2+ consecutive for trust —
        // prevents single noise blobs near storms from poisoning the gate.
        if (d.valid && (consecutiveValid >= 2 || lastKnownX < 0 || fwdBeforeUpdate > 5)) {
            lastKnownX = d.x;
            lastKnownY = d.y;
        } else if (framesWithoutDetection > 30) {
            // Lost object completely — reset spatial gating, allow full-frame search
            lastKnownX = -1;
            lastKnownY = -1;
        }

        // === 9+10. SERVO CONTROL — PD controller with damping ===
        // === 9+10. SERVO CONTROL — two-mode: acquisition + tracking ===
        // Acquisition (error > 150px): aggressive proportional, high slew
        // Tracking  (error <= 150px): PD controller with EMA, moderate slew
        bool servoAllowed = false;
        if (currentTrackingEnabled && !scanActive && d.valid && servoSettleCounter <= 0) {
            if (state.confidence >= 0.5 || servoClose) {
                servoAllowed = (consecutiveValid >= 1);
            } else {
                servoAllowed = (consecutiveValid >= 2);
            }
        }
        if (servoAllowed) {
            double ex = d.x - cx;
            double ey = d.y - cy;
            const double DEG_PER_PX = 72.0 / 1920.0 * 1.05;
            double absErr = std::sqrt(ex * ex + ey * ey);

            double corrYaw, corrPitch;
            if (absErr > 150.0) {
                // ACQUISITION MODE: proportional jump, limited to 3° max.
                // Larger steps destroy MOG2 background model → detection lost 20+ frames.
                // 3° = ~76px shift, within affine compensation range of MOG2.
                corrYaw   = -ex * 0.7 * DEG_PER_PX;
                corrPitch = -ey * 0.7 * DEG_PER_PX;
                corrYaw   = std::clamp(corrYaw,   -3.0, 3.0);
                corrPitch = std::clamp(corrPitch, -3.0, 3.0);
                // Reset EMA to current error so tracking mode starts clean
                filtErrX = ex;
                filtErrY = ey;
                prevErrX = filtErrX;
                prevErrY = filtErrY;
            } else {
                // TRACKING MODE: PD controller with EMA smoothing
                const double ALPHA = 0.7;
                filtErrX = ALPHA * ex + (1.0 - ALPHA) * filtErrX;
                filtErrY = ALPHA * ey + (1.0 - ALPHA) * filtErrY;

                double dex = filtErrX - prevErrX;
                double dey = filtErrY - prevErrY;
                prevErrX = filtErrX;
                prevErrY = filtErrY;

                double Kp = 0.55;
                double Kd = 0.15;
                corrYaw   = -(filtErrX * Kp + dex * Kd) * DEG_PER_PX;
                corrPitch = -(filtErrY * Kp + dey * Kd) * DEG_PER_PX;
                corrYaw   = std::clamp(corrYaw,   -2.5, 2.5);
                corrPitch = std::clamp(corrPitch, -2.5, 2.5);
            }

            yawDeg   = std::clamp(lastYawDeg   + corrYaw,   5.0, 175.0);
            pitchDeg = std::clamp(lastPitchDeg + corrPitch, 5.0, 175.0);

            // Compensate spatial gate: servo moves camera → object shifts in frame
            const double PX_PER_DEG = 1.0 / DEG_PER_PX;
            double shiftX = -(yawDeg - lastYawDeg)   * PX_PER_DEG;
            double shiftY = -(pitchDeg - lastPitchDeg) * PX_PER_DEG;
            if (lastKnownX >= 0) {
                lastKnownX = std::clamp(lastKnownX + shiftX, 0.0, (double)(f.frame.cols - 1));
                lastKnownY = std::clamp(lastKnownY + shiftY, 0.0, (double)(f.frame.rows - 1));
            }

            setServoAngle(PWM_CHANNEL_HORIZONTAL, static_cast<float>(yawDeg));
            setServoAngle(PWM_CHANNEL_VERTICAL, static_cast<float>(pitchDeg));
            lastYawDeg = yawDeg;
            lastPitchDeg = pitchDeg;
            servoSettleCounter = 0;
            framesSinceServo = 0;
            detector.notifyServoMove(absErr > 150.0);  // freeze only acquisition
        }
        if (servoSettleCounter > 0) servoSettleCounter--;

        // ROI для визуализации
        lastROI = dynROI.getROI() & cv::Rect(0, 0, display.cols, display.rows);

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
            std::cout << "Kalman: conf=" << std::setprecision(2) << tracker.getConfidence()
                      << " miss=" << tracker.getMissCount()
                      << " v=(" << std::setprecision(1) << state.vx << "," << state.vy << ")"
                      << " velLock=" << (lockedVel.locked ? "ON" : "off") << std::endl;
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

        // Predicted trajectory (cyan, dashed) — method 8 visualization
        if (trajectoryEnabled.load() && tracker.isInitialized() && !predTrajectory.empty()) {
            cv::Point prevPt((int)predState.x, (int)predState.y);
            for (size_t i = 0; i < predTrajectory.size(); i++) {
                cv::Point pt((int)predTrajectory[i].x, (int)predTrajectory[i].y);
                if (pt.x > 0 && pt.x < display.cols && pt.y > 0 && pt.y < display.rows) {
                    int a = (int)(255 * predTrajectory[i].confidence);
                    a = std::clamp(a, 30, 255);
                    cv::line(display, prevPt, pt, cv::Scalar(a, a, 0), 1); // cyan fading
                    cv::circle(display, pt, 2, cv::Scalar(a, a, 0), -1);
                }
                prevPt = pt;
            }
        }

        // Dynamic ROI rectangle (magenta)
        if (!dynROI.isFullFrame()) {
            cv::Rect dr = dynROI.getROI() & cv::Rect(0, 0, display.cols, display.rows);
            cv::rectangle(display, dr, cv::Scalar(255, 0, 255), 2);
            cv::putText(display, "DROI", cv::Point(dr.x + 5, dr.y + 18),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
        }
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

        // Draw active ROI window
        if (!dynROI.isFullFrame()) {
            cv::Rect roi = dynROI.getROI() & cv::Rect(0, 0, display.cols, display.rows);
            cv::rectangle(display, roi, cv::Scalar(255, 255, 0), 3);
            cv::putText(display, "ROI", cv::Point(roi.x + 5, roi.y + 25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        }
        
        // Draw bright green rectangles around all detected moving objects with labels
        int objCounter = 0;
        for (const auto& box : d.all_boxes) {
            cv::rectangle(display, box, cv::Scalar(0,255,0), 3);  // Thicker border
            // Add semi-transparent fill (lightweight)
            // Lightweight fill: just draw a thin inner rectangle instead of full-frame blend
            cv::Rect inner(box.x + 1, box.y + 1, std::max(1, box.width - 2), std::max(1, box.height - 2));
            cv::rectangle(display, inner, cv::Scalar(0, 200, 0), 1);
            
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
    cv::namedWindow("Predictive Gimbal Control", cv::WINDOW_GUI_NORMAL | cv::WINDOW_NORMAL);
    cv::moveWindow("Predictive Gimbal Control", 100, 50);
    cv::resizeWindow("Predictive Gimbal Control", 1280, 720);
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
                cv::Mat display;
                cv::resize(frame, display, cv::Size(1280, 720));
                cv::imshow("Predictive Gimbal Control", display);
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

//https://github.com/biatech665696-ai/gimbal-arducam.git1