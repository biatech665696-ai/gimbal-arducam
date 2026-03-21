#!/usr/bin/python3
#motion_tracking_30fps_11_11_25.py
#========================
# находит движущиеся объекты
#==============================
# Изменения в этой версии (кратко):
# - Добавлена обработка fisheye с защитой: попытка undistort через cv2.fisheye,
#   при ошибке возвращается исходный кадр, чтобы цикл не падал.
# - Сохранение G-канала каждого кадра в RAW-файлы в папке `raw_capture`.
# - Добавлен быстрый подсчёт средней «скорости» движения по кадрам через
#   Farneback optical flow (уменьшенное изображение) — используется как
#   метрика для адаптивной настройки чувствительности.
# - Динамическая чувствительность вычитания фона: порог бинаризации и
#   минимальная площадь контура зависят от средней скорости движения.
# - Настройки BackgroundSubtractorMOG2 скорректированы для большей чувствительности
#   (меньше history, меньше varThreshold, detectShadows=False).
# - Реализован «след» объектов (trail) с ограниченной длиной и визуальным
#   затуханием (TRAIL_LENGTH, TRAIL_FADE).
# - Рисуется прогнозная точка по вектору последнего движения для каждого следа.
# - Данные об обнаруженных объектах отправляются по UDP в JSON-формате и
#   кратко логируются в консоль для отладки.
#
# Примечания/следующие улучшения (не реализовано здесь):
# - Калибровка метрики "метры/пиксель" (можно добавить диалог ввода и
#   сохранение параметров); для точных скоростей нужна полноценная
#   калибровка камеры.
# - Более устойчивый трекер (SORT/DeepSORT) вместо простого индексирования
#   по контурной области.
#
# - Добавлена функция sharpen_image для повышения резкости изображения
#   с помощью фильтра unsharp masking. Силу резкости можно регулировать
#   параметром --sharpen.
#
import cv2
import numpy as np
import socket
import json
import os
from collections import deque
import argparse
import sys
# Try to import Picamera2 (Raspberry Pi). If unavailable, fall back to OpenCV VideoCapture.
picamera2_available = False
try:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')
    from picamera2 import Picamera2
    picamera2_available = True
except Exception:
    Picamera2 = None
    picamera2_available = False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Motion tracking with fisheye correction and UDP output.')
parser.add_argument('--width', type=int, default=2400, help='Frame width')
parser.add_argument('--height', type=int, default=1200, help='Frame height')
parser.add_argument('--udp-ip', default="192.168.1.100", help='UDP IP address')
parser.add_argument('--udp-port', type=int, default=5005, help='UDP port')
parser.add_argument('--sharpen', type=float, default=0.0, help='Sharpening strength (0.0 to disable, typical 1.5-2.5)')
parser.add_argument('--source', default='0', help='Video source: camera index or path to video file (default 0).')
parser.add_argument('--use-picam2', action='store_true', default=picamera2_available, help='Prefer Picamera2 if available (default auto-detected)')
parser.add_argument('--demo-image', default=None, help='Path to an image file to use as a static demo input')
args = parser.parse_args()

# --- Sharpening Function ---
def sharpen_image(image, strength=1.5):
    """Apply unsharp mask sharpening.

    strength: strength of the effect (0 -> no change). Typical range 0.0-3.0.
    """
    if strength <= 0.0:
        return image
    # Gaussian blur with (0,0) kernel lets OpenCV pick size from sigma
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=strength, sigmaY=strength)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# ---------------- UDP параметры ----------------
UDP_IP = args.udp_ip   # IP ПК
UDP_PORT = args.udp_port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
# ---------------- Настройки камеры ----------------
CAM_INDEX = 2
FPS = 30
FRAME_WIDTH = args.width
FRAME_HEIGHT = args.height
 
# ---------------- Настройки trail ----------------
TRAIL_LENGTH = 20   # сколько точек хранить для "следа"
TRAIL_FADE = 10    # шаг затухания (чем больше — быстрее исчезает)
 
# ---------------- Папка для RAW ----------------
RAW_DIR = "raw_capture"
os.makedirs(RAW_DIR, exist_ok=True)
 
# Очереди для хранения траекторий объектов
trails = {}
 
# ---------------- Функция: коррекция fisheye ----------------
def undistort_fisheye(frame):
    h, w = frame.shape[:2]
    # Матрица камеры (примерная, требует калибровки для точности)
    K = np.array([[w/2, 0, w/2],
                  [0, w/2, h/2],
                  [0,   0,   1]], dtype=np.float32)
    D = np.zeros(4)
    # если нет калибровочных данных, оставляем 0
    try:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3, dtype=np.float32), K, (w, h), cv2.CV_16SC2
        )
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    except Exception:
        # при ошибке возвращаем оригинал, чтобы цикл не упал
        return frame


# ---------------- Перевод пикселей в углы ----------------
def pixel_to_angle(x, y, width, height, fov_horizontal=180, fov_vertical=180):
    cx = width / 2
    cy = height / 2
    nx = (x - cx) / cx   # нормализуем [-1;1]
    ny = (y - cy) / cy
    angle_x = nx * (fov_horizontal / 2)
    angle_y = ny * (fov_vertical / 2)
    return angle_x, angle_y

# ---------------- Capture initialization ----------------
cap = None
use_picam2 = False
picam2 = None

def _init_picam2():
    p = Picamera2()
    cfg = p.create_video_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}, controls={"FrameRate": FPS})
    p.configure(cfg)
    p.start()
    return p

if args.demo_image:
    demo_img = cv2.imread(args.demo_image, cv2.IMREAD_COLOR)
    if demo_img is None:
        print(f"Failed to load demo image: {args.demo_image}")
        sys.exit(1)
    demo_img = cv2.resize(demo_img, (FRAME_WIDTH, FRAME_HEIGHT))
    def get_frame():
        return demo_img.copy()
else:
    if args.use_picam2 and picamera2_available:
        try:
            picam2 = _init_picam2()
            use_picam2 = True
            def get_frame():
                f = picam2.capture_array()
                return cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Failed to initialize Picamera2: {e}. Falling back to OpenCV VideoCapture.")
            use_picam2 = False

    if not use_picam2:
        # Interpret numeric source as camera index
        src = args.source
        try:
            idx = int(src)
            cap = cv2.VideoCapture(idx)
        except Exception:
            cap = cv2.VideoCapture(src)

        if not cap or not cap.isOpened():
            print(f"Failed to open video source: {args.source}")
            sys.exit(1)

        # set desired resolution when possible
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)
        except Exception:
            pass

        def get_frame():
            ret, frame = cap.read()
            if not ret:
                return None
            return frame

# Более чувствительная настройка вычитания фона:
# - уменьшён history чтобы модель быстрее адаптировалась к изменениям
# - уменьшён varThreshold чтобы обнаруживать более слабые изменения
# - отключено обнаружение теней (меньше ложных срабатываний)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

frame_id = 0
prev_gray_for_flow = None
flow_scale = 2  # downscale factor for optical flow to speed up computation

while True:
    frame = get_frame()
    if frame is None:
        # source ended or read error
        print("No frame received (source ended). Exiting.")
        break

    frame_id += 1

    # Сохраняем RAW G-канал
    g_channel = frame[:, :, 1]
    raw_path = os.path.join(RAW_DIR, f"frame_{frame_id:06d}.raw")
    g_channel.tofile(raw_path)

    # Коррекция fisheye
    corrected = undistort_fisheye(frame)
    # optionally apply sharpening
    if args.sharpen > 0:
        corrected = sharpen_image(corrected, strength=args.sharpen)
    #corrected = frame


    # Берём G-канал для обработки
    gray = corrected[:, :, 1]

    # Оценка средней скорости движения между кадрами (Farneback на уменьшенном изображении)
    avg_speed = 0.0
    try:
        if prev_gray_for_flow is not None:
            small_prev = cv2.resize(prev_gray_for_flow, (gray.shape[1]//flow_scale, gray.shape[0]//flow_scale))
            small_curr = cv2.resize(gray, (gray.shape[1]//flow_scale, gray.shape[0]//flow_scale))
            # convert to float32
            small_prev_f = small_prev.astype(np.uint8)
            small_curr_f = small_curr.astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(small_prev_f, small_curr_f, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_speed = float(np.mean(mag))
        # store current for next frame
        prev_gray_for_flow = gray.copy()
    except Exception:
        avg_speed = 0.0

    # Динамическая настройка чувствительности: чем выше avg_speed, тем ниже порог (более чувствительно)
    # порог бинаризации в диапазоне [10, 100]
    dynamic_thresh = int(np.clip(50 - avg_speed * 10.0, 10, 100))
    # минимальная площадь контура: чем быстрее, тем меньше (ловим маленькие быстрые объекты)
    dynamic_min_area = int(np.clip(800 - avg_speed * 50.0, 200, 5000))

    # Фон/движение
    fgmask = fgbg.apply(gray)
    # Пониженный/динамический порог бинаризации — реагируем на слабые изменения фона
    _, thresh = cv2.threshold(fgmask, dynamic_thresh, 255, cv2.THRESH_BINARY)
   
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []

    for i, cnt in enumerate(contours):
        # динамический порог по площади контура — ловим меньше объектов при медленном движении
        if cv2.contourArea(cnt) < dynamic_min_area: # фильтр шума
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # вычисляем углы
        angle_h, angle_v = pixel_to_angle(cx, cy, corrected.shape[1], corrected.shape[0])

        obj = {
            "id": i,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "angle_h": float(angle_h),
            "angle_v": float(angle_v)
        }
        objects.append(obj)

        # --- Trail обработка ---
        if i not in trails:
            trails[i] = deque(maxlen=TRAIL_LENGTH)
        trails[i].append((cx, cy))

        # Рисуем рамку и центр
        cv2.rectangle(corrected, (x, y), (x+w, y+h), (0, 255,0), 2)
        cv2.circle(corrected, (cx, cy), 4, (0, 0, 255), -1)

        # Рисуем след с затуханием
        pts = list(trails[i])
        for j in range(1, len(pts)):
            alpha = max(0, 255 - (len(pts)-j)*TRAIL_FADE)
            cv2.line(corrected, pts[j-1], pts[j], (255, alpha, alpha), 2)

        # Прогноз: продолжаем движение по последнему вектору
        if len(pts) >= 2:
            dx = pts[-1][0] - pts[-2][0]
            dy = pts[-1][1] - pts[-2][1]
            pred_x, pred_y = pts[-1][0] + dx*5, pts[-1][1] + dy*5
            cv2.circle(corrected, (pred_x, pred_y), 6, (255, 255, 0), 2)

    # --- Отправка данных на ПК ---
    if objects:
        data = json.dumps(objects).encode("utf-8")
        sock.sendto(data, (UDP_IP, UDP_PORT))
        # Выводим отправляемые данные в терминал для отладки (кратко)
        try:
            print(f"[Frame {frame_id}] Sent {len(objects)} objects to {UDP_IP}:{UDP_PORT}")
            print(json.dumps(objects, ensure_ascii=False))
        except Exception as e:
            print(f"Failed to print payload: {e}")

    # --- Показ ---
    cv2.imshow("Motion Tracking Fisheye", corrected)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
if cap is not None:
    try:
       cap.release()
    except Exception:
       pass

#if 'picam2' in globals() and use_picam2:
#    try:
#        picam2.stop()
#   except Exception:
#        pass

#[DEBUG ON]
#[DEBUG OFF]

#================================ RESTART: Shell ================================

