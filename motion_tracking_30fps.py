
#motion_tracking_30fps.pycod
#========================
#
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
import cv2
import numpy as np
import socket
import json
import os
from collections import deque

# ---------------- UDP параметры ----------------
UDP_IP = "192.168.1.100"   # IP ПК
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
# ---------------- Настройки камеры ----------------
CAM_INDEX = 0
FPS = 30
FRAME_WIDTH = 1920 #9152
FRAME_HEIGHT = 1080 #6944
 
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

# ---------------- Инициализация камеры ----------------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Более чувствительная настройка вычитания фона:
# - уменьшён history чтобы модель быстрее адаптировалась к изменениям
# - уменьшён varThreshold чтобы обнаруживать более слабые изменения
# - отключено обнаружение теней (меньше ложных срабатываний)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

frame_id = 0
prev_gray_for_flow = None
flow_scale = 2  # downscale factor for optical flow to speed up computation

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Сохраняем RAW G-канал
    g_channel = frame[:, :, 1]
    raw_path = os.path.join(RAW_DIR, f"frame_{frame_id:06d}.raw")
    g_channel.tofile(raw_path)

    # Коррекция fisheye
    corrected = undistort_fisheye(frame)
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

cap.release()
cv2.destroyAllWindows()
#[DEBUG ON]
#[DEBUG OFF]

#================================ RESTART: Shell ================================
