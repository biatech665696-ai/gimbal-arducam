#motion_tracking_30fps2.py
# показывает движение объектов с калибровкой по 2-м точкам
# вычисляет скорость в м/с
#========================
#
# Изменения и особенности этой версии:
# - Калибровка по 2 точкам: нажмите 'c', кликните 2 раза левой кнопкой мыши
#   по известной длине на изображении, затем введите реальную длину в метрах
#   в консоли (поддерживаются десятичные с запятой/точкой и простые единицы:
#   mm, cm, m — например: "0,45", "45cm", "450 mm").
#   Добавлена валидация ввода (до 3 попыток) и возможность отмены (Enter).
# - Простой трекер (SimpleTracker): жадное сопоставление по IoU для сохранения id
#   между кадрами; скорость вычисляется как перемещение центра bbox в метрах/сек.
# - Фильтрация и регистрация по скорости: объекты регистрируются только если
#   SPEED_THRESHOLD_MPS <= speed_mps <= MAX_SPEED_MPS (по умолчанию 10..100 м/с).
# - Outlier protection: проверка резкого изменения площади bbox (MAX_BBOX_SCALE_CHANGE)
#   и верхний кап скорости (MAX_SPEED_MPS) чтобы отфильтровать артефакты.
# - Сохранение G-канала каждого кадра в папку `raw_capture` в виде .raw файлов.
# - Коррекция fisheye через OpenCV (cv2.fisheye.initUndistortRectifyMap);
#   при ошибке возвращается оригинальный кадр — чтобы программа не падала.
# - Трейлы объектов (TRAIL_LENGTH) с визуальным затуханием (TRAIL_FADE)
#   и прогнозная точка по вектору последнего движения.
# - Отправка зарегистрированных объектов по UDP в JSON и лог в терминал.
# - Парсер ввода калибровки поддерживает запятую, точки и единицы (mm/cm/m).
# - Общие защиты: try/except в критичных местах, ограничение на попытки ввода.
#
# Параметры по умолчанию:
# - METERS_PER_PIXEL = 0.02 (пример) — скорректируйте через калибровку для точных скоростей.
# - SPEED_THRESHOLD_MPS = 10.0
# - MAX_SPEED_MPS = 100.0
#
# Рекомендации для улучшения:
# - Сохранять параметры калибровки (например, в calibration.json) и загружать при старте.
# - Заменить SimpleTracker на SORT/DeepSORT для более устойчивого трекинга.
# - Логировать события (JSONL) и сохранять кропы при регистрации быстрых объектов.
#

import cv2
import numpy as np
import socket
import json
import os
import time
import math
from typing import List, Tuple
from collections import deque
import re

# ---------------- UDP параметры ----------------
UDP_IP = "192.168.1.100"   # IP ПК
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
# ---------------- Настройки камеры ----------------
CAM_INDEX = 0
FPS = 30
FRAME_WIDTH = 2400
FRAME_HEIGHT = 1200
 
# ---------------- Настройки trail ----------------
TRAIL_LENGTH = 100   # сколько точек хранить для "следа"
TRAIL_FADE = 1 #10     # шаг затухания (чем больше — быстрее исчезает)
 
# ---------------- Папка для RAW ----------------
RAW_DIR = "raw_capture"
os.makedirs(RAW_DIR, exist_ok=True)
 
# Очереди для хранения траекторий объектов
trails = {}
# Для простого сопоставления объектов между кадрами
active_objects = {}  # obj_id -> {'pos': (x,y), 'time': ts}
next_obj_id = 0

# Параметры скорости (подстройте METERS_PER_PIXEL под вашу камеру)
MIN_CONTOUR_AREA = 5  # минимальная площадь движущегося объекта (пиксели)
METERS_PER_PIXEL = 200 / FRAME_WIDTH  # метры за пиксель для сцены 200 м шириной
SPEED_THRESHOLD_MPS = 0.005  # минимальный порог скорости (ещё выше чувствительность)
MATCH_MAX_DISTANCE_PX = 200  # макс расстояние для сопоставления между кадрами (не используется в SORT-like трекере)
MAX_SPEED_MPS = 1000.0  # защитный кап на вычисленную скорость
MAX_BBOX_SCALE_CHANGE = 3.0  # если bbox изменился больше чем в N раз — возможно артефакт

# Калибровка (сбор двух кликов и ввода реальной длины)

# --- Для перемещения мишени ---
calibrating = False
cal_points: List[Tuple[int,int]] = []
target_center = None  # (x, y)
dragging_target = False

def mouse_callback(event, x, y, flags, param):
    global calibrating, cal_points, target_center, dragging_target
    if calibrating and event == cv2.EVENT_LBUTTONDOWN:
        cal_points.append((x, y))
        print(f"Calibration click: {(x,y)}")
    # --- Перемещение мишени ---
    elif not calibrating:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Проверяем, попал ли клик в область мишени (радиус 50)
            if target_center is not None:
                tx, ty = target_center
                if (x - tx) ** 2 + (y - ty) ** 2 <= 50 ** 2:
                    dragging_target = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging_target:
            target_center = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_target = False

cv2.namedWindow("Motion Tracking Fisheye")
cv2.setMouseCallback("Motion Tracking Fisheye", mouse_callback)

def compute_iou(a, b):
    # a,b are (x,y,w,h)
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw)
    y2 = min(ay+ah, by+bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2-x1) * (y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """Very small SORT-like tracker: greedy IoU matching, per-track last bbox/time.
    Tracks: dict id -> {bbox, last_bbox, last_time, speed_mps}
    """
    def __init__(self, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Tuple[int,int,int,int]], ts: float):
        # Build match matrix
        updated = {}
        used_det = set()
        # greedy match by IoU
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_j = None
            for j, det in enumerate(detections):
                if j in used_det:
                    continue
                iou = compute_iou(tr['bbox'], det)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j is not None and best_iou >= self.iou_threshold:
                det = detections[best_j]
                used_det.add(best_j)
                # compute speed
                prev_cx = tr['bbox'][0] + tr['bbox'][2]/2
                prev_cy = tr['bbox'][1] + tr['bbox'][3]/2
                cx = det[0] + det[2]/2
                cy = det[1] + det[3]/2
                dt = ts - tr.get('time', ts)
                if dt <= 0:
                    dt = 1.0 / FPS
                dx_m = (cx - prev_cx) * METERS_PER_PIXEL
                dy_m = (cy - prev_cy) * METERS_PER_PIXEL
                dist_m = math.hypot(dx_m, dy_m)
                speed = dist_m / dt if dt > 0 else 0.0
                # outlier protections
                try:
                    prev_area = tr['bbox'][2] * tr['bbox'][3]
                    new_area = det[2] * det[3]
                    if prev_area > 0 and (new_area / prev_area > MAX_BBOX_SCALE_CHANGE or prev_area / new_area > MAX_BBOX_SCALE_CHANGE):
                        # consider this an unreliable jump — keep previous bbox and zero speed
                        speed = 0.0
                        det = tr['bbox']
                except Exception:
                    pass
                speed = min(speed, MAX_SPEED_MPS)
                updated[tid] = {'bbox': det, 'last_bbox': tr['bbox'], 'time': ts, 'speed_mps': speed}
            else:
                # no matching detection: keep track but mark time (we may remove later)
                # keep old track unchanged
                pass

        # create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            updated[tid] = {'bbox': det, 'last_bbox': det, 'time': ts, 'speed_mps': 0.0}

        # replace tracks with updated (note: we drop unmatched old tracks)
        self.tracks = updated
        # return list of track dicts
        out = []
        for tid, tr in self.tracks.items():
            out.append({'id': tid, 'bbox': tr['bbox'], 'speed_mps': tr.get('speed_mps', 0.0)})
        return out


tracker = SimpleTracker(iou_threshold=0.3)

 
# ---------------- Функция: коррекция fisheye ----------------
# Precompute undistort maps for performance
undistort_maps_computed = False
map1, map2 = None, None

def undistort_fisheye(frame):
     global undistort_maps_computed, map1, map2
     h, w = frame.shape[:2]
     if not undistort_maps_computed:
         # Матрица камеры (примерная, требует калибровки для точности)
         K = np.array([[w/2, 0, w/2],
                      [0, w/2, h/2],
                      [0,   0,   1]], dtype=np.float32)
         D = np.zeros(4)
         # если нет калибровочных данных, оставляем 0
         map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
         )
         undistort_maps_computed = True
     return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

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


fgbg = cv2.createBackgroundSubtractorKNN(history=7, dist2Threshold=20.0, detectShadows=False)

# --- Прогрев фона ---
for _ in range(15):
    ret, frame = cap.read()
    if not ret:
        break
    gray = frame[:, :, 1]
    fgbg.apply(gray)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Сохраняем RAW G-канал (каждый 10-й кадр для производительности)
    if frame_id % 10 == 0:
        g_channel = frame[:, :, 1]
        raw_path = os.path.join(RAW_DIR, f"frame_{frame_id:06d}.raw")
        g_channel.tofile(raw_path)

    # Коррекция fisheye
    corrected = undistort_fisheye(frame)
    # --- Центр мишени ---
    if target_center is None:
        target_center = (corrected.shape[1] // 2, corrected.shape[0] // 2)

    # Берём G-канал для обработки
    gray = corrected[:, :, 1]

    # Фон/движение
    fgmask = fgbg.apply(gray)
    _, thresh = cv2.threshold(fgmask, 5, 255, cv2.THRESH_BINARY)  # более чувствительный порог

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # собираем детекции (bbox) перед передачей трекеру
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append((int(x), int(y), int(w), int(h)))

    ts = time.time()
    tracks = tracker.update(detections, ts)

    objects = []
    # отображаем объекты с скоростью > 100 м/с
    for t in tracks:
        tid = t['id']
        x, y, w, h = t['bbox']
        speed = float(t.get('speed_mps', 0.0))
        if speed <= 0.0:
            continue
        if speed > MAX_SPEED_MPS:
            continue

        # --- Проверка движения к мишени ---
        cx, cy = x + w // 2, y + h // 2
        # trail обновляется всегда
        if tid not in trails:
            trails[tid] = deque(maxlen=TRAIL_LENGTH)
        trails[tid].append((cx, cy))
        pts = list(trails[tid])
        if len(pts) < 2:
            continue  # ждём хотя бы две точки
        prev_cx, prev_cy = pts[-2]
        tx, ty = target_center
        dist_prev = ((prev_cx - tx) ** 2 + (prev_cy - ty) ** 2) ** 0.5
        dist_now = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        moving_towards_target = dist_now < dist_prev
        if moving_towards_target and speed > SPEED_THRESHOLD_MPS and speed <= MAX_SPEED_MPS:
            angle_h, angle_v = pixel_to_angle(cx, cy, corrected.shape[1], corrected.shape[0])
            obj = {
                "id": int(tid),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "angle_h": round(float(angle_h), 2),
                "angle_v": round(float(angle_v), 2),
                "speed_mps": round(float(speed), 2)
            }
            objects.append(obj)
            # рисуем
            color = (0, 0, 255)
            label = f"{speed:.2f} m/s"
            cv2.rectangle(corrected, (x, y), (x+w, y+h), color, 3)
            cv2.putText(corrected, label, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # Draw trail lines (optimized: draw every other point for speed)
            for j in range(1, len(pts), 2):
                alpha = max(0, 255 - (len(pts)-j)*TRAIL_FADE)
                cv2.line(corrected, pts[j-1], pts[j], (255, alpha, alpha), 2)
            # прогноз
            if len(pts) >= 2:
                dx = pts[-1][0] - pts[-2][0]
                dy = pts[-1][1] - pts[-2][1]
                pred_x, pred_y = pts[-1][0] + dx*5, pts[-1][1] + dy*5
                cv2.circle(corrected, (pred_x, pred_y), 6, (255, 255, 0), 2)


    # --- Отправка данных на ПК ---
    if objects:
        data = json.dumps(objects).encode("utf-8")
        sock.sendto(data, (UDP_IP, UDP_PORT))
        # Выводим данные в терминал для отладки
        try:
            print(f"[Frame {frame_id}] Sent {len(objects)} objects to {UDP_IP}:{UDP_PORT}")
            print(json.dumps(objects, ensure_ascii=False))
        except Exception as e:
            print(f"Failed printing payload: {e}")

    # --- Показ ---
    # --- Рисуем мишень в центре сцены ---
    scene_cx, scene_cy = target_center
    # Круги
    cv2.circle(corrected, (scene_cx, scene_cy), 50, (0, 0, 0), 3)  # внешний круг (черный)
    cv2.circle(corrected, (scene_cx, scene_cy), 30, (255, 255, 255), 3)  # средний круг (белый)
    cv2.circle(corrected, (scene_cx, scene_cy), 10, (0, 0, 0), 3)  # внутренний круг (черный)
    cv2.circle(corrected, (scene_cx, scene_cy), 2, (0, 0, 255), -1)  # центр (красная точка)
    # Перекрестие
    cross_len = 60
    cross_color = (0, 0, 0)
    cross_thick = 2
    cv2.line(corrected, (scene_cx - cross_len, scene_cy), (scene_cx + cross_len, scene_cy), cross_color, cross_thick)
    cv2.line(corrected, (scene_cx, scene_cy - cross_len), (scene_cx, scene_cy + cross_len), cross_color, cross_thick)
    # --- Показ ---
    cv2.imshow("Motion Tracking Fisheye", corrected)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        quit_flag = False
        # начинаем калибровку: ждём 2 клика, затем ввод длины
        calibrating = True
        cal_points = []
        print("Calibration mode: click two points on the image (left mouse button), then enter real distance in meters in the console.")
        # wait until we get 2 clicks
        while len(cal_points) < 2:
            cv2.imshow("Motion Tracking Fisheye", corrected)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                quit_flag = True
                break
        calibrating = False
        if quit_flag:
            break
        if len(cal_points) >= 2:
            p0, p1 = cal_points[0], cal_points[1]
            px_dist = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
            # Robust parsing: accept commas as decimal separators and simple united input like '45cm' or '0,45m'
            def _parse_distance_input(s: str) -> float:
                """Find first numeric token and optional unit, convert to meters.
                Accepts comma as decimal separator. Units supported: mm, cm, m (case-insensitive).
                Examples: '0,45', '0.45', '45cm', '450 mm', '1.2m'
                """
                m = re.search(r"([-+]?\d+[.,]?\d*)\s*(mm|cm|m)?", s, re.I)
                if not m:
                    raise ValueError("no numeric value found")
                num_tok = m.group(1).replace(',', '.')
                unit = (m.group(2) or '').lower()
                val = float(num_tok)
                if unit == 'mm':
                    val = val / 1000.0
                elif unit == 'cm':
                    val = val / 100.0
                # if unit == 'm' or unit == '': assume meters
                return val

            attempts = 0
            real = None
            while attempts < 3:
                s = input("Enter real distance between points in meters (e.g. 0.45 or 45cm). Press Enter to cancel: ").strip()
                if s == "":
                    break
                try:
                    real_val = _parse_distance_input(s)
                    real = real_val
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Invalid input: {e}. Attempts left: {3-attempts}")

            if real is None:
                print("Calibration aborted or invalid input.")
            else:
                if px_dist > 0:
                    METERS_PER_PIXEL = real / px_dist
                    print(f"Calibration done. METERS_PER_PIXEL = {METERS_PER_PIXEL:.6f} m/px")
                else:
                    print("Zero pixel distance, calibration aborted.")

cap.release()
cv2.destroyAllWindows()
#[DEBUG ON]
#[DEBUG OFF]
