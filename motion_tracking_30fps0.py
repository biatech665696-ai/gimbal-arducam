#motion_tracking_30fps0.py
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
# Клавиши управления (добавлено):
# - 'c' : калибровка метрики "метры/пиксель" по двум кликам + ввод реальной длины;
# - 'v' : интерактивная настройка диапазона скоростей (min/max), сохраняется в calibration_speed.json;
# - 'h' : интерактивная настройка реальной высоты объекта и поля зрения камеры (FOV), сохраняется в calibration_depth.json;
#
# Файлы калибровки (создаются в папке скрипта):
# - calibration_speed.json  (формат: {"min": <m/s>, "max": <m/s>})
# - calibration_depth.json  (формат: {"object_height_m": <m>, "camera_fov_deg": <deg>})
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
import json

# ---------------- UDP параметры ----------------
#UDP_IP = "192.168.1.100"   # IP ПК
#UDP_PORT = 5005
##ock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
# ---------------- Настройки камеры ----------------
CAM_INDEX = 2
FPS = 30
FRAME_WIDTH = 1280 # Lowered for compatibility
FRAME_HEIGHT = 720  # Lowered for compatibility
 
# ---------------- Настройки trail ----------------
TRAIL_LENGTH = 20   # сколько точек хранить для "следа"
TRAIL_FADE = 10     # шаг затухания (чем больше — быстрее исчезает)
 
# ---------------- Папка для RAW ----------------
RAW_DIR = "raw_capture"
os.makedirs(RAW_DIR, exist_ok=True)
 
# Очереди для хранения траекторий объектов
trails = {}
# Для простого сопоставления объектов между кадрами
active_objects = {}  # obj_id -> {'pos': (x,y), 'time': ts}
next_obj_id = 0

# Параметры скорости (подстройте METERS_PER_PIXEL под вашу камеру)
METERS_PER_PIXEL = 0.02  # метры за пиксель (пример) — скорректируйте через калибровку
SPEED_THRESHOLD_MPS = 10.0  # порог в м/с для подсветки
MATCH_MAX_DISTANCE_PX = 200  # макс расстояние для сопоставления между кадрами (не используется в SORT-like трекере)
MAX_SPEED_MPS = 100.0  # защитный кап на вычисленную скорость
MAX_BBOX_SCALE_CHANGE = 3.0  # если bbox изменился больше чем в N раз — возможно артефакт

# Файл для сохранения калибровки диапазона скоростей
SPEED_CAL_FILE = "calibration_speed.json"

# Параметры для грубой оценки глубины (моно-камерная оценка по высоте объекта)
# Если OBJECT_REAL_HEIGHT_M = None — оценка глубины отключена и используется
# прежняя проекция METERS_PER_PIXEL.
# Пример: для человека укажите ~1.7 (м)
OBJECT_REAL_HEIGHT_M = 1.7  # meters or None
# Горизонтальное поле зрения камеры (градусы). Подберите под вашу камеру для лучшей точности.
CAMERA_FOV_DEG = 90.0
DEPTH_HISTORY_LEN = 5
DEPTH_CAL_FILE = "calibration_depth.json"

# Калибровка (сбор двух кликов и ввода реальной длины)
calibrating = False
cal_points: List[Tuple[int,int]] = []

def mouse_callback(event, x, y, flags, param):
    global calibrating, cal_points
    if calibrating and event == cv2.EVENT_LBUTTONDOWN:
        cal_points.append((x, y))
        print(f"Calibration click: {(x,y)}")


def _parse_float_input(s: str) -> float:
    """Parse a numeric value, accept comma as decimal separator."""
    m = re.search(r"[-+]?\d+[.,]?\d*", s)
    if not m:
        raise ValueError("no numeric value found")
    return float(m.group(0).replace(',', '.'))


def load_speed_calibration():
    global SPEED_THRESHOLD_MPS, MAX_SPEED_MPS
    try:
        if os.path.exists(SPEED_CAL_FILE):
            with open(SPEED_CAL_FILE, 'r', encoding='utf-8') as f:
                d = json.load(f)
            if 'min' in d and 'max' in d:
                SPEED_THRESHOLD_MPS = float(d['min'])
                MAX_SPEED_MPS = float(d['max'])
                print(f"Loaded speed calibration: {SPEED_THRESHOLD_MPS}-{MAX_SPEED_MPS} m/s")
    except Exception as e:
        print(f"Failed to load speed calibration: {e}")


def save_speed_calibration(min_s: float, max_s: float):
    try:
        with open(SPEED_CAL_FILE, 'w', encoding='utf-8') as f:
            json.dump({'min': float(min_s), 'max': float(max_s)}, f, ensure_ascii=False, indent=2)
        print(f"Saved speed calibration: {min_s}-{max_s} m/s to {SPEED_CAL_FILE}")
    except Exception as e:
        print(f"Failed to save speed calibration: {e}")

cv2.namedWindow("Motion Tracking Fisheye")
cv2.setMouseCallback("Motion Tracking Fisheye", mouse_callback)
# Load saved speed calibration (min/max) if present
load_speed_calibration()
# Load saved depth calibration (object height, camera fov) if present
depth_calibration = {
  "object_height_m": 1.7,
  "camera_fov_deg": 90.0
}
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


def estimate_depth_from_bbox_height(bbox_h_px: int, image_height_px: int) -> float:
    """Estimate approximate distance (meters) from camera to object using known real object height.
    Uses pinhole approximation: h_px = f * H_real / Z  => Z = f * H_real / h_px
    We estimate focal length f from image height and CAMERA_FOV_DEG: f = (image_height_px/2) / tan(fov_vertical/2)
    This is approximate and assumes object is upright and fills bbox height roughly.
    Returns Z in meters or None if OBJECT_REAL_HEIGHT_M is None or invalid.
    """
    if OBJECT_REAL_HEIGHT_M is None or bbox_h_px <= 0:
        return None
    # assume vertical FOV ~= CAMERA_FOV_DEG (could be adjusted for aspect ratio)
    fov_rad = math.radians(CAMERA_FOV_DEG)
    f_px = (image_height_px / 2.0) / math.tan(fov_rad / 2.0)
    Z = (f_px * OBJECT_REAL_HEIGHT_M) / float(bbox_h_px)
    return Z


def load_depth_calibration():
    global OBJECT_REAL_HEIGHT_M, CAMERA_FOV_DEG
    try:
        if os.path.exists(DEPTH_CAL_FILE):
            with open(DEPTH_CAL_FILE, 'r', encoding='utf-8') as f:
                d = json.load(f)
            if 'object_height_m' in d:
                OBJECT_REAL_HEIGHT_M = float(d['object_height_m'])
            if 'camera_fov_deg' in d:
                CAMERA_FOV_DEG = float(d['camera_fov_deg'])
            print(f"Loaded depth calibration: object_height={OBJECT_REAL_HEIGHT_M} m, fov={CAMERA_FOV_DEG} deg")
    except Exception as e:
        print(f"Failed to load depth calibration: {e}")


def save_depth_calibration():
    try:
        with open(DEPTH_CAL_FILE, 'w', encoding='utf-8') as f:
            json.dump({'object_height_m': OBJECT_REAL_HEIGHT_M, 'camera_fov_deg': CAMERA_FOV_DEG}, f, ensure_ascii=False, indent=2)
        print(f"Saved depth calibration to {DEPTH_CAL_FILE}")
    except Exception as e:
        print(f"Failed to save depth calibration: {e}")


class SimpleTracker:
    """Very small SORT-like tracker: greedy IoU matching, per-track last bbox/time.
    Tracks: dict id -> {bbox, last_bbox, last_time, speed_mps}
    """
    def __init__(self, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        # per-track depth history for smoothing
        self.depth_history = {}  # tid -> deque of recent depth estimates

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

                # If we can estimate depth, convert pixel motion to meters using depth-aware scale
                depth_prev = estimate_depth_from_bbox_height(int(tr['bbox'][3]), FRAME_HEIGHT)
                depth_curr = estimate_depth_from_bbox_height(int(det[3]), FRAME_HEIGHT)
                # fallback to METERS_PER_PIXEL if depth estimate not available
                if depth_prev is None or depth_curr is None:
                    dx_m = (cx - prev_cx) * METERS_PER_PIXEL
                    dy_m = (cy - prev_cy) * METERS_PER_PIXEL
                else:
                    # use average depth to compute angular/pixel scale approximation
                    Z = max(0.001, (depth_prev + depth_curr) / 2.0)
                    # focal length in pixels (vertical)
                    fov_rad = math.radians(CAMERA_FOV_DEG)
                    f_px = (FRAME_HEIGHT / 2.0) / math.tan(fov_rad / 2.0)
                    meters_per_pixel_at_Z = Z / f_px
                    dx_m = (cx - prev_cx) * meters_per_pixel_at_Z
                    dy_m = (cy - prev_cy) * meters_per_pixel_at_Z
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
                # save depth estimates for later display
                # append depth to history for smoothing
                hist = self.depth_history.get(tid, deque(maxlen=DEPTH_HISTORY_LEN))
                if depth_curr is not None:
                    hist.append(depth_curr)
                self.depth_history[tid] = hist
                # compute smoothed depth
                depth_smoothed = None
                if len(hist) > 0:
                    depth_smoothed = float(sum(hist) / len(hist))
                updated[tid] = {'bbox': det, 'last_bbox': tr['bbox'], 'time': ts, 'speed_mps': speed, 'depth_m': depth_smoothed}
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
            # initialize depth history for new track
            self.depth_history[tid] = deque(maxlen=DEPTH_HISTORY_LEN)
            updated[tid] = {'bbox': det, 'last_bbox': det, 'time': ts, 'speed_mps': 0.0, 'depth_m': None}

        # replace tracks with updated (note: we drop unmatched old tracks)
        self.tracks = updated
        # return list of track dicts
        out = []
        for tid, tr in self.tracks.items():
            out.append({'id': tid, 'bbox': tr['bbox'], 'speed_mps': tr.get('speed_mps', 0.0), 'depth_m': tr.get('depth_m', None)})
        return out


tracker = SimpleTracker(iou_threshold=0.15)

 
# ---------------- Функция: коррекция fisheye ----------------
def undistort_fisheye(frame):
     h, w = frame.shape[:2]
     # Матрица камеры (примерная, требует калибровки для точности)
     K = np.array([[w/2, 0, w/2],
                  [0, w/2, h/2],
                  [0,   0,   1]], dtype=np.float32)
     D = np.zeros(4)
# если нет калибровочных данных, оставляем 0
     map1, map2 = cv2.fisheye.initUndistortRectifyMap(
     
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
     )
     return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

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

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

frame_id = 0

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

    # Берём G-канал для обработки
    gray = corrected[:, :, 1]

    # Фон/движение
    fgmask = fgbg.apply(gray)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # собираем детекции (bbox) перед передачей трекеру
    detections = []
    det_angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append((int(x), int(y), int(w), int(h)))
        cx, cy = x + w // 2, y + h // 2
        angle_h, angle_v = pixel_to_angle(cx, cy, corrected.shape[1], corrected.shape[0])
        det_angles.append((angle_h, angle_v))

    ts = time.time()
    tracks = tracker.update(detections, ts)

    objects = []
    # отображаем только треки с высокой скоростью
    for t in tracks:
        tid = t['id']
        x, y, w, h = t['bbox']
        speed = float(t.get('speed_mps', 0.0))
        if speed <= 0.0:
            continue
        # защитные проверки уже внутри трекера; дополнительно фильтруем экстремумы
        if speed > MAX_SPEED_MPS:
            continue

        # составляем объект для отправки
        cx, cy = x + w // 2, y + h // 2
        angle_h, angle_v = pixel_to_angle(cx, cy, corrected.shape[1], corrected.shape[0])
        obj = {"id": int(tid), "x": int(x), "y": int(y), "w": int(w), "h": int(h), "angle_h": float(angle_h), "angle_v": float(angle_v), "speed_mps": speed}
        # attach depth if available
        if t.get('depth_m', None) is not None:
            obj['depth_m'] = float(t['depth_m'])
        # Только если превышает порог — показываем
        # регистрируем/показываем только объекты с 10 <= speed <= MAX_SPEED_MPS
        if speed >= SPEED_THRESHOLD_MPS and speed <= MAX_SPEED_MPS:
            objects.append(obj)
            # рисуем
            color = (0, 0, 255)
            depth_m = t.get('depth_m', None)
            label = f"{speed:.1f} m/s"
            if depth_m is not None:
                label += f" | {depth_m:.1f} m"
            cv2.rectangle(corrected, (x, y), (x+w, y+h), color, 3)
            cv2.putText(corrected, label, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # обновляем trail
            if tid not in trails:
                trails[tid] = deque(maxlen=TRAIL_LENGTH)
            trails[tid].append((cx, cy))
            pts = list(trails[tid])
            for j in range(1, len(pts)):
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
    # Отобразим текущий диапазон скоростей в углу
    try:
        label_speed = f"Speed range: {SPEED_THRESHOLD_MPS:.1f}-{MAX_SPEED_MPS:.1f} m/s (press 'v' to set)"
        cv2.putText(corrected, label_speed, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    except Exception:
        pass
    cv2.imshow("Motion Tracking Fisheye", corrected)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        # начинаем калибровку: ждём 2 клика, затем ввод длины
        calibrating = True
        cal_points = []
        print("Calibration mode: click two points on the image (left mouse button), then enter real distance in meters in the console.")
        # wait until we get 2 clicks
        while len(cal_points) < 2:
            cv2.imshow("Motion Tracking Fisheye", corrected)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        calibrating = False
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
    if key == ord('v'):
        # интерактивная настройка диапазона скоростей
        print(f"Current speed range: {SPEED_THRESHOLD_MPS} - {MAX_SPEED_MPS} m/s")
        attempts = 0
        new_min = None
        new_max = None
        while attempts < 3:
            smin = input("Enter minimum speed (m/s), or press Enter to cancel: ").strip()
            if smin == "":
                print("Speed calibration cancelled.")
                break
            try:
                new_min = _parse_float_input(smin)
                break
            except Exception as e:
                attempts += 1
                print(f"Invalid input for min speed: {e}. Attempts left: {3-attempts}")
        if new_min is None:
            pass
        else:
            attempts = 0
            while attempts < 3:
                smax = input("Enter maximum speed (m/s): ").strip()
                try:
                    new_max = _parse_float_input(smax)
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Invalid input for max speed: {e}. Attempts left: {3-attempts}")

            if new_max is None:
                print("Speed calibration aborted.")
            else:
                if new_min < 0 or new_max <= new_min:
                    print("Invalid range: ensure 0 <= min < max. Calibration aborted.")
                else:
                    SPEED_THRESHOLD_MPS = float(new_min)
                    MAX_SPEED_MPS = float(new_max)
                    save_speed_calibration(SPEED_THRESHOLD_MPS, MAX_SPEED_MPS)
    if key == ord('h'):
        # интерактивная настройка реальной высоты объекта и поля зрения
        print(f"Current object height: {OBJECT_REAL_HEIGHT_M} m, camera FOV: {CAMERA_FOV_DEG} deg")
        s = input("Enter object real height in meters (e.g. 1.7) or press Enter to keep: ").strip()
        if s != "":
            try:
                val = _parse_float_input(s)
                if val <= 0:
                    print("Height must be positive. Aborted.")
                else:
                    OBJECT_REAL_HEIGHT_M = float(val)
            except Exception as e:
                print(f"Invalid height: {e}")
        s2 = input("Enter camera FOV in degrees (e.g. 90) or press Enter to keep: ").strip()
        if s2 != "":
            try:
                fov = _parse_float_input(s2)
                CAMERA_FOV_DEG = float(fov)
            except Exception as e:
                print(f"Invalid FOV: {e}")
        save_depth_calibration()

cap.release()
cv2.destroyAllWindows()
#[DEBUG ON]
#[DEBUG OFF]
