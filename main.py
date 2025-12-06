import cv2
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# --------------------------
# CONFIG
# --------------------------
video_source = "parking2.mp4"   # видео в текущей директории
target_w = 1280
target_h = 720

# Пропиши свои полигоны здесь
parking_places = [
    [(154, 154), (213, 173), (170, 296), (104, 280), (147, 156)],
    [(258, 95), (343, 137), (264, 262), (191, 236), (257, 101)],
    [(358, 104), (433, 152), (363, 283), (281, 242), (349, 111)],
    [(455, 103), (537, 129), (465, 255), (391, 223), (453, 108)],
    [(555, 120), (614, 160), (540, 277), (461, 253), (552, 120)],
    [(643, 121), (703, 168), (642, 266), (573, 226), (638, 119)],
    [(717, 129), (787, 182), (709, 301), (641, 265), (715, 134)],
    [(814, 126), (894, 177), (798, 319), (720, 298), (810, 142)],
    [(224, 466), (299, 617), (366, 590), (299, 443), (221, 458)],
    [(818, 309), (878, 337), (977, 176), (915, 126), (817, 301)],
    [(972, 132), (1056, 184), (958, 343), (875, 314), (971, 135)],
    [(757, 484), (841, 646), (902, 620), (834, 479), (761, 482)],
    [(667, 482), (745, 635), (817, 606), (750, 468), (672, 483)],
    [(576, 473), (658, 641), (726, 612), (657, 462), (577, 471)],
    [(513, 482), (570, 635), (643, 604), (643, 604), (574, 465), (508, 477)],
    [(396, 466), (487, 626), (548, 595), (484, 447), (401, 466)],
    [(307, 462), (396, 620), (460, 595), (387, 447), (317, 457)],
]

# папки
os.makedirs("shoots", exist_ok=True)
os.makedirs("records", exist_ok=True)

# Загружаем YOLO
model = YOLO("yolov8n.pt")

vehicle_classes = {"car", "truck", "bus", "motorcycle", "motorbike"}


# ----------------------------------------------------
# Функция проверки пересечения полигона и детекции
# ----------------------------------------------------
def polygon_overlap(poly, bbox):
    """
    poly: [(x, y), ...]
    bbox: (x1,y1,x2,y2)
    Возвращает True, если центр bbox внутри полигона
    или если область пересекается значительно.
    """

    poly_np = np.array(poly, np.int32)
    x1, y1, x2, y2 = bbox

    # Центр bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Проверка: центр внутри полигона
    inside = cv2.pointPolygonTest(poly_np, (cx, cy), False)
    if inside >= 0:
        return True

    # Доп. проверка маской на пересечение
    min_x = min(poly_np[:, 0].min(), x1)
    max_x = max(poly_np[:, 0].max(), x2)
    min_y = min(poly_np[:, 1].min(), y1)
    max_y = max(poly_np[:, 1].max(), y2)

    w = max_x - min_x + 1
    h = max_y - min_y + 1
    if w <= 0 or h <= 0:
        return False

    poly_shifted = poly_np - np.array([min_x, min_y])
    bbox_shifted = (x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y)

    mask_poly = np.zeros((h, w), np.uint8)
    mask_bbox = np.zeros((h, w), np.uint8)

    cv2.fillPoly(mask_poly, [poly_shifted], 255)
    cv2.rectangle(mask_bbox,
                  (bbox_shifted[0], bbox_shifted[1]),
                  (bbox_shifted[2], bbox_shifted[3]),
                  255, -1)

    intersection = cv2.bitwise_and(mask_poly, mask_bbox)
    inter_area = np.sum(intersection == 255)
    poly_area = np.sum(mask_poly == 255)

    if poly_area == 0:
        return False

    # если пересечение больше 10% площади
    return inter_area / poly_area > 0.10


# --------------------------
# Основной цикл
# --------------------------
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Ошибка: не могу открыть видео.")
    exit()

recording = False
video_writer = None
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (target_w, target_h))
    display = frame.copy()

    # YOLO детекция
    results = model(frame, conf=0.35, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.cpu())
            cls_name = model.names[cls].lower()
            if cls_name not in vehicle_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append((x1, y1, x2, y2))

            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(display, cls_name, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

    # Проверяем парковочные места
    occupied = []
    for poly in parking_places:
        is_occ = False
        for det in detections:
            if polygon_overlap(poly, det):
                is_occ = True
                break
        occupied.append(is_occ)

    # Рисуем полигоны
    overlay = display.copy()

    for i, poly in enumerate(parking_places):
        poly_np = np.array(poly, np.int32)

        if occupied[i]:
            color = (0, 0, 255)   # занято = красный
        else:
            color = (0, 255, 0)   # свободно = зелёный

        # контур
        cv2.polylines(display, [poly_np], True, (255, 255, 255), 2)

        # заливка
        cv2.fillPoly(overlay, [poly_np], color)

    cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

    # Статистика
    total = len(parking_places)
    occ = sum(occupied)
    free = total - occ

    text = f"Total: {total}  |  Free: {free}  |  Occupied: {occ}"
    cv2.putText(display, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Видео запись
    if recording:
        if video_writer is None:
            filename = f"records/rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_writer = cv2.VideoWriter(filename, fourcc, 30, (target_w, target_h))
        video_writer.write(display)
    else:
        if video_writer is not None:
            video_writer.release()
            video_writer = None

    cv2.imshow("Parking Detection", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        filename = f"shoots/shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, display)
    elif key == ord("v"):
        recording = not recording
        print("Recording:" if recording else "Stopped recording.")

cap.release()
cv2.destroyAllWindows()
