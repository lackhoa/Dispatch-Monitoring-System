# video_app.py

import os
import cv2
import csv
import time
import json
import torch
import queue
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, messagebox
from threading import Thread
from torchvision import transforms
from ultralytics import YOLO
import timm
import subprocess
from retrain.feedback_gui import open_feedback_gui

# ------------------- Config -------------------
VIDEO_PATH = '1473_CH05_20250501133703_154216.mp4'
FRAME_SKIP = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_HALF = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
FIXED_ROI = [925, 20, 1350, 275]
USE_FIXED_ROI = False

# ------------------- Class Names -------------------
CLASS_NAMES = ['dish_empty', 'dish_kakigori', 'dish_not_empty', 'tray_empty', 'tray_kakigori', 'tray_not_empty']

# ------------------- Load Models -------------------
print("[INFO] Loading YOLOv8 detection model...")
DETECT_MODEL = YOLO('det_models/yolov8n_best.pt').to(DEVICE)
if USE_HALF:
    DETECT_MODEL.model.half()

print("[INFO] Loading classification model...")
DEFAULT_MODEL_PATH = 'classify_models/efficientnet_b0_best.pth'
RETRAINED_DIR = 'retrain/retrained_models'
os.makedirs(RETRAINED_DIR, exist_ok=True)
model_paths = sorted([os.path.join(RETRAINED_DIR, f) for f in os.listdir(RETRAINED_DIR) if f.endswith(".pth")], key=os.path.getctime)
model_path = model_paths[-1] if model_paths else DEFAULT_MODEL_PATH
current_model_index = len(model_paths) - 1 if model_paths else 0

CLASSIFY_MODEL = timm.create_model("efficientnet_b0", pretrained=False)
CLASSIFY_MODEL.classifier = torch.nn.Linear(CLASSIFY_MODEL.classifier.in_features, len(CLASS_NAMES))
CLASSIFY_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
if USE_HALF:
    CLASSIFY_MODEL = CLASSIFY_MODEL.half()
CLASSIFY_MODEL.to(DEVICE).eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------- Queues -------------------
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)

# ------------------- Shared States -------------------
last_crop_images, last_preds, last_track_ids = [], [], []
log_lines = []

# ------------------- Helper Functions -------------------
def get_label_counts(feedback_dir):
    counts = {}
    for label in os.listdir(feedback_dir):
        label_path = os.path.join(feedback_dir, label)
        if os.path.isdir(label_path):
            counts[label] = len([f for f in os.listdir(label_path) if f.endswith('.jpg')])
    return counts

def should_trigger_retrain(feedback_dir, record_path='retrain/last_counts.json', threshold=20):
    current_counts = get_label_counts(feedback_dir)
    if not os.path.exists(record_path):
        with open(record_path, 'w') as f:
            json.dump(current_counts, f)
        return False

    with open(record_path, 'r') as f:
        last_counts = json.load(f)

    all_labels = set(current_counts) & set(last_counts)
    enough_new_data = all(
        current_counts[label] - last_counts.get(label, 0) >= threshold for label in all_labels
    )

    if enough_new_data:
        with open(record_path, 'w') as f:
            json.dump(current_counts, f)
        return True
    return False

def run_retrain_and_log():
    global log_lines, model_paths, current_model_index
    process = subprocess.Popen(['python', 'retrain/retrain.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_lines.clear()
    for line in process.stdout:
        log_lines.append(line.strip())
        if len(log_lines) > 10:
            log_lines.pop(0)
    process.wait()
    model_paths = sorted([os.path.join(RETRAINED_DIR, f) for f in os.listdir(RETRAINED_DIR) if f.endswith('.pth')], key=os.path.getctime)
    current_model_index = len(model_paths) - 1

# ------------------- Reader Thread -------------------
def read_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % FRAME_SKIP == 0 and not frame_queue.full():
            frame_queue.put(frame.copy())
        frame_id += 1
    cap.release()
    frame_queue.put(None)

# ------------------- Processing Thread -------------------
def process_frames():
    global last_crop_images, last_preds, last_track_ids

    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)
            break

        display_frame = frame.copy()
        if not USE_FIXED_ROI:
            result_queue.put(display_frame)
            continue

        rx1, ry1, rx2, ry2 = FIXED_ROI
        roi_frame = frame[ry1:ry2, rx1:rx2]
        if roi_frame.size == 0:
            result_queue.put(display_frame)
            continue

        results = DETECT_MODEL.track(roi_frame, conf=0.6, iou=0.45, persist=True, verbose=False)[0]
        boxes = results.boxes
        track_ids = boxes.id.int().tolist() if boxes.id is not None else [-1] * len(boxes)

        crops, crop_images, positions = [], [], []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1_full, y1_full, x2_full, y2_full = x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1
            crop = frame[y1_full:y2_full, x1_full:x2_full]
            if crop.size == 0:
                continue
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = TRANSFORM(img_pil).half() if USE_HALF else TRANSFORM(img_pil)
            crops.append(input_tensor)
            crop_images.append(img_pil)
            positions.append((x1_full, y1_full, x2_full, y2_full, float(box.conf[0]), int(box.cls[0])))

        preds = []
        if crops:
            with torch.no_grad():
                batch_tensor = torch.stack(crops).to(DEVICE)
                outputs = CLASSIFY_MODEL(batch_tensor)
                preds = outputs.argmax(dim=1).tolist()

            for i, pred_cls in enumerate(preds):
                x1, y1, x2, y2, conf, _ = positions[i]
                label = f"ID:{track_ids[i]} {CLASS_NAMES[pred_cls]} ({conf:.2f})"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        last_crop_images, last_preds, last_track_ids = crop_images.copy(), preds.copy(), track_ids.copy()
        result_queue.put(display_frame)


# ------------------- Main App Loop -------------------
Thread(target=read_frames).start()
Thread(target=process_frames).start()

while True:
    frame = result_queue.get()
    if frame is None:
        break

    if should_trigger_retrain("retrain/feedback_data"):
        Thread(target=run_retrain_and_log).start()

    cv2.putText(frame, "Press r: Toggle ROI", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Press f: Send Feedback", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Press q: Quit", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Press n: Use Newer Model", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "Press p: Use Previous Model", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    if USE_FIXED_ROI:
        x1, y1, x2, y2 = FIXED_ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    log_y = frame.shape[0] - 100
    for i, line in enumerate(log_lines[-5:]):
        y = log_y + i * 15
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow('Detection + Classification', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('r'):
        USE_FIXED_ROI = not USE_FIXED_ROI

    elif key == ord('f'):
        if not last_crop_images or not last_preds:
            root = tk.Tk()
            root.withdraw()  
            tk.messagebox.showwarning("No objects detected.")
            root.destroy()
        else:
            open_feedback_gui(last_crop_images, last_preds, last_track_ids, CLASS_NAMES)

    elif key == ord('n'):
        if current_model_index < len(model_paths) - 1:
            current_model_index += 1
            model_path = model_paths[current_model_index]
            CLASSIFY_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
            CLASSIFY_MODEL.eval()
            print(f"[INFO] Switched to model: {model_path}")
        else:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("This is lastest model.")
            root.destroy()

    elif key == ord('p'):
        if current_model_index > 0:
            current_model_index -= 1
            model_path = model_paths[current_model_index]
            CLASSIFY_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
            CLASSIFY_MODEL.eval()
            print(f"[INFO] Reverted to model: {model_path}")
        else:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("This is oldest model. ")
        root.destroy()

cv2.destroyAllWindows()
