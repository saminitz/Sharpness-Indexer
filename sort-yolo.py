import cv2
import numpy as np
import os
import shutil
import platform
from ultralytics import YOLO
from pathlib import Path

# === Input: Model selection with default ===
print("Choose YOLOv8 model (n = nano, s = small, m = medium, l = large, x = xlarge)")
model_choice = input("Model (default: m): ").strip().lower() or "m"
model_name = f"yolov8{model_choice}.pt"
print(f"[INFO] Using model: {model_name}")

# === Paths ===
input_dir = Path("input")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load model ===
model = YOLO(model_name)

# === Configuration ===
image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))

# === Sharpness measurement ===
def measure_sharpness(image_path):
    img = cv2.imread(str(image_path))
    results = model(img)

    # Filter only cars (COCO class 2)
    cars = [box for box in results[0].boxes if int(box.cls[0]) == 2]
    if not cars:
        print(f"[INFO] No car detected in {image_path.name}")
        return 0.0

    max_sharpness = 0.0
    for box in cars:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        car_crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        max_sharpness = max(max_sharpness, lap_var)

    return max_sharpness

# === Main loop ===
for image_path in image_files:
    sharpness_score = measure_sharpness(image_path)
    new_name = f"S{sharpness_score:03.0f}__{image_path.name}"
    new_path = output_dir / new_name
    shutil.copy(image_path, new_path)
    print(f"[OK] {image_path.name} -> {new_name}")

# === Completion ===
print("\nAll images have been renamed with sharpness score and copied to 'output/'")
