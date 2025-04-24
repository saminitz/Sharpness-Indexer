import cv2
import numpy as np
import os
import shutil
import platform
from ultralytics import YOLO
from pathlib import Path

# === Model selection with default ===
print("Choose YOLOv8 model: (n = nano, s = small, m = medium, l = large, x = xlarge)")
model_choice = input("Model (default: x): ").strip().lower() or "x"
model_name = f"yolov8{model_choice}.pt"
print(f"[INFO] Using model: {model_name}")

# === Paths ===
input_dir = Path("input")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# === Load model ===
model = YOLO(model_name)

# === Image files ===
image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))

# === Function: Measure sharpness of detected car in image ===
def measure_sharpness(image_path):
    img = cv2.imread(str(image_path))
    results = model(img)

    # Class 2 = car in COCO dataset
    cars = [box for box in results[0].boxes if int(box.cls[0]) == 2]
    if not cars:
        print(f"[INFO] No car detected in {image_path.name}")
        return 0.0

    max_sharpness = 0.0
    for box in cars:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        car_crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        max_sharpness = max(max_sharpness, sharpness)

    return max_sharpness

# === Process images ===
processed_paths = []
for image_path in image_files:
    sharpness_value = measure_sharpness(image_path)
    sharpness_int = int(round(sharpness_value))
    new_filename = f"S{sharpness_int:03d}__{image_path.name}"
    new_path = output_dir / new_filename
    processed_paths.append((image_path, new_path))
    print(f"[OK] {image_path.name} -> {new_filename}")

# === Final action choice ===
print("\nWhat should happen to the processed images?")
print("1 = Copy to output folder")
print("2 = Move to output folder (remove from input folder)")
final_choice = input("Selection (default: 1): ").strip() or "1"

for src, dst in processed_paths:
    if final_choice == "2":
        shutil.move(src, dst)
    else:
        shutil.copy(src, dst)

print("\n[DONE] Images have been processed and saved in 'output/' folder.")
