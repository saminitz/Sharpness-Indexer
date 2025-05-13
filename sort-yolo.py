import cv2
import re
import shutil
import torch
from ultralytics import YOLO
from pathlib import Path
import sys

# === Paths from arguments ===
if len(sys.argv) < 2:
    print("Usage: python sort-yolo.py <input_folder> [output_folder]")
    sys.exit(1)

input_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir
output_dir.mkdir(parents=True, exist_ok=True)

# === Model selection with default ===
print("Choose YOLOv8 model: (n = nano, s = small, m = medium, l = large, x = xlarge)")
model_choice = input("Model (default: x): ").strip().lower() or "x"
model_name = f"yolov8{model_choice}.pt"
print(f"[INFO] Using model: {model_name}")

# === File action choice ===
print("\nWhat should happen to the processed images in the end?")
print("1 = Copy to output folder")
print("2 = Move to output folder (remove from input folder)")
print("3 = Create shortcuts in output folder")
file_action_choice = input("Selection (default: 1): ").strip() or "1"

# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if GPU is available
print(f'[INFO] Device used for YOLO: {device.upper()}')
model = YOLO(model_name).to(device)  # Load the model to GPU if available

# === Image files ===
image_files = [file for file in input_dir.iterdir() if re.search(r'\.(jpg|jpeg|png)$', file.suffix, re.IGNORECASE)]

# === Function: Measure sharpness of detected objects in image ===
def measure_sharpness(image_path):
    img = cv2.imread(str(image_path))
    results = model(img)

    # Class name mapping from COCO
    class_names = model.names  # e.g., {0: 'person', 1: 'bicycle', 2: 'car', ...}

    object_boxes = [box for box in results[0].boxes]
    if not object_boxes:
        print(f"[INFO] No object detected in {image_path.name}")
        return 0.0

    max_sharpness = 0.0
    for box in object_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = class_names.get(cls_id, f"Unknown({cls_id})")
        
        object_crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(object_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"[DEBUG] Detected '{cls_name}' with sharpness: {sharpness:.2f}")
        
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

# === Move, copy, or create shortcuts ===
for src, dst in processed_paths:
    if file_action_choice == "2":
        shutil.move(src, dst)
    elif file_action_choice == "3":
        dst.symlink_to(src)
    else:
        shutil.copy(src, dst)

print("\n[DONE] Images have been processed and saved in 'output/' folder.")
