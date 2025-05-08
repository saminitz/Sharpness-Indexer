import cv2
import re
import shutil
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# === Sharpness Methods ===
def laplacian_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad_sharpness(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return (gx**2 + gy**2).mean()

def brenner_sharpness(gray):
    return np.sum((gray[2:, :] - gray[:-2, :])**2)

def fft_sharpness(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return np.var(magnitude_spectrum)

def gradient_energy(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_magnitude)

sharpness_methods = {
    "Laplacian": laplacian_sharpness,
    "Tenengrad": tenengrad_sharpness,
    "Brenner": brenner_sharpness,
    "FFT": fft_sharpness,
    "Gradient": gradient_energy
}

# === Model selection ===
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
image_files = [file for file in input_dir.iterdir() if re.search(r'\.(jpg|jpeg|png)$', file.suffix, re.IGNORECASE)]

# === Data structure: {method: [(sharpness, image_path)]} ===
results_by_method = {key: [] for key in sharpness_methods}

# === Process each image ===
for image_path in image_files:
    img = cv2.imread(str(image_path))
    results = model(img)
    cars = [box for box in results[0].boxes if int(box.cls[0]) == 2]

    if not cars:
        print(f"[INFO] No car detected in {image_path.name}")
        continue

    # Take the sharpest car per method
    method_scores = {method: 0.0 for method in sharpness_methods}
    for box in cars:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        car_crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        for name, func in sharpness_methods.items():
            score = func(gray)
            method_scores[name] = max(method_scores[name], score)

    # Store sharpness values
    for method_name, score in method_scores.items():
        results_by_method[method_name].append((score, image_path))

# === Sort and rename ===
for method_name, data in results_by_method.items():
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
    for idx, (_, src_path) in enumerate(sorted_data, 1):
        new_name = f"{method_name}__{idx:03d}__{src_path.name}"
        dst_path = output_dir / new_name
        shutil.copy(src_path, dst_path)

print("\n[DONE] Images processed and sorted by multiple sharpness methods into 'output/' folder.")
