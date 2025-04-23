import cv2
import numpy as np
import os
import shutil
import platform
from ultralytics import YOLO
from pathlib import Path

# === Eingabe: Modellwahl mit Standard ===
print("Wähle YOLOv8-Modell (n = nano, s = small, m = medium, l = large, x = xlarge)")
modellwahl = input("Modell (Standard: s): ").strip().lower() or "s"
modellname = f"yolov8{modellwahl}.pt"
print(f"[INFO] Benutze Modell: {modellname}")

# === Pfade ===
input_dir = Path("input")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# === Modell laden ===
model = YOLO(modellname)

# === Konfiguration ===
SCHAERFE_GRENZE = 100.0  # nur noch informativ
image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))

# === Schärfewert ermitteln ===
def schaerfe_messung(image_path):
    img = cv2.imread(str(image_path))
    results = model(img)

    # Filtere nur Autos (COCO-Klasse 2)
    autos = [box for box in results[0].boxes if int(box.cls[0]) == 2]
    if not autos:
        print(f"[INFO] Kein Auto erkannt in {image_path.name}")
        return 0.0

    max_schaerfe = 0.0
    for box in autos:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        car_crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        max_schaerfe = max(max_schaerfe, lap_var)

    return max_schaerfe

# === Hauptlauf ===
for image_path in image_files:
    scharfwert = schaerfe_messung(image_path)
    new_name = f"S{scharfwert:.0f}__{image_path.name}"
    new_path = output_dir / new_name
    shutil.copy(image_path, new_path)
    print(f"[OK] {image_path.name} -> {new_name}")

# === Abschluss ===
print("\nAlle Bilder wurden mit Schärfewert umbenannt und nach 'output/' kopiert.")
