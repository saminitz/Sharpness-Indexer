# Sharpness Indexer

**Sharpness Indexer** was born out of a need to save time after motorsport photography sessions. When capturing panning shots—where you follow a moving car and take bursts of images—you often end up with hundreds of pictures, many of which are blurry or imperfect due to motion. Reviewing each photo manually to find the sharp ones, where the car is properly captured, can be extremely time-consuming. This script was created to automate that process.

This tool is written in Python and is designed to analyze the sharpness of cars in high-resolution images, even when the background is blurry—as is typical in panning shots. It uses YOLOv8 to detect cars and Laplacian variance to evaluate image sharpness. Once processed, each image is renamed with a prepended sharpness value, and all images are saved to a single output folder where you can then simply sort the files by filename descending and have the sharpest pictures at the top.

---

## 🚀 Features
   
- ✅ Uses YOLOv8 to detect cars in each image
- ✅ Measures sharpness (Laplacian variance) of each detected car
- ✅ Prepends sharpness score to filename (e.g., `S134__IMG001.jpg`)
- ✅ Saves all renamed images into one single output folder
- ✅ Supports multiple image formats (`.jpg`, `.jpeg`, `.png`)
- ✅ Runs on GPU if available (via PyTorch + CUDA)

---

## 🖼️ Example

| Original filename | Detected sharpness | Output filename           |
|-------------------|--------------------|---------------------------|
| `IMG_001.jpg`     | `134.23`           | `S134__IMG_001.jpg`      |
| `trackshot.png`   | `41.87`            | `S041__trackshot.png`     |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sharpness-indexer.git
   cd sharpness-indexer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv yolo
   ```

3. **Activate the virtual environment**

   - On **Windows**:
     ```bash
     yolo\Scripts\activate
     ```

   - On **macOS/Linux**:
     ```bash
     source yolo/bin/activate
     ```

4. **Install dependencies**
   > Python 3.9+ and pip required
   ```bash
   pip install -r yolo-requirements.txt
   ```

5. **Optional: Check for GPU availability**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

---

## 📁 Folder Structure

```
sharpness-indexer/
├── input/                   # Put your input images here
├── output/                  # Processed images will appear here
├── README.md                # This file
├── sort-yolo.py             # Main script
├── yolo-requirements.txt    # Requirements for `sort-yolo.py`
```

---

## 🧠 How It Works

1. You choose a YOLOv8 model (`n`, `s`, `m`, `l`, `x`) – defaults to `yolov8x.pt`.
2. The script loads each image from the `input/` folder.
3. Cars are detected using the YOLO model.
4. Sharpness is calculated for each detected car using the Laplacian operator.
5. The image is renamed using the **highest sharpness score** found in any car.
6. The renamed image is copied or moved to `output/` depending on your choice.

---

## ⚙️ Configuration

| Setting              | Description                                         | Default        |
|----------------------|-----------------------------------------------------|----------------|
| YOLOv8 model         | Detection backbone (`n`, `s`, `m`, `l`, `x`)        | `yolov8x.pt`   |
| Sharpness metric     | Laplacian variance (computed via OpenCV)            | -              |
| Output format        | `S{SHARPNESS}__{ORIGINALNAME}.{EXT}`                | e.g. `S098__IMG1.jpg` |
| Input directory      | Folder containing raw images                        | `input/`       |
| Output directory     | Folder for renamed files                            | `output/`      |

---

## 📌 Notes

- The tool uses the COCO class index `2` to detect cars.
- If no car is found in the image, the sharpness score is set to `0.00`.
- It does **not** modify the original images; it **copies and renames** them.

---

## 🛠️ Requirements

- Python 3.9+
- PyTorch (with CUDA for GPU support)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV

---

## ✅ Example Run

```bash
python sort-yolo.py
```

```
Choose YOLOv8 model (n = nano, s = small, m = medium, l = large, x = xlarge)
Model (default: x): 
[INFO] Using model: yolov8x.pt
[OK] IMG_001.jpg -> S134__IMG_001.jpg
[OK] IMG_002.jpg -> S023__IMG_002.jpg
```

---

## 📖 License

MIT License © 2025

---

## 💬 Questions?

Feel free to open an [issue](https://github.com/saminitz/Sharpness-Indexer/issues) if you have suggestions or questions!

---

