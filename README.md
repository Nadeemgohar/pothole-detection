<div align="center">

# 🚧 Pothole Detection System

**YOLOv8-powered road pothole detector — detects, classifies, and reports potholes from images, videos, or live webcam.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 What It Does

For every pothole found in a road image or video, the system reports:

| Output | Description |
|--------|-------------|
| 📍 Location | Bounding box (x, y, width, height) |
| 📏 Size | Pixel area + severity label (Small / Medium / Large) |
| 🎯 Confidence | How sure the model is (0–1) |
| 🛣️ Road name | From filename or GPS |
| 🕐 Timestamp | When detected |
| 🖼️ Image | Saved with colored boxes drawn |

---

## 🗂️ Project Structure

```
pothole-detection/
├── detect.py             ← Main detection script (run locally)
├── colab_train.py        ← Training script (run on Google Colab)
├── requirements.txt      ← Python dependencies
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/pothole-detection.git
cd pothole-detection
pip install -r requirements.txt
```

### 2. Train the model (Google Colab — free GPU)

1. Go to [Google Colab](https://colab.research.google.com) → Runtime → **GPU (T4)**
2. Open `colab_train.py` and paste cells one by one
3. Get a free API key at [roboflow.com](https://roboflow.com) and paste it in Cell 3
4. Run all cells (~30 min) → `best.pt` downloads automatically
5. Place `best.pt` in the project root

### 3. Run detection

```bash
# Single image
python detect.py --source road.jpg

# Folder of images
python detect.py --source images/

# Video
python detect.py --source dashcam.mp4

# Live webcam
python detect.py --source 0

# Show window + lower confidence threshold
python detect.py --source road.jpg --show --conf 0.3
```

---

## 📊 Output

**Annotated image** saved to `output/` with colored boxes:

- 🟡 Yellow = Small pothole  
- 🟠 Orange = Medium pothole  
- 🔴 Red = Large pothole

**CSV report** (`output/pothole_report.csv`):

```
id, road, timestamp, image, confidence, severity, area_pct, bbox_x1, bbox_y1, ...
1, Canal Road, 2026-06-29 18:31, road1.jpg, 0.91, Large, 6.3, 120, 85, 310, 220, ...
```

**Console summary:**

```
==================================================
  POTHOLE DETECTION SUMMARY
==================================================
  Total potholes found : 7
  Small    potholes    : 2
  Medium   potholes    : 4
  Large    potholes    : 1

  By road:
    Canal Road       : 4 pothole(s)
    Main Boulevard   : 3 pothole(s)

  Avg confidence     : 83.4%
==================================================
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n (nano) |
| Input size | 640×640 |
| Classes | 1 (pothole) |
| Training epochs | 50 (Colab GPU) |
| Expected mAP50 | 0.85–0.92 |
| Inference speed | ~45ms/image on CPU |

---

## 🗃️ Dataset

Uses the [Roboflow Pothole Dataset](https://universe.roboflow.com/pothole-detection-lfkrg/pothole-detection-dataset-y8kyk) — 665 annotated road images in YOLOv8 format (free, no login needed for download).

Alternative datasets:
- [Kaggle Pothole Dataset](https://www.kaggle.com/datasets/sovitrath/road-pothole-images-for-pothole-detection)
- [GitHub Pothole Dataset](https://github.com/XD-mu/Pothole-Detection-Dataset)

---

## 🚀 Improve Accuracy

In `colab_train.py`, change:

```python
model = YOLO("yolov8s.pt")   # upgrade: nano → small
epochs = 50                   # more epochs = better accuracy
imgsz = 640                   # full resolution
```

---

## 📍 Add Real GPS / Road Names

```python
# pip install geopy
from geopy.geocoders import Nominatim

def get_road_name(lat, lon):
    geolocator = Nominatim(user_agent="pothole_detector")
    location = geolocator.reverse(f"{lat}, {lon}")
    return location.raw['address'].get('road', 'Unknown Road')
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with <a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLOv8</a>
</div>
