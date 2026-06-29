# ============================================================
# POTHOLE DETECTION - GOOGLE COLAB TRAINING SCRIPT
# Run each section in a separate Colab cell (marked with # CELL)
# Runtime -> Change runtime type -> GPU (T4)
# ============================================================

# CELL 1: Install dependencies
# ----------------------------
# !pip install ultralytics roboflow -q

# CELL 2: Import & check GPU
# --------------------------
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None - switch to GPU runtime!")

# CELL 3: Download pothole dataset from Roboflow (free, no login needed)
# -----------------------------------------------------------------------
# This dataset has ~665 images of potholes, pre-annotated in YOLO format
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_FREE_API_KEY")  # Get free key at roboflow.com
project = rf.workspace("pothole-detection-lfkrg").project("pothole-detection-dataset-y8kyk")
dataset = project.version(4).download("yolov8")

# --- Alternative: use this public dataset URL instead (no API key needed) ---
# !curl -L "https://universe.roboflow.com/ds/..." > dataset.zip
# Just go to: https://universe.roboflow.com/pothole-detection-lfkrg/pothole-detection-dataset-y8kyk
# Click "Download Dataset" -> YOLOv8 format -> get your free download link


# CELL 4: Train YOLOv8 model
# --------------------------
from ultralytics import YOLO

# Load pretrained YOLOv8 small model (good balance of speed vs accuracy)
model = YOLO("yolov8s.pt")

# Train on the pothole dataset
results = model.train(
    data=f"{dataset.location}/data.yaml",  # path to dataset config
    epochs=50,           # 50 is enough for good results on Colab free tier
    imgsz=640,           # image size (640 is YOLO default)
    batch=16,            # fits in Colab T4 GPU memory
    name="pothole_v1",   # saved under runs/detect/pothole_v1/
    patience=15,         # stop early if no improvement for 15 epochs
    save=True,
    plots=True,          # saves loss curves, confusion matrix etc.
    device=0,            # GPU
    project="/content/runs",
    verbose=True
)

print("Training complete!")
print(f"Best model saved at: /content/runs/detect/pothole_v1/weights/best.pt")


# CELL 5: Evaluate the model
# --------------------------
# Validate on the test set
metrics = model.val()

print(f"\n=== Model Performance ===")
print(f"mAP50:     {metrics.box.map50:.3f}   (detection accuracy at IoU=0.5)")
print(f"mAP50-95:  {metrics.box.map:.3f}   (stricter accuracy)")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall:    {metrics.box.mr:.3f}")


# CELL 6: Test on a sample image
# --------------------------------
import cv2
from google.colab.patches import cv2_imshow

# Test on one image from the test set
test_img_path = f"{dataset.location}/test/images"
import os
test_images = os.listdir(test_img_path)
sample = os.path.join(test_img_path, test_images[0])

results = model.predict(source=sample, conf=0.25, save=True)
print(f"Detected {len(results[0].boxes)} potholes in sample image")

# Show annotated image
cv2_imshow(cv2.imread(f"/content/runs/detect/predict/{test_images[0]}"))


# CELL 7: Download the trained model weights
# -------------------------------------------
from google.colab import files

# Download best weights to your computer
files.download("/content/runs/detect/pothole_v1/weights/best.pt")

print("""
=== DONE! ===
You now have 'best.pt' downloaded.
Put it in your project folder alongside detect.py
""")
