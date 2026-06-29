"""
POTHOLE DETECTION SCRIPT
========================
Usage:
  python detect.py --source image.jpg
  python detect.py --source road_video.mp4
  python detect.py --source images_folder/
  python detect.py --source 0                # webcam

Output:
  - Annotated image/video saved to output/
  - CSV report with all detections
  - Console summary per image
"""

import argparse
import csv
import os
import cv2
import datetime
import math
from pathlib import Path
from ultralytics import YOLO


# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH   = "best.pt"       # path to your trained weights
CONF_THRESH  = 0.35            # minimum confidence to count a detection
IOU_THRESH   = 0.45            # overlap threshold for NMS
OUTPUT_DIR   = "output"        # folder where results are saved
REPORT_FILE  = "pothole_report.csv"

# Severity thresholds (based on bbox area as % of image area)
SEVERITY_SMALL  = 0.01   # < 1% of image  → small
SEVERITY_MEDIUM = 0.05   # 1–5%           → medium
SEVERITY_LARGE  = 0.05   # > 5%           → large


# ─── Helpers ─────────────────────────────────────────────────────────────────

def classify_severity(bbox_area_px, img_area_px):
    """Return severity label and area percentage."""
    ratio = bbox_area_px / img_area_px
    if ratio < SEVERITY_SMALL:
        severity = "Small"
    elif ratio < SEVERITY_MEDIUM:
        severity = "Medium"
    else:
        severity = "Large"
    return severity, round(ratio * 100, 2)


def draw_box(frame, box, label, color):
    """Draw bounding box + label on frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def severity_color(severity):
    """BGR color per severity level."""
    return {
        "Small":  (0, 200, 255),   # yellow
        "Medium": (0, 120, 255),   # orange
        "Large":  (0, 0, 220),     # red
    }.get(severity, (200, 200, 200))


def get_road_name(source_path):
    """
    In a real system this would do reverse geocoding from GPS.
    Here we use the filename as a proxy for road name.
    """
    return Path(source_path).stem.replace("_", " ").title()


# ─── Core detection function ──────────────────────────────────────────────────

def detect_image(model, img_path, road_name=None):
    """
    Run detection on a single image.
    Returns (annotated_frame, list_of_detection_dicts)
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [!] Could not read image: {img_path}")
        return None, []

    img_h, img_w = frame.shape[:2]
    img_area = img_h * img_w
    road = road_name or get_road_name(img_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run YOLO inference
    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False
    )

    detections = []

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf    = float(box.conf[0])
        w_px    = x2 - x1
        h_px    = y2 - y1
        area_px = w_px * h_px
        cx      = int((x1 + x2) / 2)
        cy      = int((y1 + y2) / 2)

        severity, area_pct = classify_severity(area_px, img_area)
        color = severity_color(severity)

        label = f"Pothole {severity} {conf:.0%}"
        frame = draw_box(frame, (x1, y1, x2, y2), label, color)

        # Pothole ID marker
        cv2.circle(frame, (cx, cy), 4, color, -1)

        detection = {
            "id":            i + 1,
            "road":          road,
            "timestamp":     timestamp,
            "image":         Path(img_path).name,
            "confidence":    round(conf, 3),
            "severity":      severity,
            "area_pct":      area_pct,
            "bbox_x1":       int(x1),
            "bbox_y1":       int(y1),
            "bbox_x2":       int(x2),
            "bbox_y2":       int(y2),
            "width_px":      int(w_px),
            "height_px":     int(h_px),
            "center_x":      cx,
            "center_y":      cy,
            "img_width":     img_w,
            "img_height":    img_h,
        }
        detections.append(detection)

    # Summary overlay on image
    n = len(detections)
    summary = f"Potholes detected: {n}"
    cv2.putText(frame, summary, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, summary, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 1)

    return frame, detections


def detect_video(model, video_path):
    """Run detection frame-by-frame on a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Could not open video: {video_path}")
        return []

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(OUTPUT_DIR, Path(video_path).stem + "_detected.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    road    = get_road_name(video_path)
    all_det = []
    frame_n = 0

    print(f"  Processing {total} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        # Only run detection every 3rd frame for speed
        if frame_n % 3 != 0:
            writer.write(frame)
            continue

        # Save frame temporarily for detection
        tmp = os.path.join(OUTPUT_DIR, "_tmp_frame.jpg")
        cv2.imwrite(tmp, frame)
        annotated, dets = detect_image(model, tmp, road_name=road)

        for d in dets:
            d["frame"] = frame_n
            d["time_sec"] = round(frame_n / fps, 1)
        all_det.extend(dets)

        if annotated is not None:
            writer.write(annotated)
        else:
            writer.write(frame)

        if frame_n % 30 == 0:
            print(f"    Frame {frame_n}/{total} — {len(dets)} potholes this frame")

    cap.release()
    writer.release()
    if os.path.exists(os.path.join(OUTPUT_DIR, "_tmp_frame.jpg")):
        os.remove(os.path.join(OUTPUT_DIR, "_tmp_frame.jpg"))

    print(f"  Video saved: {out_path}")
    return all_det


# ─── Report writer ────────────────────────────────────────────────────────────

def save_report(all_detections, report_path):
    """Write all detections to a CSV file."""
    if not all_detections:
        print("[i] No detections to report.")
        return

    fieldnames = list(all_detections[0].keys())
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_detections)

    print(f"\n Report saved: {report_path}")


def print_summary(all_detections):
    """Print a readable summary to the console."""
    if not all_detections:
        print("\n No potholes detected.")
        return

    print("\n" + "=" * 50)
    print("  POTHOLE DETECTION SUMMARY")
    print("=" * 50)
    print(f"  Total potholes found : {len(all_detections)}")

    # Count by severity
    from collections import Counter
    sev = Counter(d["severity"] for d in all_detections)
    for s, c in sev.items():
        print(f"  {s:8s} potholes    : {c}")

    # By road
    roads = Counter(d["road"] for d in all_detections)
    print("\n  By road:")
    for road, count in roads.most_common():
        print(f"    {road}: {count} pothole(s)")

    avg_conf = sum(d["confidence"] for d in all_detections) / len(all_detections)
    print(f"\n  Avg confidence       : {avg_conf:.1%}")
    print("=" * 50)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pothole Detection Script")
    parser.add_argument("--source", required=True,
                        help="Image file, video file, folder, or 0 for webcam")
    parser.add_argument("--model",  default=MODEL_PATH,
                        help=f"Path to model weights (default: {MODEL_PATH})")
    parser.add_argument("--conf",   type=float, default=CONF_THRESH,
                        help=f"Confidence threshold (default: {CONF_THRESH})")
    parser.add_argument("--show",   action="store_true",
                        help="Show results in a window (requires display)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n Loading model: {args.model}")
    model = YOLO(args.model)
    model.conf = args.conf

    source = args.source
    all_detections = []

    # ── Webcam ────────────────────────────────────────────────────────────
    if source == "0":
        print(" Running live webcam detection. Press Q to quit.")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tmp = os.path.join(OUTPUT_DIR, "_webcam_frame.jpg")
            cv2.imwrite(tmp, frame)
            annotated, dets = detect_image(model, tmp, road_name="Live Camera")
            all_detections.extend(dets)
            if annotated is not None:
                cv2.imshow("Pothole Detection (Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    # ── Single video ──────────────────────────────────────────────────────
    elif Path(source).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        print(f" Processing video: {source}")
        all_detections = detect_video(model, source)

    # ── Folder of images ──────────────────────────────────────────────────
    elif Path(source).is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in Path(source).iterdir() if p.suffix.lower() in exts]
        print(f" Processing {len(images)} images from: {source}")
        for img_path in images:
            print(f"  → {img_path.name}")
            annotated, dets = detect_image(model, img_path)
            all_detections.extend(dets)
            if annotated is not None:
                out = os.path.join(OUTPUT_DIR, img_path.name)
                cv2.imwrite(out, annotated)
                if args.show:
                    cv2.imshow("Detection", annotated)
                    cv2.waitKey(500)
            print(f"     {len(dets)} pothole(s) found")

    # ── Single image ──────────────────────────────────────────────────────
    else:
        print(f" Processing image: {source}")
        annotated, dets = detect_image(model, source)
        all_detections.extend(dets)
        if annotated is not None:
            out_name = Path(source).stem + "_detected" + Path(source).suffix
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, annotated)
            print(f"  Annotated image saved: {out_path}")
            if args.show:
                cv2.imshow("Pothole Detection", annotated)
                cv2.waitKey(0)
        print(f"  {len(dets)} pothole(s) found")

    cv2.destroyAllWindows()

    # Save CSV report and print summary
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    save_report(all_detections, report_path)
    print_summary(all_detections)


if __name__ == "__main__":
    main()
