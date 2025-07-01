from collections import Counter
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0            # default webcam
IMGSZ        = 480          # inference resolution (square)
CONF_THRES   = 0.50         # detection confidence threshold
IOU_THRES    = 0.45         # NMS IoU threshold
MODEL_PATH   = "yolov10n.pt"

TARGET_FPS   = 8            # desired processing FPS (5‑10 recommended)
SNAP_DIR     = "snapshots"  # directory to save detected frames
os.makedirs(SNAP_DIR, exist_ok=True)

# Fixed colour palette (BGR)
PALETTE = [
    (255,   0,   0),  # class‑id % 6 == 0 – Blue
    (  0, 255,   0),  #              1 – Green
    (  0,   0, 255),  #              2 – Red
    (255, 255,   0),  #              3 – Cyan
    (255,   0, 255),  #              4 – Magenta
    (  0, 255, 255),  #              5 – Yellow
]

# ──────────────────────────────────────────────────────────────
# Load model (CPU‑only PyTorch)
# ──────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH).to("cpu")

# ──────────────────────────────────────────────────────────────
# Video capture init
# ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(WEBCAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check index/permissions.")

fps_ema, prev_time, frame_idx = 0.0, time.time(), 0
session_cnt = Counter()
print("[INFO] Starting inference loop. Press 'q' to exit.")

min_frame_interval = 1.0 / TARGET_FPS

while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame grab failed. Exiting…")
        break

    # Inference
    result = model.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device="cpu",
        half=False,
        stream=False,
        verbose=False,
    )[0]

    boxes = result.boxes
    detected_this_frame = False
    if boxes is not None and len(boxes):
        clss   = boxes.cls.cpu().numpy().astype(int)
        names  = [model.names[c] for c in clss]
        counts = Counter(names)
        session_cnt.update(counts)
        detected_this_frame = True

        for box, cls_id, conf in zip(boxes.xyxy.cpu().numpy().astype(int), clss, boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            cls_name = model.names[int(cls_id)]
            color = PALETTE[int(cls_id) % len(PALETTE)]  # consistent colour per class
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Per‑frame console print
        joined = " ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"Frame {frame_idx}: {joined}")

    # Snapshot capture
    if detected_this_frame:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_path = os.path.join(SNAP_DIR, f"snap_{ts}.jpg")
        cv2.imwrite(snap_path, frame)
        print(f"[SNAP] Saved {snap_path}")

    # FPS overlay (always red for visibility)
    curr_time = time.time()
    fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Real-Time Object Detection", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Frame‑rate limiter
    elapsed = time.time() - loop_start
    sleep_time = max(0.0, min_frame_interval - elapsed)
    if sleep_time:
        time.sleep(sleep_time)

# ──────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("[INFO] Session complete. Total objects detected:")
for name, count in session_cnt.items():
    print(f"  {name}: {count}")