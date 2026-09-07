from ultralytics import YOLO
import cv2
import time


MODEL_PATH = "runs/detect/runs/detect/group_finetuned-3/weights/best.pt"


VIDEO_PATH = "video1.mp4"


model = YOLO(MODEL_PATH)

# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"ERROR: Cannot open video: {VIDEO_PATH}")

# Get original video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

print(f"Original video FPS: {fps:.2f}")

# Correct display delay
frame_delay = max(1, int(1000 / fps))

print(f"Display delay: {frame_delay} ms")

# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # TRACK GROUPS
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.10,
        iou=0.5,
        verbose=False
    )

    result = results[0]

    group_count = 0

    # --------------------------------------------------------
    # DRAW GROUP BOXES
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            confidence = float(boxes.conf[i])

            if confidence < 0.35:
                continue

            group_count += 1

            # Bounding box
            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .astype(int)
            )

            # Tracking ID
            if boxes.id is not None:
                track_id = int(boxes.id[i])
                label = f"Group {track_id}"
            else:
                label = "Group"

            # Draw box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ========================================================
    # GROUP COUNT DISPLAY
    # ========================================================

    text = f"TOTAL GROUPS: {group_count}"

    cv2.rectangle(
        frame,
        (20, 20),
        (310, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow("Group Counter", frame)

    # IMPORTANT:
    # Keep original video playback speed
    key = cv2.waitKey(frame_delay) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Group counting finished.")