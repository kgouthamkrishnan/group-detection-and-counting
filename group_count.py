from ultralytics import YOLO
import cv2
from collections import defaultdict, Counter



GROUP_MODEL_PATH = (
    "runs/detect/runs/detect/group_finetuned-3/weights/best.pt"
)

PERSON_MODEL_PATH = "yolov8l.pt"

VIDEO_PATH = "video1.mp4"



GROUP_CONF = 0.50

PERSON_CONF = 0.15

# 0.25 = 25%
OVERLAP_THRESHOLD = 0.20

HISTORY_SIZE = 8

CHANGE_CONFIRMATION = 3



print("Loading group model...")

group_model = YOLO(GROUP_MODEL_PATH)

print("Loading YOLOv8l person model...")

person_model = YOLO(PERSON_MODEL_PATH)

print("Models loaded successfully.")



cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise SystemExit(
        f"ERROR: Cannot open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:

    fps = 30.0


print(f"Original video FPS: {fps:.2f}")

frame_delay = max(
    1,
    int(1000 / fps)
)


count_history = defaultdict(list)

stable_count = {}

candidate_count = {}

candidate_frames = {}


while True:

    ret, frame = cap.read()

    if not ret:
        break



    person_results = person_model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        classes=[0],

        conf=PERSON_CONF,

        iou=0.5,

        verbose=False
    )

    person_result = person_results[0]


    people = []


    if person_result.boxes is not None:

        boxes = person_result.boxes


        for i in range(len(boxes)):

            # Person bounding box
            px1, py1, px2, py2 = (

                boxes.xyxy[i]
                .cpu()
                .numpy()
                .astype(int)
            )


            # Person tracking ID
            if boxes.id is not None:

                person_id = int(
                    boxes.id[i]
                )

            else:

                person_id = None


            people.append({

                "id": person_id,

                "box": (
                    px1,
                    py1,
                    px2,
                    py2
                )

            })


    group_results = group_model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=0.20,

        iou=0.6,

        verbose=False
    )

    group_result = group_results[0]


    group_count = 0



    if group_result.boxes is not None:

        group_boxes = group_result.boxes


        for i in range(len(group_boxes)):

            confidence = float(
                group_boxes.conf[i]
            )


            if confidence < GROUP_CONF:
                continue


            group_count += 1



            x1, y1, x2, y2 = (

                group_boxes.xyxy[i]
                .cpu()
                .numpy()
                .astype(int)
            )



            if group_boxes.id is not None:

                group_id = int(
                    group_boxes.id[i]
                )

            else:

                group_id = group_count



            current_count = 0


            for person in people:

                px1, py1, px2, py2 = (
                    person["box"]
                )



                inter_x1 = max(
                    x1,
                    px1
                )

                inter_y1 = max(
                    y1,
                    py1
                )

                inter_x2 = min(
                    x2,
                    px2
                )

                inter_y2 = min(
                    y2,
                    py2
                )



                inter_width = max(
                    0,
                    inter_x2 - inter_x1
                )

                inter_height = max(
                    0,
                    inter_y2 - inter_y1
                )


                inter_area = (
                    inter_width
                    *
                    inter_height
                )


                person_width = max(
                    1,
                    px2 - px1
                )

                person_height = max(
                    1,
                    py2 - py1
                )


                person_area = (
                    person_width
                    *
                    person_height
                )



                overlap_ratio = (
                    inter_area
                    /
                    person_area
                )

                if overlap_ratio >= OVERLAP_THRESHOLD:

                    current_count += 1



                    cv2.rectangle(

                        frame,

                        (px1, py1),

                        (px2, py2),

                        (255, 0, 0),

                        2
                    )



                    if person["id"] is not None:

                        person_label = (
                            f"P{person['id']}"
                        )

                        cv2.putText(

                            frame,

                            person_label,

                            (
                                px1,
                                max(
                                    py1 - 5,
                                    20
                                )
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.45,

                            (255, 0, 0),

                            1
                        )



            count_history[group_id].append(
                current_count
            )


            # Keep only recent values

            if (
                len(
                    count_history[group_id]
                )
                > HISTORY_SIZE
            ):

                count_history[group_id].pop(0)



            counts = count_history[group_id]


            if len(counts) > 0:

                most_common_count = (
                    Counter(counts)
                    .most_common(1)[0][0]
                )

            else:

                most_common_count = (
                    current_count
                )



            if group_id not in stable_count:

                stable_count[group_id] = (
                    most_common_count
                )



            if (
                most_common_count
                != stable_count[group_id]
            ):


                # New candidate count

                if (

                    group_id
                    not in candidate_count

                    or

                    candidate_count[group_id]
                    != most_common_count

                ):

                    candidate_count[group_id] = (
                        most_common_count
                    )

                    candidate_frames[group_id] = 1


                else:

                    candidate_frames[group_id] += 1



                if (

                    candidate_frames[group_id]
                    >= CHANGE_CONFIRMATION

                ):

                    stable_count[group_id] = (
                        most_common_count
                    )

                    candidate_frames[group_id] = 0


            else:

                candidate_frames[group_id] = 0



            displayed_count = (
                stable_count[group_id]
            )



            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )



            label = (

                f"Group {group_id}"
                f" | People: {displayed_count}"

            )


            # Text size

            (tw, th), _ = cv2.getTextSize(

                label,

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                2
            )


            # Label position

            label_y = max(
                y1 - th - 10,
                0
            )


            # Label background

            cv2.rectangle(

                frame,

                (
                    x1,
                    label_y
                ),

                (
                    x1 + tw + 10,
                    y1
                ),

                (0, 255, 0),

                -1
            )


            # Label text

            cv2.putText(

                frame,

                label,

                (
                    x1 + 5,
                    y1 - 7
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 0, 0),

                2
            )



    total_text = (
        f"TOTAL GROUPS: {group_count}"
    )


    # Background

    cv2.rectangle(

        frame,

        (20, 20),

        (340, 75),

        (0, 0, 0),

        -1
    )


    # Text

    cv2.putText(

        frame,

        total_text,

        (30, 57),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (255, 255, 255),

        2
    )



    cv2.imshow(

        "Stable Group Counter",

        frame
    )



    key = cv2.waitKey(
        frame_delay
    ) & 0xFF


    # Press Q to quit

    if key == ord("q"):

        break



cap.release()

cv2.destroyAllWindows()

print(
    "Stable group counting finished."
)