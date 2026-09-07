import argparse
import os
import cv2

# ============================================================
# CONFIG
# ============================================================

OUTPUT_ROOT = "group_dataset"

IMAGE_DIR = os.path.join(OUTPUT_ROOT, "images")
LABEL_DIR = os.path.join(OUTPUT_ROOT, "labels")

MAX_DISPLAY_HEIGHT = 800

# Keys
KEY_GROUP = ord("z")
KEY_SKIP = ord("v")
KEY_UNDO = ord("d")
KEY_QUIT = ord("q")
KEY_SAVE_NEXT = 32  # SPACE

# ============================================================
# GLOBAL STATE
# ============================================================

DISPLAY_SCALE = 1.0

drawing = False
start_x = 0
start_y = 0

# Cursor crosshair position
cursor_x = -1
cursor_y = -1

current_box = None
boxes = []


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(
            os.path.join(IMAGE_DIR, split),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(LABEL_DIR, split),
            exist_ok=True
        )


# ============================================================
# FIND NEXT FILENAME
# ============================================================

def next_filename():

    max_num = -1

    for split in ["train", "val", "test"]:

        folder = os.path.join(IMAGE_DIR, split)

        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):

            if not filename.lower().endswith(".jpg"):
                continue

            name = os.path.splitext(filename)[0]

            if name.startswith("frame_"):

                try:
                    num = int(name.replace("frame_", ""))
                    max_num = max(max_num, num)

                except ValueError:
                    pass

    return max_num + 1


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global current_box
    global boxes
    global cursor_x
    global cursor_y

    # Always keep the cursor position updated
    cursor_x = x
    cursor_y = y

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            current_box = (
                start_x,
                start_y,
                x,
                y
            )

    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False

            x1 = min(start_x, x)
            y1 = min(start_y, y)

            x2 = max(start_x, x)
            y2 = max(start_y, y)

            # Ignore accidental tiny boxes
            if (
                abs(x2 - x1) > 10
                and
                abs(y2 - y1) > 10
            ):

                boxes.append(
                    (x1, y1, x2, y2)
                )

            current_box = None


# ============================================================
# DRAW BOXES
# ============================================================

def draw_boxes(
    image,
    boxes,
    current_box=None,
    group_mode=False
):

    result = image.copy()

    # --------------------------------------------------------
    # EXISTING BOXES
    # --------------------------------------------------------

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = box

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            str(i + 1),
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # CURRENT BOX BEING DRAWN
    # --------------------------------------------------------

    if current_box is not None:

        x1, y1, x2, y2 = current_box

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    cv2.putText(
        result,
        "DRAG=ADD | D=UNDO | Z=CONFIRM | SPACE=SAVE/NEXT | V=SKIP | Q=QUIT",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Selections: {len(boxes)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    if group_mode:

        cv2.putText(
            result,
            "GROUP SELECTED - PRESS SPACE TO SAVE",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return result


# ============================================================
# SAVE IMAGE + YOLO LABEL
# ============================================================

def save_frame(frame, boxes):

    if len(boxes) == 0:
        return False

    image_number = next_filename()

    # New images go to TRAIN for now.
    # Later run your split script.
    split = "train"

    image_path = os.path.join(
        IMAGE_DIR,
        split,
        f"frame_{image_number:06d}.jpg"
    )

    label_path = os.path.join(
        LABEL_DIR,
        split,
        f"frame_{image_number:06d}.txt"
    )

    # --------------------------------------------------------
    # SAVE ORIGINAL IMAGE
    # --------------------------------------------------------

    success = cv2.imwrite(
        image_path,
        frame
    )

    if not success:

        print("ERROR: Image could not be saved!")

        return False

    # --------------------------------------------------------
    # ORIGINAL IMAGE SIZE
    # --------------------------------------------------------

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # SAVE YOLO LABEL
    # --------------------------------------------------------

    valid_labels = 0

    with open(label_path, "w") as f:

        for box in boxes:

            x1, y1, x2, y2 = box

            # Display -> original coordinates

            x1 = int(x1 / DISPLAY_SCALE)
            y1 = int(y1 / DISPLAY_SCALE)

            x2 = int(x2 / DISPLAY_SCALE)
            y2 = int(y2 / DISPLAY_SCALE)

            # Clamp coordinates

            x1 = max(
                0,
                min(x1, w - 1)
            )

            y1 = max(
                0,
                min(y1, h - 1)
            )

            x2 = max(
                0,
                min(x2, w - 1)
            )

            y2 = max(
                0,
                min(y2, h - 1)
            )

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            # YOLO normalized format

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h

            nw = bw / w
            nh = bh / h

            # Class 0 = group

            f.write(
                f"0 {cx:.6f} {cy:.6f} "
                f"{nw:.6f} {nh:.6f}\n"
            )

            valid_labels += 1

    print(
        f"SAVED: frame_{image_number:06d}.jpg "
        f"| {valid_labels} GROUPS"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    global DISPLAY_SCALE
    global boxes
    global current_box
    global cursor_x
    global cursor_y

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Video path"
    )

    parser.add_argument(
        "--every",
        type=int,
        default=15,
        help="Process every N frames"
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start processing from this frame"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CHECK EVERY
    # --------------------------------------------------------

    if args.every < 1:

        print("ERROR: --every must be >= 1")

        return

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    ensure_dirs()

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():

        print(
            f"ERROR: Could not open video: "
            f"{args.source}"
        )

        return

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print()
    print("==============================================")
    print("       MULTI GROUP DATASET COLLECTOR")
    print("==============================================")
    print()

    print("Controls:")
    print("  DRAG       = Add selection")
    print("  D          = Undo last selection")
    print("  Z          = Confirm GROUP")
    print("  SPACE      = Save + next frame")
    print("  V          = Skip frame")
    print("  Q          = Quit")
    print()

    print(
        f"Video frames : {total_frames}"
    )

    print(
        f"FPS          : {fps:.2f}"
    )

    print(
        f"Start frame  : {args.start_frame}"
    )

    print(
        f"Every        : {args.every}"
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT:
    # JUMP DIRECTLY TO START FRAME
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        args.start_frame
    )

    frame_idx = args.start_frame

    total_saved = 0
    total_skipped = 0

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print()
            print("Reached end of video.")

            break

        # ----------------------------------------------------
        # CURRENT FRAME NUMBER
        # ----------------------------------------------------

        current_frame_number = frame_idx

        frame_idx += args.every

        # ----------------------------------------------------
        # SKIP TO NEXT SELECTED FRAME
        #
        # We already jumped to the correct frame.
        # After processing, directly seek to the next one.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # DISPLAY SCALE
        # ----------------------------------------------------

        h, w = frame.shape[:2]

        DISPLAY_SCALE = 1.0

        if h > MAX_DISPLAY_HEIGHT:

            DISPLAY_SCALE = (
                MAX_DISPLAY_HEIGHT / h
            )

        display_w = int(
            w * DISPLAY_SCALE
        )

        display_h = int(
            h * DISPLAY_SCALE
        )

        base_display = cv2.resize(
            frame,
            (display_w, display_h)
        )

        # ----------------------------------------------------
        # RESET BOXES
        # ----------------------------------------------------

        boxes = []

        current_box = None

        # Reset cursor position for the new frame
        cursor_x = -1
        cursor_y = -1

        group_mode = False

        # ----------------------------------------------------
        # WINDOW
        # ----------------------------------------------------

        window_name = "GROUP LABELER"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.setMouseCallback(
            window_name,
            mouse_callback
        )

        # ====================================================
        # LABEL CURRENT FRAME
        # ====================================================

        move_to_next = False

        while True:

            shown = draw_boxes(
                base_display,
                boxes,
                current_box,
                group_mode
            )

            # Show frame number

            cv2.putText(
                shown,
                f"Video Frame: {current_frame_number}",
                (10, display_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # CURSOR CROSSHAIR
            # ------------------------------------------------
            # Full vertical + horizontal guide lines follow
            # the mouse cursor while labeling.
            if cursor_x >= 0 and cursor_y >= 0:

                cv2.line(
                    shown,
                    (cursor_x, 0),
                    (cursor_x, display_h - 1),
                    (255, 255, 255),
                    1
                )

                cv2.line(
                    shown,
                    (0, cursor_y),
                    (display_w - 1, cursor_y),
                    (255, 255, 255),
                    1
                )

            cv2.imshow(
                window_name,
                shown
            )

            key = cv2.waitKey(20) & 0xFF

            # ------------------------------------------------
            # Q = QUIT
            # ------------------------------------------------

            if key == KEY_QUIT:

                print()
                print("QUIT")

                cap.release()
                cv2.destroyAllWindows()

                print(
                    f"Frames saved: {total_saved}"
                )

                print(
                    f"Frames skipped: {total_skipped}"
                )

                return

            # ------------------------------------------------
            # D = UNDO
            # ------------------------------------------------

            elif key == KEY_UNDO:

                if len(boxes) > 0:

                    removed = boxes.pop()

                    group_mode = False

                    print(
                        f"UNDO: removed selection "
                        f"{len(boxes) + 1}"
                    )

                else:

                    print(
                        "UNDO: nothing to remove"
                    )

            # ------------------------------------------------
            # Z = CONFIRM GROUP
            # ------------------------------------------------

            elif key == KEY_GROUP:

                if len(boxes) > 0:

                    group_mode = True

                    print(
                        f"GROUP confirmed: "
                        f"{len(boxes)} selections"
                    )

                    print(
                        "Press SPACE to save "
                        "and move to next frame."
                    )

                else:

                    print(
                        "No selections!"
                    )

                    print(
                        "Drag at least one box."
                    )

            # ------------------------------------------------
            # V = SKIP
            # ------------------------------------------------

            elif key == KEY_SKIP:

                print(
                    f"SKIPPED video frame "
                    f"{current_frame_number}"
                )

                total_skipped += 1

                move_to_next = True

                break

            # ------------------------------------------------
            # SPACE = SAVE + NEXT
            # ------------------------------------------------

            elif key == KEY_SAVE_NEXT:

                if group_mode and len(boxes) > 0:

                    success = save_frame(
                        frame,
                        boxes
                    )

                    if success:

                        total_saved += 1

                        print(
                            "Saved successfully."
                        )

                        print(
                            "Moving to next frame..."
                        )

                        move_to_next = True

                        break

                elif len(boxes) == 0:

                    print(
                        "No selections."
                    )

                    print(
                        "Drag at least one box."
                    )

                else:

                    print(
                        "Press Z first to confirm "
                        "these selections as GROUP."
                    )

        # ----------------------------------------------------
        # CLOSE CURRENT WINDOW
        # ----------------------------------------------------

        cv2.destroyWindow(
            window_name
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # SEEK DIRECTLY TO NEXT SELECTED FRAME
        # ----------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

    # ========================================================
    # END
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("==============================================")
    print("DATASET COLLECTION COMPLETE")
    print("==============================================")

    print(
        f"Frames saved   : {total_saved}"
    )

    print(
        f"Frames skipped : {total_skipped}"
    )

    print(
        f"Dataset        : {OUTPUT_ROOT}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()