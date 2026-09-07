# Group Detection and Group-wise People Counting

A computer vision project that detects groups of people in video footage and estimates how many people are present in each detected group.

The system uses a custom-trained YOLO model for group detection and YOLOv8l for individual person detection. ByteTrack is used for tracking, and a stability mechanism is used to reduce fluctuations in the people count.

---

## 1. Project Objective

The objective of this project is to detect stable groups of people and display:

- Total number of groups
- Group ID
- Number of people in each group
- Bounding boxes around groups
- Bounding boxes around detected people

Example:

```text
TOTAL GROUPS: 2

Group 1 | People: 4
Group 2 | People: 3
```

The project is intended for video sources such as CCTV, fixed cameras, mobile cameras, drone footage, and recorded videos.

---

## 2. Features

### Group Detection
A custom-trained YOLO model detects groups of people in the video.

### Person Detection
YOLOv8l detects individual people inside the video.

### Group-wise People Counting
Detected people are associated with detected groups using bounding-box overlap.

### Object Tracking
ByteTrack is used to track detected people and groups across video frames.

### Stable Counting
Recent counting results are stored to reduce temporary changes in the detected number of people.

### Group Count
The system displays the total number of groups detected in the current frame.

---

## 3. Technologies

- Python
- Ultralytics YOLO
- YOLOv8l
- PyTorch
- OpenCV
- ByteTrack

---

## 4. Project Structure

```text
group2/
│
├── group.py
├── new_dataset_collector.py
├── split_new_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

Large files such as videos, datasets, model weights, virtual environments, and training outputs are excluded from Git.

---

## 5. Installation

### Step 1: Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd group2
```

### Step 2: Create a Virtual Environment

```powershell
python -m venv env
```

### Step 3: Activate the Environment

```powershell
.\env\Scripts\Activate.ps1
```

### Step 4: Install Requirements

```powershell
pip install -r requirements.txt
```

---

## 6. Dataset Creation

The group detection dataset is created by extracting frames from video footage.

Frames can be collected using:

```text
new_dataset_collector.py
```

Example:

```powershell
python new_dataset_collector.py --source sample\video16.mp4 --every 15 --start-frame 504
```

### Parameters

`--source` specifies the input video.

`--every` specifies the frame extraction interval.

`--start-frame` specifies the frame from which extraction starts.

The extracted images are then labelled for the `group` class.

---

## 7. Dataset Splitting

After collecting and labelling the images, the dataset is divided into:

- Training
- Validation
- Testing

Run:

```powershell
python split_new_dataset.py
```

The dataset used for fine-tuning contained:

```text
Total : 234
Train : 187
Val   : 23
Test  : 24
```

The YOLO dataset configuration contains one class:

```yaml
names:
  0: group
```

---

## 8. Model Fine-tuning

The custom group detection model is fine-tuned using the prepared dataset.

The previously trained group model is used as the starting point, and additional group images are used for further training.

The trained model produces:

```text
best.pt
```

The model is used to detect groups in the final video-processing system.

---

## 9. Models Used

### Group Detection Model

```text
best.pt
```

Used to detect groups.

### Person Detection Model

```text
yolov8l.pt
```

Used to detect individual people.

---

## 10. Model Configuration

In `group.py`, set the model paths:

```python
GROUP_MODEL_PATH = "path/to/best.pt"
PERSON_MODEL_PATH = "yolov8l.pt"
```

Set the input video:

```python
VIDEO_PATH = "video1.mp4"
```

---

## 11. Run the Program

Activate the virtual environment:

```powershell
.\env\Scripts\Activate.ps1
```

Then run:

```powershell
python group.py
```

The program will open the video and process it frame by frame.

Press `Q` to stop the video.

---

## 12. How the System Works

```text
Input Video
     │
     ▼
Detect People ───────► YOLOv8l
     │
     ▼
Track People ────────► ByteTrack
     │
     └──────────────┐
                    │
                    ▼
             Detect Groups
                    │
                    ▼
              Custom YOLO
                    │
                    ▼
              Track Groups
                    │
                    ▼
          Calculate Box Overlap
                    │
                    ▼
        Associate People with Groups
                    │
                    ▼
          Count People per Group
                    │
                    ▼
            Stabilize the Count
                    │
                    ▼
             Display Results
```

---

## 13. Group-wise People Counting

For every detected group, the program compares its bounding box with the bounding boxes of detected people.

If enough of a person's bounding box overlaps with the group bounding box, that person is counted as part of the group.

Example:

```text
Group 1
┌─────────────────────────────────┐
│                                 │
│  P1     P2      P3       P4     │
│                                 │
└─────────────────────────────────┘

Group 1 | People: 4
```

---

## 14. Stable People Counting

Object detection can sometimes produce unstable results.

For example, an actual group of four people may temporarily be detected as:

```text
4 → 3 → 4 → 2 → 4
```

This can happen because of:

- Occlusion
- People overlapping
- Temporary missed detections
- Camera movement
- Detection confidence changes

To reduce this problem, the program stores recent counting results and uses them to produce a more stable displayed count.

---

## 15. Important Parameters

The main parameters are:

```python
GROUP_CONF = 0.50
PERSON_CONF = 0.15
OVERLAP_THRESHOLD = 0.20
HISTORY_SIZE = 8
CHANGE_CONFIRMATION = 3
```

### GROUP_CONF

Controls the confidence threshold for group detection.

### PERSON_CONF

Controls the confidence threshold for person detection.

### OVERLAP_THRESHOLD

Controls how much a person's bounding box must overlap with a group bounding box to be counted.

### HISTORY_SIZE

Number of recent counts stored for stabilization.

### CHANGE_CONFIRMATION

Number of confirmations required before changing the displayed count.

---

## 16. Output

The processed video displays:

### Total Group Count

```text
TOTAL GROUPS: 2
```

### Group Information

```text
Group 1 | People: 4
```

### Person Detection

Individual detected people are displayed with bounding boxes and tracking IDs.

---

## 17. Stable Group Detection

The project focuses on detecting stable groups rather than simply considering every set of nearby people as a group.

### Stable Group

```text
Person A ─┐
Person B ─┼─ Stay together
Person C ─┘

        ↓

     GROUP
```

### Walking People

People who are only temporarily close while walking should ideally not be considered a stable group.

```text
P1 → → → →

P2 → → → →

P3 → → → →

Temporary proximity
        ↓
Not necessarily a group
```

Future versions can improve this using tracking history, movement direction, distance between people, and temporal group stability.

---

## 18. Limitations

The current system may have difficulties when:

- People heavily overlap.
- People are partially hidden.
- Groups are very small in the frame.
- Lighting conditions are poor.
- The camera moves significantly.
- Multiple groups merge together.
- People enter or leave a group.
- People temporarily walk close together.
- The camera angle changes significantly.

The people-per-group value is an estimate based on object detection and bounding-box overlap.

---

## 19. Future Improvements

Possible improvements include:

- Improve detection of partially occluded people.
- Improve group/person association.
- Improve group tracking.
- Improve counting stability.
- Reduce false group detections.
- Detect group merging and splitting.
- Improve stable-group recognition.
- Use movement analysis to distinguish walking people from stable groups.
- Support real-time CCTV streams.
- Support RTSP cameras.
- Improve performance on drone footage.
- Improve performance on mobile-camera footage.
- Optimize GPU inference.
- Add real-time group alerts.

---

## 20. Git

The repository excludes large and generated files such as:

```text
*.pt
*.onnx
*.mp4
*.avi
*.mov
*.mkv
*.jpg
*.jpeg
*.png
runs/
env/
venv/
```

This keeps the GitHub repository clean and lightweight.

---

## 21. Quick Start

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>

cd group2

python -m venv env

.\env\Scripts\Activate.ps1

pip install -r requirements.txt
```

Place the required model files and input video.

Update:

```python
GROUP_MODEL_PATH = "path/to/best.pt"
PERSON_MODEL_PATH = "yolov8l.pt"
VIDEO_PATH = "video1.mp4"
```

Run:

```powershell
python group.py
```

Press:

```text
Q
```

to stop the video.

---

## 22. Conclusion

This project combines custom YOLO-based group detection, YOLOv8l person detection, ByteTrack tracking, bounding-box association, and count stabilization to estimate the number of people in each detected group.

The goal is to develop a reliable group detection and group-wise people counting system that can work across different real-world video sources such as CCTV, mobile cameras, and drone footage.

---

## License

This project is intended for educational, research, and development purposes.
