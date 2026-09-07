from ultralytics import YOLO

model = YOLO("best.pt")

results = model.train(
    data="group_dataset/data.yaml",

    epochs=50,
    imgsz=640,
    batch=8,
    device=0,

    # Conservative fine-tuning
    lr0=0.001,
    lrf=0.01,
    patience=15,

    # Useful for viewpoint/scale variation
    degrees=5,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,

    fliplr=0.5,
    flipud=0.0,

    mosaic=1.0,
    mixup=0.0,

    workers=0,

    project="runs/detect",
    name="group_finetuned",
    exist_ok=False,

    save=True,
    plots=True
)