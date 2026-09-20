from ultralytics import YOLO

# Load a small YOLO model
model = YOLO("yolo11n.pt")

# Run YOLO on our test video
model.predict(
    source="../test videos/TEST.mp4",
    show=True,
    conf=0.25
)