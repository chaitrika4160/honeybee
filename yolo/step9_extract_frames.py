import cv2
import os

VIDEO_PATH = "../test videos/TEST.mp4"
OUTPUT_DIR = "dataset/images/raw"

# Save one frame every 100 frames
FRAME_INTERVAL = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % FRAME_INTERVAL == 0:

        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{saved_count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        saved_count += 1

    frame_count += 1

cap.release()

print(f"Total frames checked: {frame_count}")
print(f"Frames saved: {saved_count}")