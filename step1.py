import cv2

video = cv2.VideoCapture("test videos/TEST.mp4")

if not video.isOpened():
    print("ERROR: Could not open the video.")
    exit()

# Get total number of frames
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Frames we want to inspect
positions = [
    0,
    frame_count // 4,
    frame_count // 2,
    (3 * frame_count) // 4,
    frame_count - 1
]

for i, frame_number in enumerate(positions):

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = video.read()

    if ret:
        filename = f"frame_{i+1}.jpg"
        cv2.imwrite(filename, frame)

        print(f"Saved {filename} - frame {frame_number}")

video.release()