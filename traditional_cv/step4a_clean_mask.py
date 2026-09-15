import cv2
import numpy as np

video = cv2.VideoCapture("test videos/TEST.mp4")

if not video.isOpened():
    print("ERROR: Could not open video.")
    exit()

# ROI coordinates
x = 3
y = 216
w = 1276
h = 440

# Background subtractor
background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

# Kernel used for cleaning
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Extract ROI
    roi = frame[y:y+h, x:x+w]

    # Background subtraction
    mask = background.apply(roi)

    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill small gaps / connect nearby regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Original ROI", roi)
    cv2.imshow("Cleaned Motion Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()