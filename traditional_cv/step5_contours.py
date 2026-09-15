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

# Kernel for cleaning the mask
kernel = np.ones((5, 5), np.uint8)

# Minimum contour area
MIN_AREA = 100

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Extract ROI
    roi = frame[y:y+h, x:x+w]

    # Background subtraction
    mask = background.apply(roi)

    # Clean the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw bounding boxes
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area < MIN_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)

        cv2.rectangle(
            roi,
            (bx, by),
            (bx + bw, by + bh),
            (0, 255, 0),
            2
        )

    cv2.imshow("Detected Objects", roi)
    cv2.imshow("Cleaned Motion Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()