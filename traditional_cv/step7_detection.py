import cv2
import numpy as np

VIDEO_PATH = "test videos/TEST.mp4"

# Detection settings we selected
THRESHOLD = 25
KERNEL_SIZE = 5

# Ignore very small regions
MIN_AREA = 100

cap = cv2.VideoCapture(VIDEO_PATH)

background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=THRESHOLD,
    detectShadows=False
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ROI
    roi = frame[216:656, 3:1279]

    # Background subtraction
    mask = background.apply(roi)

    # Clean mask
    kernel = np.ones(
        (KERNEL_SIZE, KERNEL_SIZE),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw detections
    output = roi.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small regions
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Show area
        cv2.putText(
            output,
            f"Area: {int(area)}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imshow("Bee Detection", output)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()