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

# Mask cleaning
kernel = np.ones((5, 5), np.uint8)

# Ignore very small contours
MIN_AREA = 100

# Previous tracked objects:
# ID -> (center_x, center_y)
previous_objects = {}

next_id = 1

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Extract ROI
    roi = frame[y:y+h, x:x+w]

    # Background subtraction
    mask = background.apply(roi)

    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Store current detections
    current_detections = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)

        center_x = bx + bw // 2
        center_y = by + bh // 2

        current_detections.append(
            (center_x, center_y, bx, by, bw, bh)
        )

    current_objects = {}

    # Match each current detection to the closest
    # object from the previous frame
    for center_x, center_y, bx, by, bw, bh in current_detections:

        best_id = None
        best_distance = 50

        for object_id, (old_x, old_y) in previous_objects.items():

            distance = np.sqrt(
                (center_x - old_x) ** 2 +
                (center_y - old_y) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                best_id = object_id

        # No nearby previous object → new ID
        if best_id is None:
            best_id = next_id
            next_id += 1

        current_objects[best_id] = (center_x, center_y)

        # Draw box
        cv2.rectangle(
            roi,
            (bx, by),
            (bx + bw, by + bh),
            (0, 255, 0),
            2
        )

        # Draw center
        cv2.circle(
            roi,
            (center_x, center_y),
            4,
            (0, 0, 255),
            -1
        )

        # Draw ID
        cv2.putText(
            roi,
            f"ID {best_id}",
            (bx, by - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # Current objects become previous objects
    # for the next frame
    previous_objects = current_objects

    cv2.imshow("Tracking Test", roi)
    cv2.imshow("Motion Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()