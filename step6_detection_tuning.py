import cv2
import numpy as np

VIDEO_PATH = "test videos/TEST.mp4"

# Settings we want to compare
settings = [
    (15, 5),
    (25, 5),
    (40, 7)
]

cap = cv2.VideoCapture(VIDEO_PATH)

# Create background subtractors
subtractors = []

for threshold, kernel_size in settings:
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=threshold,
        detectShadows=False
    )
    subtractors.append(subtractor)

current_setting = 0

# First, let the background model learn
print("Learning background...")

for i in range(100):
    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[216:656, 3:1279]

    for subtractor in subtractors:
        subtractor.apply(roi)

print("Background learned.")
print("Find a frame with bees and press SPACE to pause.")

paused_frame = None

while True:

    # Play video until SPACE is pressed
    if paused_frame is None:

        ret, frame = cap.read()

        if not ret:
            print("Video ended.")
            break

        roi = frame[216:656, 3:1279]

        # Keep updating background models
        for subtractor in subtractors:
            subtractor.apply(roi)

    else:
        roi = paused_frame

    # Use selected setting
    threshold, kernel_size = settings[current_setting]

    # We need to calculate the mask from the paused frame.
    # Create a temporary subtractor using the selected settings.
    temp_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=threshold,
        detectShadows=False
    )

    # This temporary model needs background information,
    # so instead we'll use the already calculated mask
    # from the selected subtractor.
    mask = subtractors[current_setting].apply(roi, learningRate=0)

    # Clean mask
    kernel = np.ones(
        (kernel_size, kernel_size),
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

    # Show setting
    text = f"Setting {current_setting + 1}: Threshold={threshold}, Kernel={kernel_size}"

    cv2.putText(
        mask,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        255,
        2
    )

    cv2.imshow("Detection Tuning", mask)

    key = cv2.waitKey(30) & 0xFF

    # SPACE = pause/unpause
    if key == 32:

        if paused_frame is None:
            paused_frame = roi.copy()
            print("PAUSED - press 1, 2 or 3 to compare settings.")

        else:
            paused_frame = None
            print("PLAYING")

    # Change setting
    elif key == ord('1'):
        current_setting = 0
        print("Setting 1: Threshold=15, Kernel=5")

    elif key == ord('2'):
        current_setting = 1
        print("Setting 2: Threshold=25, Kernel=5")

    elif key == ord('3'):
        current_setting = 2
        print("Setting 3: Threshold=40, Kernel=7")

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()