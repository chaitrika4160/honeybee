import cv2

video = cv2.VideoCapture("test videos/TEST.mp4")

if not video.isOpened():
    print("ERROR: Could not open video.")
    exit()

# ROI coordinates
x = 3
y = 216
w = 1276
h = 440

# Create background subtractor
background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Extract the tube ROI
    roi = frame[y:y+h, x:x+w]

    # Apply background subtraction
    mask = background.apply(roi)

    # Show original ROI
    cv2.imshow("Original ROI", roi)

    # Show motion mask
    cv2.imshow("Motion Mask", mask)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()