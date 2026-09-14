import cv2

video = cv2.VideoCapture("test videos/TEST.mp4")

if not video.isOpened():
    print("ERROR: Could not open video.")
    exit()

# ROI coordinates from Step 2
x = 3
y = 216
w = 1276
h = 440

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Crop the frame to our ROI
    roi = frame[y:y+h, x:x+w]

    cv2.imshow("Tube ROI", roi)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()