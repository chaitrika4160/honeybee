import cv2

video = cv2.VideoCapture("test videos/TEST.mp4")

if not video.isOpened():
    print("ERROR: Could not open video.")
    exit()

ret, frame = video.read()

if not ret:
    print("ERROR: Could not read frame.")
    exit()

print("Select the tube area using your mouse.")
print("Click and drag a rectangle around the tube.")
print("Press ENTER when finished.")
print("Press ESC to cancel.")

# Select ROI
roi = cv2.selectROI("Select Tube ROI", frame)

x, y, w, h = roi

print()
print("----- ROI Coordinates -----")
print("X:", x)
print("Y:", y)
print("Width:", w)
print("Height:", h)

# Crop the selected area
cropped = frame[y:y+h, x:x+w]

cv2.imshow("Selected ROI", cropped)

print()
print("Press any key to close.")

cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()