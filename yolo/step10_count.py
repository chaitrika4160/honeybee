import cv2
from ultralytics import YOLO
import math

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = r"C:\Users\chait\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"../test videos/TEST.mp4"

CONF = 0.30

# LINE 1 = LEFT
LINE1 = ((600, 680), (700, 130))

# LINE 2 = RIGHT
LINE2 = ((800, 715), (900, 165))

MAX_DISTANCE = 70
MAX_MISSED = 8


# ==========================================
# FUNCTIONS
# ==========================================

def side_of_line(point, line):

    x, y = point

    (x1, y1), (x2, y2) = line

    return (
        (x2 - x1) * (y - y1)
        -
        (y2 - y1) * (x - x1)
    )


def center_of_box(box):

    x1, y1, x2, y2 = box

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ==========================================
# OUTPUT VIDEO
# ==========================================

output = cv2.VideoWriter(
    "step10_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# ==========================================
# TRACKS
# ==========================================

tracks = []

next_id = 0

inside_count = 0
outside_count = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model.predict(
        frame,
        conf=CONF,
        verbose=False
    )


    detections = []


    # ======================================
    # GET DETECTIONS
    # ======================================

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes.xyxy:

            box = box.cpu().numpy()

            x1, y1, x2, y2 = map(int, box)

            center = center_of_box(box)

            detections.append(center)


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.circle(
                frame,
                center,
                4,
                (0, 255, 255),
                -1
            )


    # ======================================
    # MATCH DETECTIONS TO TRACKS
    # ======================================

    matched_tracks = set()

    for detection in detections:

        best_track = None

        best_distance = MAX_DISTANCE

        best_index = -1


        for i, track in enumerate(tracks):

            if i in matched_tracks:
                continue


            d = distance(
                detection,
                track["position"]
            )


            if d < best_distance:

                best_distance = d

                best_track = track

                best_index = i


        # ==================================
        # EXISTING TRACK
        # ==================================

        if best_track is not None:

            matched_tracks.add(best_index)

            old_position = best_track["position"]

            new_position = detection


            # --------------------------------
            # CHECK LINE 1
            # --------------------------------

            old_side_1 = side_of_line(
                old_position,
                LINE1
            )

            new_side_1 = side_of_line(
                new_position,
                LINE1
            )


            if old_side_1 * new_side_1 < 0:

                if best_track["line1"] is None:

                    best_track["line1"] = True

                    if new_position[0] > old_position[0]:

                        best_track["direction"] = "OUTSIDE"

                    else:

                        best_track["direction"] = "INSIDE"


            # --------------------------------
            # CHECK LINE 2
            # --------------------------------

            old_side_2 = side_of_line(
                old_position,
                LINE2
            )

            new_side_2 = side_of_line(
                new_position,
                LINE2
            )


            if old_side_2 * new_side_2 < 0:

                if best_track["line2"] is None:

                    best_track["line2"] = True


                    if best_track["line1"]:

                        if best_track["direction"] == "OUTSIDE":

                            outside_count += 1

                            best_track["counted"] = True

                        elif best_track["direction"] == "INSIDE":

                            inside_count += 1

                            best_track["counted"] = True


            best_track["position"] = new_position

            best_track["missed"] = 0


        # ==================================
        # NEW TRACK
        # ==================================

        else:

            tracks.append({

                "id": next_id,

                "position": detection,

                "line1": None,

                "line2": None,

                "direction": None,

                "counted": False,

                "missed": 0
            })

            next_id += 1


    # ======================================
    # UPDATE MISSED TRACKS
    # ======================================

    for i, track in enumerate(tracks):

        if i not in matched_tracks:

            track["missed"] += 1


    tracks = [

        track

        for track in tracks

        if track["missed"] <= MAX_MISSED
    ]


    # ======================================
    # DRAW LINES
    # ======================================

    cv2.line(
        frame,
        LINE1[0],
        LINE1[1],
        (0, 255, 0),
        4
    )

    cv2.line(
        frame,
        LINE2[0],
        LINE2[1],
        (0, 255, 0),
        4
    )


    cv2.putText(
        frame,
        "LINE 1",
        LINE1[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "LINE 2",
        LINE2[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # COUNTS
    # ======================================

    cv2.putText(
        frame,
        f"INSIDE: {inside_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"OUTSIDE: {outside_count}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )


    output.write(frame)

    cv2.imshow(
        "Step 10 - Honeybee Counting",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# FINISH
# ==========================================

cap.release()

output.release()

cv2.destroyAllWindows()


print()
print("============================")
print("STEP 10 FINAL COUNT")
print("============================")
print("INSIDE :", inside_count)
print("OUTSIDE:", outside_count)
print("============================")