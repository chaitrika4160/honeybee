import cv2
import math
from ultralytics import YOLO

# ============================================================
# STEP 11 - UPGRADED YOLO + BYTE TRACK + ROBUST TWO-LINE COUNT
# ============================================================
#
# Main improvements over the previous version:
# 1. Stable "bee IDs" are kept even when ByteTrack changes IDs.
# 2. A dead-zone around each counting line reduces jitter crossings.
# 3. A bee must make a valid LINE1 -> LINE2 or LINE2 -> LINE1
#    transition before it is counted.
# 4. After a count, the bee must move away from both lines before
#    it can start another journey. This prevents duplicate counts.
# 5. Recently lost tracks can be re-associated using position.
#
# Direction:
#   LINE 1 -> LINE 2 = OUTSIDE
#   LINE 2 -> LINE 1 = INSIDE
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\chait\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"../test videos/TEST.mp4"

TRACKER_PATH = "bytetrack_custom.yaml"

CONF = 0.30
IMGSZ = 960

# Prevent a line from being counted repeatedly because of jitter.
CROSSING_COOLDOWN = 15

# Ignore tiny side changes close to a line.
# This is measured in pixels perpendicular to the line.
LINE_DEAD_ZONE = 8

# How far a bee must move away from both lines after a count
# before it can start another journey.
SAFE_DISTANCE = 45

# Re-identification settings for ByteTrack ID changes.
# If a new YOLO track appears close to a recently lost stable bee,
# we continue the old stable bee identity.
MAX_REID_DISTANCE = 120
MAX_REID_GAP = 35


# ============================================================
# COUNTING LINES
# ============================================================

# LEFT LINE
LINE1 = ((565, 378), (562, 585))

# RIGHT LINE
LINE2 = ((740, 417), (740, 621))


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def signed_distance_to_line(point, line):
    """
    Signed perpendicular distance from a point to a line.
    Positive/negative values indicate which side of the line
    the point is on.
    """
    x, y = point
    (x1, y1), (x2, y2) = line

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        return 0.0

    cross = dx * (y - y1) - dy * (x - x1)

    return cross / length


def stable_side(point, line):
    """
    Returns:
        +1 = clearly on one side
        -1 = clearly on the other side
         0 = inside the dead-zone around the line
    """
    distance = signed_distance_to_line(point, line)

    if distance > LINE_DEAD_ZONE:
        return 1

    if distance < -LINE_DEAD_ZONE:
        return -1

    return 0


def crossed_line(previous_side, current_side):
    """
    A crossing is accepted only when the bee goes from one
    clearly defined side to the opposite clearly defined side.
    """
    return (
        previous_side != 0
        and current_side != 0
        and previous_side != current_side
    )


def distance_between(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def distance_from_both_lines(point):
    d1 = abs(signed_distance_to_line(point, LINE1))
    d2 = abs(signed_distance_to_line(point, LINE2))
    return min(d1, d2)


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ============================================================
# OUTPUT VIDEO
# ============================================================

output = cv2.VideoWriter(
    "step11_bytetrack_upgraded_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# ============================================================
# STABLE TRACK HISTORY
# ============================================================
#
# stable_tracks:
#
# stable_id -> {
#     position,
#     yolo_id,
#     last_seen,
#     state,
#     line_side_1,
#     line_side_2,
#     last_cross_frame,
#     armed
# }
#
# state:
#     NONE
#     LINE1
#     LINE2
#
# LINE1 -> LINE2 = OUTSIDE
# LINE2 -> LINE1 = INSIDE
# ============================================================

stable_tracks = {}

# Current YOLO ID -> stable bee ID
yolo_to_stable = {}

next_stable_id = 1

inside_count = 0
outside_count = 0

frame_number = 0


# ============================================================
# STABLE ID MANAGEMENT
# ============================================================

def create_stable_track(yolo_id, center, frame_number):
    global next_stable_id

    stable_id = next_stable_id
    next_stable_id += 1

    stable_tracks[stable_id] = {
        "position": center,
        "yolo_id": yolo_id,
        "last_seen": frame_number,

        # Current journey state
        "state": "NONE",

        # Last reliable side of each counting line
        "line_side_1": stable_side(center, LINE1),
        "line_side_2": stable_side(center, LINE2),

        # Prevent repeated crossing caused by jitter
        "last_cross_frame": -999,

        # After a count, wait until the bee moves away
        # from both lines before allowing a new journey.
        "armed": True,
    }

    return stable_id


def get_stable_id(yolo_id, center, frame_number, used_stable_ids):
    """
    Match a YOLO track to an existing stable bee identity.

    Priority:
    1. Existing YOLO -> stable mapping.
    2. Recently lost stable bee close to this detection.
    3. Create a new stable bee.
    """

    # --------------------------------------------------------
    # 1. Existing mapping
    # --------------------------------------------------------

    if yolo_id in yolo_to_stable:

        stable_id = yolo_to_stable[yolo_id]

        if stable_id in stable_tracks:

            track = stable_tracks[stable_id]

            if frame_number - track["last_seen"] <= MAX_REID_GAP:

                used_stable_ids.add(stable_id)
                return stable_id


        # Mapping is stale.
        del yolo_to_stable[yolo_id]


    # --------------------------------------------------------
    # 2. Re-identify a recently lost stable bee
    # --------------------------------------------------------

    best_id = None
    best_distance = MAX_REID_DISTANCE

    for stable_id, track in stable_tracks.items():

        if stable_id in used_stable_ids:
            continue

        gap = frame_number - track["last_seen"]

        if gap <= 0 or gap > MAX_REID_GAP:
            continue

        d = distance_between(
            track["position"],
            center
        )

        if d < best_distance:

            best_distance = d
            best_id = stable_id


    if best_id is not None:

        yolo_to_stable[yolo_id] = best_id

        stable_tracks[best_id]["yolo_id"] = yolo_id

        used_stable_ids.add(best_id)

        print(
            f"YOLO ID {yolo_id} reassociated "
            f"with STABLE BEE {best_id}"
        )

        return best_id


    # --------------------------------------------------------
    # 3. Completely new bee
    # --------------------------------------------------------

    stable_id = create_stable_track(
        yolo_id,
        center,
        frame_number
    )

    yolo_to_stable[yolo_id] = stable_id

    used_stable_ids.add(stable_id)

    print(
        f"NEW STABLE BEE {stable_id} "
        f"(YOLO ID {yolo_id})"
    )

    return stable_id


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_PATH,
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False
    )


    # Track IDs detected in this frame.
    used_stable_ids = set()


    # ========================================================
    # PROCESS TRACKED OBJECTS
    # ========================================================

    if results[0].boxes.id is not None:

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        ids = (
            results[0]
            .boxes
            .id
            .int()
            .cpu()
            .tolist()
        )

        confs = (
            results[0]
            .boxes
            .conf
            .cpu()
            .numpy()
        )


        # ====================================================
        # PROCESS EACH BEE
        # ====================================================

        for box, yolo_id, confidence in zip(
            boxes,
            ids,
            confs
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )


            # ------------------------------------------------
            # GET STABLE BEE ID
            # ------------------------------------------------

            stable_id = get_stable_id(
                yolo_id,
                center,
                frame_number,
                used_stable_ids
            )

            history = stable_tracks[stable_id]

            previous = history["position"]


            # ------------------------------------------------
            # DRAW DETECTION
            # ------------------------------------------------

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
                5,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                f"BEE:{stable_id} Y:{yolo_id} {confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2
            )


            # =================================================
            # CURRENT LINE SIDES
            # =================================================

            current_side_1 = stable_side(
                center,
                LINE1
            )

            current_side_2 = stable_side(
                center,
                LINE2
            )


            # =================================================
            # RE-ARM AFTER A COMPLETED JOURNEY
            # =================================================

            if (
                not history["armed"]
                and
                distance_from_both_lines(center) >= SAFE_DISTANCE
            ):

                history["armed"] = True

                print(
                    f"BEE {stable_id} re-armed"
                )


            # =================================================
            # CROSSING LINE 1
            # =================================================

            crossed_line1 = crossed_line(
                history["line_side_1"],
                current_side_1
            )


            # =================================================
            # CROSSING LINE 2
            # =================================================

            crossed_line2 = crossed_line(
                history["line_side_2"],
                current_side_2
            )


            cooldown_finished = (
                frame_number -
                history["last_cross_frame"]
                >= CROSSING_COOLDOWN
            )


            # =================================================
            # LINE 1 EVENT
            # =================================================

            if crossed_line1 and cooldown_finished:

                history["last_cross_frame"] = frame_number


                # ------------------------------------------------
                # Start journey at LINE 1
                # ------------------------------------------------

                if (
                    history["state"] == "NONE"
                    and history["armed"]
                ):

                    history["state"] = "LINE1"

                    print(
                        f"BEE {stable_id} "
                        f"started at LINE 1"
                    )


                # ------------------------------------------------
                # LINE 2 -> LINE 1 = INSIDE
                # ------------------------------------------------

                elif (
                    history["state"] == "LINE2"
                    and history["armed"]
                ):

                    inside_count += 1

                    print(
                        "================================"
                    )

                    print(
                        f"BEE {stable_id} -> INSIDE"
                    )

                    print(
                        f"INSIDE COUNT = {inside_count}"
                    )

                    print(
                        "================================"
                    )


                    # Journey completed.
                    history["state"] = "NONE"

                    # Do not allow another count until the bee
                    # moves away from both lines.
                    history["armed"] = False


            # =================================================
            # LINE 2 EVENT
            # =================================================

            if crossed_line2 and cooldown_finished:

                history["last_cross_frame"] = frame_number


                # ------------------------------------------------
                # Start journey at LINE 2
                # ------------------------------------------------

                if (
                    history["state"] == "NONE"
                    and history["armed"]
                ):

                    history["state"] = "LINE2"

                    print(
                        f"BEE {stable_id} "
                        f"started at LINE 2"
                    )


                # ------------------------------------------------
                # LINE 1 -> LINE 2 = OUTSIDE
                # ------------------------------------------------

                elif (
                    history["state"] == "LINE1"
                    and history["armed"]
                ):

                    outside_count += 1

                    print(
                        "================================"
                    )

                    print(
                        f"BEE {stable_id} -> OUTSIDE"
                    )

                    print(
                        f"OUTSIDE COUNT = {outside_count}"
                    )

                    print(
                        "================================"
                    )


                    # Journey completed.
                    history["state"] = "NONE"

                    # Require the bee to move away before
                    # another journey can start.
                    history["armed"] = False


            # =================================================
            # UPDATE TRACK HISTORY
            # =================================================

            if current_side_1 != 0:
                history["line_side_1"] = current_side_1

            if current_side_2 != 0:
                history["line_side_2"] = current_side_2

            history["position"] = center
            history["last_seen"] = frame_number
            history["yolo_id"] = yolo_id


    # ========================================================
    # CLEAN OLD YOLO -> STABLE MAPPINGS
    # ========================================================

    stale_yolo_ids = []

    for yolo_id, stable_id in yolo_to_stable.items():

        if stable_id not in stable_tracks:
            stale_yolo_ids.append(yolo_id)
            continue

        if (
            frame_number -
            stable_tracks[stable_id]["last_seen"]
            > MAX_REID_GAP
        ):
            stale_yolo_ids.append(yolo_id)


    for yolo_id in stale_yolo_ids:

        del yolo_to_stable[yolo_id]


    # ========================================================
    # DRAW COUNTING LINES
    # ========================================================

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


    # ========================================================
    # DRAW LABELS
    # ========================================================

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


    # ========================================================
    # DRAW COUNTS
    # ========================================================

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


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output.write(frame)


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Step 11 - Upgraded ByteTrack Counting",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# FINISH
# ============================================================

cap.release()
output.release()
cv2.destroyAllWindows()


print()
print("============================")
print("FINAL COUNT")
print("============================")
print("INSIDE :", inside_count)
print("OUTSIDE:", outside_count)
print("============================")
