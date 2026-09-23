import cv2
import math
import argparse
import numpy as np
from ultralytics import YOLO

# ============================================================
# STEP 11 - ADVANCED ROBUST HONEYBEE TRACKER & COUNTER
# ============================================================
#
# Key Features:
# 1. IoS (Intersection-over-Smaller) Containment Suppression:
#    Filters out fragmented/part detections (e.g. abdomen-only boxes)
#    before tracker ingestion, stopping track oscillation at the source.
# 2. Trajectory & Velocity Continuity:
#    Maintains centroid histories, tube-axis velocity vectors, and
#    predicted positions across multi-frame occlusions.
# 3. Directional Boundary Crossing:
#    Strictly distinguishes Left-to-Right (+1 -> -1) from
#    Right-to-Left (-1 -> +1) on each counting boundary.
# 4. Anti-Jitter Dead Zones & Minimum Tube Traversal:
#    Enforces a dead zone around both lines and requires a minimum
#    net displacement (>= 80px) across the tube interior to count.
#    Turnarounds (reversing back across the start line) are aborted
#    cleanly without false counts.
# 5. Full Multi-Trip Support:
#    A bee that travels OUTSIDE can immediately turn around and
#    travel INSIDE, and vice-versa, counting every valid crossing.
# 6. Tube Journey Continuity:
#    Even if low confidence temporarily drops a track in the tube,
#    the in-progress journey persists and re-attaches upon recovery.
# ============================================================


# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\chait\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"../test videos/TEST.mp4"
OUTPUT_VIDEO_PATH = "step11_bytetrack_upgraded_output.mp4"

# Detection settings
CONF_THRESH = 0.25
IOU_THRESH = 0.45
IOS_THRESH = 0.65       # Containment threshold to suppress sub-bee boxes
IMGSZ = 960

# Geometry & Boundaries
# LINE 1 = Hive / Left boundary ((565, 378), (562, 585))
LINE1 = ((565, 378), (562, 585))

# LINE 2 = Outside / Right boundary ((740, 417), (740, 621))
LINE2 = ((740, 417), (740, 621))

# Crossing constraints
LINE_DEAD_ZONE = 10     # Pixels perpendicular to line
MIN_TRAVEL_DIST = 80    # Minimum net displacement along tube to count journey
JOURNEY_TIMEOUT = 120   # Frames before an incomplete stalled journey expires
REARM_COOLDOWN = 30     # Frames after completing journey before new journey on same line
MAX_TRACK_GAP = 30      # Frames to maintain lost tracks in memory
MAX_REID_DIST = 130     # Maximum pixel distance to re-associate lost bee


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def signed_distance_to_line(point, line):
    """
    Signed perpendicular distance from point to line.
    For our vertical-ish tube lines:
      Positive (+) indicates Left of the line (Hive side).
      Negative (-) indicates Right of the line (Outside side).
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


def get_line_side(point, line, dead_zone=LINE_DEAD_ZONE):
    """
    Returns:
        +1: Clearly Left of the line (distance > +dead_zone)
        -1: Clearly Right of the line (distance < -dead_zone)
         0: Inside the dead-zone
    """
    dist = signed_distance_to_line(point, line)
    if dist > dead_zone:
        return 1
    elif dist < -dead_zone:
        return -1
    return 0


def filter_contained_boxes(boxes, confidences, ios_threshold=IOS_THRESH):
    """
    Suppresses bounding boxes that are substantially contained inside
    a larger, higher-confidence box (e.g. bee abdomen detected alongside whole bee).
    """
    if len(boxes) <= 1:
        return list(range(len(boxes)))

    order = sorted(range(len(boxes)), key=lambda i: confidences[i], reverse=True)
    keep = []

    for i in order:
        b = boxes[i]
        area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        suppressed = False

        for k in keep:
            kb = boxes[k]
            area_k = max(1.0, (kb[2] - kb[0]) * (kb[3] - kb[1]))

            ix1 = max(b[0], kb[0])
            iy1 = max(b[1], kb[1])
            ix2 = min(b[2], kb[2])
            iy2 = min(b[3], kb[3])

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                ios = intersection / min(area_b, area_k)
                if ios >= ios_threshold:
                    suppressed = True
                    break

        if not suppressed:
            keep.append(i)

    return sorted(keep)


# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================

def process_bee_video(
    video_path=VIDEO_PATH,
    model_path=MODEL_PATH,
    output_path=OUTPUT_VIDEO_PATH,
    max_frames=None,
    show_live=True
):
    print("==================================================")
    print("HONEYBEE DIRECTIONAL TRACKER & COUNTER INITIALIZED")
    print("==================================================")
    print(f"Model:  {model_path}")
    print(f"Video:  {video_path}")
    print(f"Output: {output_path}")
    print("==================================================")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 46.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # STATE TRACKERS & REGISTRIES
    # --------------------------------------------------------
    # tracks: track_id -> dict
    tracks = {}
    next_track_id = 1

    # Active journeys currently traveling inside the tube:
    # list of dicts: {journey_id, direction ('OUT' or 'IN'), start_frame, start_pos,
    #                 last_pos, last_frame, track_id}
    active_tube_journeys = []
    next_journey_id = 1

    inside_count = 0
    outside_count = 0

    frame_number = 0
    recent_crossing_events = [] # For on-screen notification overlay

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if max_frames and frame_number > max_frames:
            break

        # ----------------------------------------------------
        # 1. YOLO INFERENCE
        # ----------------------------------------------------
        results = model.predict(
            frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMGSZ,
            verbose=False
        )[0]

        # Extract raw detections
        raw_boxes = []
        raw_confs = []
        if len(results.boxes) > 0:
            raw_boxes = results.boxes.xyxy.cpu().numpy()
            raw_confs = results.boxes.conf.cpu().numpy()

        # Apply IoS containment filter to remove sub-box duplicates
        valid_indices = filter_contained_boxes(raw_boxes, raw_confs, IOS_THRESH)

        detections = []
        for idx in valid_indices:
            box = raw_boxes[idx]
            conf = raw_confs[idx]
            center = (
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0
            )
            detections.append({
                "box": box,
                "conf": conf,
                "center": center
            })

        # ----------------------------------------------------
        # 2. ASSOCIATE DETECTIONS WITH TRACKS
        # ----------------------------------------------------
        # Active tracks within MAX_TRACK_GAP
        active_track_ids = [
            tid for tid, t in tracks.items()
            if frame_number - t["last_seen"] <= MAX_TRACK_GAP
        ]

        matched_dets = set()
        matched_tracks = set()

        if active_track_ids and detections:
            cost_matrix = np.full((len(active_track_ids), len(detections)), 1e6)

            for i, tid in enumerate(active_track_ids):
                t = tracks[tid]
                gap = frame_number - t["last_seen"]

                # Position prediction using smoothed velocity
                pred_x = t["pos"][0] + t["vel"][0] * gap
                pred_y = t["pos"][1] + t["vel"][1] * gap

                for j, d in enumerate(detections):
                    dist_pred = math.hypot(d["center"][0] - pred_x, d["center"][1] - pred_y)
                    dist_last = math.hypot(d["center"][0] - t["pos"][0], d["center"][1] - t["pos"][1])
                    eff_dist = min(dist_pred, dist_last)

                    max_gate = 45 + 5 * gap
                    if eff_dist < max_gate:
                        cost_matrix[i, j] = eff_dist

            # Greedy Hungarian-style matching
            while True:
                min_val = cost_matrix.min()
                if min_val >= 1e5:
                    break

                i, j = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
                tid = active_track_ids[i]
                matched_tracks.add(tid)
                matched_dets.add(j)
                cost_matrix[i, :] = 1e6
                cost_matrix[:, j] = 1e6

                # Update track with matched detection
                t = tracks[tid]
                d = detections[j]
                gap = frame_number - t["last_seen"]
                pos = d["center"]

                # Smoothed velocity
                vx = (pos[0] - t["pos"][0]) / max(gap, 1)
                vy = (pos[1] - t["pos"][1]) / max(gap, 1)
                t["vel"] = (0.7 * t["vel"][0] + 0.3 * vx, 0.7 * t["vel"][1] + 0.3 * vy)

                t["pos"] = pos
                t["box"] = d["box"]
                t["conf"] = d["conf"]
                t["last_seen"] = frame_number
                t["history"].append(pos)
                if len(t["history"]) > 60:
                    t["history"].pop(0)

        # ----------------------------------------------------
        # 3. UNMATCHED DETECTIONS: RECONNECT OR CREATE TRACK
        # ----------------------------------------------------
        for j, d in enumerate(detections):
            if j in matched_dets:
                continue

            pos = d["center"]

            # First priority: check if this detection can re-attach to an orphaned active tube journey
            best_journey = None
            best_j_dist = MAX_REID_DIST

            for j_info in active_tube_journeys:
                if j_info["track_id"] not in matched_tracks:
                    gap = frame_number - j_info["last_frame"]
                    dist = math.hypot(pos[0] - j_info["last_pos"][0], pos[1] - j_info["last_pos"][1])
                    if gap <= 60 and dist < best_j_dist:
                        # Verify directional motion consistency
                        if j_info["direction"] == "OUT" and pos[0] >= j_info["last_pos"][0] - 25:
                            best_j_dist = dist
                            best_journey = j_info
                        elif j_info["direction"] == "IN" and pos[0] <= j_info["last_pos"][0] + 25:
                            best_j_dist = dist
                            best_journey = j_info

            # Second priority: check if can re-identify a recently lost track without journey
            reid_tid = None
            if best_journey is None:
                best_t_dist = MAX_REID_DIST
                for tid, t in tracks.items():
                    if tid in matched_tracks:
                        continue
                    gap = frame_number - t["last_seen"]
                    if 0 < gap <= 45:
                        dist = math.hypot(pos[0] - t["pos"][0], pos[1] - t["pos"][1])
                        if dist < best_t_dist:
                            best_t_dist = dist
                            reid_tid = tid

            if reid_tid is not None:
                # Re-associated with existing lost track
                tid = reid_tid
                t = tracks[tid]
                t["pos"] = pos
                t["box"] = d["box"]
                t["conf"] = d["conf"]
                t["last_seen"] = frame_number
                t["history"].append(pos)
                matched_tracks.add(tid)
            else:
                # Create a new track
                tid = next_track_id
                next_track_id += 1
                s1 = get_line_side(pos, LINE1)
                s2 = get_line_side(pos, LINE2)

                tracks[tid] = {
                    "id": tid,
                    "pos": pos,
                    "box": d["box"],
                    "conf": d["conf"],
                    "vel": (0.0, 0.0),
                    "last_seen": frame_number,
                    "history": [pos],
                    "side1": s1,
                    "side2": s2,
                    "journey": None,
                    "last_count_frame": -999
                }
                matched_tracks.add(tid)

                if best_journey is not None:
                    best_journey["track_id"] = tid
                    tracks[tid]["journey"] = best_journey
                    print(f"[Frame {frame_number}] Bee {tid} reconnected to in-flight Journey {best_journey['journey_id']} ({best_journey['direction']})")

        # ----------------------------------------------------
        # 4. DIRECTIONAL BOUNDARY CROSSING & STATE MACHINE
        # ----------------------------------------------------
        for tid in matched_tracks:
            t = tracks[tid]
            pos = t["pos"]
            cur_s1 = get_line_side(pos, LINE1)
            cur_s2 = get_line_side(pos, LINE2)

            # Keep attached journey last known position updated
            if t["journey"] is not None:
                t["journey"]["last_pos"] = pos
                t["journey"]["last_frame"] = frame_number

            # --- LINE 1 BOUNDARY EVALUATION ---
            if t["side1"] != 0 and cur_s1 != 0 and cur_s1 != t["side1"]:
                prev_s1 = t["side1"]
                t["side1"] = cur_s1

                # Left to Right (+1 -> -1): Bee enters the tube for an OUTSIDE journey
                if prev_s1 == 1 and cur_s1 == -1:
                    if t["journey"] is None and (frame_number - t["last_count_frame"] > REARM_COOLDOWN):
                        j_info = {
                            "journey_id": next_journey_id,
                            "direction": "OUT",
                            "start_frame": frame_number,
                            "start_pos": pos,
                            "last_pos": pos,
                            "last_frame": frame_number,
                            "track_id": tid
                        }
                        next_journey_id += 1
                        active_tube_journeys.append(j_info)
                        t["journey"] = j_info
                        print(f"[Frame {frame_number}] Bee {tid} STARTED OUTSIDE journey {j_info['journey_id']} at LINE 1 (L->R)")

                # Right to Left (-1 -> +1): Bee completes an INSIDE journey or aborts OUTSIDE
                elif prev_s1 == -1 and cur_s1 == 1:
                    if t["journey"] is not None and t["journey"]["direction"] == "IN":
                        j_info = t["journey"]
                        travel = abs(pos[0] - j_info["start_pos"][0])
                        if travel >= MIN_TRAVEL_DIST:
                            inside_count += 1
                            t["last_count_frame"] = frame_number
                            event_msg = f"Bee {tid} -> INSIDE +1 (travel={travel:.0f}px)"
                            recent_crossing_events.append((frame_number, event_msg, (0, 255, 0)))
                            print("================================")
                            print(f"[Frame {frame_number}] {event_msg}")
                            print(f"INSIDE COUNT = {inside_count}")
                            print("================================")
                        if j_info in active_tube_journeys:
                            active_tube_journeys.remove(j_info)
                        t["journey"] = None

                    elif t["journey"] is not None and t["journey"]["direction"] == "OUT":
                        # Bee turned back towards hive before reaching LINE 2
                        print(f"[Frame {frame_number}] Bee {tid} turned back across LINE 1 (R->L), cleanly aborting OUTSIDE journey")
                        if t["journey"] in active_tube_journeys:
                            active_tube_journeys.remove(t["journey"])
                        t["journey"] = None

            elif cur_s1 != 0:
                t["side1"] = cur_s1

            # --- LINE 2 BOUNDARY EVALUATION ---
            if t["side2"] != 0 and cur_s2 != 0 and cur_s2 != t["side2"]:
                prev_s2 = t["side2"]
                t["side2"] = cur_s2

                # Right to Left (-1 -> +1): Bee enters the tube for an INSIDE journey
                if prev_s2 == -1 and cur_s2 == 1:
                    if t["journey"] is None and (frame_number - t["last_count_frame"] > REARM_COOLDOWN):
                        j_info = {
                            "journey_id": next_journey_id,
                            "direction": "IN",
                            "start_frame": frame_number,
                            "start_pos": pos,
                            "last_pos": pos,
                            "last_frame": frame_number,
                            "track_id": tid
                        }
                        next_journey_id += 1
                        active_tube_journeys.append(j_info)
                        t["journey"] = j_info
                        print(f"[Frame {frame_number}] Bee {tid} STARTED INSIDE journey {j_info['journey_id']} at LINE 2 (R->L)")

                # Left to Right (+1 -> -1): Bee completes an OUTSIDE journey or aborts INSIDE
                elif prev_s2 == 1 and cur_s2 == -1:
                    if t["journey"] is not None and t["journey"]["direction"] == "OUT":
                        j_info = t["journey"]
                        travel = abs(pos[0] - j_info["start_pos"][0])
                        if travel >= MIN_TRAVEL_DIST:
                            outside_count += 1
                            t["last_count_frame"] = frame_number
                            event_msg = f"Bee {tid} -> OUTSIDE +1 (travel={travel:.0f}px)"
                            recent_crossing_events.append((frame_number, event_msg, (0, 0, 255)))
                            print("================================")
                            print(f"[Frame {frame_number}] {event_msg}")
                            print(f"OUTSIDE COUNT = {outside_count}")
                            print("================================")
                        if j_info in active_tube_journeys:
                            active_tube_journeys.remove(j_info)
                        t["journey"] = None

                    elif t["journey"] is not None and t["journey"]["direction"] == "IN":
                        # Bee turned back towards outside world before reaching LINE 1
                        print(f"[Frame {frame_number}] Bee {tid} turned back across LINE 2 (L->R), cleanly aborting INSIDE journey")
                        if t["journey"] in active_tube_journeys:
                            active_tube_journeys.remove(t["journey"])
                        t["journey"] = None

            elif cur_s2 != 0:
                t["side2"] = cur_s2

        # ----------------------------------------------------
        # 5. EXPIRATION OF STALLED JOURNEYS
        # ----------------------------------------------------
        stale_journeys = [
            j for j in active_tube_journeys
            if frame_number - j["last_frame"] > JOURNEY_TIMEOUT
        ]
        for j in stale_journeys:
            print(f"[Frame {frame_number}] Tube journey {j['journey_id']} ({j['direction']}) timed out without completing")
            active_tube_journeys.remove(j)

        # ----------------------------------------------------
        # 6. VISUAL DEBUGGING OVERLAY & ANNOTATIONS
        # ----------------------------------------------------
        # Draw Counting Boundaries
        cv2.line(frame, LINE1[0], LINE1[1], (0, 255, 255), 3)
        cv2.line(frame, LINE2[0], LINE2[1], (0, 165, 255), 3)

        cv2.putText(
            frame, "LINE 1 (HIVE)", (LINE1[0][0] - 80, LINE1[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )
        cv2.putText(
            frame, "LINE 2 (OUTSIDE)", (LINE2[0][0] - 20, LINE2[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2
        )

        # Draw Tracks
        for tid in matched_tracks:
            t = tracks[tid]
            box = t.get("box", None)
            if box is None:
                continue

            x1, y1, x2, y2 = map(int, box)
            center = (int(t["pos"][0]), int(t["pos"][1]))

            # Determine color and status badge based on journey state
            if t["journey"] is not None:
                if t["journey"]["direction"] == "OUT":
                    color = (0, 0, 255) # Red for outside journey
                    state_text = f"BEE {tid}: GOING OUT"
                else:
                    color = (0, 255, 0) # Green for inside journey
                    state_text = f"BEE {tid}: GOING IN"
            else:
                color = (255, 200, 0)
                state_text = f"BEE {tid}: IDLE"

            # Draw bounding box & center point
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 4, (0, 255, 255), -1)

            # Draw trajectory trail
            if len(t["history"]) > 1:
                pts = np.array(t["history"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 2)

            # Draw track label
            cv2.putText(
                frame,
                f"{state_text} ({t.get('conf', 0.0):.2f})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # Draw Header Dashboard Banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (450, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (20, 20), (450, 170), (255, 255, 255), 1)

        cv2.putText(
            frame, f"HONEYBEE TUBE COUNTER", (35, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"INSIDE  (INTO TUBE): {inside_count}", (35, 88),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"OUTSIDE (OUT OF TUBE): {outside_count}", (35, 126),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2
        )
        cv2.putText(
            frame, f"Frame: {frame_number}/{total_video_frames} | Active in Tube: {len(active_tube_journeys)}", (35, 155),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1
        )

        # Draw recent crossing notifications
        recent_crossing_events = [
            (f_num, msg, col) for f_num, msg, col in recent_crossing_events
            if frame_number - f_num <= 45
        ]
        y_offset = 205
        for f_num, msg, col in recent_crossing_events:
            cv2.putText(
                frame, f">> {msg}", (30, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2
            )
            y_offset += 28

        # Write frame to output video
        writer.write(frame)

        if show_live:
            cv2.imshow("Honeybee Tube Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Interrupted by user keypress 'q'.")
                break

    # --------------------------------------------------------
    # FINISH & SUMMARY
    # --------------------------------------------------------
    cap.release()
    writer.release()
    if show_live:
        cv2.destroyAllWindows()

    print()
    print("============================")
    print("FINAL COUNT")
    print("============================")
    print(f"INSIDE : {inside_count}")
    print(f"OUTSIDE: {outside_count}")
    print("============================")
    print(f"Annotated output saved to: {output_path}")
    print(f"Processed frames: {frame_number}")

    return inside_count, outside_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honeybee Tube Counter")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, help="Path to video file")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to trained YOLO model")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to output annotated video")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process (for testing)")
    parser.add_argument("--no_show", action="store_true", help="Disable cv2.imshow live window")

    args = parser.parse_args()

    process_bee_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        max_frames=args.max_frames,
        show_live=not args.no_show
    )
