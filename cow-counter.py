import cv2
from ultralytics import YOLOv10
import numpy as np
from collections import defaultdict
import time
import argparse
import os

# --------------------------------------------------
# Command Line Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="yolov10m.pt",
    help="YOLOv10 model path"
)

parser.add_argument(
    "--video",
    type=str,
    default="inference/cow.mp4",
    help="Video path or webcam index"
)

parser.add_argument(
    "--conf",
    type=float,
    default=0.25,
    help="Confidence threshold"
)

parser.add_argument(
    "--save",
    action="store_true",
    help="Save output video"
)

parser.add_argument(
    "--project",
    type=str,
    default="results",
    help="Output directory"
)

args = parser.parse_args()


# --------------------------------------------------
# FPS Display
# --------------------------------------------------
def show_fps(frame, fps):
    x, y, w, h = 10, 10, 350, 50

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 52),
        cv2.FONT_HERSHEY_PLAIN,
        3.5,
        (0, 255, 0),
        3
    )


# --------------------------------------------------
# Counter Display
# --------------------------------------------------
def show_counter(frame, counter):
    x, y, w, h = 535, 10, 250, 60

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (253, 110, 13),
        -1
    )

    cv2.putText(
        frame,
        f"{counter} Cow",
        (537, 60),
        cv2.FONT_HERSHEY_PLAIN,
        4,
        (255, 255, 255),
        5
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    video_input = args.video

    if video_input.isdigit():
        video_input = int(video_input)

    cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        exit()

    conf_thres = args.conf

    model = YOLOv10(args.model)

    track_history = defaultdict(list)

    start_time = time.time()

    # --------------------------------------------------
    # Video Writer Setup
    # --------------------------------------------------
    out = None

    if args.save:
        os.makedirs(args.project, exist_ok=True)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps_video = cap.get(cv2.CAP_PROP_FPS)

        if fps_video <= 0:
            fps_video = 30

        output_path = os.path.join(
            args.project,
            "cow_tracking.mp4"
        )

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_video,
            (width, height)
        )

        print(f"Saving output to: {output_path}")

    # --------------------------------------------------
    # Count Line
    # --------------------------------------------------
    count_line = {
        "x1": 190,
        "y1": 412,
        "x2": 966,
        "y2": 412,
    }

    offset = 80

    tracked_cow_ids = []
    cow_count = 0

    # --------------------------------------------------
    # Processing Loop
    # --------------------------------------------------
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # Draw counting line
        cv2.line(
            frame,
            (count_line["x1"], count_line["y1"]),
            (count_line["x2"], count_line["y2"]),
            (0, 127, 255),
            3
        )

        # Tracking
        results = model.track(
            frame,
            classes=19,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_thres
        )

        annotated_frame = results[0].plot()

        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes.id is not None:

            track_ids = results[0].boxes.id.int().cpu().tolist()

            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(
                boxes,
                track_ids,
                class_ids
            ):

                x, y, w, h = box

                track = track_history[track_id]

                track.append((float(x), float(y)))

                if len(track) > 90:
                    track.pop(0)

                points = np.hstack(track).astype(
                    np.int32
                ).reshape((-1, 1, 2))

                cv2.polylines(
                    annotated_frame,
                    [points],
                    False,
                    (255, 0, 0),
                    3
                )

                # ------------------------------------------
                # Counting Logic
                # ------------------------------------------
                bottom_y = y + (h / 2)

                if count_line["y1"] <= int(bottom_y) <= count_line["y1"] + offset:

                    if track_id not in tracked_cow_ids:
                        cow_count += 1
                        tracked_cow_ids.append(track_id)

        # Draw count line again on annotated frame
        cv2.line(
            annotated_frame,
            (count_line["x1"], count_line["y1"]),
            (count_line["x2"], count_line["y2"]),
            (0, 127, 255),
            3
        )

        # Counter
        show_counter(
            annotated_frame,
            cow_count
        )

        # FPS
        end_time = time.time()

        fps = 1 / max(end_time - start_time, 0.0001)

        start_time = end_time

        show_fps(
            annotated_frame,
            round(fps, 2)
        )

        # Save frame
        if out is not None:
            out.write(annotated_frame)

        # Display
        cv2.imshow(
            "YOLOv10 Cow Tracking",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    cap.release()

    if out is not None:
        out.release()

    cv2.destroyAllWindows()

    print(f"Final Cow Count: {cow_count}")