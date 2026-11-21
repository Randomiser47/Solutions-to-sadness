#!/usr/bin/env python3
"""
lectureFaceLogger.py - record a lecture and log facial features.

Captures webcam video, extracts facial landmarks with MediaPipe's Tasks API and
saves a video (MP4) plus a time-series CSV (eye openness, mouth openness, head
pose) for later unsupervised analysis with analyze_lecture_states.py.

Everything stays on your machine. The only network access is a one-time
download of MediaPipe's face model (face_landmarker.task) on first run.
"""

import argparse
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)

FEATURE_NAMES = ["pitch_deg", "yaw_deg", "roll_deg", "left_ear", "right_ear", "mouth_ar"]
CSV_COLUMNS = ["timestamp_sec", "video_time_sec", "frame_id", "face_detected"] + FEATURE_NAMES

# ----------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices.
# "left"/"right" below mean the left/right side of the IMAGE (not the person's).
#
# Eyes, in the order the EAR formula expects:
#   [corner A, upper lid 1, upper lid 2, corner B, lower lid 2, lower lid 1]
# EAR = (|upper1 - lower1| + |upper2 - lower2|) / (2 * |cornerA - cornerB|)
# ----------------------------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth, inner lips: [left corner, right corner, upper lip, lower lip]
MOUTH = [78, 308, 13, 14]

# Six landmarks used for head pose and the matching 3D reference face.
POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]  # nose tip, chin, eye corners, mouth corners
# Reference face in OpenCV camera axes (x right, y down, z away from the camera),
# in millimetres, nose tip at the origin. A face looking straight at the camera
# therefore has the identity rotation, so all three angles read 0.
MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],      # nose tip (1)
        [0.0, 66.0, 13.0],    # chin (199)
        [-45.0, -34.0, 27.0],  # outer eye corner, image-left (33)
        [45.0, -34.0, 27.0],   # outer eye corner, image-right (263)
        [-30.0, 30.0, 25.0],   # mouth corner, image-left (61)
        [30.0, 30.0, 25.0],    # mouth corner, image-right (291)
    ],
    dtype=np.float64,
)


def get_landmark_indices():
    """Return the landmark index lists used for the features."""
    return {"left_eye": LEFT_EYE, "right_eye": RIGHT_EYE, "mouth": MOUTH}


# ----------------------------------------------------------------------
# Feature calculations (all expect landmarks in PIXEL coordinates)
# ----------------------------------------------------------------------
def eye_aspect_ratio(landmarks, eye_indices):
    """Eye aspect ratio from six landmarks: ~0.3 open, falls towards 0 when closed."""
    p1, p2, p3, p4, p5, p6 = (np.asarray(landmarks[i], dtype=np.float64)[:2] for i in eye_indices)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, mouth_indices):
    """Mouth aspect ratio (inner-lip opening / mouth width): ~0 closed, larger when open."""
    left, right, top, bottom = (np.asarray(landmarks[i], dtype=np.float64)[:2] for i in mouth_indices)
    width = np.linalg.norm(right - left)
    if width == 0:
        return 0.0
    return np.linalg.norm(bottom - top) / width


def estimate_head_pose(landmarks_px, image_size):
    """
    Estimate head pose with solvePnP.

    landmarks_px: array of landmarks in PIXEL coordinates (at least 292 of them).
    image_size:   (height, width) of the frame.
    Returns (pitch, yaw, roll) in degrees, or (nan, nan, nan) if it can't be solved.
      pitch: + looking up,  - looking down
      yaw:   + turning towards the right of the image,  - towards the left
      roll:  + head tilted clockwise in the image
    All three are ~0 when facing the camera.
    """
    nan3 = (np.nan, np.nan, np.nan)
    h, w = image_size
    if len(landmarks_px) <= max(POSE_LANDMARKS):
        return nan3
    image_points = np.asarray(landmarks_px, dtype=np.float64)[POSE_LANDMARKS, :2]

    focal_length = float(w)
    camera_matrix = np.array(
        [[focal_length, 0, w / 2.0], [0, focal_length, h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or tvec[2, 0] <= 0:  # no solution, or face "behind" the camera
        return nan3

    R, _ = cv2.Rodrigues(rvec)
    # R = Rz(roll) @ Ry(yaw) @ Rx(pitch) in camera axes
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        about_x = np.arctan2(R[2, 1], R[2, 2])
        about_y = np.arctan2(-R[2, 0], sy)
        about_z = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock (looking sideways by ~90 degrees)
        about_x = np.arctan2(-R[1, 2], R[1, 1])
        about_y = np.arctan2(-R[2, 0], sy)
        about_z = 0.0
    # Camera y points down, so flip signs to get "up" and "image-right" as positive.
    return float(-np.degrees(about_x)), float(-np.degrees(about_y)), float(np.degrees(about_z))


def extract_features(landmarks_px, image_size):
    """All per-frame features from pixel-coordinate landmarks."""
    pitch, yaw, roll = estimate_head_pose(landmarks_px, image_size)
    return {
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "roll_deg": roll,
        "left_ear": eye_aspect_ratio(landmarks_px, LEFT_EYE),
        "right_ear": eye_aspect_ratio(landmarks_px, RIGHT_EYE),
        "mouth_ar": mouth_aspect_ratio(landmarks_px, MOUTH),
    }


# ----------------------------------------------------------------------
# Model download
# ----------------------------------------------------------------------
def ensure_model(model_path):
    """Download MediaPipe's face model once, if it isn't there already."""
    if os.path.exists(model_path):
        return
    print("Downloading face_landmarker.task model (one-time download)...")
    partial = model_path + ".part"
    urllib.request.urlretrieve(MODEL_URL, partial)
    os.replace(partial, model_path)  # never leave a half-downloaded model behind
    print("Download complete.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Record a lecture with facial features (MediaPipe Tasks API)")
    parser.add_argument("--output_dir", type=str, default="./lecture_data",
                        help="Directory to save video and CSV")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second to record and log (camera frames are dropped to match)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Extract features from 1 out of N recorded frames (reduces CPU)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Where to keep face_landmarker.task (downloaded if missing)")
    parser.add_argument("--no_preview", action="store_true",
                        help="Don't open a preview window (stop with Ctrl+C instead of q)")
    args = parser.parse_args(argv)
    if args.fps < 1 or args.skip_frames < 1:
        parser.error("--fps and --skip_frames must be at least 1")

    # Imported here so the feature maths above can be used/tested without MediaPipe installed.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 1
    cap.set(cv2.CAP_PROP_FPS, args.fps)       # best effort; many cameras ignore it
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # best effort; avoids lagging behind real time

    ensure_model(args.model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model_path),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"lecture_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    video_path = os.path.join(session_dir, "video.mp4")
    csv_path = os.path.join(session_dir, "features.csv")

    show_preview = not args.no_preview
    rows = []
    out_video = None            # created on the first frame, using the real frame size
    frame_count = 0             # frames recorded (= frames written to the video)
    face_detected = False
    frame_interval = 1.0 / args.fps
    start_time = time.time()
    next_frame_time = start_time

    print(f"Recording to {session_dir}")
    print("Press 'q' in the preview window to stop." if show_preview else "Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stopped delivering frames.")
                break
            now = time.time()
            if show_preview and (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            # Cameras usually run at 30 fps: drop frames so we really record at --fps.
            # (A small tolerance stops timing jitter from skipping a frame we wanted.)
            if now < next_frame_time - 0.2 * frame_interval:
                continue
            next_frame_time = max(next_frame_time + frame_interval, now)

            if out_video is None:
                h, w = frame.shape[:2]
                out_video = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
                )
            out_video.write(frame)  # written before any overlay, so the video stays clean

            if frame_count % args.skip_frames == 0:
                features = dict.fromkeys(FEATURE_NAMES, np.nan)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
                face_detected = bool(result.face_landmarks)
                if face_detected:
                    h, w = frame.shape[:2]
                    landmarks_px = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]])
                    features = extract_features(landmarks_px, (h, w))
                rows.append({
                    "timestamp_sec": now - start_time,
                    "video_time_sec": frame_count / args.fps,
                    "frame_id": frame_count,
                    "face_detected": int(face_detected),
                    **features,
                })
            frame_count += 1

            if show_preview:
                elapsed = int(now - start_time)
                label = f"Face: {'yes' if face_detected else 'NO'}   {elapsed // 60:02d}:{elapsed % 60:02d}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Lecture Face Logger", frame)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Always save what we have, even after Ctrl+C or a crash mid-lecture.
        cap.release()
        if out_video is not None:
            out_video.release()
        if show_preview:
            cv2.destroyAllWindows()
        detector.close()
        pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)

    elapsed = time.time() - start_time
    video_len = frame_count / args.fps
    print(f"\nSaved video to {video_path}")
    print(f"Saved {len(rows)} feature rows to {csv_path}")
    print(f"Feature rate: {len(rows) / max(elapsed, 1e-9):.2f} rows/s over {elapsed:.0f}s")
    if elapsed > 5 and abs(video_len - elapsed) > 0.1 * elapsed:
        print(
            f"Warning: the video is {video_len:.0f}s long but the recording ran {elapsed:.0f}s, "
            f"so this machine could not keep up with --fps {args.fps}. Try a lower --fps or a "
            f"higher --skip_frames. Use video_time_sec (not timestamp_sec) in features.csv to "
            f"find a moment in the video."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
