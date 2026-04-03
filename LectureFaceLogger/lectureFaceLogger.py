#!/usr/bin/env python3
"""
LectureFaceLogger.py – Stable MediaPipe Tasks API Version
Captures webcam video, extracts facial landmarks using MediaPipe's Tasks API,
saves video (MP4) and a CSV of features for later unsupervised analysis.
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
import os
import urllib.request

# The standard, reliable import for MediaPipe
import mediapipe as mp

# Import specific components from the tasks API for cleaner code
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------
# Helper functions (feature calculations - unchanged)
# ----------------------------------------------------------------------
def get_landmark_indices():
    """Return dictionary of relevant landmark indices."""
    left_eye = [33, 133, 157, 158, 159, 160, 161, 173]
    right_eye = [362, 263, 387, 386, 385, 384, 398, 466]
    mouth = [61, 291, 0, 17, 269, 405]
    return {'left_eye': left_eye, 'right_eye': right_eye, 'mouth': mouth}
def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate EAR from landmark coordinates."""
    if len(eye_indices) >= 4:
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        vert1 = np.linalg.norm(p3 - p2)
        vert2 = np.linalg.norm(p4 - p2)
        horiz = np.linalg.norm(p2 - p1)
        if horiz == 0:
            return 0.0
        return (vert1 + vert2) / (2.0 * horiz)
    return 0.0

def mouth_aspect_ratio(landmarks, mouth_indices):
    """Calculate MAR (height/width)."""
    if len(mouth_indices) >= 4:
        left = landmarks[mouth_indices[0]]
        right = landmarks[mouth_indices[1]]
        top = landmarks[mouth_indices[2]]
        bottom = landmarks[mouth_indices[3]]
        width = np.linalg.norm(right - left)
        height = np.linalg.norm(bottom - top)
        if width == 0:
            return 0.0
        return height / width
    return 0.0

def estimate_head_pose(landmarks, image_size):
    """
    Estimate head pose (pitch, yaw, roll) using solvePnP.
    Returns (pitch, yaw, roll) in degrees.
    """
    # 3D model points of a generic face (in mm)
    model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -63.0, -12.0],      # Chin
        [-30.0, 28.0, -30.0],     # Left eye corner
        [30.0, 28.0, -30.0],      # Right eye corner
        [-25.0, 12.0, -25.0],     # Left mouth corner
        [25.0, 12.0, -25.0]       # Right mouth corner
    ], dtype=np.float64)

    # Corresponding landmark indices in MediaPipe
    idx_map = [1, 199, 33, 263, 61, 291]
    h, w = image_size

    image_points = []
    for idx in idx_map:
        if idx >= len(landmarks):
            return 0.0, 0.0, 0.0
        x = landmarks[idx][0] * w
        y = landmarks[idx][1] * h
        image_points.append([x, y])
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(-R[1,2], R[1,1])
        roll = 0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Record lecture with facial features (Tasks API)")
    parser.add_argument("--output_dir", type=str, default="./lecture_data",
                        help="Directory to save video and CSV")
    parser.add_argument("--fps", type=int, default=15,
                        help="Target frames per second")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Process 1 out of N frames (reduce CPU)")
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, f"lecture_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    video_path = os.path.join(session_dir, "video.mp4")
    csv_path = os.path.join(session_dir, "features.csv")

    # ------------------------------------------------------------------
    # Initialize FaceLandmarker (Tasks API)
    # ------------------------------------------------------------------
    # Download the model file if it doesn't exist
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading face_landmarker.task model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

    # Create the FaceLandmarker options and object
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Webcam and video writer setup
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))

    data_rows = []
    landmark_indices = get_landmark_indices()
    frame_count = 0
    process_count = 0
    start_time = time.time()

    print(f"Recording to {session_dir}")
    print("Press 'q' to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_video.write(frame)

        # Process every 'skip_frames' frame for feature extraction
        if frame_count % args.skip_frames == 0:
            # Convert the frame toj RGB and then to a MediaPipe Image object
            # This is the correct, stable way to create an image from a numpy array
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform face landmark detection
            detection_result = detector.detect(mp_image)

            # Default values if no face detected
            pitch = yaw = roll = left_ear = right_ear = mouth_ar = 0.0
            face_detected = False

            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                face_landmarks = detection_result.face_landmarks[0]
                h, w = frame.shape[:2]

                # Convert normalized landmarks to pixel coordinates
                landmarks_px = []
                for lm in face_landmarks:
                    landmarks_px.append(np.array([lm.x * w, lm.y * h]))
                landmarks_px = np.array(landmarks_px)

                # Calculate features
                left_ear = eye_aspect_ratio(landmarks_px, landmark_indices['left_eye'])
                right_ear = eye_aspect_ratio(landmarks_px, landmark_indices['right_eye'])
                mouth_ar = mouth_aspect_ratio(landmarks_px, landmark_indices['mouth'])
                pitch, yaw, roll = estimate_head_pose(landmarks_px, (h, w))
                face_detected = True

            elapsed = time.time() - start_time
            data_rows.append({
                'timestamp_sec': elapsed,
                'frame_id': frame_count,
                'face_detected': int(face_detected),
                'pitch_deg': pitch,
                'yaw_deg': yaw,
                'roll_deg': roll,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'mouth_ar': mouth_ar
            })
            process_count += 1

        frame_count += 1

        # Display the frame with basic info
        if frame_count % 30 == 0:
            cv2.putText(frame, f"Face: {face_detected}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Lecture Face Logger", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    detector.close()

    # Save the data to CSV
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved video to {video_path}")
    print(f"Saved {len(data_rows)} feature rows to {csv_path}")
    print(f"Average processing rate: {process_count / (time.time() - start_time):.2f} fps")

if __name__ == "__main__":
    main()
