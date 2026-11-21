#!/usr/bin/env python3
"""
Webcam-free self-checks for LectureFaceLogger.

    python test_features.py        (or: pytest test_features.py)

Covers the feature maths (head pose, EAR, MAR), the analysis pipeline on
synthetic data, and the recorder loop with a fake camera and fake detector.
"""

import glob
import os
import sys
import tempfile
import time
import types

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import analyze_lecture_states as als  # noqa: E402
import lectureFaceLogger as lfl  # noqa: E402


# ---------------------------------------------------------------- helpers
def rotation(pitch_up, yaw_right, roll_cw):
    """Camera-axes rotation for the documented sign conventions."""
    a, b, g = np.radians([-pitch_up, -yaw_right, roll_cw])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    return rz @ ry @ rx


def pose_landmarks(R, translation, width, height):
    """478 landmarks (pixels) where only the six pose landmarks are meaningful."""
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], float)
    pts = (R @ lfl.MODEL_POINTS.T).T + np.asarray(translation, float)
    uv = (camera @ pts.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    lm = np.zeros((478, 2))
    lm[lfl.POSE_LANDMARKS] = uv
    return lm


def synthetic_recording(minutes=12, fps=15, gap=(360, 450), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, minutes * 60, 1 / fps)
    n = len(t)
    state = (t // 120).astype(int) % 3
    df = pd.DataFrame({
        "timestamp_sec": t, "frame_id": np.arange(n), "face_detected": 1,
        "pitch_deg": np.array([-2, 5, -1])[state] + rng.normal(0, 1.5, n),
        "yaw_deg": rng.normal(0, 3, n),
        "roll_deg": rng.normal(0, 2, n),
        "left_ear": np.array([0.32, 0.18, 0.28])[state] + rng.normal(0, 0.02, n),
        "right_ear": np.array([0.32, 0.18, 0.28])[state] + rng.normal(0, 0.02, n),
        "mouth_ar": np.array([0.08, 0.10, 0.21])[state] + rng.normal(0, 0.01, n),
    })
    away = (df.timestamp_sec > gap[0]) & (df.timestamp_sec < gap[1])
    df.loc[away, "face_detected"] = 0
    df.loc[away, als.METRICS] = np.nan
    return df


# ------------------------------------------------------------------ tests
def test_head_pose_recovers_known_angles():
    for (w, h) in [(640, 480), (1280, 720)]:
        for pitch, yaw, roll in [(0, 0, 0), (10, 20, 5), (-15, -25, 10), (25, 0, -15), (0, 30, 0)]:
            lm = pose_landmarks(rotation(pitch, yaw, roll), (40, -20, 650), w, h)
            got = lfl.estimate_head_pose(lm, (h, w))
            assert np.allclose(got, (pitch, yaw, roll), atol=1.0), (w, h, (pitch, yaw, roll), got)


def test_head_pose_handles_bad_input():
    assert np.isnan(lfl.estimate_head_pose(np.zeros((10, 2)), (480, 640))).all()


def test_eye_aspect_ratio_open_vs_closed():
    def eye(half_opening):
        lm = np.zeros((478, 2))
        # corners 30 px apart, lids +/- half_opening around the middle
        lm[33], lm[133] = (0, 0), (30, 0)
        lm[160], lm[158] = (10, -half_opening), (20, -half_opening)
        lm[144], lm[153] = (10, half_opening), (20, half_opening)
        return lfl.eye_aspect_ratio(lm, lfl.LEFT_EYE)
    assert abs(eye(5) - 2 * 5 / 30) < 1e-9
    assert eye(5) > 0.25 > 0.2 > eye(1)


def test_mouth_aspect_ratio():
    lm = np.zeros((478, 2))
    lm[78], lm[308], lm[13], lm[14] = (0, 0), (50, 0), (25, -3), (25, 3)
    assert abs(lfl.mouth_aspect_ratio(lm, lfl.MOUTH) - 6 / 50) < 1e-9


def test_blink_counting():
    df = pd.DataFrame({"timestamp_sec": np.arange(0, 40, 1 / 15)})
    df["frame_id"], df["face_detected"] = np.arange(len(df)), 1
    for col in als.METRICS:
        df[col] = 0.3
    for start in (50, 150, 300):                       # three 3-frame blinks
        df.loc[start:start + 2, ["left_ear", "right_ear"]] = 0.1
    windows = als.window_features(df, window_sec=30, step_sec=15)
    assert windows.loc[0, "blink_count"] == 3


def test_load_data_fills_short_gaps_only():
    df = synthetic_recording()
    short = (df.timestamp_sec > 100) & (df.timestamp_sec < 100.5)   # ~0.5 s: should be filled
    df.loc[short, "face_detected"] = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "features.csv")
        df.to_csv(path, index=False)
        loaded = als.load_data(path)
    assert loaded.loc[loaded.timestamp_sec.between(100, 100.5), "left_ear"].notna().all()
    assert loaded.loc[loaded.timestamp_sec.between(380, 430), "left_ear"].isna().all()


def test_analysis_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "features.csv")
        synthetic_recording().to_csv(path, index=False)
        features = als.window_features(als.load_data(path))
        assert len(features) > als.MIN_WINDOWS

        for method in ("hdbscan", "kmeans"):
            labels, _ = als.cluster_windows(features, method=method, n_clusters=4)
            assert len(labels) == len(features)

        # HDBSCAN should separate the three behavioural states
        labels, _ = als.cluster_windows(features, method="hdbscan")
        modes = []
        for s in range(3):
            inside = (
                ((features.window_start // 120) % 3 == s)
                & (features.window_start % 120 + 30 <= 120)
                & (features.face_missing_frac == 0)
            )
            chosen = pd.Series(labels[inside.values])
            assert len(chosen) >= 3
            mode = chosen.mode().iloc[0]
            assert mode != -1 and (chosen == mode).mean() >= 0.9, (s, chosen.tolist())
            modes.append(mode)
        assert len(set(modes)) == 3

        # leaving the camera must always be flagged
        anomalies = als.detect_anomalies(features)
        away = features.face_missing_frac.values >= 0.5
        assert away.sum() >= 5 and (anomalies[away] == -1).all()

        png = os.path.join(tmp, "timeline.png")
        als.plot_timeline(features.assign(cluster=labels), labels, anomalies, output_image=png, show=False)
        assert os.path.getsize(png) > 10_000


def test_recorder_with_fake_camera_and_detector():
    width, height = 640, 480
    landmarks = pose_landmarks(rotation(10, 15, 5), (0, 0, 650), width, height)
    # eyes and mouth: put the extra points at known offsets from the pose landmarks
    base_l, base_r = landmarks[33].copy(), landmarks[263].copy()
    for idx, off in {133: (30, 0), 160: (10, -5), 158: (20, -5), 144: (10, 5), 153: (20, 5)}.items():
        landmarks[idx] = base_l + off
    for idx, off in {362: (-30, 0), 385: (-20, -5), 387: (-10, -5), 380: (-20, 5), 373: (-10, 5)}.items():
        landmarks[idx] = base_r + off
    landmarks[78], landmarks[308] = landmarks[61] + (5, 0), landmarks[291] - (5, 0)
    mid = (landmarks[78] + landmarks[308]) / 2
    landmarks[13], landmarks[14] = mid - (0, 3), mid + (0, 3)
    expected_mar = 6 / np.linalg.norm(landmarks[308] - landmarks[78])

    class FakeCapture:
        def __init__(self, *_a, **_k):
            self.frames = 0
        def isOpened(self): return True
        def set(self, *_a): return True
        def read(self):
            if self.frames >= 60:                      # 2 s of a 30 fps camera
                return False, None
            time.sleep(1 / 30)
            self.frames += 1
            return True, np.full((height, width, 3), 127, np.uint8)
        def release(self): pass

    class FakeDetector:
        def detect(self, _image):
            face = [types.SimpleNamespace(x=u / width, y=v / height) for u, v in landmarks]
            return types.SimpleNamespace(face_landmarks=[face])
        def close(self): pass

    mp = types.ModuleType("mediapipe")
    mp.Image = lambda image_format, data: data
    mp.ImageFormat = types.SimpleNamespace(SRGB=1)
    tasks, python, vision = (types.ModuleType(n) for n in
                             ("mediapipe.tasks", "mediapipe.tasks.python", "mediapipe.tasks.python.vision"))
    python.BaseOptions = lambda **kw: types.SimpleNamespace(**kw)
    vision.FaceLandmarkerOptions = lambda **kw: types.SimpleNamespace(**kw)
    vision.FaceLandmarker = types.SimpleNamespace(create_from_options=lambda _o: FakeDetector())
    mp.tasks, tasks.python, python.vision = tasks, python, vision
    fakes = {"mediapipe": mp, "mediapipe.tasks": tasks,
             "mediapipe.tasks.python": python, "mediapipe.tasks.python.vision": vision}

    real_capture, saved_modules = cv2.VideoCapture, {k: sys.modules.get(k) for k in fakes}
    cv2.VideoCapture = FakeCapture
    sys.modules.update(fakes)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "face_landmarker.task")
            open(model, "wb").close()                  # pretend the model is already downloaded
            code = lfl.main(["--output_dir", tmp, "--fps", "10", "--no_preview", "--model_path", model])
            assert code == 0
            session = glob.glob(os.path.join(tmp, "lecture_*"))[0]
            data = pd.read_csv(os.path.join(session, "features.csv"))
            assert list(data.columns) == lfl.CSV_COLUMNS
            assert 12 <= len(data) <= 28, len(data)    # 30 fps camera recorded at ~10 fps for 2 s
            assert (data.face_detected == 1).all()
            assert np.allclose(data.left_ear, 1 / 3, atol=0.01)
            assert np.allclose(data.right_ear, 1 / 3, atol=0.01)
            assert np.allclose(data.mouth_ar, expected_mar, atol=0.005)
            assert np.allclose(data[["pitch_deg", "yaw_deg", "roll_deg"]].iloc[0], (10, 15, 5), atol=1.0)
            assert abs(data.video_time_sec.iloc[-1] - data.timestamp_sec.iloc[-1]) < 0.5
            assert os.path.getsize(os.path.join(session, "video.mp4")) > 0
    finally:
        cv2.VideoCapture = real_capture
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
