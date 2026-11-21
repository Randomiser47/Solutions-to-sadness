# SADNESS STRUCK AGAIN
## and we made this

**Privacy‑first, unsupervised analysis of your own attention, energy, and focus during lectures.**

This system records your webcam during a lecture, extracts facial features (eye openness, mouth movement, head pose) using MediaPipe, and then uses unsupervised machine learning (clustering, anomaly detection) to discover natural patterns in your behaviour – without any manual labelling or pre‑trained emotion models.

Built for **2e / ADHD** brains who learn by building and want personalised, local‑first tools.

## Why this exists


After another lecture where I zoned out for 20 minutes and didn't notice until the end, I got sad. Then angry. Then I thought: if companies can mine my face for profit, why can't I mine my own face for self-understanding? So I built this.
No corporate dashboards. No "wellness" emails. Just raw data about *my* brain, in *my* hands, interpreted by *me*.

---

## 🔒 Your data, your machine

**This tool has zero telemetry, zero cloud uploads, zero hidden analytics.**

- Every frame, every landmark, every CSV stays on your computer.
- The only network request is a one-time download of Google's public MediaPipe face model (`face_landmarker.task`) on first run. Nothing of yours is ever uploaded.
- Recordings and CSVs are git-ignored by default (see `.gitignore`), so you can't accidentally commit your face.
- No account, no login, no "share anonymous usage stats".
- The only "cloud" here is the one you stare at during a boring lecture.

Corps take your face data to sell you things. You take your face data to understand yourself. That's the trade.

## What it does

1. **Capture** – Records video and saves facial landmarks as a time‑series CSV.
2. **Analyse** – Slides a window over the data, computes statistics, then clusters the windows to reveal hidden states (e.g., high focus, low energy, talking/engaged).
3. **Visualise** – Produces a colour‑coded timeline of your behavioural states across the lecture.
4. **Anomaly detection** – Flags unusual moments (e.g., leaving the camera, sudden distraction).

No cloud, no manual labels, no biased emotion classifiers.

---

## How it works

- **Feature extraction** – MediaPipe Face Landmarker (Tasks API) gives 478 3D landmarks per frame (468 face + 10 iris).
- **Derived metrics** – Eye Aspect Ratio (EAR, six landmarks per eye: lid-to-lid distance over eye width), Mouth Aspect Ratio (MAR, inner-lip opening over mouth width), head pose (pitch/yaw/roll via `solvePnP` on six landmarks).
- **Windowing** – 30‑second windows (15 s overlap) → each window becomes a vector of statistics (mean, std, slope, blink count, etc.).
- **Unsupervised learning** – HDBSCAN or k‑means finds natural clusters. Isolation Forest flags anomalies.
- **Output** – Timeline plot + CSV with cluster labels per window.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Randomiser47/Solutions-to-sadness.git
cd Solutions-to-sadness/LectureFaceLogger

# Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows

# Install dependencies (HDBSCAN comes from scikit-learn >= 1.3, no extra package)
pip install -r requirements.txt

# Optional: check the maths and the analysis without a webcam
python test_features.py
```

> **Note:** MediaPipe works best with Python 3.8–3.11. The script uses MediaPipe's Tasks API; its feature maths and analysis were checked on Python 3.13 with MediaPipe 0.10.32.

## Usage

### 1. Record a lecture

```bash
python lectureFaceLogger.py --output_dir ./my_lectures --fps 15 --skip_frames 2
```

- `--output_dir` – where session folders are saved (created automatically)
- `--fps` – video frame rate (default 15)
- `--skip_frames` – process every Nth frame for features (reduce CPU, default 1)

More options: `--camera N` (which webcam), `--model_path` (where the face model lives), `--no_preview` (no window; stop with Ctrl+C). `--fps` is now a real recording rate: a 30 fps camera has frames dropped to match it.

Press `q` in the preview window to stop recording. Whatever was recorded is saved even if you stop with Ctrl+C.  
The script saves:
- `video.mp4` – the raw recording
- `features.csv` – time‑series of EAR, MAR, head pose, face detection flag

### 2. Analyse the recording (unsupervised)

```bash
python analyze_lecture_states.py --csv ./my_lectures/lecture_YYYYMMDD_HHMMSS/features.csv --cluster_method kmeans --n_clusters 4
```

Options:
- `--window` – window length in seconds (default 30)
- `--step` – step size in seconds (default 15)
- `--cluster_method` – `hdbscan` (auto‑discovers clusters) or `kmeans` (fixed number)
- `--n_clusters` – for k‑means only

Extra: `--no_show` saves the plot without opening a window.

`features.csv` has a `video_time_sec` column as well as `timestamp_sec`. Use `video_time_sec` to find a moment in `video.mp4` (they only differ if your machine couldn't keep up with `--fps`, and the recorder warns you when that happens). Frames without a face have empty feature values.

Outputs:
- `features_windows_with_clusters.csv` – each row = one time window + cluster label + anomaly flag
- `features_timeline.png` – colour‑coded timeline of states, with eye/mouth/head pose overlays
- Console prints cluster distribution and characteristic averages

---

## Interpreting the results

### Cluster summary (example)

| cluster | left_ear_mean | mouth_ar_mean | pitch_deg_mean | blink_count | interpretation        |
|---------|---------------|---------------|----------------|-------------|-----------------------|
| 0       | 0.32          | 0.08          | -2.1           | 0.02        | Focused listening     |
| 1       | 0.18          | 0.12          | 5.4            | 0.45        | Low energy / drowsy   |
| 2       | 0.28          | 0.21          | -1.2           | 0.08        | Talking / engaged     |

- **EAR** (0.2–0.4) – lower when eyes close (blinks or fatigue).
- **MAR** (<0.1 = mouth closed, >0.15 = talking or yawning).
- **Pitch** (nodding) – positive = looking up, negative = looking down.
- **Anomalies** (red X on timeline) – windows that don’t fit any cluster (e.g., you left the camera).

> **Calibrate to yourself.** The numbers above are typical starting points, not truths: they shift with your face, glasses and camera. Clusters learn *your* baseline, so compare clusters with each other rather than with the table. Head pose signs: pitch + = looking up, yaw + = turning towards the right of the image, roll + = head tilted clockwise in the image; all are ~0 when you face the camera.
>
> **Anomalies** (red X) are windows Isolation Forest finds unusual, plus every window where you were out of frame for at least half the time. HDBSCAN "noise" (cluster -1, grey) means a window that didn't fit any cluster.

### Timeline plot

- **Top panel:** one dot per 30 s window, coloured by the state it belongs to (grey = noise, red × = anomaly).
- **Middle panel:** eye openness (EAR) and mouth movement (MAR) over time.
- **Bottom panel:** head pose (pitch = nodding, yaw = shaking).

You can visually align state changes with lecture content (e.g., a drop in EAR after 60 minutes might signal fatigue).

---

## Project structure

```
LectureFaceLogger/
├── lectureFaceLogger.py        # Recording script (MediaPipe Tasks API)
├── analyze_lecture_states.py   # Unsupervised analysis + visualisation
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── test_features.py            # Webcam-free self-checks (python test_features.py)
├── .gitignore                  # Keeps recordings and CSVs out of git
└── my_lectures/                # Session folders (created automatically)
    └── lecture_20260403_143956/
        ├── video.mp4
        ├── features.csv
        ├── features_windows_with_clusters.csv
        └── features_timeline.png
```

---

## Limitations & biases (acknowledged)

- **Lighting & camera angle** – best results with fixed webcam at eye level, good light on your face.
- **Glasses / facial hair** – MediaPipe still works, but EAR values may shift (unsupervised learning adapts to your baseline).
- **No emotion labels** – clusters are behavioural states, not emotions. You interpret them by watching the video clips of each cluster.
- **Head pose accuracy** – relative changes are reliable; absolute angles may have small bias due to generic face model.
- **Python version** – MediaPipe Tasks API works on Python 3.8–3.11. Python 3.12+ may work but is not officially tested.
- **Thresholds are starting points** – the blink threshold (EAR below 0.2) and the mouth/eye ranges above were not tuned on real faces. The head-pose reference face is a generic one, so absolute angles can be a few degrees off. The pose maths was checked against known synthetic rotations, not against a real webcam.

---

## Next steps (ideas)

- **Real‑time energy meter** – Once clusters are defined, train a simple classifier (e.g., nearest centroid) to predict your state live.
- **Correlate with lecture content** – Extract audio or slide timestamps, overlay on the timeline.
- **Self‑experimentation** – Try different window sizes (20s, 45s) or feature sets (add head velocity, blink rate variability).

---

## Contributing

This is a personal tool, but issues and ideas are welcome. Fork it, break it, make it yours.

## License

MIT – use it, learn from it, don't sell it as a "focus tracker" to schools.

## Author

Built by neurocoder – 2e, ADHD, learning by building.

