#!/usr/bin/env python3
"""
analyze_lecture_states.py

Loads features.csv from a lecture recording, creates time windows,
extracts features per window, clusters the windows (unsupervised),
and visualises engagement/energy patterns.

No manual labels - discovers natural states from your behaviour.
"""

import argparse
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

METRICS = ['left_ear', 'right_ear', 'mouth_ar', 'pitch_deg', 'yaw_deg', 'roll_deg']
BLINK_EAR_THRESHOLD = 0.2       # eye aspect ratio below this counts as "eyes closed"
MIN_WINDOWS = 8                 # fewer windows than this can't be clustered meaningfully


# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------
def load_data(csv_path, max_gap_sec=2.0):
    """
    Load features.csv. Frames without a face become NaN; gaps of up to
    `max_gap_sec` seconds (blinks of the tracker, a quick look away) are
    interpolated, longer gaps stay NaN so "left the camera" is not faked over.
    """
    df = pd.read_csv(csv_path).sort_values('timestamp_sec').reset_index(drop=True)
    df.loc[df['face_detected'] == 0, METRICS] = np.nan

    dt = df['timestamp_sec'].diff().median()
    max_gap_frames = max(1, int(round(max_gap_sec / dt))) if dt and dt > 0 else 30

    for col in METRICS:
        series = df[col]
        missing = series.isna()
        run_id = (missing != missing.shift()).cumsum()
        run_length = missing.groupby(run_id).transform('sum')
        fill_here = missing & (run_length <= max_gap_frames)
        interpolated = series.interpolate(method='linear', limit_area='inside')
        df[col] = series.where(~fill_here, interpolated)
    return df


# ------------------------------------------------------------
# 2. Create sliding windows and extract per-window features
# ------------------------------------------------------------
def window_features(df, window_sec=30, step_sec=15):
    """
    Returns a DataFrame where each row is a time window, with statistical
    features (mean, std, min, max, slope per second) for each metric, a blink
    count and the fraction of the window in which no face was detected.
    """
    duration = df['timestamp_sec'].max()
    windows = []
    start = 0.0
    while start + window_sec <= duration:
        end = start + window_sec
        window_data = df[(df['timestamp_sec'] >= start) & (df['timestamp_sec'] < end)]
        if len(window_data) < 10:   # skip windows with too few frames
            start += step_sec
            continue

        features = {'window_start': start, 'window_end': end}
        for metric in METRICS:
            series = window_data[metric].dropna()
            if len(series) > 0:
                features[f'{metric}_mean'] = series.mean()
                features[f'{metric}_std'] = series.std() if len(series) > 1 else 0.0
                features[f'{metric}_min'] = series.min()
                features[f'{metric}_max'] = series.max()
                # slope: linear trend per second (rising / falling over the window)
                t = window_data.loc[series.index, 'timestamp_sec'].values
                if len(series) > 1 and np.ptp(t) > 0:
                    features[f'{metric}_slope'] = np.polyfit(t - t[0], series.values, 1)[0]
                else:
                    features[f'{metric}_slope'] = 0.0
            else:
                for stat in ('mean', 'std', 'min', 'max', 'slope'):
                    features[f'{metric}_{stat}'] = 0.0

        # Blinks: falling edges of the eye aspect ratio through the threshold.
        # Missing frames (NaN) are never counted as a blink.
        ear = window_data[['left_ear', 'right_ear']].mean(axis=1, skipna=True).values
        valid = ~np.isnan(ear)
        closed = valid & (np.nan_to_num(ear, nan=1.0) < BLINK_EAR_THRESHOLD)
        was_open = valid & (np.nan_to_num(ear, nan=0.0) >= BLINK_EAR_THRESHOLD)
        features['blink_count'] = int(np.sum(closed[1:] & was_open[:-1]))

        # How much of the window had no face in view (1.0 = you were away)
        features['face_missing_frac'] = 1.0 - window_data['face_detected'].mean()

        windows.append(features)
        start += step_sec

    return pd.DataFrame(windows)


# ------------------------------------------------------------
# 3. Cluster windows
# ------------------------------------------------------------
def _feature_matrix(feature_df):
    feature_cols = [c for c in feature_df.columns if c not in ['window_start', 'window_end']]
    X = feature_df[feature_cols].fillna(0).values
    return StandardScaler().fit_transform(X)


def _make_hdbscan(min_cluster_size=5, min_samples=3):
    try:
        from sklearn.cluster import HDBSCAN   # scikit-learn >= 1.3, no extra package needed
    except ImportError as exc:
        raise SystemExit("HDBSCAN needs scikit-learn >= 1.3 (pip install -U scikit-learn)") from exc
    try:
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True)
    except TypeError:   # older scikit-learn has no `copy` argument
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)


def cluster_windows(feature_df, method='hdbscan', n_clusters=4):
    """
    Apply clustering to the window feature vectors (time columns excluded).
    Returns cluster labels (-1 = noise for HDBSCAN) and the number of clusters asked for.
    """
    X_scaled = _feature_matrix(feature_df)
    if method == 'hdbscan':
        labels = _make_hdbscan().fit_predict(X_scaled)
    else:  # kmeans
        k = min(n_clusters, len(X_scaled))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    return labels, n_clusters


# ------------------------------------------------------------
# 4. Anomaly detection
# ------------------------------------------------------------
def detect_anomalies(feature_df, contamination=0.05, missing_threshold=0.5):
    """
    -1 = anomaly, 1 = normal. Isolation Forest finds unusual windows; on top of
    that, any window where you were out of frame for at least `missing_threshold`
    of the time is always flagged ("left the camera").
    """
    X_scaled = _feature_matrix(feature_df)
    anomalies = IsolationForest(contamination=contamination, random_state=42).fit_predict(X_scaled)
    if 'face_missing_frac' in feature_df.columns:
        anomalies[feature_df['face_missing_frac'].values >= missing_threshold] = -1
    return anomalies


# ------------------------------------------------------------
# 5. Visualisation
# ------------------------------------------------------------
def plot_timeline(feature_df, labels, anomalies=None, output_image='lecture_states.png', show=True):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Timeline of cluster states
    ax1 = axes[0]
    times = feature_df['window_start'].values
    labels = np.asarray(labels)
    palette = plt.get_cmap('tab10')
    real_clusters = [l for l in sorted(set(labels)) if l != -1]
    for label in sorted(set(labels)):
        mask = labels == label
        if label == -1:
            color, name = 'lightgray', 'Noise (no clear state)'
        else:
            color, name = palette(real_clusters.index(label) % 10), f'State {label}'
        ax1.scatter(times[mask], [label] * int(mask.sum()), color=color, s=20, alpha=0.8, label=name)
    if anomalies is not None:
        anomaly_mask = np.asarray(anomalies) == -1
        if anomaly_mask.any():
            ax1.scatter(times[anomaly_mask], [-2] * int(anomaly_mask.sum()),
                        c='red', marker='x', s=50, label='Anomaly')
    ax1.set_ylabel('Cluster')
    ax1.set_title('Discovered Behavioural States (Unsupervised)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Key metrics over time (mean EAR and mouth openness)
    ax2 = axes[1]
    ax2.plot(times, feature_df['left_ear_mean'], label='Left EAR (mean)', alpha=0.7)
    ax2.plot(times, feature_df['mouth_ar_mean'], label='Mouth AR (mean)', alpha=0.7)
    ax2.set_ylabel('Ratio')
    ax2.set_title('Eye Openness & Mouth Movement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Head pose
    ax3 = axes[2]
    ax3.plot(times, feature_df['pitch_deg_mean'], label='Pitch (nodding)', alpha=0.7)
    ax3.plot(times, feature_df['yaw_deg_mean'], label='Yaw (turning)', alpha=0.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Degrees')
    ax3.set_title('Head Pose')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Convert x-axis to hours:minutes:seconds
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: str(timedelta(seconds=int(x)))))
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    print(f"Saved timeline plot to {output_image}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Unsupervised analysis of lecture facial features')
    parser.add_argument('--csv', type=str, required=True, help='Path to features.csv')
    parser.add_argument('--window', type=int, default=30, help='Window length in seconds')
    parser.add_argument('--step', type=int, default=15, help='Step size in seconds')
    parser.add_argument('--cluster_method', choices=['hdbscan', 'kmeans'], default='hdbscan')
    parser.add_argument('--n_clusters', type=int, default=4, help='For k-means only')
    parser.add_argument('--no_show', action='store_true', help='Save the plot without opening a window')
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} rows, duration = {df['timestamp_sec'].max() / 60:.1f} minutes")

    print(f"Creating windows of {args.window}s every {args.step}s...")
    feature_df = window_features(df, window_sec=args.window, step_sec=args.step)
    print(f"Generated {len(feature_df)} windows")
    if len(feature_df) < MIN_WINDOWS:
        raise SystemExit(
            f"Only {len(feature_df)} windows - need at least {MIN_WINDOWS} to find states. "
            f"Record a longer session or use a smaller --window/--step."
        )

    print(f"Clustering using {args.cluster_method}...")
    labels, _ = cluster_windows(feature_df, method=args.cluster_method, n_clusters=args.n_clusters)
    feature_df['cluster'] = labels
    print("Cluster distribution (-1 = noise):")
    print(pd.Series(labels).value_counts().sort_index())

    print("Detecting anomalies...")
    anomalies = detect_anomalies(feature_df, contamination=0.05)
    feature_df['anomaly'] = anomalies
    print(f"Flagged {int((anomalies == -1).sum())} anomalous windows")

    base, _ = os.path.splitext(args.csv)
    output_csv = base + '_windows_with_clusters.csv'
    feature_df.to_csv(output_csv, index=False)
    print(f"Saved windowed data with clusters to {output_csv}")

    plot_timeline(feature_df, labels, anomalies, output_image=base + '_timeline.png',
                  show=not args.no_show)

    print("\n--- Cluster characteristics (mean values) ---")
    cluster_summary = feature_df.drop(columns=['window_start', 'window_end', 'anomaly']).groupby('cluster').mean()
    print(cluster_summary.round(3))


if __name__ == '__main__':
    main()
