#!/usr/bin/env python3

"""
analyze_lecture_states.py

Loads features.csv from a lecture recording, creates time windows,
extracts features per window, clusters the windows (unsupervised),
and visualises engagement/energy patterns.

No manual labels – discovers natural states from your behaviour.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import hdbscan
import argparse
from datetime import timedelta
import os

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure timestamp is float, sort by frame_id just in case
    df = df.sort_values('timestamp_sec')
    # Replace zeros or missing face detection with NaN for ear/mouth/pose
    # (we'll interpolate short gaps later)
    for col in ['left_ear', 'right_ear', 'mouth_ar', 'pitch_deg', 'yaw_deg', 'roll_deg']:
        df.loc[df['face_detected'] == 0, col] = np.nan
    # Interpolate short gaps (up to 2 seconds) – assumes you briefly looked away
    df = df.interpolate(method='linear', limit=30)  # 30 frames ~2 seconds at 15 fps
    return df

# ------------------------------------------------------------
# 2. Create sliding windows and extract per‑window features
# ------------------------------------------------------------
def window_features(df, window_sec=30, step_sec=15, fps=15):
    """
    Returns a DataFrame where each row is a time window,
    with statistical features (mean, std, slope, etc.) for each original column.
    """
    duration = df['timestamp_sec'].max()
    windows = []
    start = 0.0
    while start + window_sec <= duration:
        end = start + window_sec
        mask = (df['timestamp_sec'] >= start) & (df['timestamp_sec'] < end)
        window_data = df.loc[mask]
        if len(window_data) < 10:   # skip windows with too few frames
            start += step_sec
            continue
        
        features = {'window_start': start, 'window_end': end}
        # For each metric, compute statistics
        for metric in ['left_ear', 'right_ear', 'mouth_ar', 'pitch_deg', 'yaw_deg', 'roll_deg']:
            series = window_data[metric].dropna()
            if len(series) > 0:
                features[f'{metric}_mean'] = series.mean()
                features[f'{metric}_std'] = series.std()
                features[f'{metric}_min'] = series.min()
                features[f'{metric}_max'] = series.max()
                # slope (linear trend) – indicates rising/falling over window
                if len(series) > 1:
                    x = np.arange(len(series))
                    slope = np.polyfit(x, series.values, 1)[0]
                    features[f'{metric}_slope'] = slope
                else:
                    features[f'{metric}_slope'] = 0.0
            else:
                # fill with zeros if no data (rare after interpolation)
                features[f'{metric}_mean'] = 0.0
                features[f'{metric}_std'] = 0.0
                features[f'{metric}_min'] = 0.0
                features[f'{metric}_max'] = 0.0
                features[f'{metric}_slope'] = 0.0
        
        # Additional feature: blink rate (quick drop in EAR)
        ear = window_data['left_ear'].fillna(0).values
        # simple blink detection: EAR below threshold (e.g., 0.2) for 1-3 frames
        blinks = np.sum((ear < 0.2) & (np.roll(ear, 1) >= 0.2))  # falling edge
        features['blink_count'] = blinks
        
        windows.append(features)
        start += step_sec
    
    return pd.DataFrame(windows)

# ------------------------------------------------------------
# 3. Cluster windows
# ------------------------------------------------------------
def cluster_windows(feature_df, method='hdbscan', n_clusters=4):
    """
    Apply clustering to the feature vectors (excluding time columns).
    Returns cluster labels and the fitted scaler.
    """
    # Select all feature columns (exclude window_start/end)
    feature_cols = [c for c in feature_df.columns if c not in ['window_start', 'window_end']]
    X = feature_df[feature_cols].fillna(0).values
    
    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, prediction_data=True)
        labels = clusterer.fit_predict(X_scaled)
    else:  # kmeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)
    
    return labels, scaler

# ------------------------------------------------------------
# 4. Anomaly detection
# ------------------------------------------------------------
def detect_anomalies(feature_df, contamination=0.05):
    feature_cols = [c for c in feature_df.columns if c not in ['window_start', 'window_end']]
    X = feature_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)  # -1 = anomaly
    return anomalies

# ------------------------------------------------------------
# 5. Visualisation
# ------------------------------------------------------------
def plot_timeline(feature_df, labels, anomalies=None, output_image='lecture_states.png'):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Timeline of cluster states
    ax1 = axes[0]
    times = feature_df['window_start'].values
    # colour map for clusters (excluding noise -1)
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(times[mask], [label]*np.sum(mask), c=[cmap(i)], s=20, alpha=0.7, label=f'State {label}')
    if anomalies is not None:
        anomaly_mask = anomalies == -1
        ax1.scatter(times[anomaly_mask], [-2]*np.sum(anomaly_mask), c='red', marker='x', s=50, label='Anomaly')
    ax1.set_ylabel('Cluster')
    ax1.set_title('Discovered Behavioural States (Unsupervised)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot key metrics over time (mean EAR and mouth openness)
    ax2 = axes[1]
    ax2.plot(times, feature_df['left_ear_mean'], label='Left EAR (mean)', alpha=0.7)
    ax2.plot(times, feature_df['mouth_ar_mean'], label='Mouth AR (mean)', alpha=0.7)
    ax2.set_ylabel('Ratio')
    ax2.set_title('Eye Openness & Mouth Movement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Head pose pitch (nodding)
    ax3 = axes[2]
    ax3.plot(times, feature_df['pitch_deg_mean'], label='Pitch (nodding)', alpha=0.7)
    ax3.plot(times, feature_df['yaw_deg_mean'], label='Yaw (shaking)', alpha=0.7)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Degrees')
    ax3.set_title('Head Pose')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Convert x-axis to hours:minutes
    def format_time(x, pos):
        return str(timedelta(seconds=int(x)))
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    print(f"Saved timeline plot to {output_image}")
    plt.show()

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
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.csv}...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} rows, duration = {df['timestamp_sec'].max()/60:.1f} minutes")
    
    # Create windows
    print(f"Creating windows of {args.window}s every {args.step}s...")
    feature_df = window_features(df, window_sec=args.window, step_sec=args.step)
    print(f"Generated {len(feature_df)} windows")
    
    # Cluster
    print(f"Clustering using {args.cluster_method}...")
    labels, _ = cluster_windows(feature_df, method=args.cluster_method, n_clusters=args.n_clusters)
    feature_df['cluster'] = labels
    print("Cluster distribution:")
    print(pd.Series(labels).value_counts().sort_index())
    
    # Anomaly detection
    print("Detecting anomalies...")
    anomalies = detect_anomalies(feature_df, contamination=0.05)
    feature_df['anomaly'] = anomalies
    
    # Save results
    output_csv = args.csv.replace('.csv', '_windows_with_clusters.csv')
    feature_df.to_csv(output_csv, index=False)
    print(f"Saved windowed data with clusters to {output_csv}")
    
    # Plot
    plot_timeline(feature_df, labels, anomalies, output_image=args.csv.replace('.csv', '_timeline.png'))
    
    # Optional: print summary of each cluster's characteristics
    print("\n--- Cluster characteristics (mean values) ---")
    cluster_summary = feature_df.groupby('cluster').mean()
    print(cluster_summary.round(3))

if __name__ == '__main__':
    main()
