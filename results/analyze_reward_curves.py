"""analyze_reward_curves.py

Reads an absolute CSV of experimental results and plots convergence (reward per episode)
for each instance_id: for that instance, plot each algorithm's mean reward curve (mean ± std
across seeds/runs) on the same figure.

Usage:
    python results/analyze_reward_curves.py /absolute/path/to/all_experiment_results.csv

The script will try to find a column containing per-episode reward series. Supported column
names (checked in order): 'episode_rewards_series', 'reward_curve', 'reward_curve_list',
 'extra_info' (if it contains a 'reward_curve' key).

Output:
    results/convergence_plots/instance_<instance_id>_convergence.png

This file is written to be defensive about malformed rows and will skip rows it cannot parse.
"""

import sys
import os
import ast
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Any, Tuple


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isabs(path):
        raise ValueError("Please provide an absolute path to the CSV file")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    # try default read first; if it fails, use engine='python' and skip bad lines
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
        except TypeError:
            # older pandas may not accept on_bad_lines param
            df = pd.read_csv(path, engine='python')
    return df


def parse_reward_series_from_row(row: pd.Series, candidate_cols: List[str]) -> Optional[List[float]]:
    """Try to extract a list[float] reward series from the row using various columns."""
    for col in candidate_cols:
        if col not in row or pd.isna(row[col]):
            continue
        val = row[col]
        # If it's already a list/ndarray
        if isinstance(val, (list, tuple, np.ndarray)):
            try:
                return [float(x) for x in val]
            except Exception:
                continue
        # If it's a string representation of a list like "-1.0,-2.0" or "[-1,-2]"
        if isinstance(val, str):
            s = val.strip()
            # plain comma-separated numbers (no brackets)
            if s and all(c in '0123456789-+.,eE ' for c in s):
                parts = [p.strip() for p in s.split(',') if p.strip()!='']
                try:
                    return [float(p) for p in parts]
                except Exception:
                    pass
            # try literal_eval (handles Python list strings)
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [float(x) for x in parsed]
            except Exception:
                pass
            # sometimes the series is JSON-like
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except Exception:
                pass
    return None


def pad_and_aggregate(curves: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad shorter curves with NaN to the same length, then compute mean and std across axis=0."""
    if not curves:
        return np.array([]), np.array([])
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main(csv_path: Optional[str] = None):
    # If csv_path provided (e.g., from CLI), use it. Otherwise default to CSV in the same folder as this script.
    if csv_path is None:
        # Try command-line argument first
        if len(sys.argv) >= 2 and sys.argv[1]:
            csv_path = sys.argv[1]
        else:
            # fallback: CSV file placed next to this script (use forward slashes to avoid escape issues)
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'algorithm_comparison.csv'))

    # Normalize path to avoid backslash-escape issues
    csv_path = os.path.abspath(csv_path)
    df = load_csv(csv_path)

    # Candidate columns to look for reward series (in order)
    candidate_cols = ['episode_rewards_series', 'episode_reward_series', 'reward_curve', 'reward_curve_list', 'episode_rewards', 'episode_reward', 'episode_rewards_series_str', 'episode_rewards_series_csv', 'episode_rewards_series_list', 'episode_rewards_series']

    # if 'episode_rewards_series' not present but there is 'episode_rewards_series' in header (typo variations), we'll still try
    # Also check for a special column 'extra_info' that may contain a dict with 'reward_curve'
    if 'extra_info' in df.columns:
        candidate_cols.append('extra_info')

    # Use 'instance_id' (or 'instance' as fallback) to group
    instance_col = 'instance_id' if 'instance_id' in df.columns else ('instance' if 'instance' in df.columns else None)
    if instance_col is None:
        raise RuntimeError("CSV must contain an 'instance_id' or 'instance' column to group runs by problem instance")

    algo_col = 'algo_name' if 'algo_name' in df.columns else ('algorithm_name' if 'algorithm_name' in df.columns else ('algorithm' if 'algorithm' in df.columns else None))
    if algo_col is None:
        raise RuntimeError("CSV must contain an algorithm name column (algo_name / algorithm_name / algorithm)")

    out_dir = os.path.join(os.path.dirname(csv_path), 'convergence_plots')
    ensure_out_dir(out_dir)

    summary = []

    grouped = df.groupby(instance_col)
    for instance_id, group in grouped:
        plt.figure(figsize=(10, 6))
        plotted = 0
        for algo, algo_df in group.groupby(algo_col):
            curves = []
            for _, row in algo_df.iterrows():
                series = parse_reward_series_from_row(row, candidate_cols)
                if series is None and 'extra_info' in row and isinstance(row['extra_info'], str):
                    # try parse extra_info as dict and extract 'reward_curve'
                    try:
                        info = ast.literal_eval(row['extra_info'])
                        if isinstance(info, dict) and 'reward_curve' in info:
                            series = parse_reward_series_from_row(pd.Series({'reward_curve': info['reward_curve']}), ['reward_curve'])
                    except Exception:
                        pass
                if series is not None and len(series) > 0:
                    curves.append(series)
            if not curves:
                continue
            mean_curve, std_curve = pad_and_aggregate(curves)
            episodes = np.arange(1, len(mean_curve) + 1)
            plt.plot(episodes, mean_curve, label=f"{algo}")
            plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
            plotted += 1
            summary.append((instance_id, algo, len(curves), float(np.nanmean(mean_curve) if mean_curve.size>0 else math.nan)))

        if plotted == 0:
            print(f"Instance {instance_id}: no parsable reward-series found for any algorithm, skipping plot")
            plt.close()
            continue

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Convergence for instance {instance_id}')
        plt.legend()
        plt.grid(alpha=0.3)
        out_file = os.path.join(out_dir, f'instance_{instance_id}_convergence.png')
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved plot for instance {instance_id} -> {out_file}")

    # Save a small CSV summary
    if summary:
        summary_df = pd.DataFrame(summary, columns=['instance_id', 'algorithm', 'num_runs', 'mean_of_mean_curve'])
        summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
        print(f"Wrote summary to {os.path.join(out_dir, 'summary.csv')}")
    else:
        print("No plots generated; no parsed reward series found in the CSV.")


if __name__ == '__main__':
   

   
    main()
   