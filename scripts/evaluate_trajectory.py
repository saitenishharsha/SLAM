#!/usr/bin/env python3
"""Evaluate an estimated trajectory against ground truth: ATE + RPE.

Usage:
    evaluate-trajectory \\
        --estimate results/my_run/trajectory.txt \\
        --groundtruth /data/my_dataset/groundtruth.txt \\
        [--rpe-delta 1] [--plot results/my_run/eval_plot.png]
"""

from __future__ import annotations

import argparse

from slam.common.evaluation import compute_ate, compute_rpe
from slam.common.trajectory import Trajectory, plot_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", required=True, help="TUM-format estimated trajectory")
    parser.add_argument("--groundtruth", required=True, help="TUM-format ground-truth trajectory")
    parser.add_argument(
        "--rpe-delta", type=int, default=1, help="Frame gap for Relative Pose Error (default: 1)"
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.02,
        help="Max timestamp gap (s) for associating poses (default: 0.02)",
    )
    parser.add_argument("--plot", default=None, help="Optional output path for a trajectory plot")
    args = parser.parse_args()

    est = Trajectory.from_tum_file(args.estimate)
    gt = Trajectory.from_tum_file(args.groundtruth)
    print(f"Estimate: {len(est)} poses | Ground truth: {len(gt)} poses")

    ate = compute_ate(est, gt, max_time_diff=args.max_time_diff)
    print("\nAbsolute Trajectory Error (ATE), meters:")
    print(f"  RMSE   {ate.rmse:.4f}")
    print(f"  mean   {ate.mean:.4f}")
    print(f"  median {ate.median:.4f}")
    print(f"  std    {ate.std:.4f}")
    print(f"  max    {ate.max:.4f}")

    try:
        rpe = compute_rpe(est, gt, delta=args.rpe_delta, max_time_diff=args.max_time_diff)
        print(f"\nRelative Pose Error (RPE, delta={args.rpe_delta}), meters:")
        print(f"  RMSE   {rpe.rmse:.4f}")
        print(f"  mean   {rpe.mean:.4f}")
        print(f"  median {rpe.median:.4f}")
        print(f"  std    {rpe.std:.4f}")
        print(f"  max    {rpe.max:.4f}")
    except ValueError as e:
        print(f"\nCould not compute RPE: {e}")

    if args.plot:
        plot_trajectory(est, args.plot, groundtruth=gt, title="Estimate vs. ground truth")
        print(f"\nSaved plot: {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
