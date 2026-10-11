"""Trajectory evaluation: Absolute Trajectory Error (ATE) and
Relative Pose Error (RPE), following the standard TUM RGB-D benchmark
definitions (Sturm et al., 2012).

ATE measures global consistency after aligning the estimate to ground truth
with Umeyama/Horn's method (rigid, no scale by default). RPE measures local
drift over a fixed delta between pose pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.common.trajectory import Trajectory


@dataclass
class ATEResult:
    rmse: float
    mean: float
    median: float
    std: float
    max: float
    per_pose_error: np.ndarray


@dataclass
class RPEResult:
    rmse: float
    mean: float
    median: float
    std: float
    max: float
    per_pair_error: np.ndarray


def _associate_by_timestamp(
    est: Trajectory, gt: Trajectory, max_diff: float = 0.02
) -> list[tuple[int, int]]:
    """Match each estimate and reference pose at most once by timestamp proximity."""
    gt_times = gt.timestamps
    candidates = sorted(
        (abs(t_est - t_gt), i, j)
        for i, t_est in enumerate(est.timestamps)
        for j, t_gt in enumerate(gt_times)
        if abs(t_est - t_gt) <= max_diff
    )
    used_est, used_gt = set(), set()
    matches = []
    for _, i, j in candidates:
        if i in used_est or j in used_gt:
            continue
        matches.append((i, j))
        used_est.add(i)
        used_gt.add(j)
    matches.sort()
    if not matches:
        raise ValueError(
            "No timestamp matches between estimate and ground truth "
            f"within max_diff={max_diff}s. Check that both trajectories use "
            "the same time base."
        )
    return matches


def _umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find rigid rotation R and translation t minimizing ||R @ src + t - dst||.
    src, dst: (N, 3) corresponding point sets. Returns (R, t).
    """
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_dst - R @ mu_src
    return R, t


def compute_ate(est: Trajectory, gt: Trajectory, max_time_diff: float = 0.02) -> ATEResult:
    """Absolute Trajectory Error, in the same units as the trajectories (meters)."""
    matches = _associate_by_timestamp(est, gt, max_time_diff)
    est_pos = np.array([est.positions[i] for i, _ in matches])
    gt_pos = np.array([gt.positions[j] for _, j in matches])

    R, t = _umeyama_alignment(est_pos, gt_pos)
    aligned = (R @ est_pos.T).T + t

    errors = np.linalg.norm(aligned - gt_pos, axis=1)
    return ATEResult(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        max=float(np.max(errors)),
        per_pose_error=errors,
    )


def compute_rpe(
    est: Trajectory, gt: Trajectory, delta: int = 1, max_time_diff: float = 0.02
) -> RPEResult:
    """Relative Pose Error over `delta` frames apart (translation part, meters)."""
    if delta < 1:
        raise ValueError(f"delta must be at least 1, got {delta}")

    matches = _associate_by_timestamp(est, gt, max_time_diff)
    est_poses = [est.poses[i] for i, _ in matches]
    gt_poses = [gt.poses[j] for _, j in matches]

    errors = []
    for k in range(len(matches) - delta):
        rel_est = np.linalg.inv(est_poses[k]) @ est_poses[k + delta]
        rel_gt = np.linalg.inv(gt_poses[k]) @ gt_poses[k + delta]
        rel_err = np.linalg.inv(rel_gt) @ rel_est
        errors.append(np.linalg.norm(rel_err[:3, 3]))

    if not errors:
        raise ValueError(
            f"Trajectory too short for RPE with delta={delta} ({len(matches)} matched poses)."
        )
    errors = np.array(errors)
    return RPEResult(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        max=float(np.max(errors)),
        per_pair_error=errors,
    )
