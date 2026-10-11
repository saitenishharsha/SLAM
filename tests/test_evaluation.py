import numpy as np
import pytest

from slam.common.evaluation import compute_ate, compute_rpe
from slam.common.trajectory import Trajectory


def _straight_line_trajectory(n=10, step=1.0, offset=None):
    offset = np.zeros(3) if offset is None else offset
    timestamps = np.arange(n, dtype=float)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        poses[i, :3, 3] = [i * step, 0, 0] + offset
    return Trajectory(timestamps=timestamps, poses=poses)


def test_ate_zero_for_identical_trajectories():
    gt = _straight_line_trajectory()
    est = _straight_line_trajectory()
    result = compute_ate(est, gt)
    assert result.rmse == pytest.approx(0.0, abs=1e-8)


def test_ate_nonzero_for_offset_trajectory():
    gt = _straight_line_trajectory()
    # A rigid offset is removed by alignment; add noise instead so ATE is nonzero.
    est = _straight_line_trajectory()
    rng = np.random.default_rng(0)
    est.poses[:, :3, 3] += rng.normal(scale=0.05, size=(len(est), 3))
    result = compute_ate(est, gt)
    assert result.rmse > 0.0


def test_rpe_zero_for_identical_trajectories():
    gt = _straight_line_trajectory()
    est = _straight_line_trajectory()
    result = compute_rpe(est, gt, delta=1)
    assert result.rmse == pytest.approx(0.0, abs=1e-8)


def test_ate_raises_on_no_timestamp_overlap():
    gt = _straight_line_trajectory()
    est = _straight_line_trajectory()
    est.timestamps = est.timestamps + 1000.0  # no longer within max_time_diff
    with pytest.raises(ValueError):
        compute_ate(est, gt, max_time_diff=0.02)


@pytest.mark.parametrize("delta", [0, -1])
def test_rpe_rejects_nonpositive_delta(delta):
    trajectory = _straight_line_trajectory()
    with pytest.raises(ValueError, match="delta must be at least 1"):
        compute_rpe(trajectory, trajectory, delta=delta)
