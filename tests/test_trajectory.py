import numpy as np
import pytest

from slam.common.trajectory import Trajectory, plot_trajectory


def _random_trajectory(n=5, seed=0):
    rng = np.random.default_rng(seed)
    timestamps = np.arange(n, dtype=float)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, 3] = rng.normal(size=(n, 3))
    return Trajectory(timestamps=timestamps, poses=poses)


def test_tum_roundtrip(tmp_path):
    traj = _random_trajectory()
    path = tmp_path / "traj.txt"
    traj.to_tum_file(path)

    loaded = Trajectory.from_tum_file(path)
    assert len(loaded) == len(traj)
    np.testing.assert_allclose(loaded.timestamps, traj.timestamps)
    np.testing.assert_allclose(loaded.positions, traj.positions, atol=1e-5)


def test_positions_shape():
    traj = _random_trajectory(n=7)
    assert traj.positions.shape == (7, 3)


def test_from_tum_file_rejects_malformed_line(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0.0 1.0 2.0\n")  # too few fields
    try:
        Trajectory.from_tum_file(path)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_plot_trajectory_rejects_invalid_coordinate_plane(tmp_path):
    with pytest.raises(ValueError, match="plane must contain two coordinate axes"):
        plot_trajectory(_random_trajectory(), tmp_path / "trajectory.png", plane=(0, 3))
