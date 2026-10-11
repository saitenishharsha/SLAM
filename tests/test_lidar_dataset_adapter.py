import numpy as np
import pytest

from slam.lidar.dataset_adapter import GenericPointCloudAdapter, KITTIOdometryAdapter


def _make_kitti_dataset(root, n=3, points_per_scan=5):
    (root / "velodyne").mkdir(parents=True)
    times, poses = [], []
    for i in range(n):
        scan = np.random.default_rng(i).normal(size=(points_per_scan, 4)).astype(np.float32)
        scan.tofile(root / "velodyne" / f"{i:06d}.bin")
        times.append(f"{i * 0.1:.6f}")
        pose = np.eye(3, 4)
        pose[0, 3] = i
        poses.append(" ".join(f"{v:.6f}" for v in pose.flatten()))
    (root / "times.txt").write_text("\n".join(times))
    (root / "poses.txt").write_text("\n".join(poses))
    (root / "calib.txt").write_text("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")


def test_kitti_adapter_loads_scans_and_poses(tmp_path):
    _make_kitti_dataset(tmp_path, n=3, points_per_scan=5)
    adapter = KITTIOdometryAdapter(tmp_path)

    assert adapter.num_scans() == 3
    scan0 = adapter.load_scan(0)
    assert scan0.points.shape == (5, 3)
    assert scan0.groundtruth_pose is not None

    scan2 = adapter.load_scan(2)
    np.testing.assert_allclose(scan2.groundtruth_pose[:3, 3], [2, 0, 0], atol=1e-6)


def test_kitti_adapter_scans_iterator_matches_load_scan(tmp_path):
    _make_kitti_dataset(tmp_path, n=3, points_per_scan=5)
    adapter = KITTIOdometryAdapter(tmp_path)
    all_scans = list(adapter.scans())
    assert len(all_scans) == adapter.num_scans()
    np.testing.assert_allclose(all_scans[1].points, adapter.load_scan(1).points)


def test_kitti_adapter_rejects_mismatched_metadata(tmp_path):
    _make_kitti_dataset(tmp_path, n=3, points_per_scan=5)
    (tmp_path / "times.txt").write_text("0.0\n0.1")

    with pytest.raises(ValueError, match="timestamps for 3 scans"):
        KITTIOdometryAdapter(tmp_path)


def test_kitti_adapter_converts_camera_pose_to_velodyne_pose(tmp_path):
    _make_kitti_dataset(tmp_path, n=1, points_per_scan=5)
    (tmp_path / "calib.txt").write_text("Tr_velo_to_cam: 1 0 0 10 0 1 0 0 0 0 1 0\n")

    scan = KITTIOdometryAdapter(tmp_path).load_scan(0)

    np.testing.assert_allclose(scan.groundtruth_pose[:3, 3], [10, 0, 0], atol=1e-6)


def test_kitti_adapter_accepts_standard_tr_calibration_key(tmp_path):
    _make_kitti_dataset(tmp_path, n=1, points_per_scan=5)
    (tmp_path / "calib.txt").write_text("Tr: 1 0 0 10 0 1 0 0 0 0 1 0\n")

    scan = KITTIOdometryAdapter(tmp_path).load_scan(0)

    np.testing.assert_allclose(scan.groundtruth_pose[:3, 3], [10, 0, 0], atol=1e-6)


def test_generic_pointcloud_adapter_respects_explicit_binary_width(tmp_path):
    scans_dir = tmp_path / "scans"
    scans_dir.mkdir()
    np.arange(12, dtype=np.float32).tofile(scans_dir / "000000.bin")

    scan = GenericPointCloudAdapter(tmp_path, bin_columns=3).load_scan(0)

    assert scan.points.shape == (4, 3)
