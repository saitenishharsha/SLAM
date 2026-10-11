import numpy as np
import pytest

from slam.visual.dataset_adapter import (
    CameraIntrinsics,
    GenericRGBDAdapter,
    TUMRGBDAdapter,
    _associate,
)

cv2 = pytest.importorskip("cv2")


def _make_tum_dataset(root, n=3):
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)

    rgb_lines, depth_lines, gt_lines = [], [], []
    for i in range(n):
        t = i * 0.1
        rgb_img = np.full((4, 4, 3), i * 10, dtype=np.uint8)
        depth_img = np.full((4, 4), 1000 * (i + 1), dtype=np.uint16)
        cv2.imwrite(str(root / "rgb" / f"{i}.png"), rgb_img)
        cv2.imwrite(str(root / "depth" / f"{i}.png"), depth_img)
        rgb_lines.append(f"{t:.6f} rgb/{i}.png")
        depth_lines.append(f"{t:.6f} depth/{i}.png")
        gt_lines.append(f"{t:.6f} {i} 0 0 0 0 0 1")

    (root / "rgb.txt").write_text("\n".join(rgb_lines))
    (root / "depth.txt").write_text("\n".join(depth_lines))
    (root / "groundtruth.txt").write_text("\n".join(gt_lines))


def test_tum_adapter_loads_frames_and_groundtruth(tmp_path):
    _make_tum_dataset(tmp_path, n=3)
    intrinsics = CameraIntrinsics(fx=500, fy=500, cx=2, cy=2, width=4, height=4)
    adapter = TUMRGBDAdapter(tmp_path, intrinsics, depth_scale=5000.0)

    frames = adapter.frames()
    assert len(frames) == 3
    assert frames[0].groundtruth_pose is not None
    np.testing.assert_allclose(frames[1].groundtruth_pose[:3, 3], [1, 0, 0], atol=1e-6)
    assert adapter.depth_scale() == 5000.0
    assert adapter.intrinsics().width == 4


def test_tum_adapter_raises_on_no_matches(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "depth").mkdir()
    (tmp_path / "rgb.txt").write_text("0.0 rgb/0.png")
    (tmp_path / "depth.txt").write_text("100.0 depth/0.png")  # way out of sync
    intrinsics = CameraIntrinsics(fx=500, fy=500, cx=2, cy=2, width=4, height=4)
    adapter = TUMRGBDAdapter(tmp_path, intrinsics)
    with pytest.raises(ValueError):
        adapter.frames()


def test_associate_uses_each_depth_frame_once():
    matches = _associate(
        [(0.000, "rgb/0.png"), (0.001, "rgb/1.png")],
        [(0.0005, "depth/0.png")],
        max_diff=0.01,
    )

    assert matches == [(0.0, "rgb/0.png", "depth/0.png")]


def test_generic_rgbd_uses_frame_order_when_timestamps_are_omitted(tmp_path):
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()
    for index in range(2):
        (rgb_dir / f"{index}.png").write_bytes(b"")
        (depth_dir / f"{index}.png").write_bytes(b"")

    adapter = GenericRGBDAdapter(
        tmp_path,
        "rgb",
        "depth",
        "",
        CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=1, height=1),
    )

    assert [frame.timestamp for frame in adapter.frames()] == [0, 1]
