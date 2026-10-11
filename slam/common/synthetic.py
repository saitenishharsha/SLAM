"""Synthetic datasets with exact ground truth, for smoke-testing the pipelines.

Both generators ray-cast an analytic scene from a camera/sensor moving along a
known path, so you can check that a fresh install works end to end (and that
the estimated trajectory matches the truth) without downloading anything.

- `generate_rgbd_sequence` writes a TUM-RGB-D-layout sequence
  (`rgb.txt`, `depth.txt`, `groundtruth.txt`, `rgb/`, `depth/`).
- `generate_lidar_sequence` writes a KITTI-odometry-layout sequence
  (`velodyne/*.bin`, `times.txt`, `poses.txt`, `calib.txt`).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from slam.common.trajectory import Trajectory

# --------------------------------------------------------------------------- RGB-D

RGBD_WIDTH = 160
RGBD_HEIGHT = 120
RGBD_FX = RGBD_FY = 200.0
RGBD_CX = RGBD_WIDTH / 2
RGBD_CY = RGBD_HEIGHT / 2
RGBD_DEPTH_SCALE = 5000.0  # same convention as TUM RGB-D

# Room bounds in the world frame (camera convention: x right, y down, z forward).
_ROOM_MIN = np.array([-1.2, -0.9, -1.0])
_ROOM_MAX = np.array([1.2, 0.9, 4.0])


def _camera_pose(s: float) -> np.ndarray:
    """Camera -> world pose at path parameter s in [0, 1]. Starts at identity."""
    pose = np.eye(4)
    yaw = np.deg2rad(12.0) * np.sin(np.pi * s)
    pose[:3, :3] = Rotation.from_euler("y", yaw).as_matrix()
    pose[:3, 3] = [0.4 * np.sin(2 * np.pi * s), 0.1 * np.sin(np.pi * s), 1.0 * s]
    return pose


def _texture(points: np.ndarray) -> np.ndarray:
    """Deterministic RGB texture as a function of 3D position, (N, 3) -> (N, 3) uint8."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cell = np.floor(x / 0.2) + np.floor(y / 0.2) + np.floor(z / 0.2)
    checker = (cell % 2) * 60.0
    r = 120 + 50 * np.sin(5 * x + 1.0) * np.cos(4 * z) + checker
    g = 120 + 50 * np.sin(6 * y + 2.0) * np.cos(5 * x) + checker
    b = 120 + 50 * np.sin(4 * z + 0.5) * np.cos(6 * y) + checker
    return np.clip(np.stack([r, g, b], axis=1), 0, 255).astype(np.uint8)


def _render_rgbd(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast the room from `pose`. Returns (rgb HxWx3 uint8, depth HxW uint16)."""
    u, v = np.meshgrid(np.arange(RGBD_WIDTH), np.arange(RGBD_HEIGHT))
    d_cam = np.stack(
        [(u - RGBD_CX) / RGBD_FX, (v - RGBD_CY) / RGBD_FY, np.ones_like(u, dtype=float)], axis=-1
    ).reshape(-1, 3)
    d_world = d_cam @ pose[:3, :3].T
    origin = pose[:3, 3]

    # Exit distance of each ray from the axis-aligned room (camera is inside it).
    with np.errstate(divide="ignore", invalid="ignore"):
        t_axes = np.where(
            d_world > 0,
            (_ROOM_MAX - origin) / d_world,
            (_ROOM_MIN - origin) / d_world,
        )
    t_axes[d_world == 0] = np.inf  # parallel to this pair of walls: never exits through them
    t = np.nanmin(t_axes, axis=1)  # == z-depth because d_cam has unit z component
    hit = origin + d_world * t[:, None]

    rgb = _texture(hit).reshape(RGBD_HEIGHT, RGBD_WIDTH, 3)
    depth = np.round(t * RGBD_DEPTH_SCALE).clip(0, 65535).astype(np.uint16)
    return rgb, depth.reshape(RGBD_HEIGHT, RGBD_WIDTH)


def generate_rgbd_sequence(out_dir: str | Path, n_frames: int = 60) -> dict:
    """Write a synthetic TUM-layout RGB-D sequence plus its camera intrinsics YAML.

    Returns a dict with the paths: `root`, `intrinsics_file`.
    """
    out = Path(out_dir)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    rgb_lines = ["# timestamp filename"]
    depth_lines = ["# timestamp filename"]
    timestamps, poses = [], []
    for i in range(n_frames):
        t = 1000.0 + i / 30.0
        pose = _camera_pose(i / max(n_frames - 1, 1))
        rgb, depth = _render_rgbd(pose)
        cv2.imwrite(str(out / "rgb" / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "depth" / f"{i:06d}.png"), depth)
        rgb_lines.append(f"{t:.6f} rgb/{i:06d}.png")
        depth_lines.append(f"{t:.6f} depth/{i:06d}.png")
        timestamps.append(t)
        poses.append(pose)

    (out / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (out / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    Trajectory(np.array(timestamps), np.stack(poses)).to_tum_file(out / "groundtruth.txt")

    intrinsics_file = out / "camera.yaml"
    intrinsics_file.write_text(
        yaml.safe_dump(
            {
                "fx": RGBD_FX,
                "fy": RGBD_FY,
                "cx": RGBD_CX,
                "cy": RGBD_CY,
                "width": RGBD_WIDTH,
                "height": RGBD_HEIGHT,
            }
        )
    )
    return {"root": out, "intrinsics_file": intrinsics_file}


# --------------------------------------------------------------------------- LiDAR

# Scene in the world frame, KITTI sensor convention (x forward, y left, z up).
_LIDAR_ROOM_MIN = np.array([-6.0, -5.0, -1.5])
_LIDAR_ROOM_MAX = np.array([26.0, 5.0, 3.0])
# Axis-aligned boxes (min corner, max corner) that break up the long corridor.
_LIDAR_BOXES = [
    (np.array([5.0, 2.0, -1.5]), np.array([6.0, 3.0, 1.5])),
    (np.array([9.0, -4.0, -1.5]), np.array([10.5, -2.5, 0.5])),
    (np.array([14.0, 1.0, -1.5]), np.array([15.0, 2.5, 2.0])),
    (np.array([18.0, -3.5, -1.5]), np.array([19.5, -2.0, 1.0])),
    (np.array([22.0, 0.0, -1.5]), np.array([23.0, 1.5, 2.5])),
]
# KITTI calib.txt `Tr`: velodyne -> left camera (rotation + a small lever arm).
_KITTI_TR = np.array(
    [
        [0.0, -1.0, 0.0, -0.01],
        [0.0, 0.0, -1.0, -0.08],
        [1.0, 0.0, 0.0, -0.27],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _sensor_pose(s: float) -> np.ndarray:
    """Sensor -> world pose at path parameter s in [0, 1]. Starts at identity."""
    pose = np.eye(4)
    yaw = np.deg2rad(8.0) * np.sin(2 * np.pi * s)
    pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    pose[:3, 3] = [15.0 * s, 1.0 * np.sin(2 * np.pi * s), 0.0]
    return pose


def _ray_directions(n_beams: int = 16, n_azimuth: int = 360) -> np.ndarray:
    elevation = np.deg2rad(np.linspace(-15.0, 15.0, n_beams))
    azimuth = np.deg2rad(np.linspace(-180.0, 180.0, n_azimuth, endpoint=False))
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    return np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
    ).reshape(-1, 3)


def _lidar_scan(pose: np.ndarray, rng: np.random.Generator, noise: float = 0.01) -> np.ndarray:
    """Ray-cast the scene from `pose`; returns (N, 3) points in the sensor frame."""
    dirs_sensor = _ray_directions()
    d = dirs_sensor @ pose[:3, :3].T
    o = pose[:3, 3]

    with np.errstate(divide="ignore", invalid="ignore"):
        t_exit = np.where(d > 0, (_LIDAR_ROOM_MAX - o) / d, (_LIDAR_ROOM_MIN - o) / d)
        t_exit[d == 0] = np.inf
        t_room = np.nanmin(t_exit, axis=1)
        t_best = t_room.copy()
        for bmin, bmax in _LIDAR_BOXES:
            t1 = (bmin - o) / d
            t2 = (bmax - o) / d
            t_near = np.nanmax(np.minimum(t1, t2), axis=1)
            t_far = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (t_near <= t_far) & (t_near > 0.5)
            t_best = np.where(hit & (t_near < t_best), t_near, t_best)

    t_best = t_best + rng.normal(scale=noise, size=t_best.shape)
    keep = (t_best > 0.5) & (t_best < 60.0)
    return (dirs_sensor * t_best[:, None])[keep].astype(np.float32)


def generate_lidar_sequence(out_dir: str | Path, n_scans: int = 40, seed: int = 0) -> Path:
    """Write a synthetic KITTI-odometry-layout LiDAR sequence. Returns its root."""
    out = Path(out_dir)
    (out / "velodyne").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    times, pose_lines = [], []
    tr_inv = np.linalg.inv(_KITTI_TR)
    for i in range(n_scans):
        pose = _sensor_pose(i / max(n_scans - 1, 1))
        points = _lidar_scan(pose, rng)
        scan = np.hstack([points, np.full((len(points), 1), 0.5, dtype=np.float32)])
        scan.astype(np.float32).tofile(out / "velodyne" / f"{i:06d}.bin")
        times.append(f"{i * 0.1:.6f}")
        # KITTI ground-truth poses are expressed in the *camera* frame.
        pose_cam = _KITTI_TR @ pose @ tr_inv
        pose_lines.append(" ".join(f"{v:.9e}" for v in pose_cam[:3, :4].flatten()))

    (out / "times.txt").write_text("\n".join(times) + "\n")
    (out / "poses.txt").write_text("\n".join(pose_lines) + "\n")
    calib_row = " ".join(f"{v:.9e}" for v in _KITTI_TR[:3, :4].flatten())
    (out / "calib.txt").write_text(f"Tr: {calib_row}\n")
    return out
