"""Dataset adapters for Visual SLAM.

An adapter's only job is to turn whatever a dataset looks like on disk into
a standard sequence of `Frame` objects. The tracking/mapping code never
knows (or cares) which dataset it came from — this is the "Bring Your Own
Dataset" boundary.

To support a new dataset layout, write a new adapter class that implements
`VisualDatasetAdapter` and register it in `build_adapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from slam.common.config import Config


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> CameraIntrinsics:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            width=data["width"],
            height=data["height"],
        )

    def as_open3d(self):
        import open3d as o3d

        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )


@dataclass
class Frame:
    """One synchronized RGB-D observation."""

    timestamp: float
    rgb_path: Path
    depth_path: Path
    groundtruth_pose: np.ndarray | None = None  # 4x4, world <- camera, if available


class VisualDatasetAdapter(ABC):
    """Interface every visual-dataset adapter must implement."""

    @abstractmethod
    def frames(self) -> list[Frame]:
        """Return the full, timestamp-sorted sequence of frames."""

    @abstractmethod
    def intrinsics(self) -> CameraIntrinsics:
        """Return the camera intrinsics for this sequence."""

    @abstractmethod
    def depth_scale(self) -> float:
        """Factor to divide raw depth-image values by to get meters."""


def _read_tum_list(path: Path) -> list[tuple[float, str]]:
    """Parse a TUM-style `rgb.txt` / `depth.txt` file: `timestamp path` per line."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t, rel_path = line.split(maxsplit=1)
            entries.append((float(t), rel_path))
    return entries


def _associate(rgb_list, depth_list, max_diff: float = 0.02) -> list[tuple[float, str, str]]:
    """Match RGB and depth entries by nearest timestamp (TUM `associate.py` logic)."""
    candidates = sorted(
        (abs(t_rgb - t_depth), i, j)
        for i, (t_rgb, _) in enumerate(rgb_list)
        for j, (t_depth, _) in enumerate(depth_list)
        if abs(t_rgb - t_depth) <= max_diff
    )
    used_rgb, used_depth = set(), set()
    matched = []
    for _, i, j in candidates:
        if i in used_rgb or j in used_depth:
            continue
        t_rgb, rgb_path = rgb_list[i]
        matched.append((t_rgb, rgb_path, depth_list[j][1]))
        used_rgb.add(i)
        used_depth.add(j)
    return sorted(matched)


class TUMRGBDAdapter(VisualDatasetAdapter):
    """Adapter for the TUM RGB-D SLAM Dataset layout:

        <root>/rgb.txt            # "timestamp rgb/xxxx.png" per line
        <root>/depth.txt          # "timestamp depth/xxxx.png" per line
        <root>/groundtruth.txt    # optional, TUM pose format
        <root>/rgb/*.png
        <root>/depth/*.png

    Depth images are 16-bit PNGs in millimeters (depth_scale = 5000 for
    most TUM sequences — see the sequence's own README).
    """

    def __init__(
        self,
        root: str | Path,
        intrinsics: CameraIntrinsics,
        depth_scale: float = 5000.0,
        max_time_diff: float = 0.02,
    ):
        self.root = Path(root)
        self._intrinsics = intrinsics
        self._depth_scale = depth_scale
        self._max_time_diff = max_time_diff

    def frames(self) -> list[Frame]:
        rgb_list = _read_tum_list(self.root / "rgb.txt")
        depth_list = _read_tum_list(self.root / "depth.txt")
        matched = _associate(rgb_list, depth_list, self._max_time_diff)
        if not matched:
            raise ValueError(
                f"No RGB/depth matches found under {self.root} "
                f"(max_time_diff={self._max_time_diff}s)"
            )

        gt_path = self.root / "groundtruth.txt"
        gt_poses = self._load_groundtruth(gt_path) if gt_path.exists() else None

        frames = []
        for t, rgb_rel, depth_rel in matched:
            pose = None
            if gt_poses is not None:
                gt_times = np.array([p[0] for p in gt_poses])
                j = int(np.argmin(np.abs(gt_times - t)))
                if abs(gt_times[j] - t) <= self._max_time_diff:
                    pose = gt_poses[j][1]
            frames.append(
                Frame(
                    timestamp=t,
                    rgb_path=self.root / rgb_rel,
                    depth_path=self.root / depth_rel,
                    groundtruth_pose=pose,
                )
            )
        return frames

    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def depth_scale(self) -> float:
        return self._depth_scale

    @staticmethod
    def _load_groundtruth(path: Path) -> list[tuple[float, np.ndarray]]:
        from scipy.spatial.transform import Rotation

        poses = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                t, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in line.split())
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                pose[:3, 3] = [tx, ty, tz]
                poses.append((t, pose))
        return poses


class GenericRGBDAdapter(VisualDatasetAdapter):
    """Adapter for any RGB-D dataset already laid out like TUM
    (rgb_dir/, depth_dir/, timestamps_file) but under different folder names —
    the common case for a custom capture. Configure via `configs/visual_slam.yaml`.

    If no timestamp file is configured, frame order is used as the timestamp.
    """

    def __init__(
        self,
        root: str | Path,
        rgb_dir: str,
        depth_dir: str,
        timestamps_file: str,
        intrinsics: CameraIntrinsics,
        depth_scale: float = 1000.0,
    ):
        self.root = Path(root)
        self.rgb_dir = self.root / rgb_dir
        self.depth_dir = self.root / depth_dir
        timestamp_path = self.root / timestamps_file if timestamps_file else None
        self.timestamps_file = timestamp_path
        self._intrinsics = intrinsics
        self._depth_scale = depth_scale

    def frames(self) -> list[Frame]:
        image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
        rgb_files = sorted(
            path
            for path in self.rgb_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
        depth_files = sorted(
            path
            for path in self.depth_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
        if len(rgb_files) != len(depth_files):
            raise ValueError(
                f"RGB ({len(rgb_files)}) and depth ({len(depth_files)}) frame "
                "counts differ — generic adapter assumes a 1:1, already-synced layout."
            )
        if self.timestamps_file is not None and self.timestamps_file.exists():
            timestamps = [float(x) for x in self.timestamps_file.read_text().split()]
        else:
            timestamps = list(range(len(rgb_files)))
        if len(timestamps) != len(rgb_files):
            raise ValueError(
                f"Timestamp count ({len(timestamps)}) does not match frame count ({len(rgb_files)})"
            )
        return [
            Frame(timestamp=t, rgb_path=r, depth_path=d)
            for t, r, d in zip(timestamps, rgb_files, depth_files, strict=True)
        ]

    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def depth_scale(self) -> float:
        return self._depth_scale


def build_adapter(cfg: Config) -> VisualDatasetAdapter:
    """Construct the right adapter from a loaded `visual_slam.yaml` config."""
    cfg.require("dataset.type", "dataset.root", "camera.intrinsics_file")
    dataset_type = cfg.dataset.type
    intrinsics = CameraIntrinsics.from_yaml(cfg.camera.intrinsics_file)
    depth_scale = cfg.camera.get("depth_scale", 5000.0)

    if dataset_type == "tum_rgbd":
        return TUMRGBDAdapter(cfg.dataset.root, intrinsics, depth_scale)
    if dataset_type == "rgbd":
        cfg.require("dataset.rgb_dir", "dataset.depth_dir")
        return GenericRGBDAdapter(
            root=cfg.dataset.root,
            rgb_dir=cfg.dataset.rgb_dir,
            depth_dir=cfg.dataset.depth_dir,
            timestamps_file=cfg.dataset.get("timestamps_file", ""),
            intrinsics=intrinsics,
            depth_scale=depth_scale,
        )
    raise ValueError(
        f"Unknown dataset.type '{dataset_type}'. Supported: 'tum_rgbd', 'rgbd'. "
        "Add a new adapter class + case here to support another layout."
    )
