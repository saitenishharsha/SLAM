"""Dataset adapters for LiDAR SLAM.

Same boundary as the visual adapters: turn whatever a dataset looks like on
disk into a standard sequence of `Scan` objects. Odometry/mapping code never
touches file formats directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slam.common.config import Config


@dataclass
class Scan:
    """One LiDAR point cloud scan."""

    timestamp: float
    points: np.ndarray  # (N, 3) float32, sensor frame
    groundtruth_pose: np.ndarray | None = None  # 4x4, world <- sensor, if available


class LidarDatasetAdapter(ABC):
    """Interface every LiDAR-dataset adapter must implement."""

    @abstractmethod
    def num_scans(self) -> int: ...

    @abstractmethod
    def load_scan(self, index: int) -> Scan: ...

    def scans(self):
        """Lazily yield all scans in order. Override if your format allows
        a cheaper streaming read than one-by-one random access."""
        for i in range(self.num_scans()):
            yield self.load_scan(i)


class KITTIOdometryAdapter(LidarDatasetAdapter):
    """Adapter for the KITTI Odometry benchmark's velodyne `.bin` scan layout:

        <root>/velodyne/000000.bin, 000001.bin, ...   # float32 x,y,z,reflectance
        <root>/times.txt                                # optional, seconds per scan
        <root>/poses.txt                                # optional, 3x4 row-major per line

    `.bin` files have no embedded timestamp, so `times.txt` (one float per
    line) supplies it; if absent, scan index is used as a surrogate timestamp.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.velodyne_dir = self.root / "velodyne"
        if not self.velodyne_dir.exists():
            raise FileNotFoundError(f"Expected velodyne scans at {self.velodyne_dir}")
        self._scan_files = sorted(self.velodyne_dir.glob("*.bin"))
        if not self._scan_files:
            raise FileNotFoundError(f"No .bin scans found in {self.velodyne_dir}")

        times_file = self.root / "times.txt"
        self._timestamps = (
            [float(x) for x in times_file.read_text().split()]
            if times_file.exists()
            else list(range(len(self._scan_files)))
        )
        if len(self._timestamps) != len(self._scan_files):
            raise ValueError(
                f"times.txt contains {len(self._timestamps)} timestamps for "
                f"{len(self._scan_files)} scans"
            )

        poses_file = self.root / "poses.txt"
        self._gt_poses = self._load_kitti_poses(poses_file) if poses_file.exists() else None
        if self._gt_poses is not None and len(self._gt_poses) != len(self._scan_files):
            raise ValueError(
                f"poses.txt contains {len(self._gt_poses)} poses for {len(self._scan_files)} scans"
            )
        calib_file = self.root / "calib.txt"
        if self._gt_poses is not None and not calib_file.exists():
            raise FileNotFoundError(
                f"{calib_file} is required when poses.txt is present so camera poses can be "
                "converted to the Velodyne frame"
            )
        self._velo_to_cam = (
            self._load_kitti_calibration(calib_file) if calib_file.exists() else np.eye(4)
        )

    def num_scans(self) -> int:
        return len(self._scan_files)

    def load_scan(self, index: int) -> Scan:
        raw = np.fromfile(self._scan_files[index], dtype=np.float32).reshape(-1, 4)
        points = raw[:, :3]
        pose = None
        if self._gt_poses is not None:
            pose = self._gt_poses[index] @ self._velo_to_cam
        return Scan(timestamp=self._timestamps[index], points=points, groundtruth_pose=pose)

    @staticmethod
    def _load_kitti_poses(path: Path) -> list[np.ndarray]:
        poses = []
        with open(path) as f:
            for line in f:
                vals = [float(x) for x in line.split()]
                pose = np.eye(4)
                pose[:3, :4] = np.array(vals).reshape(3, 4)
                poses.append(pose)
        return poses

    @staticmethod
    def _load_kitti_calibration(path: Path) -> np.ndarray:
        for line in path.read_text().splitlines():
            key, separator, values = line.partition(":")
            if key.strip() in {"Tr", "Tr_velo_to_cam"} and separator:
                transform = np.eye(4)
                transform[:3, :4] = np.asarray(values.split(), dtype=float).reshape(3, 4)
                return transform
        raise ValueError(f"No Tr_velo_to_cam entry found in {path}")


class GenericPointCloudAdapter(LidarDatasetAdapter):
    """Adapter for a folder of per-scan point cloud files (`.bin` float32 Nx3/Nx4,
    or `.pcd`/`.ply` readable by Open3D) — the common case for a custom capture.
    """

    def __init__(
        self,
        root: str | Path,
        scans_dir: str = "scans",
        timestamps_file: str = "",
        bin_columns: int = 4,
    ):
        if bin_columns not in (3, 4):
            raise ValueError(f"bin_columns must be 3 or 4, got {bin_columns}")

        self.root = Path(root)
        self.scans_dir = self.root / scans_dir
        self.timestamps_file = (self.root / timestamps_file) if timestamps_file else None
        self.bin_columns = bin_columns
        self._scan_files = sorted(
            p for p in self.scans_dir.iterdir() if p.suffix.lower() in (".bin", ".pcd", ".ply")
        )
        if not self._scan_files:
            raise FileNotFoundError(f"No scan files (.bin/.pcd/.ply) found in {self.scans_dir}")

        self._timestamps = (
            [float(x) for x in self.timestamps_file.read_text().split()]
            if self.timestamps_file and self.timestamps_file.exists()
            else list(range(len(self._scan_files)))
        )
        if len(self._timestamps) != len(self._scan_files):
            raise ValueError(
                f"{self.timestamps_file.name if self.timestamps_file else 'timestamps'} contains "
                f"{len(self._timestamps)} timestamps for {len(self._scan_files)} scans"
            )

    def num_scans(self) -> int:
        return len(self._scan_files)

    def load_scan(self, index: int) -> Scan:
        path = self._scan_files[index]
        if path.suffix.lower() == ".bin":
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size % self.bin_columns:
                raise ValueError(
                    f"Binary scan {path} has {raw.size} values, which is not divisible by "
                    f"bin_columns={self.bin_columns}"
                )
            points = raw.reshape(-1, self.bin_columns)[:, :3]
        else:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points, dtype=np.float32)
        return Scan(timestamp=self._timestamps[index], points=points)


def build_adapter(cfg: Config) -> LidarDatasetAdapter:
    """Construct the right adapter from a loaded `lidar_slam.yaml` config."""
    cfg.require("dataset.type", "dataset.root")
    dataset_type = cfg.dataset.type

    if dataset_type == "kitti":
        return KITTIOdometryAdapter(cfg.dataset.root)
    if dataset_type == "pointcloud":
        return GenericPointCloudAdapter(
            root=cfg.dataset.root,
            scans_dir=cfg.dataset.get("scans_dir", "scans"),
            timestamps_file=cfg.dataset.get("timestamps_file", ""),
            bin_columns=cfg.dataset.get("bin_columns", 4),
        )
    raise ValueError(
        f"Unknown dataset.type '{dataset_type}'. Supported: 'kitti', 'pointcloud'. "
        "Add a new adapter class + case here to support another layout."
    )
