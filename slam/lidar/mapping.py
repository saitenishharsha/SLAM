"""Incremental 3D map construction from tracked LiDAR scans."""

from __future__ import annotations

import numpy as np
import open3d as o3d


class LidarMapper:
    """Accumulates world-frame scans into a single map point cloud,
    periodically voxel-downsampled to keep memory bounded on long sequences.
    """

    def __init__(self, voxel_size: float = 0.1, downsample_every: int = 10):
        self.voxel_size = voxel_size
        self.downsample_every = downsample_every
        self._map = o3d.geometry.PointCloud()
        self._scans_since_downsample = 0

    def integrate(self, scan: o3d.geometry.PointCloud, pose: np.ndarray) -> None:
        """pose: sensor frame -> world."""
        scan_world = o3d.geometry.PointCloud(scan).transform(pose)
        self._map += scan_world
        self._scans_since_downsample += 1
        if self._scans_since_downsample >= self.downsample_every:
            self._map = self._map.voxel_down_sample(self.voxel_size)
            self._scans_since_downsample = 0

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        return self._map.voxel_down_sample(self.voxel_size)
