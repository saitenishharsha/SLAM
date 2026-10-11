"""Dense 3D reconstruction from tracked RGB-D frames.

Two mapping modes are provided:
- TSDF volumetric fusion (recommended): produces a clean, denoised mesh/point
  cloud and is memory-bounded regardless of sequence length.
- Raw point cloud concatenation: faster, keeps every back-projected point —
  useful for short sequences or debugging tracking quality frame by frame.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from slam.visual.dataset_adapter import CameraIntrinsics
from slam.visual.preprocessing import RGBDImage, point_cloud_from_rgbd, to_open3d_rgbd


class TSDFMapper:
    def __init__(
        self, intrinsics: CameraIntrinsics, voxel_size: float = 0.01, depth_trunc: float = 4.0
    ):
        self.intrinsics = intrinsics.as_open3d()
        self.depth_trunc = depth_trunc
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(self, image: RGBDImage, pose: np.ndarray) -> None:
        """pose: camera -> world. Open3D's integrate wants the inverse (world -> camera)."""
        rgbd = to_open3d_rgbd(image, self.depth_trunc)
        extrinsic = np.linalg.inv(pose)
        self.volume.integrate(rgbd, self.intrinsics, extrinsic)

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        return self.volume.extract_point_cloud()

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


class PointCloudMapper:
    """Simpler alternative: concatenate per-frame point clouds, voxel-downsampled
    periodically to keep memory bounded on long sequences.
    """

    def __init__(self, voxel_size: float = 0.01, depth_trunc: float = 4.0):
        self.voxel_size = voxel_size
        self.depth_trunc = depth_trunc
        self._map = o3d.geometry.PointCloud()
        self._frames_since_downsample = 0

    def integrate(self, image: RGBDImage, pose: np.ndarray, intrinsics: CameraIntrinsics) -> None:
        pcd = point_cloud_from_rgbd(image, intrinsics, pose=pose, depth_trunc=self.depth_trunc)
        self._map += pcd
        self._frames_since_downsample += 1
        if self._frames_since_downsample >= 20:
            self._map = self._map.voxel_down_sample(self.voxel_size)
            self._frames_since_downsample = 0

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        return self._map.voxel_down_sample(self.voxel_size)
