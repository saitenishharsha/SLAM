"""LiDAR point cloud preprocessing: downsampling, outlier removal, normals."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from slam.lidar.dataset_adapter import Scan


def to_open3d(scan: Scan) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan.points.astype(np.float64))
    return pcd


def preprocess_scan(
    scan: Scan,
    voxel_size: float = 0.2,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    estimate_normals: bool = True,
) -> o3d.geometry.PointCloud:
    """Voxel-downsample, remove statistical outliers, and (optionally) estimate
    normals — normals are required for point-to-plane ICP in `odometry.py`.
    """
    pcd = to_open3d(scan)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    return pcd
