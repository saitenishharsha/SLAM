"""RGB-D frame loading and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from slam.visual.dataset_adapter import CameraIntrinsics, Frame


@dataclass
class RGBDImage:
    color: np.ndarray  # (H, W, 3) uint8, RGB
    depth: np.ndarray  # (H, W) float32, meters
    timestamp: float


def load_rgbd(frame: Frame, depth_scale: float, depth_trunc: float = 4.0) -> RGBDImage:
    """Load a color/depth pair and convert depth to meters.

    depth_trunc: depths beyond this (meters) are zeroed out — standard practice
    since far/noisy depth readings hurt odometry more than they help.
    """
    color = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"Could not read RGB image: {frame.rgb_path}")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {frame.depth_path}")
    depth = depth_raw.astype(np.float32) / depth_scale
    depth[depth > depth_trunc] = 0.0

    return RGBDImage(color=color, depth=depth, timestamp=frame.timestamp)


def to_open3d_rgbd(image: RGBDImage, depth_trunc: float = 4.0) -> o3d.geometry.RGBDImage:
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(image.color))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(image.depth))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,  # already converted to meters in load_rgbd
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


def point_cloud_from_rgbd(
    image: RGBDImage,
    intrinsics: CameraIntrinsics,
    pose: np.ndarray | None = None,
    depth_trunc: float = 4.0,
) -> o3d.geometry.PointCloud:
    """Back-project an RGB-D frame to a point cloud, optionally transformed into
    the world frame by `pose` (camera -> world).
    """
    rgbd = to_open3d_rgbd(image, depth_trunc)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics.as_open3d())
    if pose is not None:
        pcd.transform(pose)
    return pcd
