"""Frame-to-frame camera tracking for RGB-D Visual SLAM.

Uses Open3D's RGB-D odometry (hybrid photometric + geometric term), which is
a solid, dependency-light baseline. Swapping in a different backend (e.g. an
ORB-feature-based tracker, or wrapping ORB-SLAM3) only means replacing
`RGBDTracker.track` — everything else (adapters, mapping, evaluation) is
unaffected, by design.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from slam.visual.dataset_adapter import CameraIntrinsics
from slam.visual.preprocessing import RGBDImage, to_open3d_rgbd


class RGBDTracker:
    """Estimates camera pose for each frame relative to the first frame."""

    def __init__(self, intrinsics: CameraIntrinsics, depth_trunc: float = 4.0):
        self.intrinsics = intrinsics.as_open3d()
        self.depth_trunc = depth_trunc
        self.option = o3d.pipelines.odometry.OdometryOption()
        self._prev_rgbd: o3d.geometry.RGBDImage | None = None
        self._world_pose = np.eye(4)  # current frame -> world

    def track(self, image: RGBDImage) -> tuple[np.ndarray, bool]:
        """Feed the next frame. Returns (pose_in_world, tracking_ok).

        The first call always succeeds trivially (identity pose, defines the
        map origin). Subsequent calls estimate the relative transform from
        the previous frame and compose it onto the running world pose.
        """
        rgbd = to_open3d_rgbd(image, self.depth_trunc)

        if self._prev_rgbd is None:
            self._prev_rgbd = rgbd
            return self._world_pose.copy(), True

        success, T_prev_cur, _info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd,
            self._prev_rgbd,
            self.intrinsics,
            np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            self.option,
        )
        self._prev_rgbd = rgbd

        if not success:
            # Tracking lost on this frame: hold the last known pose rather than
            # inserting a wild/undefined transform into the trajectory.
            return self._world_pose.copy(), False

        # T_prev_cur maps current-frame points into the previous frame; compose
        # to get current-frame -> world.
        self._world_pose = self._world_pose @ T_prev_cur
        return self._world_pose.copy(), True
