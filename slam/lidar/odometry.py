"""LiDAR odometry via point-to-plane ICP.

Frame-to-frame ICP alone drifts quickly on long sequences, so this also
keeps a small rolling local map (in world coordinates) and aligns each new
scan directly against it (scan-to-map), which is the standard cheap way to
cut drift without a full pose-graph/loop-closure backend. Swapping in a
different registration method (NDT, GICP, a learned backend, ...) only means
replacing `LidarOdometry.track` — adapters and mapping are unaffected.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


class LidarOdometry:
    def __init__(
        self,
        voxel_size: float = 0.2,
        max_correspondence_distance: float = 1.0,
        use_scan_to_map: bool = True,
        local_map_max_points: int = 200_000,
    ):
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        self.use_scan_to_map = use_scan_to_map
        self.local_map_max_points = local_map_max_points

        self._prev_scan: o3d.geometry.PointCloud | None = (
            None  # sensor frame, used if not scan-to-map
        )
        self._local_map: o3d.geometry.PointCloud | None = None  # world frame
        self._world_pose = np.eye(4)  # current scan's sensor frame -> world

    def track(self, scan: o3d.geometry.PointCloud) -> tuple[np.ndarray, bool]:
        """Feed the next preprocessed, normal-estimated scan (sensor frame).
        Returns (pose_in_world, tracking_ok). First call defines the map origin.
        """
        if self._prev_scan is None:
            self._prev_scan = scan
            self._local_map = o3d.geometry.PointCloud(scan)  # already at identity pose = world
            return self._world_pose.copy(), True

        # Constant-pose initial guess: assume the sensor hasn't moved much
        # since the last frame. registration_icp solves source(scan, sensor
        # frame) -> target(world frame), so `init` must map sensor -> world.
        init_guess = self._world_pose.copy()

        if self.use_scan_to_map:
            target = self._local_map
        else:
            target = o3d.geometry.PointCloud(self._prev_scan).transform(self._world_pose)

        result = o3d.pipelines.registration.registration_icp(
            scan,
            target,
            self.max_correspondence_distance,
            init_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        ok = result.fitness > 0.1  # heuristic: too little overlap = tracking lost
        if ok:
            self._world_pose = result.transformation
        self._prev_scan = scan

        if ok and self.use_scan_to_map:
            self._update_local_map(scan)

        return self._world_pose.copy(), ok

    def _update_local_map(self, scan: o3d.geometry.PointCloud) -> None:
        scan_world = o3d.geometry.PointCloud(scan).transform(self._world_pose)
        self._local_map += scan_world
        if len(self._local_map.points) > self.local_map_max_points:
            self._local_map = self._local_map.voxel_down_sample(self.voxel_size)
