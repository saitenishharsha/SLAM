#!/usr/bin/env python3
"""Run the LiDAR SLAM pipeline end to end: load config, track sensor pose
over a dataset, build a 3D map, and write trajectory/map/plot outputs.

Usage:
    run-lidar-slam --config configs/lidar_slam.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a lidar_slam.yaml config")
    args = parser.parse_args()

    import numpy as np
    import open3d as o3d

    from slam.common.config import Config
    from slam.common.trajectory import Trajectory, plot_trajectory
    from slam.lidar.dataset_adapter import build_adapter
    from slam.lidar.mapping import LidarMapper
    from slam.lidar.odometry import LidarOdometry
    from slam.lidar.preprocessing import preprocess_scan

    cfg = Config.load(args.config)
    cfg.require("dataset.type", "dataset.root", "output.directory")

    adapter = build_adapter(cfg)
    n_scans = adapter.num_scans()
    if n_scans == 0:
        print(f"No scans found for dataset at {cfg.dataset.root}", file=sys.stderr)
        return 1
    print(f"Loaded {n_scans} LiDAR scans from {cfg.dataset.root}")

    pre_cfg = cfg.get("preprocessing", Config({}))
    voxel_size = pre_cfg.get("voxel_size", 0.2)

    odom_cfg = cfg.get("odometry", Config({}))
    odometry = LidarOdometry(
        voxel_size=voxel_size,
        max_correspondence_distance=odom_cfg.get("max_correspondence_distance", 1.0),
        use_scan_to_map=odom_cfg.get("use_scan_to_map", True),
        local_map_max_points=odom_cfg.get("local_map_max_points", 200_000),
    )

    map_cfg = cfg.get("mapping", Config({}))
    mapper = LidarMapper(
        voxel_size=map_cfg.get("voxel_size", 0.1),
        downsample_every=map_cfg.get("downsample_every", 10),
    )

    timestamps, poses = [], []
    gt_timestamps, gt_poses = [], []
    lost_count = 0
    t_start = time.time()

    for i, scan in enumerate(adapter.scans()):
        pcd = preprocess_scan(
            scan,
            voxel_size=voxel_size,
            nb_neighbors=pre_cfg.get("nb_neighbors", 20),
            std_ratio=pre_cfg.get("std_ratio", 2.0),
        )
        pose, ok = odometry.track(pcd)
        if not ok:
            lost_count += 1

        timestamps.append(scan.timestamp)
        poses.append(pose)
        if ok:
            mapper.integrate(pcd, pose)

        if scan.groundtruth_pose is not None:
            gt_timestamps.append(scan.timestamp)
            gt_poses.append(scan.groundtruth_pose)

        if (i + 1) % 50 == 0 or i == n_scans - 1:
            print(f"  tracked {i + 1}/{n_scans} scans ({lost_count} lost so far)")

    elapsed = time.time() - t_start
    print(
        f"Odometry done in {elapsed:.1f}s "
        f"({n_scans / elapsed:.1f} scans/s), {lost_count} scans lost."
    )

    trajectory = Trajectory(timestamps=np.array(timestamps), poses=np.stack(poses))

    out_dir = Path(cfg.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output.get("save_trajectory", True):
        traj_path = out_dir / "trajectory.txt"
        trajectory.to_tum_file(traj_path)
        print(f"Saved trajectory: {traj_path}")

    gt_trajectory = None
    if gt_poses:
        gt_trajectory = Trajectory(timestamps=np.array(gt_timestamps), poses=np.stack(gt_poses))
        gt_trajectory.to_tum_file(out_dir / "groundtruth.txt")

    if cfg.output.get("save_trajectory_plot", True):
        plot_path = out_dir / "trajectory.png"
        plot_trajectory(
            trajectory,
            plot_path,
            groundtruth=gt_trajectory,
            title="LiDAR SLAM trajectory",
            plane=(0, 1),
        )
        print(f"Saved trajectory plot: {plot_path}")

    if cfg.output.get("save_pointcloud", True):
        pcd = mapper.extract_point_cloud()
        pcd_path = out_dir / "map.ply"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"Saved point cloud map ({len(pcd.points)} points): {pcd_path}")

    if gt_trajectory is not None:
        from slam.common.evaluation import compute_ate

        try:
            ate = compute_ate(trajectory, gt_trajectory)
            print(f"ATE RMSE: {ate.rmse:.4f} m  (mean {ate.mean:.4f}, median {ate.median:.4f})")
        except ValueError as e:
            print(f"Could not compute ATE: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
