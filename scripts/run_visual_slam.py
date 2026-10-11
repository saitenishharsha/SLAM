#!/usr/bin/env python3
"""Run the Visual SLAM pipeline end to end: load config, track camera pose
over a dataset, fuse a 3D map, and write trajectory/map/plot outputs.

Usage:
    run-visual-slam --config configs/visual_slam.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a visual_slam.yaml config")
    args = parser.parse_args()

    import numpy as np
    import open3d as o3d

    from slam.common.config import Config
    from slam.common.trajectory import Trajectory, plot_trajectory
    from slam.visual.dataset_adapter import build_adapter
    from slam.visual.mapping import PointCloudMapper, TSDFMapper
    from slam.visual.preprocessing import load_rgbd
    from slam.visual.tracking import RGBDTracker

    cfg = Config.load(args.config)
    cfg.require("dataset.type", "dataset.root", "camera.intrinsics_file", "output.directory")

    adapter = build_adapter(cfg)
    frames = adapter.frames()
    if not frames:
        print(f"No frames found for dataset at {cfg.dataset.root}", file=sys.stderr)
        return 1
    print(f"Loaded {len(frames)} synchronized RGB-D frames from {cfg.dataset.root}")

    intrinsics = adapter.intrinsics()
    depth_scale = adapter.depth_scale()
    depth_trunc = cfg.camera.get("depth_trunc", 4.0)

    tracker = RGBDTracker(intrinsics, depth_trunc=depth_trunc)

    mapping_method = cfg.get("mapping", Config({})).get("method", "tsdf")
    voxel_size = cfg.get("mapping", Config({})).get("voxel_size", 0.01)
    if mapping_method == "tsdf":
        mapper = TSDFMapper(intrinsics, voxel_size=voxel_size, depth_trunc=depth_trunc)
    elif mapping_method == "pointcloud":
        mapper = PointCloudMapper(voxel_size=voxel_size, depth_trunc=depth_trunc)
    else:
        raise ValueError("mapping.method must be either 'tsdf' or 'pointcloud'")

    timestamps, poses = [], []
    gt_timestamps, gt_poses = [], []
    lost_count = 0
    t_start = time.time()

    for i, frame in enumerate(frames):
        image = load_rgbd(frame, depth_scale=depth_scale, depth_trunc=depth_trunc)
        pose, ok = tracker.track(image)
        if not ok:
            lost_count += 1

        timestamps.append(frame.timestamp)
        poses.append(pose)

        if ok:
            if mapping_method == "tsdf":
                mapper.integrate(image, pose)
            else:
                mapper.integrate(image, pose, intrinsics)

        if frame.groundtruth_pose is not None:
            gt_timestamps.append(frame.timestamp)
            gt_poses.append(frame.groundtruth_pose)

        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            print(f"  tracked {i + 1}/{len(frames)} frames ({lost_count} lost so far)")

    elapsed = time.time() - t_start
    print(
        f"Tracking done in {elapsed:.1f}s "
        f"({len(frames) / elapsed:.1f} FPS), {lost_count} frames lost."
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
            trajectory, plot_path, groundtruth=gt_trajectory, title="Visual SLAM trajectory"
        )
        print(f"Saved trajectory plot: {plot_path}")

    if cfg.output.get("save_pointcloud", True):
        pcd = mapper.extract_point_cloud()
        pcd_path = out_dir / "map.ply"
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        pcd_pcd_path = out_dir / "map.pcd"
        o3d.io.write_point_cloud(str(pcd_pcd_path), pcd)
        print(f"Saved point cloud map ({len(pcd.points)} points): {pcd_path}")

    if mapping_method == "tsdf" and cfg.output.get("save_mesh", False):
        mesh = mapper.extract_mesh()
        mesh_path = out_dir / "map_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Saved mesh: {mesh_path}")

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
