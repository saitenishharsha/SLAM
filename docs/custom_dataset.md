# Bringing your own dataset

The framework only knows about the standard `Frame` (visual) and `Scan` (LiDAR) types defined
in `slam/visual/dataset_adapter.py` and `slam/lidar/dataset_adapter.py`. Everything downstream
(tracking, mapping, evaluation) works off those types, not off file formats. So there are two
ways to plug in your own data:

1. **Your layout already matches a generic adapter** (`rgbd` or `pointcloud`) — just point a
   config at it.
2. **Your layout is different** — write a small adapter class (usually ~30-60 lines).

## Option 1: generic RGB-D layout

If you already have synced RGB + depth images, use `dataset.type: rgbd`:

```yaml
dataset:
  type: rgbd
  root: /data/my_dataset
  rgb_dir: rgb          # folder of images, sorted = frame order
  depth_dir: depth      # folder of 16-bit depth PNGs, same count and order as rgb_dir
  timestamps_file: timestamps.txt   # optional; one float per line, same order

camera:
  intrinsics_file: configs/my_camera.yaml
  depth_scale: 1000.0    # 1000 = millimeters -> meters; set to your depth units
```

`configs/my_camera.yaml` needs `fx, fy, cx, cy, width, height` — see
`docs/camera_calibration.md` for how to get these.

## Option 2: generic point cloud layout

If you have a folder of per-scan `.bin` (float32 Nx3 or Nx4), `.pcd`, or `.ply` files, use
`dataset.type: pointcloud`:

```yaml
dataset:
  type: pointcloud
  root: /data/my_lidar_dataset
  scans_dir: scans
  timestamps_file: timestamps.txt   # optional
  bin_columns: 4                    # 3 for XYZ, 4 for XYZR binary scans
```

## Option 3: write your own adapter

Subclass `VisualDatasetAdapter` or `LidarDatasetAdapter` and implement its required methods —
see `TUMRGBDAdapter` / `KITTIOdometryAdapter` in the respective `dataset_adapter.py` for a
complete example. Then add a case for it in that file's `build_adapter()` function so it's
selectable from a config via `dataset.type`.

## Using a single phone video (no depth sensor)

The RGB-D visual pipeline in this repo needs a depth channel per frame — a phone's normal video
camera doesn't record one. You have two practical options:

- **Monocular SLAM/SfM instead of RGB-D.** Feed the video frames into a monocular pipeline
  (e.g. ORB-SLAM3 in monocular mode, or COLMAP for a full structure-from-motion
  reconstruction) to get camera poses and a sparse/dense point cloud without depth input.
  This repo's `slam/visual` module is written for RGB-D; adding a monocular tracking path is a
  natural next adapter/tracker to add (replace `RGBDTracker` with a feature-based monocular
  tracker — the dataset adapter boundary in `dataset_adapter.py` would also change, since there's
  no depth to synchronize).
- **Generate depth first, then use the RGB-D pipeline as-is.** Run a monocular depth estimator
  (e.g. an off-the-shelf depth model) on each extracted video frame to produce a depth map per
  frame, save those as 16-bit PNGs, and point `dataset.type: rgbd` at the resulting
  `rgb/` + `depth/` folders. This is the faster route to a working demo with the code already
  in this repo, though depth quality (and so map quality) will be lower than a real sensor's.

Either way, extract frames first:

```bash
ffmpeg -i my_video.mov -vf fps=10 datasets/my_video/rgb/%06d.png
```
