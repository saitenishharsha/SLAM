# LiDAR configuration

## Config fields (`configs/lidar_slam.yaml`)

```yaml
dataset:
  type: kitti | pointcloud
  root: path/to/dataset

preprocessing:
  voxel_size: 0.2            # meters; downsampling resolution before ICP
  nb_neighbors: 20           # statistical outlier removal: neighbors to check
  std_ratio: 2.0             # statistical outlier removal: std-dev threshold

odometry:
  max_correspondence_distance: 1.0   # meters; ICP point-matching search radius
  use_scan_to_map: true              # false = frame-to-frame only (drifts faster)
  local_map_max_points: 200000       # cap on the rolling local map's size

mapping:
  voxel_size: 0.1             # final map resolution (finer than preprocessing voxel_size)
  downsample_every: 10        # re-voxelize the accumulated map every N scans
```

## Tuning notes

- **`voxel_size` too large** → thin/featureless geometry, ICP correspondences get sloppy,
  trajectory drifts. **Too small** → slow, and outdoor LiDAR noise dominates. 0.1-0.3m is a
  reasonable starting range for outdoor automotive LiDAR (KITTI-scale); indoor/handheld LiDAR
  usually wants smaller, 0.02-0.1m.
- **`max_correspondence_distance`** should be a few times `voxel_size` — too tight and ICP can't
  find correspondences after fast motion; too loose and it matches the wrong structure.
- **`use_scan_to_map: false`** is useful for debugging a new adapter (isolates frame-to-frame
  behavior) but will drift noticeably faster than scan-to-map on anything but short sequences.
- Low `result.fitness` (tracking reported as lost in the script's log) usually means either too
  little overlap between consecutive scans (motion too fast relative to scan rate) or a bad
  initial guess — the current odometry uses a constant-pose initial guess, so very fast rotation
  is the most common failure mode.

## Adding a new LiDAR dataset

See `docs/custom_dataset.md` — the same adapter pattern used for visual datasets applies here
via `LidarDatasetAdapter` in `slam/lidar/dataset_adapter.py`.
