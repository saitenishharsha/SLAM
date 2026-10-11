# Installation

## Requirements

- Python 3.10+
- Linux/macOS (Open3D wheels also exist for Windows, untested here)

## Install

```bash
git clone <your-repo-url> slam
cd slam
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

This installs the package in editable mode plus `ruff`, `pytest`, and `pre-commit`.

Enable the lint/format pre-commit hook (optional but recommended, matches CI):

```bash
pre-commit install
```

## Docker (no local Python setup needed)

```bash
docker build -t slam -f docker/Dockerfile .
docker run --rm -v "$(pwd)/datasets:/workspace/datasets" \
           -v "$(pwd)/results:/workspace/results" \
           slam -m scripts.run_visual_slam --config configs/visual_slam.yaml
```

## Get a dataset

### Visual SLAM — TUM RGB-D (recommended first dataset)

```bash
mkdir -p datasets/tum && cd datasets/tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
```

This gives you `rgb.txt`, `depth.txt`, `rgb/`, `depth/`, and `groundtruth.txt` directly under
`datasets/tum/rgbd_dataset_freiburg1_xyz/` — matching `dataset.type: tum_rgbd` in
`configs/visual_slam.yaml` with no changes needed. The matching camera intrinsics are already
in `configs/camera_tum_freiburg1.yaml`.

Other good TUM sequences to try once `xyz` works: `rgbd_dataset_freiburg1_desk` (tests loop
closure), `rgbd_dataset_freiburg3_long_office_household` (longer trajectory, larger space).

### LiDAR SLAM — KITTI Odometry

The KITTI Odometry benchmark requires a (free) registration to download. Get the "velodyne
laser data" + "odometry ground truth poses" downloads from
<https://www.cvlibs.net/datasets/kitti/eval_odometry.php>, then lay sequence `00` out as:

```
datasets/kitti/sequences/00/
├── velodyne/*.bin
├── times.txt
├── poses.txt        # only sequences 00-10 have public ground truth
└── calib.txt        # required when poses.txt is present
```

This matches `dataset.type: kitti` in `configs/lidar_slam.yaml` with no changes needed.

## Run it

```bash
run-visual-slam --config configs/visual_slam.yaml
run-lidar-slam  --config configs/lidar_slam.yaml
```

Outputs land directly in the configured `output.directory`: `trajectory.txt` (TUM format),
`map.ply` (point cloud, Visual SLAM also writes `map_mesh.ply`), and `trajectory.png`.
If the dataset had ground truth, ATE is printed at the end and `groundtruth.txt` is saved
alongside the estimate.
