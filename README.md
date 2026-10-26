# SLAM
Camera and LiDAR SLAM pipelines for building 3D maps from your own sensor data

Visual SLAM and LiDAR SLAM pipelines for building 3D maps from custom sensor data.

The project separates dataset loading from tracking, odometry, mapping, and evaluation. To use
your own dataset, configure an existing adapter or add a small adapter for your file layout.

## Features

### Visual SLAM

- RGB-D dataset support with TUM-style and generic folder adapters.
- Timestamp association for RGB and depth frames.
- Configurable camera intrinsics, depth scale, and depth range.
- Open3D RGB-D odometry for frame-to-frame tracking.
- TSDF or point-cloud mapping.
- TUM-format trajectory output and ATE evaluation.

### LiDAR SLAM

- KITTI Odometry and generic point-cloud adapters.
- `.bin`, `.pcd`, and `.ply` scan support for generic datasets.
- Voxel downsampling and statistical outlier removal.
- Point-to-plane ICP odometry with optional scan-to-map refinement.
- Incremental point-cloud map generation.
- Trajectory output and ATE evaluation when ground truth is available.

## Requirements

- Python 3.10 or newer.
- Linux or macOS. Windows is not currently tested.
- An Open3D-compatible CPU for running the actual SLAM pipelines.

The automated tests do not require a real dataset. They validate configuration loading, trajectory
evaluation, dataset adapters, and other reusable components.

## Installation

```bash
git clone https://github.com/sandeepmulagiri225-ux/SLAM.git
cd SLAM

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the checks after installation:

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m pip check
```

## Visual SLAM Quick Start

Download the TUM RGB-D Freiburg 1 `xyz` sequence:

```bash
mkdir -p datasets/tum
cd datasets/tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
cd ../..
```

The default configuration already points to this dataset and its camera calibration:

```bash
run-visual-slam --config configs/visual_slam.yaml
```

The output directory is configured in `configs/visual_slam.yaml`. A successful run can produce:

```text
trajectory.txt
groundtruth.txt
trajectory.png
map.ply
map_mesh.ply
```

## LiDAR SLAM Quick Start

Download KITTI Odometry sequence data and arrange it as follows:

```text
datasets/kitti/sequences/00/
├── velodyne/*.bin
├── times.txt
├── poses.txt        # required for ground-truth evaluation
└── calib.txt        # required when poses.txt is present
```

Then run:

```bash
run-lidar-slam --config configs/lidar_slam.yaml
```

KITTI ground-truth poses are camera-frame poses. The calibration file is used to convert them to
the Velodyne scan frame before evaluation.

## Custom Datasets

For a generic RGB-D dataset, configure:

```yaml
dataset:
	type: rgbd
	root: /data/my_rgbd_dataset
	rgb_dir: rgb
	depth_dir: depth
	timestamps_file: timestamps.txt  # optional
```

For a generic point-cloud dataset, configure:

```yaml
dataset:
	type: pointcloud
	root: /data/my_lidar_dataset
	scans_dir: scans
	timestamps_file: timestamps.txt  # optional
	bin_columns: 4                  # 3 for XYZ, 4 for XYZR binary scans
```

See [docs/custom_dataset.md](docs/custom_dataset.md), [docs/camera_calibration.md](docs/camera_calibration.md),
and [docs/lidar_configuration.md](docs/lidar_configuration.md) for details.

## Docker

Docker packages the Python dependencies and system libraries into a reproducible runtime. It does
not include datasets, which are mounted from the host at runtime.

Build the image:

```bash
docker build -t slam:latest -f docker/Dockerfile .
```

Run the tests inside the image:

```bash
docker run --rm slam:latest -m pytest -q
```

Run Visual SLAM with host datasets and output directories mounted:

```bash
docker run --rm \
	-v "$(pwd)/datasets:/workspace/datasets" \
	-v "$(pwd)/results:/workspace/results" \
	slam:latest \
	-m scripts.run_visual_slam \
	--config configs/visual_slam.yaml
```

Docker does not change the host CPU instruction set. Open3D may require a newer CPU than the
machine running Docker provides.

## Trajectory Evaluation

Evaluate an estimated TUM-format trajectory against ground truth:

```bash
evaluate-trajectory \
	--estimate results/visual_slam_tum_fr1_xyz/trajectory.txt \
	--groundtruth results/visual_slam_tum_fr1_xyz/groundtruth.txt
```

## Repository Structure

```text
SLAM/
├── configs/       Example Visual SLAM and LiDAR SLAM configurations
├── docker/        Dockerfile for reproducible runtime setup
├── docs/          Installation and dataset documentation
├── examples/      Dataset-specific usage examples
├── scripts/       Installed command-line entry points
├── slam/          Reusable SLAM library code
│   ├── common/    Configuration, trajectories, evaluation, and utilities
│   ├── lidar/     LiDAR adapters, preprocessing, odometry, and mapping
│   └── visual/    RGB-D adapters, preprocessing, tracking, and mapping
├── tests/         Automated tests
├── datasets/      Local datasets; contents are excluded from Git
└── results/       Generated outputs; contents are excluded from Git
```

## Contributors

This project was developed collaboratively by:

- **[Sandeep Mulagiri](https://github.com/sandeepmulagiri225-ux)** — Contributed to the development and implementation of the Visual SLAM and LiDAR SLAM pipelines.
- **[Saitenish Harsha Lankipalli](https://github.com/saitenishharsha)** — Contributed to the development and implementation of the SLAM project.

Both contributors collaborated on building reusable pipelines for 3D mapping from custom RGB-D camera and LiDAR sensor data.


