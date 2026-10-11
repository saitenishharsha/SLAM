# Example: Visual SLAM on TUM RGB-D

```bash
mkdir -p datasets/tum && cd datasets/tum
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
cd ../..

run-visual-slam --config configs/visual_slam.yaml
```

The default `configs/visual_slam.yaml` already points at this sequence and its calibration
(`configs/camera_tum_freiburg1.yaml`). Since the sequence ships ground truth, the run prints an
ATE RMSE at the end and writes `groundtruth.txt` + a comparison plot alongside the estimate in
`results/visual_slam_tum_fr1_xyz/`.

Try a harder sequence next by changing `dataset.root` to
`datasets/tum/rgbd_dataset_freiburg1_desk` (download it the same way) — it has loop closures
that this repo's frame-to-frame tracker does not explicitly close, so you'll see drift grow
over the trajectory; that's expected and a good illustration of why scan-to-map/loop-closure
matters.
