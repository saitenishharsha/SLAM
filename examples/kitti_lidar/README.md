# Example: LiDAR SLAM on KITTI Odometry

1. Register (free) and download sequence `00`'s velodyne laser data + ground-truth poses from
   <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>.
2. Lay it out as:

   ```
   datasets/kitti/sequences/00/
   ├── velodyne/*.bin
   ├── times.txt
   ├── poses.txt
   └── calib.txt
   ```

3. Run:

   ```bash
   run-lidar-slam --config configs/lidar_slam.yaml
   ```

The default `configs/lidar_slam.yaml` already points at this sequence. Since KITTI sequence 00
has public ground truth, the run prints an ATE RMSE and writes a comparison plot.

Note: sequence 00 is ~4.5k scans and scan-to-map ICP over the full sequence will take a while on
CPU — start with a shorter sequence (e.g. 04, which is a few hundred scans) to sanity-check your
setup first by changing `dataset.root` to `datasets/kitti/sequences/04`.
