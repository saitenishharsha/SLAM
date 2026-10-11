# datasets/

This folder is intentionally **not tracked in git** (see `.gitignore`) — datasets are large and
dataset-specific, and don't belong in the repo's history.

Download instructions for the two example datasets used by the default configs are in
`docs/installation.md`. For your own data, see `docs/custom_dataset.md`.

Expected layout for the default configs:

```
datasets/
├── tum/
│   └── rgbd_dataset_freiburg1_xyz/
│       ├── rgb.txt
│       ├── depth.txt
│       ├── groundtruth.txt
│       ├── rgb/
│       └── depth/
└── kitti/
    └── sequences/
        └── 00/
            ├── velodyne/
            ├── times.txt
            ├── poses.txt
            └── calib.txt             # required with poses.txt
```
