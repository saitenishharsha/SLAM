"""Trajectory I/O and plotting.

Poses are stored in TUM format: one line per pose,
    timestamp tx ty tz qx qy qz qw
(translation in meters, rotation as a unit quaternion). This format is a
de-facto standard for SLAM trajectories and is what `evo`, the TUM tools,
and most benchmark scripts expect, so files written here can be dropped
straight into those tools too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Trajectory:
    """A sequence of timestamped 4x4 poses (camera/sensor -> world)."""

    timestamps: np.ndarray  # (N,)
    poses: np.ndarray  # (N, 4, 4)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) translation component of each pose."""
        return self.poses[:, :3, 3]

    @classmethod
    def from_tum_file(cls, path: str | Path) -> Trajectory:
        path = Path(path)
        timestamps, poses = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    raise ValueError(f"Malformed TUM trajectory line: {line!r}")
                t, tx, ty, tz, qx, qy, qz, qw = (float(x) for x in parts)
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                pose[:3, 3] = [tx, ty, tz]
                timestamps.append(t)
                poses.append(pose)
        if not poses:
            raise ValueError(f"No poses found in {path}")
        return cls(timestamps=np.array(timestamps), poses=np.stack(poses))

    def to_tum_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for t, pose in zip(self.timestamps, self.poses, strict=True):
                tx, ty, tz = pose[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def plot_trajectory(
    trajectory: Trajectory,
    output_path: str | Path,
    groundtruth: Trajectory | None = None,
    title: str = "Trajectory",
    plane: tuple[int, int] = (0, 2),
) -> None:
    """Save a top-down trajectory plot using the selected coordinate plane.

    If `groundtruth` is given, it is overlaid for visual comparison. This is
    deliberately simple (matplotlib, no interactivity) so it works headless
    in CI/Docker. Camera trajectories typically use X-Z; KITTI LiDAR uses X-Y.
    """
    if len(plane) != 2 or any(axis not in (0, 1, 2) for axis in plane):
        raise ValueError(f"plane must contain two coordinate axes from 0, 1, 2; got {plane}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = trajectory.positions
    ax.plot(pos[:, plane[0]], pos[:, plane[1]], label="Estimate", linewidth=2)
    if groundtruth is not None:
        gt_pos = groundtruth.positions
        ax.plot(
            gt_pos[:, plane[0]],
            gt_pos[:, plane[1]],
            label="Ground truth",
            linewidth=1.5,
            linestyle="--",
        )
    axis_labels = ("X", "Y", "Z")
    ax.set_xlabel(f"{axis_labels[plane[0]]} [m]")
    ax.set_ylabel(f"{axis_labels[plane[1]]} [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
