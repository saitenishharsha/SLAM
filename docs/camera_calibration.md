# Camera calibration

The visual pipeline needs pinhole camera intrinsics: `fx, fy, cx, cy, width, height`
(radial/tangential distortion is assumed already removed — undistort first if your source
images aren't already rectified).

## Using a published dataset's calibration

TUM RGB-D, KITTI, EuRoC, and ETH3D all publish per-sequence intrinsics on their dataset pages.
`configs/camera_tum_freiburg1.yaml` is already filled in for the TUM freiburg1 (Kinect v1)
sequences as an example.

## Calibrating your own camera (e.g. a phone)

Use OpenCV's standard checkerboard calibration:

```python
import cv2
import glob
import numpy as np

pattern_size = (9, 6)  # inner corners of your checkerboard
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)

obj_points, img_points = [], []
for fname in glob.glob("calibration_images/*.jpg"):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        obj_points.append(objp)
        img_points.append(corners)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)
print("fx, fy, cx, cy:", K[0, 0], K[1, 1], K[0, 2], K[1, 2])
print("distortion:", dist)
```

Take 15-20 photos of the checkerboard from varied angles/distances first. Save the resulting
`fx, fy, cx, cy, width, height` into a YAML file matching the format of
`configs/camera_tum_freiburg1.yaml`, and reference it from `camera.intrinsics_file` in your
dataset config.

If your phone camera does lens correction/distortion removal automatically (most do for normal
photo/video mode), you can usually skip the `dist` coefficients and treat the output as already
rectified — but it's worth spot-checking that straight lines near the image edges stay straight.
