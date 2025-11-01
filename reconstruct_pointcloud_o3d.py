#!/usr/bin/env python3
"""
Reconstruct and visualize a 3D point cloud in the WORLD frame for a given frame index
using data recorded by `isaac_data_recorder_husky.py`.

Inputs expected (by default):
- Depth PNG:  /workspace/isaaclab/IsaacSimData/<scene_name>/results/depth%06d.png
- RGB  JPG:   /workspace/isaaclab/IsaacSimData/<scene_name>/results/frame%06d.jpg (optional for color)
- Trajectory: /workspace/isaaclab/IsaacSimData/<scene_name>/traj.txt (each line is Twc row-major)

Usage example:
  python reconstruct_pointcloud_o3d.py -s UR5-Peg-In-Hole_02 -f 12 \
         --fx 1550.0 --fy 1552.0 --cx 640.0 --cy 360.0 \
         --zmin 0.01 --zmax 10.0 --stride 2 --save_ply world_pc_frame12.ply

If you have an intrinsics json saved as {"fx":..., "fy":..., "cx":..., "cy":...},
you can use --intrinsics intrinsics.json instead of passing the four numbers.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit(
        "open3d is required. Install with: pip install open3d"
    ) from e


PNG_MAX_VALUE = 65535.0  # matches writer in isaac_data_recorder_husky.py


def read_traj_line_as_matrix(traj_path: str, frame_index: int) -> np.ndarray:
    """Read the (frame_index)-th 4x4 Twc matrix from traj.txt (row-major).

    Twc maps camera-frame coordinates to world-frame coordinates.
    """
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    with open(traj_path, "r") as f:
        for i, line in enumerate(f):
            if i == frame_index:
                vals = [float(x) for x in line.strip().split()]
                if len(vals) != 16:
                    raise ValueError(
                        f"Expected 16 floats for a 4x4 matrix, got {len(vals)} on line {i}"
                    )
                Twc = np.array(vals, dtype=np.float64).reshape(4, 4)
                return Twc

    raise IndexError(
        f"Frame index {frame_index} not found in {traj_path} (file has fewer lines)."
    )


def decode_depth_to_meters(depth_u16: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Invert the linear scaling used during recording to get metric depth in meters.

    z_uint16 = round((z - zmin) / (zmax - zmin) * 65535)
    => z = zmin + z_uint16/65535 * (zmax - zmin)
    """
    depth_u16 = depth_u16.astype(np.float32)
    z = zmin + depth_u16 / PNG_MAX_VALUE * (zmax - zmin)
    return z


def backproject_to_camera_points(
    z: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 1,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project depth to camera-frame 3D points.

    Returns (Nx3 points_cam, mask_idx) where mask_idx are indices into the flattened image.
    """
    H, W = z.shape[:2]

    # Downsample grid if stride > 1 for speed
    v = np.arange(0, H, stride, dtype=np.float32)
    u = np.arange(0, W, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    z_s = z[::stride, ::stride]

    if valid_min is None:
        valid_min = z.min()
    if valid_max is None:
        valid_max = z.max()

    valid = np.isfinite(z_s) & (z_s > valid_min) & (z_s < valid_max)
    if not np.any(valid):
        raise ValueError("No valid depth pixels after masking")

    # Back-project
    X = (uu - cx) / fx * z_s
    Y = (vv - cy) / fy * z_s
    Z = z_s

    Xv = X[valid]
    Yv = Y[valid]
    Zv = Z[valid]
    pts_cam = np.stack([Xv, Yv, Zv], axis=1)

    # mask indices relative to the strided grid; for coloring we can compute flat indices
    mask_idx = np.flatnonzero(valid.reshape(-1))
    return pts_cam.astype(np.float32), mask_idx


def transform_cam_to_world(Twc: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Apply Twc to camera points -> world points."""
    N = pts_cam.shape[0]
    ones = np.ones((N, 1), dtype=pts_cam.dtype)
    Pc_h = np.hstack([pts_cam, ones])  # Nx4
    Pw_h = (Twc @ Pc_h.T).T            # Nx4
    Pw = Pw_h[:, :3] / Pw_h[:, 3:4]
    return Pw


def maybe_load_intrinsics_json(path: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Intrinsics json not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    try:
        return float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])
    except Exception as e:
        raise ValueError(
            "Intrinsics json must contain keys: fx, fy, cx, cy"
        ) from e


def main():
    ap = argparse.ArgumentParser(description="Reconstruct world-frame point cloud and visualize with Open3D")
    ap.add_argument("-s", "--scene", required=True, help="Scene name (e.g., UR5-Peg-In-Hole_02)")
    ap.add_argument("-f", "--frame", type=int, required=True, help="Frame index (0-based)")
    ap.add_argument("-r", "--root", default="/workspace/isaaclab/IsaacSimData", help="Dataset root directory")
    ap.add_argument("--results-dirname", default="results", help="Subdir containing images")
    ap.add_argument("--image-prefix", default="frame", help="RGB image file prefix")
    ap.add_argument("--depth-prefix", default="depth", help="Depth image file prefix")
    ap.add_argument("--intrinsics", default=None, help="Path to intrinsics json with fx,fy,cx,cy")
    ap.add_argument("--fx", type=float, default=None, help="Focal length in pixels (x)")
    ap.add_argument("--fy", type=float, default=None, help="Focal length in pixels (y)")
    ap.add_argument("--cx", type=float, default=None, help="Principal point x (pixels)")
    ap.add_argument("--cy", type=float, default=None, help="Principal point y (pixels)")
    ap.add_argument("--zmin", type=float, default=0.01, help="Minimum depth used during encoding")
    ap.add_argument("--zmax", type=float, default=10.0, help="Maximum depth used during encoding")
    ap.add_argument("--stride", type=int, default=1, help="Pixel stride for downsampling (>=1)")
    ap.add_argument("--no-color", action="store_true", help="Do not colorize points from RGB image")
    ap.add_argument("--save-ply", default=None, help="Optional path to save PLY of world point cloud")
    args = ap.parse_args()

    scene_dir = os.path.join(args.root, args.scene)
    results_dir = os.path.join(scene_dir, args.results_dirname)
    traj_path = os.path.join(scene_dir, "traj.txt")

    depth_path = os.path.join(results_dir, f"{args.depth_prefix}{args.frame:06d}.png")
    rgb_path = os.path.join(results_dir, f"{args.image_prefix}{args.frame:06d}.jpg")

    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    # Load depth and decode
    depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_u16 is None or depth_u16.dtype != np.uint16:
        raise ValueError(
            f"Failed to read uint16 depth from {depth_path}. Ensure it was saved as 16-bit PNG."
        )

    z = decode_depth_to_meters(depth_u16, args.zmin, args.zmax)

    # Intrinsics
    intr = maybe_load_intrinsics_json(args.intrinsics)
    if intr is None:
        if None in (args.fx, args.fy, args.cx, args.cy):
            H, W = z.shape[:2]
            raise SystemExit(
                "Missing intrinsics. Provide --fx --fy --cx --cy or --intrinsics JSON. "
                f"Depth size is {W}x{H}; common cx,cy would be {W/2:.1f},{H/2:.1f}."
            )
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    else:
        fx, fy, cx, cy = intr

    # Read Twc for this frame
    Twc = read_traj_line_as_matrix(traj_path, args.frame)

    # Back-project to camera frame (with mask and stride)
    pts_cam, mask_idx = backproject_to_camera_points(
        z, fx=fx, fy=fy, cx=cx, cy=cy, stride=max(1, args.stride), valid_min=args.zmin, valid_max=args.zmax
    )

    # Colorize (optional)
    colors = None
    if not args.no_color and os.path.isfile(rgb_path):
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is not None:
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            # Downsample and mask the RGB image exactly like depth
            rgb_s = rgb[::args.stride, ::args.stride, :]
            rgb_flat = rgb_s.reshape(-1, 3)
            colors = rgb_flat[mask_idx].astype(np.float32) / 255.0
        else:
            print(f"Warning: Failed to read RGB image {rgb_path}. Proceeding without color.")
    else:
        if args.no_color:
            print("Colorization disabled by --no-color")
        else:
            print(f"RGB image not found at {rgb_path}; proceeding without color.")

    # Transform to world frame
    pts_world = transform_cam_to_world(Twc, pts_cam)

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world.astype(np.float64))
    if colors is not None and colors.shape[0] == pts_world.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if args.save_ply:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_ply)), exist_ok=True)
        o3d.io.write_point_cloud(args.save_ply, pcd)
        print(f"Saved point cloud to {args.save_ply}")

    # Visualize
    print("Opening Open3D visualizer. Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
