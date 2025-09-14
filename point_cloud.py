import os
import carb
import pickle
import numpy as np

from isaacsim import SimulationApp

# ------------------- Config -------------------
BACKGROUND_USD_PATH = "/workspace/isaaclab/SG/HuskyLab_assets/Franka-Peg-In-Hole.usd"
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": True}
simulation_app = SimulationApp(CONFIG)

import omni.usd
from pxr import Usd, UsdGeom, Gf

# ------------------- Helpers -------------------
def sample_points_from_mesh(mesh_prim, num_samples=2048):
    """Sample points from mesh using vertices (simple version)."""
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr().Get()
    if not points_attr:
        return None
    points = np.array(points_attr, dtype=np.float32)

    # Up-sample if not enough
    if len(points) >= num_samples:
        idx = np.random.choice(len(points), num_samples, replace=False)
    else:
        idx = np.random.choice(len(points), num_samples, replace=True)
    sampled = points[idx]

    # Transform to world coordinates
    xform = UsdGeom.Xformable(mesh_prim)
    transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # Gf.Matrix4d
    mat = np.array(transform)  # Directly convert to numpy

    sampled_h = np.concatenate([sampled, np.ones((len(sampled), 1))], axis=1)
    sampled_world = (mat @ sampled_h.T).T[:, :3]
    return sampled_world

def get_bbox_world(prim):
    """Return bbox center and extents in world coordinates."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    range_ = bbox.ComputeAlignedRange()
    center = range_.GetMidpoint()
    extent = range_.GetSize()
    return np.array(center), np.array(extent)

# ------------------- Main -------------------
stage = omni.usd.get_context().open_stage(BACKGROUND_USD_PATH)
stage = omni.usd.get_context().get_stage()

results = {}

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        name = prim.GetPath().pathString
        print(f"Processing {name}")

        pc = sample_points_from_mesh(prim)
        if pc is None:
            continue

        center, extent = get_bbox_world(prim)
        results[name] = {
            "pointcloud": pc,
            "bbox_center": center,
            "bbox_extent": extent,
        }

# ------------------- Save -------------------
out_path = "/workspace/isaaclab/IsaacSimData/Franka-Peg-In-Hole/scene_pointclouds.pkl"
with open(out_path, "wb") as f:
    pickle.dump(results, f)

print(f"Saved point clouds to {out_path}")
simulation_app.close()
