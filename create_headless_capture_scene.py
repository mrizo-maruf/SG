# create_table_ur5_capture.py
# Run inside Isaac Lab / Isaac Lab container (headless):
#   export HEADLESS=1
#   python3 scripts/my_scene/create_table_ur5_capture.py
#
# Outputs:
#   /workspace/isaaclab/IsaacSimData/table_scene/results/   -> rgb/depth/seg images
#   /workspace/isaaclab/IsaacSimData/table_scene/traj.txt   -> flattened 4x4 camera->world per line

import os
import sys
import time
import json
import numpy as np
import cv2
import math

import carb
from isaacsim import SimulationApp

# SimulationApp configuration: headless True
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": True}
simulation_app = SimulationApp(CONFIG)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

# world, camera
from omni.isaac.core import World
from omni.isaac.sensor import Camera
# stage utils
from isaacsim.core.utils import stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.rotations import euler_angles_to_quat
import omni.replicator.core as rep
import omni.graph.core as og
from omni.kit.viewport.utility import get_active_viewport
from pxr import Sdf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils

# optional: if you have HuskyLab.package_root like in your repo
try:
    from HuskyLab import package_root
except Exception:
    package_root = "/workspace/isaaclab/SG"



# -------- USER: set these to your actual files if different --------
UR5_USD = os.path.join(package_root, "HuskyLab_assets", "ur5.usdz")               # change if needed
TABLE_USD = os.path.join(package_root, "HuskyLab_assets", "table.usd")           # change if needed
DEX_CUBE_USD = os.path.join(package_root, "HuskyLab_assets", "mug.usd")  # change if needed
# -------------------------------------------------------------------

print('*'*50, ISAAC_NUCLEUS_DIR)

# create world (physics and rendering dt same as your code)
physics_dt = 1.0 / 20.0
rendering_dt = 1.0 / 20.0
world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Reference the three assets into the stage under /World/<Name>
stage = stage_utils.get_current_stage()

def reference_usd(usd_path: str, target_path: str):
    try:
        stage_utils.add_reference_to_stage(usd_path, target_path)
        print(f"[INFO] Referenced {usd_path} -> {target_path}")
    except Exception as e:
        carb.log_warn(f"Failed to reference {usd_path} -> {target_path}: {e}")

# Clear any previous /World children if you want - not done here to be safe

# Reference assets to /World/UR5, /World/Table, /World/Object
reference_usd(TABLE_USD, "/World/Table")
reference_usd(UR5_USD, "/World/UR5")
reference_usd(DEX_CUBE_USD, "/World/Object")

# Small helpers to set transforms (translate/scale). These use ChangeProperty on xformOp attrs.
import omni.kit.commands

# add at top of file:
from pxr import Sdf, Gf

def set_translate(prim_path: str, translate):
    """Set translate using a Gf.Vec3d to avoid USD type-mismatch errors."""
    try:
        vec = Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2]))
        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=Sdf.Path(f"{prim_path}.xformOp:translate"),
            value=vec,
            prev=None,
        )
    except Exception as e:
        carb.log_warn(f"set_translate failed for {prim_path} -> {translate}: {e}")

def set_scale(prim_path: str, scale):
    """Set non-uniform scale using Gf.Vec3d. If scale is a single float, pass it as Gf.Vec3d(s,s,s)."""
    try:
        if isinstance(scale, (int, float)):
            vec = Gf.Vec3d(float(scale), float(scale), float(scale))
        else:
            vec = Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2]))
        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=Sdf.Path(f"{prim_path}.xformOp:scale"),
            value=vec,
            prev=None,
        )
    except Exception as e:
        carb.log_warn(f"set_scale failed for {prim_path} -> {scale}: {e}")


# Apply the same transforms you supplied earlier
# table pos [0.5, 0, -0.63]
set_translate("/World/Table", [0.0, 0.0, 0.0])

# object pos [0.5, 0, 0.0] and scale (0.6,0.6,0.6)
set_translate("/World/Object", [0.5, 0.0, 0.68])
# set_scale("/World/Object", [0.1, 0.1, 0.1])

# UR5 at origin (ensure it's placed)
set_translate("/World/UR5", [-0.5, 0.0, 0.63])

# cfg_light_distant = sim_utils.DistantLightCfg(
#     intensity=200.0,
#     color=(0.75, 0.75, 0.75),
# )

# Reset world so everything is registered in simulator
world.reset()

# Create camera and configure
CAMERA_STAGE_PATH = "/World/Camera"
camera = Camera(prim_path=CAMERA_STAGE_PATH)
camera.initialize()
# initial pose; camera will be moved by keyframes
camera.set_local_pose(np.array([3.0, 0.0, 1.5]), euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), degrees=True), camera_axes="world")

# If the camera prim has focalLength/horizontalAperture attributes, adjust horizontal aperture as you had
cam_prim = stage.GetPrimAtPath(CAMERA_STAGE_PATH)
if cam_prim.IsValid():
    try:
        horiz = cam_prim.GetAttribute("horizontalAperture")
        if horiz:
            horiz.Set(80.0)
    except Exception:
        pass

# Create a small on-demand graph to ensure render product exists (same as your code)
ROS_CAMERA_GRAPH_PATH = "/ROS2_Camera"
keys = og.Controller.Keys
og.Controller.edit(
    {
        "graph_path": ROS_CAMERA_GRAPH_PATH,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
            ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
            ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 0),
            ("setCamera.inputs:cameraPrim", [Sdf.Path(CAMERA_STAGE_PATH)]),
        ],
    },
)

# Replicator annotators (LdrColor, distance_to_image_plane, semantic_segmentation)
rep_depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
rep_rgb_annotator   = rep.AnnotatorRegistry.get_annotator("LdrColor")
rep_seg_annotator   = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

# Keyframes (kept from your original file)
keyframes_move = [
    {'time': 0, 'translation': [0, 1.7, 2], 'euler_angles': [0, 30, -90]},
    {'time': 3, 'translation': [0, 3, 2], 'euler_angles': [0, 30, -90]},
    {'time': 6, 'translation': [2, 0, 2.2], 'euler_angles': [0, 30, -180]},
    {'time': 9, 'translation': [4, 0, 2.2], 'euler_angles': [0, 30, -180]},
    {'time': 12, 'translation': [0, -2, 2.2], 'euler_angles': [0, 30, -240]},
]

def interpolate_keyframes_with_euler(keyframes, t):
    for j in range(len(keyframes) - 1):
        t0, t1 = keyframes[j]['time'], keyframes[j + 1]['time']
        if t0 <= t <= t1:
            kf0, kf1 = keyframes[j], keyframes[j + 1]
            break
    else:
        return None, None
    alpha = (t - t0) / float((t1 - t0))
    next_translation = (1 - alpha) * np.array(kf0['translation']) + alpha * np.array(kf1['translation'])
    euler0 = np.array(kf0['euler_angles'])
    euler1 = np.array(kf1['euler_angles'])
    interpolated_euler = (1 - alpha) * euler0 + alpha * euler1
    next_orientation = euler_angles_to_quat(interpolated_euler, degrees=True)
    return next_translation, next_orientation

def transformation_matrix(position, orientation):
    """Create 4x4 transform from position and quaternion.
       Accepts quaternion as (x,y,z,w) (common in Isaac APIs)."""
    q = np.array(orientation, dtype=float)
    if q.size != 4:
        raise ValueError("orientation must have 4 elements")
    # convert to (w,x,y,z)
    w, x, y, z = q[3], q[0], q[1], q[2]
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=float)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.array(position, dtype=float)
    return T

def create_color_map(num_classes):
    colors = [[0,0,0]]
    import colorsys
    for i in range(1, num_classes):
        golden_ratio = 0.618033988749895
        hue = (i * golden_ratio) % 1.0
        saturation = 0.6 + (i % 4) * 0.1
        value = 0.7 + (i % 3) * 0.15
        r,g,b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(b*255), int(g*255), int(r*255)])  # BGR for OpenCV
    return np.array(colors, dtype=np.uint8)

def apply_color_map(seg_image, color_map):
    if seg_image.size == 0:
        return np.zeros((1,1,3), dtype=np.uint8)
    h,w = seg_image.shape
    colored = np.zeros((h,w,3), dtype=np.uint8)
    uids = np.unique(seg_image)
    for uid in uids:
        idx = int(uid)
        if idx < len(color_map):
            colored[seg_image == uid] = color_map[idx]
    return colored

# Output directories and traj file
scene_name = "table_scene"
base_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}/results"
traj_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(traj_dir, exist_ok=True)
traj_file_path = os.path.join(traj_dir, "traj.txt")
# start fresh
open(traj_file_path, "w").close()

image_prefix = "frame"
depth_prefix = "depth"
seg_prefix = "semantic"

# Warm-up: step world a few times before capture (ensures render product exists)
viewport_api = get_active_viewport()
render_product_path = viewport_api.get_render_product_path()

# attach annotators to the render product
rep_depth_annotator.attach([render_product_path])
rep_rgb_annotator.attach([render_product_path])
rep_seg_annotator.attach([render_product_path])

# tiny warm-up steps
for _ in range(100):
    world.step(render=True)
    simulation_app.update()

# Main capture loop
t = 0
frame = 0
max_frames = 200

print("[INFO] Starting camera motion + capture loop (table, object, ur5)...")
while simulation_app.is_running() and frame < max_frames:
    
    
    trans, quat = interpolate_keyframes_with_euler(keyframes_move, t)
    if trans is None:
        break

    # set camera pose (world axes)
    camera.set_local_pose(trans, quat, camera_axes="world")

    # step a few times so annotators have fresh renders
    for _ in range(50):
        world.step(render=True)
    simulation_app.update()

    # Grab annotator data
    depth = rep_depth_annotator.get_data()   # float32 HxW (meters)
    rgba  = rep_rgb_annotator.get_data()     # uint8 HxWx4 (RGBA)
    seg   = rep_seg_annotator.get_data()     # dict {'data', 'info'}
    seg_img = seg['data'] if seg and 'data' in seg else np.array([])
    seg_info = seg['info']['idToLabels'] if seg and 'info' in seg else {}

    # Compose filenames
    img_path = os.path.join(base_dir, f"{image_prefix}{frame:06d}.jpg")
    depth_path = os.path.join(base_dir, f"{depth_prefix}{frame:06d}.png")
    seg_colored_path = os.path.join(base_dir, f"{seg_prefix}{frame:06d}.png")
    seg_info_path = os.path.join(base_dir, f"{seg_prefix}{frame:06d}_info.json")

    if depth is not None and rgba is not None and depth.size != 0 and rgba.size != 0:
        # depth -> uint16 (clip & norm)
        min_val, max_val = 0.01, 10.0
        clipped = np.clip(depth, min_val, max_val)
        normalized = ((clipped - min_val) / (max_val - min_val)) * 65535.0
        depth_u16 = normalized.astype(np.uint16)
        cv2.imwrite(depth_path, depth_u16)

        # RGB: take first 3 channels, convert to BGR for saving
        rgb = rgba[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, bgr)

        # Segmentation: colorize and save
        seg_np = seg_img.astype(np.uint8) if seg_img.size != 0 else np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        max_seg_id = int(np.max(seg_np)) if seg_np.size != 0 else 0
        num_classes_seg = max_seg_id + 1
        num_classes_info = len(seg_info) if seg_info else 0
        num_classes = max(num_classes_seg, num_classes_info, 1)
        cmap = create_color_map(num_classes)
        colored_seg = apply_color_map(seg_np, cmap)
        cv2.imwrite(seg_colored_path, colored_seg)

        # Save seg_info with assigned colors
        enhanced = {}
        for sid, lbl in seg_info.items():
            sid_int = int(sid)
            enhanced[sid] = {"label": lbl, "color_bgr": cmap[sid_int].tolist() if sid_int < len(cmap) else [0,0,0]}
        with open(seg_info_path, "w") as jf:
            json.dump(enhanced, jf, indent=2)

        # Trajectory: camera world pose in ROS axes
        pos_w, ori_w = camera.get_world_pose(camera_axes="ros")
        tf = transformation_matrix(pos_w, ori_w)
        with open(traj_file_path, "a") as tfh:
            tfh.write(" ".join(map(str, tf.flatten())) + "\n")

        print(f"[frame {frame}] saved rgb:{img_path} depth:{depth_path} seg:{seg_colored_path}")
    else:
        carb.log_warn("[WARN] Annotators returned no data for this frame; consider increasing warmup.")

    frame += 1
    t += 1

# Cleanup
simulation_app.close()
print("[DONE] Saved outputs ->", base_dir, "traj ->", traj_file_path)
