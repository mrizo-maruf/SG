# from isaacsim import SimulationApp
import carb
from isaacsim import SimulationApp
import sys

# ---------------------- Configuration ----------------------
BACKGROUND_STAGE_PATH = "/background"
# BACKGROUND_USD_PATH = "/workspace/isaaclab/SG/scene_optimized.usd"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)

# Image / depth settings
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0  # meters
PNG_MAX_VALUE = 65535  # 16-bit depth image

# Warm-up steps before recording (improves stability)
WARMUP_STEPS = 50
RENDER_SUBSTEPS = 100  # inner steps per saved frame for better visuals

# ---------------------- Imports ----------------------
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from isaacsim.core.utils import stage
from isaacsim.storage.native import get_assets_root_path
import os
import omni.replicator.core as rep
import omni
from isaacsim.core.utils.rotations import euler_angles_to_quat
import cv2
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
import json
from pxr import Sdf

# ---------------------- Extensions ----------------------
res = enable_extension("isaacsim.ros2.bridge")

# ---------------------- World ----------------------
physics_dt = 1.0 / 20.0
rendering_dt = 1.0 / 20.0
my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# ---------------------- Environment ----------------------
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
stage.add_reference_to_stage(assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH)

# ---------------------- Utility Functions ----------------------

def hide_prim(prim_path: str):
    set_prim_visibility_attribute(prim_path, "invisible")

def show_prim(prim_path: str):
    set_prim_visibility_attribute(prim_path, "inherited")

def set_prim_visibility_attribute(prim_path: str, value: str):
    prop_path = f"{prim_path}.visibility"
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(prop_path), value=value, prev=None
    )

def transformation_matrix(position, orientation):
    # orientation expected as (w, x, y, z)
    w, x, y, z = orientation
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T

def interpolate_keyframes_with_euler(keyframes, i):
    for j in range(len(keyframes) - 1):
        t0, t1 = keyframes[j]['time'], keyframes[j + 1]['time']
        if t0 <= i <= t1:
            kf0, kf1 = keyframes[j], keyframes[j + 1]
            break
    else:
        return None, None
    alpha = (i - t0) / (t1 - t0)
    next_translation = (1 - alpha) * np.array(kf0['translation']) + alpha * np.array(kf1['translation'])
    euler0 = kf0['euler_angles']
    euler1 = kf1['euler_angles']
    interpolated_euler = (1 - alpha) * np.array(euler0) + alpha * np.array(euler1)
    next_orientation = euler_angles_to_quat(interpolated_euler, degrees=True)
    return next_translation, next_orientation

def create_color_map(num_classes):
    colors = [[0, 0, 0]]  # background
    import colorsys
    golden_ratio = 0.618033988749895
    for i in range(1, num_classes):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.6 + (i % 4) * 0.1
        value = 0.7 + (i % 3) * 0.15
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(b * 255), int(g * 255), int(r * 255)])
    return np.array(colors, dtype=np.uint8)

def apply_color_map(seg_image, color_map):
    h, w = seg_image.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for seg_id in np.unique(seg_image):
        if seg_id < len(color_map):
            colored_seg[seg_image == seg_id] = color_map[seg_id]
    return colored_seg

def compute_intrinsics(camera_prim, width, height):
    focal_length_attr = camera_prim.GetAttribute("focalLength")
    h_aperture_attr = camera_prim.GetAttribute("horizontalAperture")
    v_aperture_attr = camera_prim.GetAttribute("verticalAperture")
    focal_length = float(focal_length_attr.Get())
    h_aperture = float(h_aperture_attr.Get())
    v_aperture = float(v_aperture_attr.Get()) if v_aperture_attr and v_aperture_attr.Get() else h_aperture * height / width
    if v_aperture_attr and (not v_aperture_attr.Get()):
        v_aperture_attr.Set(v_aperture)
    fx = focal_length / h_aperture * width
    fy = focal_length / v_aperture * height
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

# ---------------------- Camera ----------------------
camera = Camera(
    prim_path="/World/Camera",
    position = np.array([3, 0, 1.5]),
    resolution=(IMAGE_HEIGHT, IMAGE_WIDTH),
    orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True, extrinsic=True)
)
my_world.reset()
camera.initialize()
camera.add_distance_to_camera_to_frame()

stage_ref = get_current_stage()
camera_prim = stage_ref.GetPrimAtPath("/World/Camera")

# Optional focal settings
focal_length_attr = camera_prim.GetAttribute("focalLength")
horizontal_aperture_attr = camera_prim.GetAttribute("horizontalAperture")
print("focal length:", focal_length_attr.Get())
print("horizontal aperture:", horizontal_aperture_attr.Get())

# Adjust horizontal aperture (affects intrinsics); set vertical aperture to maintain aspect
horizontal_aperture_attr.Set(80)
vertical_aperture_attr = camera_prim.GetAttribute("verticalAperture")
vertical_aperture_attr.Set(80 * IMAGE_HEIGHT / IMAGE_WIDTH)

# Set camera pose
# camera.set_local_pose(
#     np.array([3, 0, 1.5]),
#     rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True, extrinsic=True),
#     camera_axes="world"
# )

# Set resolution if API exposed
if hasattr(camera, "set_resolution"):
    camera.set_resolution([IMAGE_WIDTH, IMAGE_HEIGHT])

# Compute and print intrinsics & depth scale BEFORE capturing images
fx, fy, cx, cy = compute_intrinsics(camera_prim, IMAGE_WIDTH, IMAGE_HEIGHT)
png_depth_scale = (MAX_DEPTH - MIN_DEPTH) / PNG_MAX_VALUE
print(f"Camera intrinsics -> fx: {fx:.3f}, fy: {fy:.3f}, cx: {cx:.3f}, cy: {cy:.3f}, png_depth_scale: {png_depth_scale:.8f}")

# ---------------------- Keyframes ----------------------
keyframes_move = [
    {'time': 0, 'translation': [5, 4, 2.2], 'euler_angles': [0, 15, -90]},
    {'time': 6, 'translation': [8.2, -3, 2.2], 'euler_angles': [0, 15, -180]},
    {'time': 12, 'translation': [3.5, -6.3, 2.2], 'euler_angles': [0, 15, -280]},
    {'time': 18, 'translation': [-5.4, -7.2, 2.2], 'euler_angles': [0, 15, -240]},
    {'time': 21, 'translation': [-2, -3, 2.2], 'euler_angles': [0, 15, -180]},
    {'time': 24, 'translation': [-5, 1, 2.2], 'euler_angles': [0, 15, -120]},
    {'time': 27, 'translation': [-3, 5, 2.2], 'euler_angles': [0, 15, -180]},
    {'time': 30, 'translation': [-7.5, 10, 2.2], 'euler_angles': [0, 15, -270]},
    {'time': 33, 'translation': [-1.5, 12, 2.2], 'euler_angles': [0, 15, -200]},
]
record_keyframe = keyframes_move

# ---------------------- ROS2 Camera Graph ----------------------
import usdrt.Sdf
CAMERA_STAGE_PATH = "/World/Camera"
ROS_CAMERA_GRAPH_PATH = "/ROS2_Camera"
keys = og.Controller.Keys
(ros_camera_graph, _, _, _) = og.Controller.edit(
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
            ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_STAGE_PATH)]),
        ],
    },
)

from omni.kit.viewport.utility import get_active_viewport
viewport_api = get_active_viewport()
render_product_path = viewport_api.get_render_product_path()

# ---------------------- Output Paths ----------------------
scene_name = "warehouse"
base_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}/results"
traj_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}"
os.makedirs(base_dir, exist_ok=True)
image_prefix = "frame"
depth_prefix = "depth"
seg_prefix = "semantic"
traj_file_path = os.path.join(traj_dir, "traj.txt")
if not os.path.exists(traj_file_path):
    open(traj_file_path, 'w').close()
    print(f"File '{traj_file_path}' has been created.")
else:
    print(f"File '{traj_file_path}' already exists.")

# ---------------------- Warm-up ----------------------
for _ in range(WARMUP_STEPS):
    next_translation, next_orientation = interpolate_keyframes_with_euler(record_keyframe, 0)
    if _ == 0:
        camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    my_world.step(render=True)
    simulation_app.update()

# ---------------------- Main Recording Loop ----------------------
i = 0
frame_index = 0
while simulation_app.is_running():
    next_translation, next_orientation = interpolate_keyframes_with_euler(record_keyframe, i)
    if next_translation is None:
        break
    camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    position_, orientation_ = camera.get_local_pose(camera_axes="world")
    _ = transformation_matrix(position_, orientation_)
    print(f"Iteration: {i}")
    i += 1

    # Extra internal render steps (smoother outputs)
    for _ in range(RENDER_SUBSTEPS):
        my_world.step(render=True)
    simulation_app.update()

    viewport_api = get_active_viewport()
    render_product_path = viewport_api.get_render_product_path()

    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    depth_ann.attach([render_product_path])
    depth_image = depth_ann.get_data()

    rgb_ann = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_ann.attach([render_product_path])
    rgba_image = rgb_ann.get_data()

    seg_ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    seg_ann.attach([render_product_path])
    seg_data = seg_ann.get_data()
    seg_info = seg_data['info']['idToLabels']
    seg_image = seg_data['data'].astype(np.uint8)

    img_path = os.path.join(base_dir, f"{image_prefix}{frame_index:06d}.jpg")
    depth_path = os.path.join(base_dir, f"{depth_prefix}{frame_index:06d}.png")
    seg_colored_path = os.path.join(base_dir, f"{seg_prefix}{frame_index:06d}.png")
    seg_info_path = os.path.join(base_dir, f"{seg_prefix}{frame_index:06d}_info.json")

    if depth_image.size != 0 and rgba_image.size != 0:
        clipped_depth = np.clip(depth_image, MIN_DEPTH, MAX_DEPTH)
        normalized_depth = ((clipped_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)) * PNG_MAX_VALUE
        depth_image_uint16 = normalized_depth.astype("uint16")
        cv2.imwrite(depth_path, depth_image_uint16)

        max_seg_id = np.max(seg_image) if seg_image.size > 0 else 0
        num_classes = max(max_seg_id + 1, len(seg_info) if seg_info else 0, 1)
        color_map = create_color_map(num_classes)
        colored_seg_image = apply_color_map(seg_image, color_map)
        cv2.imwrite(seg_colored_path, colored_seg_image)

        rgb = rgba_image[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, bgr)

        position_ros, orientation_ros = camera.get_world_pose(camera_axes="ros")
        T_ros = transformation_matrix(position_ros, orientation_ros)
        with open(traj_file_path, "a") as traj_file:
            traj_file.write(' '.join(map(str, T_ros.flatten())) + "\n")

        enhanced_seg_info = {}
        for seg_id, label in seg_info.items():
            seg_id_int = int(seg_id)
            enhanced_seg_info[seg_id] = {
                "label": label,
                "color_bgr": color_map[seg_id_int].tolist() if seg_id_int < len(color_map) else [0, 0, 0]
            }
        with open(seg_info_path, "w") as json_file:
            json.dump(enhanced_seg_info, json_file, indent=4)

    frame_index += 1

# ---------------------- Shutdown ----------------------
simulation_app.close()
