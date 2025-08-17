# from isaacsim import SimulationApp
import carb
from isaacsim import SimulationApp
import sys

BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)


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
import numpy as np
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
import json

res = enable_extension("isaacsim.ros2.bridge")
# World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
physics_dt=1.0 / 20.0
rendering_dt=1.0 / 20.0
my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
stage.add_reference_to_stage(assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH)


from pxr import Sdf


def hide_prim(prim_path: str):
    """Hide a prim

    Args:
        prim_path (str, required): The prim path of the prim to hide
    """
    set_prim_visibility_attribute(prim_path, "invisible")


def show_prim(prim_path: str):
    """Show a prim

    Args:
        prim_path (str, required): The prim path of the prim to show
    """
    set_prim_visibility_attribute(prim_path, "inherited")


def set_prim_visibility_attribute(prim_path: str, value: str):
    """Set the prim visibility attribute at prim_path to value

    Args:
        prim_path (str, required): The path of the prim to modify
        value (str, required): The value of the visibility attribute
    """
    # You can reference attributes using the path syntax by appending the
    # attribute name with a leading `.`
    prop_path = f"{prim_path}.visibility"
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(prop_path), value=value, prev=None
    )


def transformation_matrix(position, orientation):
    """
    Convert position and quaternion orientation to a 4x4 transformation matrix.

    Parameters:
        position: 3D coordinates in the form (x, y, z)
        orientation: Quaternion in the form (qx, qy, qz, qw)

    Returns:
        4x4 transformation matrix
    """
    # Create rotation matrix
    # r = Rotation.from_quat(orientation)
    # rotation_matrix = r.as_matrix()  # 3x3 rotation matrix
    w, x, y, z = orientation

    # Calculate 3x3 rotation matrix based on quaternion
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])

    # Combine rotation and translation into 4x4 matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = position

    return transformation

def interpolate_keyframes_with_euler(keyframes, i):
    # Find the keyframe interval that contains the current time step
    for j in range(len(keyframes) - 1):
        t0, t1 = keyframes[j]['time'], keyframes[j + 1]['time']
        if t0 <= i <= t1:
            kf0, kf1 = keyframes[j], keyframes[j + 1]
            break
    else:
        return None, None  # Out of range

    # Linear interpolation for position calculation
    alpha = (i - t0) / (t1 - t0)
    next_translation = (1 - alpha) * np.array(kf0['translation']) + alpha * np.array(kf1['translation'])

    # Use Euler angle linear interpolation and convert to quaternion
    euler0 = kf0['euler_angles']
    euler1 = kf1['euler_angles']
    interpolated_euler = (1 - alpha) * np.array(euler0) + alpha * np.array(euler1)
    next_orientation = euler_angles_to_quat(interpolated_euler, degrees=True)

    return next_translation, next_orientation

def create_color_map(num_classes):
    """Create a color map for any number of segmentation classes"""
    colors = []
    
    # First color is black for background/unlabeled
    colors.append([0, 0, 0])
    
    for i in range(1, num_classes):
        # Generate distinct colors using golden ratio for optimal distribution
        import colorsys
        
        # Use golden ratio (0.618...) for hue spacing to maximize color distinction
        golden_ratio = 0.618033988749895
        hue = (i * golden_ratio) % 1.0
        
        # Vary saturation and value to create more distinct colors
        saturation = 0.6 + (i % 4) * 0.1  # 0.6, 0.7, 0.8, 0.9
        value = 0.7 + (i % 3) * 0.15       # 0.7, 0.85, 1.0
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(b*255), int(g*255), int(r*255)])  # BGR format for OpenCV
    
    return np.array(colors, dtype=np.uint8)

def apply_color_map(seg_image, color_map):
    """Apply color map to segmentation image"""
    h, w = seg_image.shape
    colored_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_ids = np.unique(seg_image)
    for seg_id in unique_ids:
        if seg_id < len(color_map):
            colored_seg[seg_image == seg_id] = color_map[seg_id]
    
    return colored_seg


camera = Camera(
    prim_path="/World/Camera",
)

my_world.reset()

camera.initialize()
camera.add_distance_to_camera_to_frame()

stage = get_current_stage()

# camera_prim = stage.GetPrimAtPath("/World/Camera")

# camera.set_focal_length(20 / 10.0)  # 20/10
camera.set_local_pose(np.array([3, 0, 1.5]), rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True, extrinsic=True), camera_axes="world")

camera_prim= stage.GetPrimAtPath("/World/Camera")
focal_length = camera_prim.GetAttribute("focalLength")
horizontal_aperture = camera_prim.GetAttribute("horizontalAperture")
# get value
print("focal length:", focal_length.Get())
print("horizontal aperture:", horizontal_aperture.Get())
# # set value
# focal_length.Set(10.0)
horizontal_aperture.Set(80)  # 60 --> fx: 1066.667    fy: 1066.667     cx: 640.0    cy: 360.0

stage = get_current_stage()

# (3.5, 3) is center, the height of object read is 1 meter
keyframes_move = [
    {'time': 0, 'translation': [0, 3, 2.2], 'euler_angles': [0, 15, -45]},
    {'time': 20, 'translation': [5.5, 3, 2.2], 'euler_angles': [0, 15, -120]},
    {'time': 30, 'translation': [-5, 3, 2.2], 'euler_angles': [0, 15, -200]},
    {'time': 50, 'translation': [-5, 8, 2.2], 'euler_angles': [0, 15, -280]},
    # {'time': 500, 'translation': [-1.3, 0, 1.5], 'euler_angles': [0, 0, 290]},
    # {'time': 750, 'translation': [-1.3, 0, 1.5], 'euler_angles': [0, 0, 70]},
    # {'time': 850, 'translation': [-1.3, 0, 1.5], 'euler_angles': [0, 30, 70]},
    # {'time': 1100, 'translation': [-1.3, 0, 1.5], 'euler_angles': [0, 30, 290]},
    # {'time': 1300, 'translation': [-1.3, 0, 2.5], 'euler_angles': [0, 90, 180]},
]

# Select one floor or interior of elevator for recording
record_keyframe = keyframes_move 

# Initialize
translation, orientation = camera.get_world_pose(camera_axes="world")

import usdrt.Sdf
CAMERA_STAGE_PATH = "/World/Camera"
ROS_CAMERA_GRAPH_PATH = "/ROS2_Camera"
# Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers
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

scene_name = "warehouse"
base_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}/results"
traj_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}"
os.makedirs(base_dir, exist_ok=True)
image_prefix = "frame"
depth_prefix = "depth"
seg_prefix = "semantic"
traj_file_path = os.path.join(traj_dir, "traj.txt")
if not os.path.exists(traj_file_path):
    with open(traj_file_path, 'w') as file:
        pass  # Create an empty traj.txt file
    print(f"File '{traj_file_path}' has been created.")
else:
    print(f"File '{traj_file_path}' already exists.")

for j in range(50):
    next_translation, next_orientation = interpolate_keyframes_with_euler(record_keyframe, 0) 
    if j == 0:
        camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    print(j)
    my_world.step(render=True)
    simulation_app.update()

i = 0
frame = 0
j=0
# record = False
# while simulation_app.is_running():
#     my_world.step(render=True)
while simulation_app.is_running() :

    # translation, orientation = camera.get_local_pose(camera_axes="world")
    # print("translation", translation)
    # print("orientation", orientation)
    # translation, orientation = camera.get_world_pose(camera_axes="world")
    # print("translation", translation)
    # print("orientation", orientation)

    next_translation, next_orientation = interpolate_keyframes_with_euler(record_keyframe, i)
    if next_translation is None:
        break
    
    camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    
    position_, orientation_ = camera.get_local_pose(camera_axes="world")  # ros or world or usd
    transformation_matrix_result = transformation_matrix(position_, orientation_)
    # print(transformation_matrix_result)
    print(f"Iteration: {i}")

    i += 1
    
    for k in range(10):
        my_world.step(render=True)

    viewport_api = get_active_viewport()
    render_product_path = viewport_api.get_render_product_path()

    depth_annotators = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane") # distance_to_image_plane  distance_to_camera
    # depth_annotators.attach([camera._render_product_path])
    depth_annotators.attach([render_product_path])
    depth_image = depth_annotators.get_data()

    rgb_annotators = rep.AnnotatorRegistry.get_annotator("LdrColor")
    rgb_annotators.attach([render_product_path])
    rgba_image = rgb_annotators.get_data()

    seg_annotators = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    seg_annotators.attach([render_product_path])
    # seg_image = seg_annotators.get_data()['data']
    seg_data = seg_annotators.get_data()
    seg_info = seg_data['info']['idToLabels']
    seg_image = seg_data['data']
    seg_image = seg_image.astype(np.uint8)

    img_path = os.path.join(base_dir, f"{image_prefix}{j:06d}.jpg")
    depth_path = os.path.join(base_dir, f"{depth_prefix}{j:06d}.png")
    # seg_path = os.path.join(base_dir, f"{seg_prefix}{j:06d}.png")
    seg_colored_path = os.path.join(base_dir, f"{seg_prefix}{j:06d}.png")
    seg_info_path = os.path.join(base_dir, f"{seg_prefix}{j:06d}_info.json")

    min_val = 0.01
    max_val = 10.0
    if depth_image.size!=0 and rgba_image.size!=0:
        clipped_depth = np.clip(depth_image, min_val, max_val)
        normalized_depth = ((clipped_depth - min_val) / (max_val - min_val)) * 65535
        depth_image_uint16 = normalized_depth.astype("uint16")
        cv2.imwrite(depth_path, depth_image_uint16)
        
        # Create and save colored segmentation
        # Dynamically determine number of classes from the actual segmentation data
        max_seg_id = np.max(seg_image) if seg_image.size > 0 else 0
        num_classes_from_seg = max_seg_id + 1
        num_classes_from_info = len(seg_info) if seg_info else 0
        
        # Use the maximum to ensure we have enough colors
        num_classes = max(num_classes_from_seg, num_classes_from_info, 1)
        
        color_map = create_color_map(num_classes)
        colored_seg_image = apply_color_map(seg_image, color_map)
        cv2.imwrite(seg_colored_path, colored_seg_image)

        rgb = rgba_image[:, :, :3] # you can log rbga if you want
        # image = Image.fromarray(rgb)
        # image.save(img_path, format="JPEG")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, bgr)

        position_, orientation_ = camera.get_world_pose(camera_axes="ros")  # ros or world or usd
        transformation_matrix_result = transformation_matrix(position_, orientation_)
        # print(transformation_matrix_result)
    
        # Convert transformation matrix to string format and write to file
        with open(traj_file_path, "a") as traj_file:  # "a" mode means append mode
            transform_str = ' '.join(map(str, transformation_matrix_result.flatten()))
            traj_file.write(transform_str + "\n")
        # Save seg_info to JSON file
        
        # Enhanced seg_info with color mapping
        enhanced_seg_info = {}
        for seg_id, label in seg_info.items():
            seg_id_int = int(seg_id)
            enhanced_seg_info[seg_id] = {
                "label": label,
                "color_bgr": color_map[seg_id_int].tolist() if seg_id_int < len(color_map) else [0, 0, 0]
            }
        
        with open(seg_info_path, "w") as json_file:
            json.dump(enhanced_seg_info, json_file, indent=4)  # Save enhanced seg_info in JSON format
         
    j=j+1
simulation_app.close()
