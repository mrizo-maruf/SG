import carb
from isaacsim import SimulationApp
import sys
import numpy as np

# ---------------------- Configuration ----------------------
MAIN_SCENE_PATH = "/World/Kitchen"
MAIN_SCENE_USD = "/workspace/isaaclab/SG/aloha_assets_new/scenes/scenes_sber_kitchen_for_BBQ/kitchen_new_simple.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)

# Image / depth settings
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0
PNG_MAX_VALUE = 65535
FOCAL_LENGTH = 50
HORIZONTAL_APARTURE = 80
VERTICAL_APARTURE = 45
WARMUP_STEPS = 50
RENDER_SUBSTEPS = 100

# Recording area boundaries
SCENE_BOUNDS = {
    'x_min': 0, 'x_max': 10,
    'y_min': 0.5, 'y_max': 7.5
}

# ---------------------- Imports ----------------------
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils import stage
from isaacsim.storage.native import get_assets_root_path
import os
import omni.replicator.core as rep
import omni
from isaacsim.core.utils.rotations import euler_angles_to_quat
import cv2
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
import json
from pxr import Sdf, Gf, UsdGeom
import omni.kit.commands
from omni.isaac.core.prims import XFormPrim
# ---------------------- Extensions ----------------------
res = enable_extension("isaacsim.ros2.bridge")

# ---------------------- World ----------------------
physics_dt = 1.0 / 20.0
rendering_dt = 1.0 / 20.0
my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# ---------------------- Load Main Scene ----------------------
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

add_reference_to_stage(usd_path=MAIN_SCENE_USD, prim_path=MAIN_SCENE_PATH)

# ---------------------- Object Configuration ----------------------
OBJECT_CONFIG = {
    "objects": [
        {
            "name": "table",
            "type": ["static_obstacle", "surface_provider", "changeable_color"],
            "count": 3,
            "size": [0.5, 0.5, 0.75],
            "usd_paths": ["scenes/scenes_sber_kitchen_for_BBQ/table/table.usd"],
            "placement": {
                "strategy": "grid",
                "grid_coordinates": [
                    [-4.5, -1.1, 0.0],
                    [-4.5, 0.0, 0.0],
                    [-4.5, 1.1, 0.0]
                ]
            }
        },
        {
            "name": "chair",
            "type": ["movable_obstacle"],
            "count": 3,
            "size": [0.4, 0.4, 0.9],
            "usd_paths": ["scenes/obstacles/chair2.usd"],
            "placement": {
                "strategy": "grid",
                "grid_coordinates": [
                    [-2.2, -1.1, 0.0],
                    [-2.2, 0.0, 0.0],
                    [-2.2, 1.1, 0.0]
                ]
            }
        },
        {
            "name": "cabinet",
            "type": ["staff_obstacle"],
            "count": 6,
            "size": [0.4, 0.4, 0.9],
            "usd_paths": ["scenes/obstacles/cabinet.usd"],
            "placement": {
                "strategy": "grid",
                "grid_coordinates": [
                    [-0.5, -3.2, 0.0],
                    [-2, -3.2, 0.0],
                    [-3.5, -3.2, 0.0],
                    [-0.5, 3.2, 0.0],
                    [-2, 3.2, 0.0],
                    [-3.5, 3.2, 0.0]
                ]
            }
        },
        {
            "name": "bowl",
            "type": ["possible_goal", "surface_only"],
            "count": 1,
            "size": [0.15, 0.15, 0.05],
            "usd_paths": ["objects/bowl.usd"],
            "placement": {
                "strategy": "on_surface",
                "surface_types": ["surface_provider"],
                "margin": 0.1
            }
        }
    ]
}


def get_scene_obj_coords(json_path="/workspace/isaaclab/SG/eval_scenes_64.json"):
    # Step 1: read JSON
    with open(json_path, "r") as f:
        data = json.load(f)   # list of dicts

    # Step 2: build dict {row_string: raw_row}
    result_dict = {}

    for row in data:
        parts = []
        for obj, coords_list in row.items():
            for coords in coords_list:
                coords = coords[:2]  # only x,y
                coords_str = "_".join(str(c) for c in coords)
                parts.append(f"{obj}_{coords_str}")
        row_key = "_".join(parts)
        result_dict[row_key] = row

    return result_dict


# Scene configurations - each entry is one scene setup
SCENE_CONFIGS = get_scene_obj_coords()

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


def compute_intrinsics(camera_prim, width, height):
    focal_length_attr = camera_prim.GetAttribute("focalLength").Get()
    h_aperture_attr = camera_prim.GetAttribute("horizontalAperture").Get()
    camera_prim.GetAttribute("verticalAperture").Set(VERTICAL_APARTURE)
    v_aperture_attr = camera_prim.GetAttribute("verticalAperture").Get()
    
    fx = focal_length_attr / h_aperture_attr * width
    fy = focal_length_attr / v_aperture_attr * height
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

# ---------------------- Object Management Functions ----------------------

def load_all_objects():
    """Load all objects from configuration into the scene"""
    loaded_objects = {}
    base_path = "/workspace/isaaclab/SG/aloha_assets_new/"
    
    for obj_config in OBJECT_CONFIG["objects"]:
        obj_name = obj_config["name"]
        loaded_objects[obj_name] = []
        
        # Get placement coordinates
        if obj_config["placement"]["strategy"] == "grid":
            coordinates = obj_config["placement"]["grid_coordinates"]
        else:
            # For surface placement, use table positions
            coordinates = [[-4.5, -1.1, 0.75], [-4.5, 0.0, 0.75], [-4.5, 1.1, 0.75]]
        
        # Load each instance
        for idx, coord in enumerate(coordinates):
            prim_path = f"/World/Objects/{obj_name}_{idx}"
            usd_path = base_path + obj_config["usd_paths"][0]
            
            # Add object to stage
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            
            # Create XFormPrim for easier manipulation
            xform_prim = XFormPrim(prim_path=prim_path)
            
            # Store object info
            loaded_objects[obj_name].append({
                "prim_path": prim_path,
                "xform": xform_prim,
                "original_position": coord,
                "index": idx
            })
            
            # Initially hide all objects
            # hide_prim(prim_path)
    
    return loaded_objects

import omni.physx.bindings as physx

# def reset_velocities(prim_path):
#     try:
#         physx.set_rigid_body_linear_velocity(prim_path, (0.0, 0.0, 0.0))
#         physx.set_rigid_body_angular_velocity(prim_path, (0.0, 0.0, 0.0))
#     except Exception:
#         pass
    
def setup_scene(loaded_objects, scene_config):
    """Configure objects for a specific scene"""
    # Storage position for unused objects (far outside the scene)
    storage_position = [50, 50, 0]
    
    # Transform from original coordinates to scene coordinates
    def transform_to_scene(pos):
        # Add offset to move objects into the recording area
        x_offset = 5   # Center of recording area (0-10)
        y_offset = 4   # Center of recording area (0.5-7.5)
        return [pos[0] + x_offset, pos[1] + y_offset, pos[2]]
    
    for obj_name, obj_instances in loaded_objects.items():
        for obj_info in obj_instances:
            obj_info["xform"].set_world_pose(position=storage_position, orientation=np.array([1, 0, 0, 0]))
            
    # Process each object type
    for obj_name, obj_instances in loaded_objects.items():
        if obj_name in scene_config:
            target_positions = scene_config[obj_name]

            for idx, obj_info in enumerate(obj_instances):
                if idx < len(target_positions):
                    # Move to scene position
                    scene_pos = transform_to_scene(target_positions[idx])
                    obj_info["xform"].set_world_pose(position=scene_pos, orientation=np.array([1, 0, 0, 0]))
                    # reset_velocities(obj_info["prim_path"])
                    # show_prim(obj_info["prim_path"])
                    print(f"Placed {obj_name}_{idx} at {scene_pos}")
                else:
                    # Move to storage
                    obj_info["xform"].set_world_pose(position=storage_position, orientation=np.array([1, 0, 0, 0]))
                    # reset_velocities(obj_info["prim_path"])
                    # hide_prim(obj_info["prim_path"])
                    print(f"Stored {obj_name}_{idx}")
        else:
            # Hide all instances of this object type
            for obj_info in obj_instances:
                obj_info["xform"].set_world_pose(position=storage_position, orientation=np.array([1, 0, 0, 0]))
                # reset_velocities(obj_info["prim_path"])
                # hide_prim(obj_info["prim_path"])

def generate_camera_keyframes(bounds):
    """Generate camera keyframes that stay within the scene bounds"""
    # Camera moves around the perimeter of the recording area at height 2m
    keyframes = [
            {'time': 0, 'translation': [0.6, 1.5, 2], 'euler_angles': [0, 25, 65]},
            {'time': 3, 'translation': [0.6, 1.5, 2], 'euler_angles': [0, 25, 25]},
            {'time': 6, 'translation': [4.75, 1.5, 2], 'euler_angles': [0, 25, 90]},
            {'time': 9, 'translation': [9.5, 1.5, 2], 'euler_angles': [0, 25, 115]},
            {'time': 12, 'translation': [9.5, 6.5, 2], 'euler_angles': [0, 25, 225]},
            {'time': 15, 'translation': [0.6, 6.5, 2], 'euler_angles': [0, 25, 290]},
            {'time': 18, 'translation': [0.6, 6.5, 2], 'euler_angles': [0, 25, 360]},
    ]
    return keyframes

# ---------------------- Camera Setup ----------------------
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([3, 0, 1.5]),
    resolution=(IMAGE_HEIGHT, IMAGE_WIDTH),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True, extrinsic=True)
)

my_world.reset()
camera.initialize()
camera.add_distance_to_camera_to_frame()

stage_ref = get_current_stage()
camera_prim = stage_ref.GetPrimAtPath("/World/Camera")
horizontal_aperture = camera_prim.GetAttribute("horizontalAperture")
horizontal_aperture.Set(HORIZONTAL_APARTURE)

fx, fy, cx, cy = compute_intrinsics(camera_prim, IMAGE_WIDTH, IMAGE_HEIGHT)
png_depth_scale = (MAX_DEPTH - MIN_DEPTH) / PNG_MAX_VALUE
print(f"Camera intrinsics -> fx: {fx:.3f}, fy: {fy:.3f}, cx: {cx:.3f}, cy: {cy:.3f}")

# ---------------------- ROS2 Camera Graph (keeping your original setup) ----------------------
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

# ---------------------- Load All Objects Once ----------------------
print("Loading all objects...")
loaded_objects = load_all_objects()
print(f"Loaded {len(loaded_objects)} object types")

# ---------------------- Process Each Scene ----------------------
for scene_name, scene_config in SCENE_CONFIGS.items():
    my_world.reset()
    print(f"\n{'='*50}")
    print(f"Processing {scene_name}")
    print(f"{'='*50}")
    
    # Setup scene
    setup_scene(loaded_objects, scene_config)
    
    # Create output directories
    base_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}/results"
    traj_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    
    traj_file_path = os.path.join(traj_dir, "traj.txt")
    open(traj_file_path, 'w').close()
    
    # Generate camera keyframes for this scene
    camera_keyframes = generate_camera_keyframes(SCENE_BOUNDS)
    
    # Warm-up
    for _ in range(WARMUP_STEPS):
        next_translation, next_orientation = interpolate_keyframes_with_euler(camera_keyframes, 0)
        if _ == 0:
            camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
        my_world.step(render=True)
        simulation_app.update()
    
    # Recording loop for this scene
    i = 0
    frame_index = 0
    max_time = camera_keyframes[-1]['time']
    
    while frame_index <= max_time:
        next_translation, next_orientation = interpolate_keyframes_with_euler(camera_keyframes, i)
        if next_translation is None:
            break
            
        camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
        position_, orientation_ = camera.get_local_pose(camera_axes="world")

        print(f"Scene - Frame {frame_index}, Time: {i}")
        i += 1  # Time step
        
        # Render
        for _ in range(RENDER_SUBSTEPS):
            my_world.step(render=True)
        simulation_app.update()
        
        # Capture data
        render_product_path = viewport_api.get_render_product_path()
        
        depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_ann.attach([render_product_path])
        depth_image = depth_ann.get_data()
        
        rgb_ann = rep.AnnotatorRegistry.get_annotator("LdrColor")
        rgb_ann.attach([render_product_path])
        rgba_image = rgb_ann.get_data()
        
        
        # Save data
        img_path = os.path.join(base_dir, f"frame{frame_index:06d}.jpg")
        depth_path = os.path.join(base_dir, f"depth{frame_index:06d}.png")
        # seg_colored_path = os.path.join(base_dir, f"semantic{frame_index:06d}.png")
        # seg_info_path = os.path.join(base_dir, f"semantic{frame_index:06d}_info.json")
        
        if depth_image.size != 0 and rgba_image.size != 0:
            # Save depth
            clipped_depth = np.clip(depth_image, MIN_DEPTH, MAX_DEPTH)
            normalized_depth = ((clipped_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)) * PNG_MAX_VALUE
            depth_image_uint16 = normalized_depth.astype("uint16")
            cv2.imwrite(depth_path, depth_image_uint16)
            
            
            # Save RGB
            rgb = rgba_image[:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, bgr)
            
            # Save trajectory
            position_ros, orientation_ros = camera.get_world_pose(camera_axes="ros")
            T_ros = transformation_matrix(position_ros, orientation_ros)
            with open(traj_file_path, "a") as traj_file:
                traj_file.write(' '.join(map(str, T_ros.flatten())) + "\n")
            
        
        frame_index += 1
    
    print(f"Completed {scene_name}: {frame_index} frames recorded")

# ---------------------- Shutdown ----------------------
print("\nAll scenes processed successfully!")
simulation_app.close()