# Camera Data Collector for 3D Scene Graph Pipeline
# Captures RGB + Point Clouds from Isaac Sim camera data

import carb
from isaacsim import SimulationApp
import sys
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# ---------------------- Configuration ----------------------
BACKGROUND_STAGE_PATH = "/World/env"
scene_name = "UR5-Peg-In-Hole_02"
BACKGROUND_USD_PATH = f"/workspace/isaaclab/SG/Franka-Cabinet/{scene_name}.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)

# Camera settings
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0
FOCAL_LENGTH = 50
HORIZONTAL_APERTURE = 80
VERTICAL_APERTURE = 45

# Data collection settings
SIMULATION_TIME = 5.0  # seconds
FPS = 10  # frames per second for data collection
WARMUP_STEPS = 50
RENDER_SUBSTEPS = 5  # reduced for faster collection

# ---------------------- Imports ----------------------
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils import stage
from isaacsim.storage.native import get_assets_root_path
import omni.replicator.core as rep
import omni
from isaacsim.core.utils.rotations import euler_angles_to_quat
import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Sdf
from omni.kit.viewport.utility import get_active_viewport

# ---------------------- Extensions ----------------------
enable_extension("isaacsim.ros2.bridge")

# ---------------------- World Setup ----------------------
physics_dt = 1.0 / 60.0
rendering_dt = 1.0 / 60.0
my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# ---------------------- Environment ----------------------
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

add_reference_to_stage(usd_path=BACKGROUND_USD_PATH, prim_path=BACKGROUND_STAGE_PATH)

# ---------------------- Data Collection Classes ----------------------

class CameraIntrinsics:
    """Store camera intrinsic parameters"""
    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

class FrameData:
    """Store data for a single frame"""
    def __init__(self, frame_id: int, timestamp: float, rgb: np.ndarray, 
                 depth: np.ndarray, camera_pose: np.ndarray, point_cloud: np.ndarray):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.rgb = rgb  # (H, W, 3)
        self.depth = depth  # (H, W)
        self.camera_pose = camera_pose  # 4x4 transformation matrix
        self.point_cloud = point_cloud  # (N, 3) world coordinates

class CameraDataCollector:
    """Main class for collecting camera data"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.frames_data: List[FrameData] = []
        self.intrinsics: Optional[CameraIntrinsics] = None
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pointclouds"), exist_ok=True)
        
    def setup_camera(self, camera_prim) -> CameraIntrinsics:
        """Setup camera and compute intrinsics"""
        # Set camera parameters
        camera_prim.GetAttribute("horizontalAperture").Set(HORIZONTAL_APERTURE)
        camera_prim.GetAttribute("verticalAperture").Set(VERTICAL_APERTURE)
        camera_prim.GetAttribute("focalLength").Set(FOCAL_LENGTH)
        
        # Compute intrinsics
        focal_length_attr = camera_prim.GetAttribute("focalLength").Get()
        h_aperture_attr = camera_prim.GetAttribute("horizontalAperture").Get()
        v_aperture_attr = camera_prim.GetAttribute("verticalAperture").Get()
        
        fx = focal_length_attr / h_aperture_attr * IMAGE_WIDTH
        fy = focal_length_attr / v_aperture_attr * IMAGE_HEIGHT
        cx = IMAGE_WIDTH / 2.0
        cy = IMAGE_HEIGHT / 2.0
        
        self.intrinsics = CameraIntrinsics(fx, fy, cx, cy, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        print(f"Camera intrinsics -> fx: {fx:.3f}, fy: {fy:.3f}, cx: {cx:.3f}, cy: {cy:.3f}")
        return self.intrinsics
    
    def depth_to_pointcloud(self, depth: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
        """Convert depth image to 3D point cloud in world coordinates"""
        if self.intrinsics is None:
            raise ValueError("Camera intrinsics not set")
        
        # Get valid depth pixels
        h, w = depth.shape
        valid_mask = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u[valid_mask]
        v = v[valid_mask]
        z = depth[valid_mask]
        
        # Back-project to camera coordinates
        x = (u - self.intrinsics.cx) * z / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * z / self.intrinsics.fy
        
        # Stack to homogeneous coordinates
        points_cam = np.vstack([x, y, z, np.ones_like(x)])
        
        # Transform to world coordinates
        points_world = camera_pose @ points_cam
        
        return points_world[:3].T  # Return as (N, 3)
    
    def capture_frame(self, camera, frame_id: int, timestamp: float) -> FrameData:
        """Capture a single frame of data"""
        # Get camera pose
        position, orientation = camera.get_world_pose()
        camera_pose = self.pose_to_matrix(position, orientation)
        
        # Get viewport and render product
        viewport_api = get_active_viewport()
        render_product_path = viewport_api.get_render_product_path()
        
        # Capture RGB
        rgb_ann = rep.AnnotatorRegistry.get_annotator("LdrColor")
        rgb_ann.attach([render_product_path])
        rgba_image = rgb_ann.get_data()
        rgb_image = rgba_image[:, :, :3]  # Remove alpha channel
        
        # Capture Depth
        depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_ann.attach([render_product_path])
        depth_image = depth_ann.get_data()
        
        # Generate point cloud
        point_cloud = self.depth_to_pointcloud(depth_image, camera_pose)
        
        # Create frame data
        frame_data = FrameData(frame_id, timestamp, rgb_image, depth_image, camera_pose, point_cloud)
        
        return frame_data
    
    def save_frame(self, frame_data: FrameData):
        """Save frame data to disk"""
        frame_id = frame_data.frame_id
        
        # Save RGB
        rgb_path = os.path.join(self.output_dir, "rgb", f"frame_{frame_id:06d}.jpg")
        rgb_bgr = cv2.cvtColor(frame_data.rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_path, rgb_bgr)
        
        # Save Depth (16-bit PNG)
        depth_path = os.path.join(self.output_dir, "depth", f"depth_{frame_id:06d}.png")
        # Normalize depth to 16-bit
        depth_normalized = np.clip(frame_data.depth, MIN_DEPTH, MAX_DEPTH)
        depth_normalized = ((depth_normalized - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)) * 65535
        depth_uint16 = depth_normalized.astype(np.uint16)
        cv2.imwrite(depth_path, depth_uint16)
        
        # Save Point Cloud (NPZ format for efficiency)
        pc_path = os.path.join(self.output_dir, "pointclouds", f"pointcloud_{frame_id:06d}.npz")
        np.savez_compressed(pc_path, points=frame_data.point_cloud)
        
        # Store frame metadata
        self.frames_data.append(frame_data)
    
    def save_metadata(self):
        """Save collection metadata"""
        metadata = {
            "scene_name": scene_name,
            "total_frames": len(self.frames_data),
            "simulation_time": SIMULATION_TIME,
            "fps": FPS,
            "camera_intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "K": self.intrinsics.K.tolist()
            },
            "depth_range": {
                "min_depth": MIN_DEPTH,
                "max_depth": MAX_DEPTH
            },
            "frames": []
        }
        
        for frame_data in self.frames_data:
            frame_info = {
                "frame_id": frame_data.frame_id,
                "timestamp": frame_data.timestamp,
                "camera_pose": frame_data.camera_pose.tolist(),
                "num_points": len(frame_data.point_cloud),
                "rgb_file": f"rgb/frame_{frame_data.frame_id:06d}.jpg",
                "depth_file": f"depth/depth_{frame_data.frame_id:06d}.png",
                "pointcloud_file": f"pointclouds/pointcloud_{frame_data.frame_id:06d}.npz"
            }
            metadata["frames"].append(frame_info)
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata for {len(self.frames_data)} frames to {metadata_path}")
    
    @staticmethod
    def pose_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Convert position and quaternion to 4x4 transformation matrix"""
        # orientation is (w, x, y, z)
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

# ---------------------- Trajectory Functions ----------------------

def interpolate_keyframes_with_euler(keyframes: List[Dict], t: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Interpolate camera pose from keyframes at time t"""
    for j in range(len(keyframes) - 1):
        t0, t1 = keyframes[j]['time'], keyframes[j + 1]['time']
        if t0 <= t <= t1:
            kf0, kf1 = keyframes[j], keyframes[j + 1]
            break
    else:
        return None, None
    
    alpha = (t - t0) / (t1 - t0)
    next_translation = (1 - alpha) * np.array(kf0['translation']) + alpha * np.array(kf1['translation'])
    euler0 = kf0['euler_angles']
    euler1 = kf1['euler_angles']
    interpolated_euler = (1 - alpha) * np.array(euler0) + alpha * np.array(euler1)
    next_orientation = euler_angles_to_quat(interpolated_euler, degrees=True)
    return next_translation, next_orientation

# ---------------------- Main Collection Function ----------------------

def collect_camera_data():
    """Main function to collect camera data"""
    
    # Setup output directory
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"/workspace/isaaclab/IsaacSimData/{scene_name}/camera_data_{timestamp_str}"
    
    collector = CameraDataCollector(output_dir)
    
    # Setup camera
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([3.5, -2.5, 1]),
        resolution=(IMAGE_HEIGHT, IMAGE_WIDTH),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True, extrinsic=True)
    )
    
    my_world.reset()
    camera.initialize()
    camera.add_distance_to_camera_to_frame()
    
    # Setup camera intrinsics
    stage_ref = get_current_stage()
    camera_prim = stage_ref.GetPrimAtPath("/World/Camera")
    collector.setup_camera(camera_prim)
    
    # Define camera trajectory (similar to your existing keyframes)
    keyframes_move = [
        {'time': 0, 'translation': [3, -2, 1.5], 'euler_angles': [0, 20, 125]},
        {'time': 2, 'translation': [3, 2, 1.5], 'euler_angles': [0, 20, 225]},
        {'time': 5, 'translation': [-2, 2, 1.5], 'euler_angles': [0, 20, 315]},
        # {'time': 22.5, 'translation': [-2, -2, 1.5], 'euler_angles': [0, 20, 380]},
        # {'time': 30, 'translation': [3, -2, 1.5], 'euler_angles': [0, 20, 485]},  # Return to start
    ]
    
    # Warm-up
    print("Warming up simulation...")
    for _ in range(WARMUP_STEPS):
        translation, orientation = interpolate_keyframes_with_euler(keyframes_move, 0)
        if _ == 0:
            camera.set_local_pose(translation, orientation, camera_axes="world")
        my_world.step(render=True)
        simulation_app.update()
    
    # Main collection loop
    print(f"Starting data collection for {SIMULATION_TIME} seconds at {FPS} FPS...")
    
    start_time = time.time()
    frame_interval = 1.0 / FPS
    next_capture_time = 0.0
    frame_id = 0
    
    while simulation_app.is_running():
        current_sim_time = time.time() - start_time
        
        if current_sim_time >= SIMULATION_TIME:
            break
        
        # Update camera pose
        translation, orientation = interpolate_keyframes_with_euler(keyframes_move, current_sim_time)
        if translation is None:
            break
        
        camera.set_local_pose(translation, orientation, camera_axes="world")
        
        # Step simulation
        for _ in range(RENDER_SUBSTEPS):
            my_world.step(render=True)
        simulation_app.update()
        
        # Capture frame if it's time
        if current_sim_time >= next_capture_time:
            print(f"Capturing frame {frame_id} at t={current_sim_time:.2f}s")
            
            frame_data = collector.capture_frame(camera, frame_id, current_sim_time)
            collector.save_frame(frame_data)
            
            frame_id += 1
            next_capture_time += frame_interval
    
    # Save metadata
    collector.save_metadata()
    
    print(f"Data collection complete! Collected {frame_id} frames")
    print(f"Data saved to: {output_dir}")
    
    return output_dir

# ---------------------- Main Execution ----------------------

if __name__ == "__main__":
    try:
        output_path = collect_camera_data()
        print(f"Success! Camera data saved to: {output_path}")
    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
