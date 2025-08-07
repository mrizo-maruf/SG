# from isaacsim import SimulationApp
import carb
from isaacsim import SimulationApp
import sys

BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)

import matplotlib.pyplot as plt
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
# from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils import stage, extensions, nucleus
# from PIL import Image
from sensor_msgs.msg import Image
import os
import omni.replicator.core as rep
import omni
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import imageio
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import omni.graph.core as og

import rclpy
from cv_bridge import CvBridge
from vision_msgs.msg import Detection3DArray

from omni.isaac.core_nodes.scripts.utils import set_target_prims

from pxr import UsdGeom, Gf, Usd

from omni.isaac.core.utils.extensions import enable_extension

res = enable_extension("isaacsim.ros2.bridge")
print("*"*20, res)
my_world = World(stage_units_in_meters=1.0)

# asset_path = "/home/zhang/.local/share/ov/pkg/isaac_sim-2023.1.0/BBQ_dataset/simple_env/example.usd"    
# asset_path = "/workspace/isaaclab/code_pack/scene_11_21.usd"  
# add_reference_to_stage(usd_path=asset_path, prim_path="/World/env")

# Locate Isaac Sim assets folder to load environment and robot stages
assets_root_path = nucleus.get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()
stage.add_reference_to_stage(assets_root_path + BACKGROUND_USD_PATH, BACKGROUND_STAGE_PATH)



def transformation_matrix(position, orientation):
    """
    将位置和四元数方向转换为4x4的变换矩阵。

    参数:
        position: 形如 (x, y, z) 的三维坐标
        orientation: 形如 (qx, qy, qz, qw) 的四元数

    返回:
        4x4 的变换矩阵
    """
    # 创建旋转矩阵
    # r = Rotation.from_quat(orientation)
    # rotation_matrix = r.as_matrix()  # 3x3旋转矩阵
    w, x, y, z = orientation

    # 根据四元数计算3x3旋转矩阵
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])

    # 组合旋转和平移到4x4矩阵
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = position

    return transformation


def create_action_graph():
    keys = og.Controller.Keys
    controller = og.Controller(graph_id="camera_graph")
    camera_frame_id = "/World/Camera".split("/")[-1]
    controller.edit(
        {"graph_path": "/World/camera_graph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("camera_HelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("camera_HelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("camera_HelperBbox", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("camera_HelperInstance", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("camera_HelperInfo", "isaacsim.ros2.bridge.ROS2PublishCameraInfo"),
                ("create_render", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("read_camera_info", "isaacsim.core.nodes.IsaacReadCameraInfo"),
                ("PublishTF_"+camera_frame_id, "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                # ("PublishRawTF_"+camera_frame_id+"_world", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),              
                ("read_sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "create_render.inputs:execIn"),
                (
                    "create_render.outputs:execOut",
                    "camera_HelperRgb.inputs:execIn",
                ),
                (
                    "create_render.outputs:execOut",
                    "camera_HelperDepth.inputs:execIn",
                ),
                (
                    "create_render.outputs:execOut",
                    "camera_HelperBbox.inputs:execIn",
                ),
                (
                    "create_render.outputs:execOut",
                    "camera_HelperInstance.inputs:execIn",
                ),
                (
                    "create_render.outputs:execOut",
                    "camera_HelperInfo.inputs:execIn",
                ),
                (
                    "create_render.outputs:renderProductPath",
                    "camera_HelperRgb.inputs:renderProductPath",
                ),
                (
                    "create_render.outputs:renderProductPath",
                    "camera_HelperDepth.inputs:renderProductPath",
                ),
                (
                    "create_render.outputs:renderProductPath",
                    "camera_HelperBbox.inputs:renderProductPath",
                ),
                (
                    "create_render.outputs:renderProductPath",
                    "camera_HelperInstance.inputs:renderProductPath",
                ),
                (
                    "create_render.outputs:renderProductPath",
                    "read_camera_info.inputs:renderProductPath",
                ),
                # (
                #     "read_camera_info.outputs:focalLength",
                #     "camera_HelperInfo.inputs:focalLength",
                # ),
                (
                    "read_camera_info.outputs:height",
                    "camera_HelperInfo.inputs:height",
                ),
                # (
                #     "read_camera_info.outputs:horizontalAperture",
                #     "camera_HelperInfo.inputs:horizontalAperture",
                # ),
                # (
                #     "read_camera_info.outputs:horizontalOffset",
                #     "camera_HelperInfo.inputs:horizontalOffset",
                # ),
                # (
                #     "read_camera_info.outputs:verticalAperture",
                #     "camera_HelperInfo.inputs:verticalAperture",
                # ),
                # (
                #     "read_camera_info.outputs:verticalOffset",
                #     "camera_HelperInfo.inputs:verticalOffset",
                # ),
                (
                    "read_camera_info.outputs:width",
                    "camera_HelperInfo.inputs:width",
                ),
                ("OnPlaybackTick.outputs:tick", "PublishTF_"+camera_frame_id+".inputs:execIn"),
                # ("OnPlaybackTick.outputs:tick", "PublishRawTF_"+camera_frame_id+"_world.inputs:execIn"),
                ("read_sim_time.outputs:simulationTime", "PublishTF_"+camera_frame_id+".inputs:timeStamp")
            ],
            keys.SET_VALUES: [
                (
                    "create_render.inputs:cameraPrim",
                    "/World/Camera",
                ),
                (
                    "camera_HelperDepth.inputs:topicName",
                    "depth",
                ),
                (
                    "camera_HelperDepth.inputs:type",
                    "depth",
                ),
                (
                    "camera_HelperBbox.inputs:topicName",
                    "bbox_3d",
                ),
                (
                    "camera_HelperBbox.inputs:type",
                    "bbox_3d",
                ),
                (
                    "camera_HelperInstance.inputs:topicName",
                    "instance",
                ),
                (
                    "camera_HelperInstance.inputs:type",
                    "instance_segmentation",
                ),
                ("PublishTF_"+camera_frame_id+".inputs:topicName", "/tf"),
                # ("PublishRawTF_"+camera_frame_id+"_world.inputs:topicName", "/tf"),
                # ("PublishRawTF_"+camera_frame_id+"_world.inputs:parentFrameId", camera_frame_id),
                # ("PublishRawTF_"+camera_frame_id+"_world.inputs:childFrameId", camera_frame_id+"_world"),
                # Static transform from ROS camera convention to world (+Z up, +X forward) convention:
                # ("PublishRawTF_"+camera_frame_id+"_world.inputs:rotation", [0.5, -0.5, 0.5, 0.5]),
  
            ],
        },
    )

    # set_targets(
    #     prim=stage.GetPrimAtPath(cfg.camera_action_graph_stage_path + f"/set_Camera"),
    #     attribute="inputs:cameraPrim",
    #     target_prim_paths=[camera_prim],
    # )
    # Add target prims for the USD pose. All other frames are static.
    set_target_prims(
        primPath="/World/camera_graph"+"/PublishTF_"+camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=["/World/Camera"],
    )
    set_target_prims(
        primPath="/World/camera_graph"+"/PublishTF_"+camera_frame_id,
        inputName="inputs:parentPrim",
        targetPrimPaths=["/World"],
    )
    simulation_app.update()


# P: [3054.16357421875, 0.0, 640.0, 0.0, 0.0, 3054.16357421875, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
# K: [1221.66552734375, 0.0, 640.0, 0.0, 1221.6654052734375, 360.0, 0.0, 0.0, 1.0]  #  camera.set_focal_length(20 / 10.0)
# P = { fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0 }
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([-3.11, -1.87, 1.0]),
    frequency=20,
    resolution=(256, 256),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
)
camera.initialize()

create_action_graph()
my_world.reset()

# stage = get_current_stage()

# camera_prim = stage.GetPrimAtPath("/World/Camera")

camera.set_focal_length(20 / 10.0)
camera.set_local_pose(np.array([3, 0, 1.5]), rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True, extrinsic=True), camera_axes="world")

# Initialize
translation, orientation = camera.get_world_pose(camera_axes="world")
for j in range(30):
    next_orientation = euler_angles_to_quat(np.array([0, 10, 180]), degrees=True, extrinsic=True) 
    next_translation = translation 
    camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    # translation, orientation = camera.get_local_pose(camera_axes="world")
    # print("translation", translation)
    # print("orientation", orientation)
    print(j)
    my_world.step(render=True)
    simulation_app.update()

i = 0
# record = False
while simulation_app.is_running():
    my_world.step(render=True)

    translation, orientation = camera.get_local_pose(camera_axes="world")
    print("translation", translation)
    print("orientation", orientation)
    translation, orientation = camera.get_world_pose(camera_axes="world")
    print("translation", translation)
    print("orientation", orientation)

    # 让camera动起来
    if i < 20:   
        next_orientation = orientation
        next_translation = np.array([3 + 0.1*i, 0, 1.5])
    elif i>=20 and i<40:
        next_orientation = euler_angles_to_quat(np.array([0, 10-0.5*(i-20), 180]), degrees=True, extrinsic=True) # world
        next_translation = np.array([3 + 0.1*20, 0, 1.5-0.025*(i-20)])
        # next_translation = translation     
    elif i>=40 and i<80:
        next_orientation = euler_angles_to_quat(np.array([0, 0, 180+1*(i-40)]), degrees=True, extrinsic=True) # world
        next_translation = translation     
    elif i>=80 and i<160:
        next_orientation = euler_angles_to_quat(np.array([0, 0, 220-(i-80)]), degrees=True, extrinsic=True) # world
        next_translation = translation  
    else:
        break
    
    camera.set_local_pose(next_translation, next_orientation, camera_axes="world")
    
    position_, orientation_ = camera.get_local_pose(camera_axes="world")  # ros 或者 world  或者usd
    transformation_matrix_result = transformation_matrix(position_, orientation_)
    print(transformation_matrix_result)

    i += 1

simulation_app.close()