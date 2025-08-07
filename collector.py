import os
import json
import cv2
import rclpy
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import numpy as np

# --- CONFIG ---
ENV_NAME = "rizo"
BASE_DIR = f"/workspace/isaaclab/isaac_sim/{ENV_NAME}/results"
TRAJ_FILE_PATH = f"/workspace/isaaclab/isaac_sim/{ENV_NAME}/traj.txt"
os.makedirs(BASE_DIR, exist_ok=True)

IMAGE_PREFIX = "frame"
DEPTH_PREFIX = "depth"
INSTANCE_PREFIX = "instance"

# --- STATE ---
rgb_queue = []
depth_queue = []
instance_queue = []
tf_queue = []
counter = 0

bridge = CvBridge()
TIME_TOLERANCE = 0.1


def to_float_time(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def transformation_matrix(position, orientation):
    w, x, y, z = orientation
    rotation_matrix = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    return transform


def find_closest(queue, target_time):
    closest_msg = None
    closest_diff = float("inf")
    for msg_time, msg in queue:
        diff = abs(msg_time - target_time)
        if diff < closest_diff:
            closest_diff = diff
            closest_msg = msg
    return closest_msg if closest_diff <= TIME_TOLERANCE else None


def save_data(rgb_msg, depth_msg, instance_msg, tf_transform, node):
    global counter

    try:
        # RGB
        rgb_img = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        rgb_path = os.path.join(BASE_DIR, f"{IMAGE_PREFIX}{counter:06d}.jpg")
        cv2.imwrite(rgb_path, rgb_img)

        # Depth
        depth_img = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        min_val, max_val = 0.01, 10.0
        clipped = np.clip(depth_img, min_val, max_val)
        norm_depth = ((clipped - min_val) / (max_val - min_val)) * 65535
        depth_img_uint16 = norm_depth.astype("uint16")
        depth_path = os.path.join(BASE_DIR, f"{DEPTH_PREFIX}{counter:06d}.png")
        cv2.imwrite(depth_path, depth_img_uint16)

        # Instance Seg
        instance_img = bridge.imgmsg_to_cv2(instance_msg, desired_encoding="passthrough")
        instance_path = os.path.join(BASE_DIR, f"{INSTANCE_PREFIX}{counter:06d}.png")
        cv2.imwrite(instance_path, instance_img)

        # TF Pose
        if tf_transform.child_frame_id == "floating_camera_world":
            pos = tf_transform.transform.translation
            ori = tf_transform.transform.rotation
            position = (pos.x, pos.y, pos.z)
            orientation = (ori.w, ori.x, ori.y, ori.z)
            matrix = transformation_matrix(position, orientation)
            with open(TRAJ_FILE_PATH, "a") as f:
                f.write(" ".join(map(str, matrix.flatten())) + "\n")

        node.get_logger().info(f"[✓] Saved frame {counter}")
        counter += 1

    except Exception as e:
        node.get_logger().error(f"[✗] Error saving data: {e}")


def process_messages(node):
    if not rgb_queue or not depth_queue or not tf_queue:
        return

    rgb_time, rgb_msg = rgb_queue.pop(0)
    depth_msg = find_closest(depth_queue, rgb_time)
    instance_msg = find_closest(instance_queue, rgb_time)
    tf_msg = find_closest(tf_queue, rgb_time)

    if all([depth_msg, instance_msg, tf_msg]):
        save_data(rgb_msg, depth_msg, instance_msg, tf_msg, node)


def rgb_callback(msg):
    rgb_queue.append((to_float_time(msg.header.stamp), msg))
    process_messages(node)


def depth_callback(msg):
    depth_queue.append((to_float_time(msg.header.stamp), msg))


def instance_callback(msg):
    instance_queue.append((to_float_time(msg.header.stamp), msg))


def tf_callback(msg):
    for transform in msg.transforms:
        if transform.child_frame_id == "floating_camera_world":
            tf_queue.append((to_float_time(transform.header.stamp), transform))


def main():
    global node

    rclpy.init()
    node = rclpy.create_node("isaac_sim_data_saver_script")

    qos = QoSProfile(depth=10)

    node.create_subscription(Image, "floating_camera_rgb", rgb_callback, qos)
    node.create_subscription(Image, "floating_camera_depth", depth_callback, qos)
    node.create_subscription(Image, "floating_camera_instance_seg", instance_callback, qos)
    node.create_subscription(TFMessage, "/tf", tf_callback, qos)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
