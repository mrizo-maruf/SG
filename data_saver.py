#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge


class IsaacDataSaver(Node):
    def __init__(self):
        super().__init__('isaac_sim_data_saver')

        # ==== PARAMETERS ====
        env_name = 'rizo'
        base = f"/workspace/isaaclab/code_pack/{env_name}/results"
        self.dirs = {
            'rgb':      os.path.join(base, 'rgb'),
            'depth':    os.path.join(base, 'depth'),
            'instance': os.path.join(base, 'instance'),
            'bbox':     os.path.join(base, 'bbox_3d'),
            'traj':     os.path.join(base, 'traj'),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.traj_file = os.path.join(self.dirs['traj'], 'traj.txt')
        self.time_tolerance = 0.1  # seconds

        # prefixes
        self.prefix = {
            'rgb':      'frame',
            'depth':    'depth',
            'instance': 'inst',
            'bbox':     'bbox',
        }

        # message queues
        self.queues = {k: [] for k in ['rgb','depth','instance','tf','bbox']}

        self.counter = 0
        self.bridge = CvBridge()

        # ==== SUBSCRIPTIONS ====
        self.create_subscription(Image,        'rgb',      self.rgb_callback,      10)
        self.create_subscription(Image,        'depth',    self.depth_callback,    10)
        self.create_subscription(Image,        'instance', self.instance_callback, 10)
        self.create_subscription(Detection3DArray,'bbox_3d',self.bbox_callback,    10)
        self.create_subscription(TFMessage,    'tf',       self.tf_callback,       10)

        self.get_logger().info('IsaacDataSaver node started.')

    @staticmethod
    def to_seconds(tstamp):
        """Convert builtin time to float seconds."""
        return tstamp.sec + tstamp.nanosec * 1e-9

    @staticmethod
    def make_transform_matrix(position, orientation):
        """Build a 4×4 matrix from translation and quaternion (w,x,y,z)."""
        w, x, y, z = orientation
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1-2*(x*x+z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x+y*y) ]
        ])
        M = np.eye(4)
        M[:3,:3] = R
        M[:3, 3] = position
        return M

    def find_closest(self, queue, target_t):
        """Find message in queue whose timestamp is closest to target_t."""
        best, best_diff = None, float('inf')
        for t, msg in queue:
            diff = abs(t - target_t)
            if diff < best_diff:
                best_diff, best = diff, msg
        return best if best_diff <= self.time_tolerance else None

    def colorize_instance(self, inst_map):
        """
        Map each label ID in inst_map to a unique color.
        We'll hash the label value to a color for consistency.
        """
        inst_u8 = inst_map.astype(np.uint16)
        h, w = inst_u8.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        unique_ids = np.unique(inst_u8)
        rng = np.random.default_rng(0)  # fixed seed
        id_to_color = {uid: rng.integers(0,255,3,dtype=np.uint8) for uid in unique_ids}
        for uid, col in id_to_color.items():
            mask = (inst_u8 == uid)
            color_img[mask] = col
        return color_img

    def save_frame(self, rgb_msg, depth_msg, inst_msg, tf_msg, bbox_msg):
        """Convert and save all modalities for one synchronized frame."""
        try:
            idx = f"{self.counter:06d}"

            # — RGB —
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            cv2.imwrite(os.path.join(self.dirs['rgb'],
                                     f"{self.prefix['rgb']}{idx}.jpg"), rgb)

            # — Depth (normalized to [0.01m,10m] → uint16) —
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            d = np.clip(depth, 0.01, 10.0)
            d = ((d - 0.01) / (9.99) * 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(self.dirs['depth'],
                                     f"{self.prefix['depth']}{idx}.png"), d)

            # — Instance Segmentation (colorized) —
            inst = self.bridge.imgmsg_to_cv2(inst_msg, 'passthrough')
            inst_color = self.colorize_instance(inst)
            cv2.imwrite(os.path.join(self.dirs['instance'],
                                     f"{self.prefix['instance']}{idx}.png"), inst_color)

            # — Camera Pose —
            for tf in tf_msg.transforms:
                if tf.header.frame_id=='World' and tf.child_frame_id=='Camera':
                    t = (tf.transform.translation.x,
                         tf.transform.translation.y,
                         tf.transform.translation.z)
                    q = (tf.transform.rotation.w,
                         tf.transform.rotation.x,
                         tf.transform.rotation.y,
                         tf.transform.rotation.z)
                    M = self.make_transform_matrix(t, q)
                    with open(self.traj_file, 'a') as f:
                        f.write(' '.join(map(str, M.flatten())) + '\n')
                    break

            # — 3D BBoxes —
            boxes = []
            for det in bbox_msg.detections:
                if not det.results:
                    continue
                r = det.results[0].hypothesis
                B = det.bbox
                boxes.append({
                    'id': r.class_id,
                    'center': {
                        'pos':{ 'x':B.center.position.x,
                                'y':B.center.position.y,
                                'z':B.center.position.z },
                        'orient':{ 'x':B.center.orientation.x,
                                   'y':B.center.orientation.y,
                                   'z':B.center.orientation.z,
                                   'w':B.center.orientation.w },
                    },
                    'size': { 'x':B.size.x, 'y':B.size.y, 'z':B.size.z }
                })

            with open(os.path.join(self.dirs['bbox'], f"{self.prefix['bbox']}{idx}.json"), 'w') as jf:
                json.dump(boxes, jf, indent=4)

            self.get_logger().info(f"Saved frame {idx}")
            self.counter += 1

        except Exception as e:
            self.get_logger().error(f"Error saving frame: {e}")

    def try_process(self):
        """Attempt to find synchronized messages and save a frame."""
        queues = self.queues
        if not all(queues[k] for k in ['rgb','depth','instance','tf','bbox']):
            return

        t_rgb, rgb_msg = queues['rgb'].pop(0)
        depth_msg  = self.find_closest(queues['depth'],    t_rgb)
        inst_msg   = self.find_closest(queues['instance'], t_rgb)
        tf_msg     = self.find_closest(queues['tf'],       t_rgb)
        bbox_msg   = self.find_closest(queues['bbox'],     t_rgb)

        if depth_msg and inst_msg and tf_msg and bbox_msg:
            self.save_frame(rgb_msg, depth_msg, inst_msg, tf_msg, bbox_msg)

    # ==== CALLBACKS ====

    def rgb_callback(self, msg):
        self.queues['rgb'].append((self.to_seconds(msg.header.stamp), msg))
        self.try_process()

    def depth_callback(self, msg):
        self.queues['depth'].append((self.to_seconds(msg.header.stamp), msg))
        self.try_process()

    def instance_callback(self, msg):
        self.queues['instance'].append((self.to_seconds(msg.header.stamp), msg))
        self.try_process()

    def bbox_callback(self, msg):
        self.queues['bbox'].append((self.to_seconds(msg.header.stamp), msg))
        self.try_process()

    def tf_callback(self, msg):
        # store the whole TFMessage
        for tf in msg.transforms:
            if tf.header.frame_id=='World' and tf.child_frame_id=='Camera':
                self.queues['tf'].append((self.to_seconds(tf.header.stamp), msg))
                break
        self.try_process()


def main(args=None):
    rclpy.init(args=args)
    node = IsaacDataSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
