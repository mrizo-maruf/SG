# multi_scene_capture.py
from isaacsim import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)

import os, json, cv2, numpy as np
import omni
import omni.replicator.core as rep
import omni.graph.core as og
from pxr import Sdf, UsdGeom
from isaacsim.core.utils.extensions import enable_extension
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import euler_angles_to_quat

# --------------------------------------------------
# Paths / config
# --------------------------------------------------
SCENE_ITEMS_JSON = "/workspace/isaaclab/SG/scene_items.json"  # replace with your extended JSON version
OUTPUT_ROOT = "/workspace/isaaclab/IsaacSimData"
STASH_POS = (1000.0, 1000.0, -1000.0)  # far away
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720
MIN_DEPTH, MAX_DEPTH = 0.01, 10.0
PNG_MAX_VALUE = 65535
WARMUP_STEPS = 10
FRAMES_PER_SCENE = 40
RENDER_SUBSTEPS = 10

# Scene signature definitions (edit as needed)
SCENES = {
    "sc1_0ch_1b":    {"tables":[1], "chairs":[],      "cabinets":[0,3], "bowl":{"on_table":1}},
    "sc1_1chl_1b":   {"tables":[1], "chairs":[0],     "cabinets":[0,3], "bowl":{"on_table":1}},
    "sc1_1chr_1b":   {"tables":[1], "chairs":[2],     "cabinets":[0,3], "bowl":{"on_table":1}},
    "sc1_2ch_1b":    {"tables":[1], "chairs":[0,2],   "cabinets":[0,3], "bowl":{"on_table":1}},
}

enable_extension("isaacsim.ros2.bridge")

world = World(physics_dt=1/30.0, rendering_dt=1/30.0, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = get_current_stage()

# --------------------------------------------------
# Load config & preload all assets
# --------------------------------------------------
with open(SCENE_ITEMS_JSON, "r") as f:
    cfg = json.load(f)["objects"]

# Registry: {class_name: {"usd": str, "size": (x,y,z), "grid": [...], "prim_paths": [...]} }
registry = {}
for obj in cfg:
    name = obj["name"]
    usd_path = obj["usd_paths"][0]
    size = tuple(obj["size"])
    placement = obj["placement"][0]
    grid = placement.get("grid_coordinates", [])
    count = obj["count"]
    prim_paths = []
    for i in range(count):
        prim_path = f"/World/Assets/{name}_{i}"
        add_reference_to_stage(usd_path=os.path.join("/workspace/isaaclab/SG/aloha_assets_new", usd_path), prim_path=prim_path)
        prim_paths.append(prim_path)
    registry[name] = {
        "usd": usd_path,
        "size": size,
        "grid": grid,
        "prim_paths": prim_paths,
        "type": obj["type"],
        "is_surface": "surface_provider" in obj["type"],
        "surface_only": "surface_only" in obj["type"],
        "change_color": "changeable_color" in obj["type"],
    }

world.reset()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def set_visibility(prim_path, visible: bool):
    prop_path = f"{prim_path}.visibility"
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(prop_path),
        value="inherited" if visible else "invisible", prev=None
    )

def move_prim(prim_path, pos):
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if ops:
        ops[0].Set(Gf.Vec3d(*pos))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

def hide_all():
    for cat in registry.values():
        for p in cat["prim_paths"]:
            set_visibility(p, False)

def activate_indexed(name, indices):
    info = registry[name]
    for i, prim_path in enumerate(info["prim_paths"]):
        if i in indices:
            set_visibility(prim_path, True)
            # place at its grid coordinate if available
            if i < len(info["grid"]):
                pos = info["grid"][i]
                move_prim_to(prim_path, pos)
        else:
            set_visibility(prim_path, False)

def move_prim_to(prim_path, pos_xyz):
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    # set translate
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(tuple(pos_xyz))
            break
    else:
        xform.AddTranslateOp().Set(tuple(pos_xyz))

def place_bowl(bowl_cfg, scene_cfg):
    bowl_info = registry["bowl"]
    bowl_prim = bowl_info["prim_paths"][0]
    set_visibility(bowl_prim, True)
    tbl_idx = bowl_cfg["on_table"]
    table_info = registry["table"]
    if tbl_idx >= len(table_info["grid"]):
        # fallback
        tbl_idx = 0
    table_pos = table_info["grid"][tbl_idx]
    table_h = table_info["size"][2]
    bowl_h = bowl_info["size"][2]
    bowl_pos = (table_pos[0], table_pos[1], table_pos[2] + table_h + bowl_h / 2.0)
    move_prim_to(bowl_prim, bowl_pos)

def activate_scene(signature, scene_cfg):
    hide_all()
    # tables
    activate_indexed("table", scene_cfg.get("tables", []))
    # chairs
    activate_indexed("chair", scene_cfg.get("chairs", []))
    # cabinets
    activate_indexed("cabinet", scene_cfg.get("cabinets", []))
    # bowl
    if "bowl" in scene_cfg:
        place_bowl(scene_cfg["bowl"], scene_cfg)

def create_color_map(num_classes):
    colors = [[0,0,0]]
    import colorsys
    golden = 0.61803398875
    for i in range(1, num_classes):
        h = (i * golden) % 1.0
        s = 0.6
        v = 0.85
        r,g,b = colorsys.hsv_to_rgb(h,s,v)
        colors.append([int(b*255), int(g*255), int(r*255)])
    return np.array(colors, dtype=np.uint8)

# --------------------------------------------------
# Camera + graph
# --------------------------------------------------
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([0,0,2.0]),
    resolution=(IMAGE_HEIGHT, IMAGE_WIDTH),
    orientation=rot_utils.euler_angles_to_quats(np.array([0,0,0]), degrees=True, extrinsic=True)
)
world.reset()
camera.initialize()

import omni.usd
from pxr import Gf
from omni.kit.viewport.utility import get_active_viewport
camera_graph_path = "/ROS2_Camera"
keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": camera_graph_path,
     "evaluator_name":"push",
     "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND},
    {
        keys.CREATE_NODES: [
            ("OnTick","omni.graph.action.OnTick"),
            ("CreateVP","isaacsim.core.nodes.IsaacCreateViewport"),
            ("GetRP","isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("SetCam","isaacsim.core.nodes.IsaacSetCameraOnRenderProduct")
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick","CreateVP.inputs:execIn"),
            ("CreateVP.outputs:execOut","GetRP.inputs:execIn"),
            ("CreateVP.outputs:viewport","GetRP.inputs:viewport"),
            ("GetRP.outputs:execOut","SetCam.inputs:execIn"),
            ("GetRP.outputs:renderProductPath","SetCam.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("CreateVP.inputs:viewportId",0),
            ("SetCam.inputs:cameraPrim",[Sdf.Path("/World/Camera")])
        ]
    }
)
viewport_api = get_active_viewport()

# Keyframes (reuse yours or edit)
KEYFRAMES = [
    {'time': 0, 'translation': [0.6, 1.5, 2], 'euler_angles': [0, 25, 65]},
    {'time': 3, 'translation': [0.6, 1.5, 2], 'euler_angles': [0, 25, 25]},
    {'time': 6, 'translation': [4.75, 1.5, 2], 'euler_angles': [0, 25, 90]},
    {'time': 12, 'translation': [9.5, 1.5, 2], 'euler_angles': [0, 25, 115]},
    {'time': 18, 'translation': [9.5, 6.5, 2], 'euler_angles': [0, 25, 225]},
    {'time': 24, 'translation': [0.6, 6.5, 2], 'euler_angles': [0, 25, 315]},
]

def interp_kf(kfs, t):
    for j in range(len(kfs)-1):
        t0,t1 = kfs[j]['time'], kfs[j+1]['time']
        if t0 <= t <= t1:
            kf0, kf1 = kfs[j], kfs[j+1]; break
    else: return None,None
    a = (t - t0)/(t1 - t0 + 1e-8)
    tr = (1-a)*np.array(kf0['translation']) + a*np.array(kf1['translation'])
    e0, e1 = np.array(kf0['euler_angles']), np.array(kf1['euler_angles'])
    e = (1-a)*e0 + a*e1
    q = euler_angles_to_quat(e, degrees=True)
    return tr, q

def transform_matrix(pos, quat):
    w,x,y,z = quat
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x),   1-2*(x*x+y*y)]
    ])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = pos
    return T

# def apply_color_map(seg_image, color_map):
#     h,w = seg_image.shape
#     out = np.zeros((h,w,3), dtype=np.uint8)
#     for sid in np.unique(seg_image):
#         if sid < len(color_map):
#             out[seg_image==sid] = color_map[sid]
#     return out

def capture_scene(signature, scene_cfg):
    print(f"[SCENE] {signature}")
    activate_scene(signature, scene_cfg)

    out_dir = os.path.join(OUTPUT_ROOT, signature, "results")
    os.makedirs(out_dir, exist_ok=True)
    traj_file = os.path.join(OUTPUT_ROOT, signature, "traj.txt")
    if not os.path.exists(os.path.dirname(traj_file)):
        os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    open(traj_file, 'w').close()

    # Warm-up
    for _ in range(WARMUP_STEPS):
        world.step(render=True); simulation_app.update()

    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_ann   = rep.AnnotatorRegistry.get_annotator("LdrColor")
    seg_ann   = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

    frame_idx = 0
    t = 0
    while frame_idx < FRAMES_PER_SCENE:
        tr, q = interp_kf(KEYFRAMES, t)
        if tr is None: break
        camera.set_local_pose(tr, q, camera_axes="world")
        for _ in range(RENDER_SUBSTEPS):
            world.step(render=True)
        simulation_app.update()

        render_product_path = viewport_api.get_render_product_path()
        depth_ann.attach([render_product_path])
        rgb_ann.attach([render_product_path])
        seg_ann.attach([render_product_path])

        depth = depth_ann.get_data()
        rgba = rgb_ann.get_data()
        seg   = seg_ann.get_data()
        seg_img = seg['data'].astype(np.uint8)
        seg_info = seg['info']['idToLabels']

        if depth.size and rgba.size:
            clipped = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
            norm = ((clipped - MIN_DEPTH)/(MAX_DEPTH - MIN_DEPTH))*PNG_MAX_VALUE
            depth_u16 = norm.astype(np.uint16)
            rgb = rgba[..., :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            max_seg = int(np.max(seg_img)) if seg_img.size else 0
            num_classes = max(max_seg+1, len(seg_info) if seg_info else 0, 1)
            cmap = create_color_map(num_classes)
            # seg_color = apply_color_map(seg_img, cmap)

            cv2.imwrite(os.path.join(out_dir, f"frame{frame_idx:06d}.jpg"), bgr)
            cv2.imwrite(os.path.join(out_dir, f"depth{frame_idx:06d}.png"), depth_u16)
            # cv2.imwrite(os.path.join(out_dir, f"semantic{frame_idx:06d}.png"), seg_color)

            # Pose matrix (ROS axes)
            pos_ros, quat_ros = camera.get_world_pose(camera_axes="ros")
            T = transform_matrix(pos_ros, quat_ros)
            with open(traj_file, "a") as f:
                f.write(' '.join(map(str, T.flatten())) + "\n")

            # Save seg info
            # info_path = os.path.join(out_dir, f"semantic{frame_idx:06d}_info.json")
            # enhanced = {}
            # for sid, label in seg_info.items():
            #     i_sid = int(sid)
            #     enhanced[sid] = {
            #         "label": label,
            #         "color_bgr": cmap[i_sid].tolist() if i_sid < len(cmap) else [0,0,0]
            #     }
            # with open(info_path, "w") as jf:
            #     json.dump(enhanced, jf, indent=2)

        frame_idx += 1
        t += 0.5  # time step along keyframes

    print(f"[DONE] {signature} frames={frame_idx}")

# --------------------------------------------------
# Run all scenes
# --------------------------------------------------
for sig, scfg in SCENES.items():
    capture_scene(sig, scfg)

simulation_app.close()