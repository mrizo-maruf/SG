# filepath: /workspace/isaaclab/multi_scene_capture_graveyard.py
import os, json, math, itertools, cv2, numpy as np
from isaacsim import SimulationApp
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}
simulation_app = SimulationApp(CONFIG)

import omni
import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.prims import XFormPrim
from pxr import UsdGeom, Sdf, Gf
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import euler_angles_to_quat

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OBJECT_CONFIG_PATH = "/workspace/isaaclab/SG/scene_items_extended.json"   # <- your extended JSON (tables/chairs/cabinets/bowl)
SCENES_JSON = "/workspace/isaaclab/SG/eval_scenes_64.json"               # list of scenes with object coordinates
ASSET_BASE = "/workspace/isaaclab/SG/aloha_assets_new"                   # base folder for usd_paths
OUTPUT_ROOT = "/workspace/isaaclab/IsaacSimData"
MAIN_SCENE_PATH = "/World/Kitchen"
MAIN_SCENE_USD = "/workspace/isaaclab/SG/aloha_assets_new/scenes/scenes_sber_kitchen_for_BBQ/kitchen_new_simple.usd"

GRAVEYARD_ORIGIN = (50, 50, -10.0)   # far away parking
GRAVEYARD_SPACING = 2.0
SHIFT_X, SHIFT_Y = 5.0, 4.0                # shift negative room coords into positive capture zone
BOWL_MARGIN_XY_JITTER = 0.0                # set >0 for random jitter on bowl
WARMUP_STEPS = 20
RENDER_SUBSTEPS = 10
MAX_SCENES = None  # set int to limit

FOCAL_LENGTH = 50
HORIZONTAL_APARTURE = 80
VERTICAL_APARTURE = 45
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0
PNG_MAX_VALUE = 65535

# Camera path and keyframes
CAMERA_PATH = "/World/Camera"
KEYFRAMES = [
    {'time': 0, 'translation': [0.6, 1.5, 2], 'euler_angles': [0, 25, 65]},
    {'time': 3, 'translation': [0.6, 1.5, 2], 'euler_angles': [0, 25, 25]},
    {'time': 6, 'translation': [4.75, 1.5, 2], 'euler_angles': [0, 25, 90]},
    {'time': 9, 'translation': [9.5, 1.5, 2], 'euler_angles': [0, 25, 115]},
    {'time': 12, 'translation': [9.5, 6.5, 2], 'euler_angles': [0, 25, 225]},
    {'time': 15, 'translation': [0.6, 6.5, 2], 'euler_angles': [0, 25, 290]},
    {'time': 18, 'translation': [0.6, 6.5, 2], 'euler_angles': [0, 25, 360]},
]

IMAGE_W, IMAGE_H = 1280, 720
MIN_DEPTH, MAX_DEPTH = 0.01, 10.0
PNG_MAX = 65535

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

# ---------------------------------------------------------------------------
# LOAD CONFIG
# ---------------------------------------------------------------------------
with open(OBJECT_CONFIG_PATH, "r") as f:
    OBJ_CFG = json.load(f)["objects"]

def load_scene_specs(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Expect each element like: {"table":[[x,y,z],...], "chair":[...], ...}
    scenes = {}
    for i, row in enumerate(data):
        # Build deterministic signature from object names + xy coords
        parts = []
        for obj_name, lst in sorted(row.items()):
            # list of coords
            for c in lst:
                parts.append(f"{obj_name}_{round(c[0],2)}_{round(c[1],2)}")
        sig = "sc1_" + "_".join(parts)
        scenes[sig] = row
    return scenes

SCENE_SPECS = load_scene_specs(SCENES_JSON)
if MAX_SCENES:
    # truncate deterministically
    SCENE_SPECS = dict(list(SCENE_SPECS.items())[:MAX_SCENES])

# ---------------------------------------------------------------------------
# WORLD + CAMERA
# ---------------------------------------------------------------------------
world = World(physics_dt=1/30.0, rendering_dt=1/30.0, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path=MAIN_SCENE_USD, prim_path=MAIN_SCENE_PATH)


stage = get_current_stage()

camera = Camera(
    prim_path=CAMERA_PATH,
    position=np.array([3, 0, 2.0]),
    resolution=(IMAGE_H, IMAGE_W),
    orientation=rot_utils.euler_angles_to_quats(np.array([0,0,0]), degrees=True, extrinsic=True)
)
world.reset()
camera.initialize()
camera.add_distance_to_camera_to_frame()

stage_ref = get_current_stage()
camera_prim = stage_ref.GetPrimAtPath("/World/Camera")
horizontal_aperture = camera_prim.GetAttribute("horizontalAperture")
horizontal_aperture.Set(HORIZONTAL_APARTURE)

fx, fy, cx, cy = compute_intrinsics(camera_prim, IMAGE_WIDTH, IMAGE_HEIGHT)
png_depth_scale = (MAX_DEPTH - MIN_DEPTH) / PNG_MAX_VALUE
print(f"Camera intrinsics -> fx: {fx:.3f}, fy: {fy:.3f}, cx: {cx:.3f}, cy: {cy:.3f}")


# Build ROS viewport graph
import usdrt.Sdf
keys = og.Controller.Keys
og.Controller.edit(
    {
        "graph_path": "/ROS2_Camera",
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND
    },
    {
        keys.CREATE_NODES: [
            ("OnTick","omni.graph.action.OnTick"),
            ("CreateVP","isaacsim.core.nodes.IsaacCreateViewport"),
            ("GetRP","isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("SetCam","isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
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
            ("SetCam.inputs:cameraPrim",[usdrt.Sdf.Path(CAMERA_PATH)]),
        ],
    }
)

from omni.kit.viewport.utility import get_active_viewport
viewport_api = get_active_viewport()

# ---------------------------------------------------------------------------
# OBJECT REGISTRY + GRAVEYARD
# ---------------------------------------------------------------------------
class InstanceRecord:
    __slots__ = ("prim_path","xform","home_slot","obj_name","idx")
    def __init__(self, prim_path, xform, home_slot, obj_name, idx):
        self.prim_path = prim_path
        self.xform = xform
        self.home_slot = home_slot  # grid coordinate or None
        self.obj_name = obj_name
        self.idx = idx

registry = {}  # name -> list[InstanceRecord]
sizes = {}     # name -> (sx,sy,sz)
surface_provider_names = set()
surface_only_names = set()

def ensure_translate_op(prim):
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xf.AddTranslateOp()

def set_world_position(prim_path, pos):
    prim = stage.GetPrimAtPath(prim_path)
    op = ensure_translate_op(prim)
    op.Set(Gf.Vec3d(*pos))

def spawn_all():
    gx0, gy0, gz0 = GRAVEYARD_ORIGIN
    grave_idx = 0

    stage = get_current_stage()

    for obj in OBJ_CFG:
        name = obj["name"]
        registry[name] = []
        sizes[name] = tuple(obj["size"])
        if "surface_provider" in obj["type"]:
            surface_provider_names.add(name)
        if "surface_only" in obj["type"]:
            surface_only_names.add(name)

        usd_rel = obj["usd_paths"][0]
        usd_full = os.path.join(ASSET_BASE, usd_rel)

        # slots (grid coords)
        grid_coords = []
        for placement in obj.get("placement", []):
            if obj["placement"]["strategy"] == "grid":
                grid_coords.extend(obj["placement"].get("grid_coordinates", []))
        count = obj["count"]
        if grid_coords:
            if len(grid_coords) < count:
                reps = (count + len(grid_coords) - 1) // len(grid_coords)
                grid_coords = (grid_coords * reps)[:count]
            else:
                grid_coords = grid_coords[:count]
        else:
            grid_coords = [(0,0,0)] * count

        # Spawn instances (direct references, no proto indirection)
        for i in range(count):
            prim_path = f"/World/Objects/{name}_{i}"
            add_reference_to_stage(usd_path=usd_full, prim_path=prim_path)

            # park in graveyard
            gx = gx0 + (grave_idx % 8) * GRAVEYARD_SPACING
            gy = gy0 + (grave_idx // 8) * GRAVEYARD_SPACING
            grave_idx += 1
            set_world_position(prim_path, (gx, gy, gz0))

            xform = XFormPrim(prim_path=prim_path)
            registry[name].append(InstanceRecord(prim_path, xform, grid_coords[i], name, i))

    print(f"[INFO] Spawned objects (referenced): " + ", ".join(f"{k}:{len(v)}" for k,v in registry.items()))

spawn_all()

def park_all():
    gx0, gy0, gz0 = GRAVEYARD_ORIGIN
    for name, inst_list in registry.items():
        for k, inst in enumerate(inst_list):
            gx = gx0 + (k % 8) * GRAVEYARD_SPACING
            gy = gy0 + (k // 8) * GRAVEYARD_SPACING
            inst.xform.set_world_pose(position=np.array([gx, gy, gz0]), orientation=np.array([1,0,0,0]))

def shift_position(p):
    # shift XY for negative zone -> positive capture area
    return [p[0] + SHIFT_X, p[1] + SHIFT_Y, p[2]]

def place_bowl(bowl_inst, chosen_table_pos, table_size, bowl_size):
    pos = shift_position(chosen_table_pos)
    z = pos[-1] + 0.05
    jitter_x = (np.random.rand() - 0.5) * 2 * BOWL_MARGIN_XY_JITTER
    jitter_y = (np.random.rand() - 0.5) * 2 * BOWL_MARGIN_XY_JITTER
    bowl_pos = [pos[0] + jitter_x, pos[1] + jitter_y, z]
    bowl_inst.xform.set_world_pose(position=np.array(bowl_pos), orientation=np.array([1,1,0,0]))

# ---------------------------------------------------------------------------
# CAMERA HELPERS
# ---------------------------------------------------------------------------
def interp_keyframes(kfs, t):
    for j in range(len(kfs)-1):
        t0, t1 = kfs[j]['time'], kfs[j+1]['time']
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0 + 1e-8)
            p0 = np.array(kfs[j]['translation'])
            p1 = np.array(kfs[j+1]['translation'])
            e0 = np.array(kfs[j]['euler_angles'])
            e1 = np.array(kfs[j+1]['euler_angles'])
            pos = (1-a)*p0 + a*p1
            euler = (1-a)*e0 + a*e1
            quat = euler_angles_to_quat(euler, degrees=True)
            return pos, quat
    return None, None

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

# ---------------------------------------------------------------------------
# SCENE ACTIVATION
# scene_spec format (from SCENE_SPECS): { "table":[[x,y,z],...], "chair":[...], "cabinet":[...], ... }
# We assume bowl not listed (we add automatically on first active table).
# ---------------------------------------------------------------------------
def activate_scene(signature, scene_spec):
    park_all()
    # Activate standard objects
    active_tables_world = []
    bowl_pos = []
    # 1) Place tables / chairs / cabinets (anything except bowl)
    for obj_name, coords_list in scene_spec.items():
        if obj_name not in registry:
            continue
        inst_list = registry[obj_name]
        for i, coord in enumerate(coords_list):
            if i >= len(inst_list):
                break
            world_pos = shift_position(coord)
            inst_list[i].xform.set_world_pose(position=np.array(world_pos), orientation=np.array([1,0,0,0]))
            if obj_name == "table":
                active_tables_world.append(world_pos)
        # Remaining instances stay parked

    # 2) Bowl placement (exactly one bowl if present)
    if "bowl" in registry and active_tables_world:
        for i in range(len(registry["bowl"])):
            
            bowl_inst = registry["bowl"][i]
            table_size = sizes["table"]
            bowl_size = sizes["bowl"]
            print(f"Placing bowl on table at {scene_spec['bowl'][i]}")
            place_bowl(bowl_inst, scene_spec['bowl'][i], table_size, bowl_size)

# ---------------------------------------------------------------------------
# ANNOTATORS (initialized once)
# ---------------------------------------------------------------------------
render_product_path = viewport_api.get_render_product_path()
depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
rgb_ann = rep.AnnotatorRegistry.get_annotator("LdrColor")
depth_ann.attach([render_product_path])
rgb_ann.attach([render_product_path])

# ---------------------------------------------------------------------------
# MAIN LOOP OVER SCENES
# ---------------------------------------------------------------------------
print(f"[INFO] Total scenes to process: {len(SCENE_SPECS)}")

for scene_idx, (sig, spec) in enumerate(SCENE_SPECS.items()):
    print(f"\n=== Processing scene {scene_idx+1}/{len(SCENE_SPECS)}: {sig} ===")
    activate_scene(sig, spec)

    # Output dirs
    out_dir = os.path.join(OUTPUT_ROOT, sig, "results")
    traj_dir = os.path.join(OUTPUT_ROOT, sig)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    traj_file = os.path.join(traj_dir, "traj.txt")
    open(traj_file, "w").close()

    # Warmup
    for _ in range(WARMUP_STEPS):
        world.step(render=True)
        simulation_app.update()

    t = 0.0
    frame = 0
    max_time = KEYFRAMES[-1]['time']
    while t <= max_time:
        pos, quat = interp_keyframes(KEYFRAMES, t)
        if pos is None:
            break
        camera.set_local_pose(pos, quat, camera_axes="world")

        # extra substeps
        for _ in range(RENDER_SUBSTEPS):
            world.step(render=True)
        simulation_app.update()

        depth = depth_ann.get_data()
        rgba = rgb_ann.get_data()

        if depth.size and rgba.size:
            # depth -> 16U
            clipped = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
            norm = ((clipped - MIN_DEPTH)/(MAX_DEPTH - MIN_DEPTH))*PNG_MAX
            depth_u16 = norm.astype(np.uint16)
            cv2.imwrite(os.path.join(out_dir, f"depth{frame:06d}.png"), depth_u16)

            rgb = rgba[..., :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"frame{frame:06d}.jpg"), bgr)

            # pose (ROS axes)
            pos_ros, quat_ros = camera.get_world_pose(camera_axes="ros")
            T = transform_matrix(pos_ros, quat_ros)
            with open(traj_file, "a") as f:
                f.write(" ".join(map(str, T.flatten())) + "\n")

        frame += 1
        t += 1.0  # 1 second step between keyframe samples (adjust if needed)

    print(f"[DONE] {sig} -> {frame} frames")

print("\nAll scenes processed.")
simulation_app.close()