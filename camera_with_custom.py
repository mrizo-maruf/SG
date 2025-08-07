import omni.usd
from pxr import UsdGeom
import omni
import numpy as np
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import stage, extensions, nucleus
import omni.graph.core as og
import omni.replicator.core as rep
# from omni.syntheticdata._syntheticdata import SensorType
import omni.syntheticdata._syntheticdata as sd

from isaacsim.core.utils.prims import set_targets
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.nodes.scripts.utils import set_target_prims
# Create a new stage if needed
stage = omni.usd.get_context().get_stage()

# Create a cube prim
cube_prim_path = "/World/Cube"
UsdGeom.Cube.Define(stage, cube_prim_path)

# Move and scale the cube
cube_prim = stage.GetPrimAtPath(cube_prim_path)
xform = UsdGeom.Xformable(cube_prim)
xform.AddTranslateOp().Set((0.0, 0.0, 0.5))   # 0.5 height to sit on the ground
xform.AddScaleOp().Set((0.5, 0.5, 0.5))       # smaller cube

cube = rep.get.prims(path=cube_prim_path)

with cube:
    rep.modify.semantics([("class", "cube")])

camera = Camera(
    prim_path="/World/floating_camera",
    position=np.array([2.0, 0.0, 1.0]),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, -45, -90]), degrees=True),
    frequency=20,
    resolution=(256, 256)
)


bbox_annotator = rep.annotators.get("bounding_box_3d")
bbox_annotator.attach(render_product)

async def capture_bbox_data():
    await rep.orchestrator.step_async()
    data = bbox_annotator.get_data()
    print("3D Bounding Box Data:", data)
    return data

asyncio.ensure_future(capture_bbox_data())
