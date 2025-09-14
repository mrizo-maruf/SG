import os
import pickle
import numpy as np

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# --- Configuration ---
# --- IMPORTANT: Make sure this path is correct for your system ---
BACKGROUND_USD_PATH = "/workspace/isaaclab/SG/HuskyLab_assets/Franka-Peg-In-Hole.usd"
OUTPUT_PICKLE_FILE = "/workspace/isaaclab/IsaacSimData/Franka-Peg-In-Hole/scene_geometry_data.pkl"
NUM_POINTS_PER_OBJECT = 2048  # Number of points to sample from each mesh surface
# --- End Configuration ---

# Launch Isaac Sim in headless mode
print("Starting Isaac Sim...")
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": True})

# Make sure the USD file exists before proceeding
if not os.path.exists(BACKGROUND_USD_PATH):
    print(f"Error: The specified USD file does not exist: {BACKGROUND_USD_PATH}")
    simulation_app.close()
    exit()

# Import core Isaac Sim classes after the app has been initialized
from omni.isaac.core import World
from omni.isaac.core.utils.bounds import compute_bound
import omni.usd
from omni.physx.scripts.utils import sample_mesh_surface
from pxr import Usd, UsdGeom

try:
    # Load the desired USD stage
    print(f"Loading stage: {BACKGROUND_USD_PATH}")
    omni.usd.get_context().open_stage(BACKGROUND_USD_PATH)

    # We need to create a World object and step the simulation once to ensure
    # that the physics scene is initialized, which is required for surface sampling.
    world = World()
    world.initialize_physics()
    # A single step is sufficient to load and process the scene geometry
    simulation_app.update()
    print("Stage loaded and simulation context initialized.")

    stage = omni.usd.get_context().get_stage()
    all_objects_data = []

    print("-" * 50)
    print("Starting prim traversal and data extraction...")

    # Traverse all prims in the loaded stage
    for prim in stage.Traverse():
        # We only care about prims that are visible geometric meshes
        if prim.IsA(UsdGeom.Mesh):
            # Check if the prim is visible. This avoids extracting data from hidden objects.
            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility(Usd.TimeCode.Default()) == 'invisible':
                # print(f"Skipping invisible prim: {prim.GetPath()}")
                continue

            prim_path = str(prim.GetPath())
            print(f"Processing mesh: {prim_path}")

            try:
                # 1. Get Point Cloud by sampling the mesh surface
                # The 'sample_mesh_surface' utility directly accesses the geometry.
                # It returns points and normals in WORLD coordinates.
                sampled_points, _ = sample_mesh_surface(
                    mesh_prim=prim,
                    num_samples=NUM_POINTS_PER_OBJECT,
                )
                point_cloud = np.array(sampled_points)

                if point_cloud.shape[0] == 0:
                    print(f"  - WARNING: Could not sample any points from {prim_path}. Skipping.")
                    continue

                # 2. Get the 3D Bounding Box in world coordinates
                # compute_bound returns a tuple of (min_corner, max_corner)
                bound_low, bound_high = compute_bound(prim)
                
                # We store it as a single array: [min_x, min_y, min_z, max_x, max_y, max_z]
                bbox_3d = np.concatenate([bound_low, bound_high])

                # 3. Calculate the Bounding Box Center
                bbox_center = (np.array(bound_low) + np.array(bound_high)) / 2.0

                # Store all collected data in a dictionary
                object_data = {
                    "prim_path": prim_path,
                    "point_cloud": point_cloud,
                    "bbox_3d": bbox_3d,
                    "bbox_center": bbox_center
                }
                all_objects_data.append(object_data)
                print(f"  - Success: Extracted {point_cloud.shape[0]} points and bounding box.")

            except Exception as e:
                print(f"  - ERROR: Failed to process prim {prim_path}: {e}")

    # After iterating through all prims, save the collected data
    print("-" * 50)
    if all_objects_data:
        print(f"Saving data for {len(all_objects_data)} objects to '{OUTPUT_PICKLE_FILE}'...")
        with open(OUTPUT_PICKLE_FILE, 'wb') as f:
            pickle.dump(all_objects_data, f)
        print("Save complete!")
    else:
        print("No mesh objects were found to process.")

finally:
    # Always shut down the simulation app cleanly
    print("Shutting down Isaac Sim.")
    simulation_app.close()