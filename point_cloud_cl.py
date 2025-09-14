import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from omni.isaac.kit import SimulationApp

# Configuration
BACKGROUND_USD_PATH = "/workspace/isaaclab/SG/HuskyLab_assets/Franka-Peg-In-Hole.usd"
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "hide_ui": False}

# Initialize Isaac Sim
simulation_app = SimulationApp(CONFIG)

# Import Isaac Sim modules after app initialization
import omni.usd
from pxr import Usd, UsdGeom, Gf, Vt
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.world import World


class PointCloudExtractor:
    def __init__(self):
        self.world = World()
        
    def sample_points_from_mesh(self, mesh_prim, num_points: int = 1024) -> np.ndarray:
        """
        Sample points from mesh surface using uniform sampling
        """
        # Get mesh geometry
        mesh = UsdGeom.Mesh(mesh_prim)
        
        # Get vertices and faces
        points_attr = mesh.GetPointsAttr()
        face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()
        face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()
        
        if not points_attr or not face_vertex_indices_attr:
            return np.array([])
            
        vertices = np.array(points_attr.Get())
        face_indices = np.array(face_vertex_indices_attr.Get())
        face_counts = np.array(face_vertex_counts_attr.Get()) if face_vertex_counts_attr else None
        
        if len(vertices) == 0 or len(face_indices) == 0:
            return np.array([])
        
        # Convert faces to triangles
        triangles = self._convert_to_triangles(face_indices, face_counts)
        
        if len(triangles) == 0:
            return np.array([])
        
        # Sample points from triangular faces
        sampled_points = self._sample_points_from_triangles(vertices, triangles, num_points)
        
        return sampled_points
    
    def _convert_to_triangles(self, face_indices: np.ndarray, face_counts: np.ndarray = None) -> np.ndarray:
        """Convert polygon faces to triangles"""
        triangles = []
        
        if face_counts is None:
            # Assume all faces are triangles
            face_counts = np.full(len(face_indices) // 3, 3)
        
        idx = 0
        for count in face_counts:
            if count == 3:
                # Already a triangle
                triangles.append(face_indices[idx:idx+3])
            elif count > 3:
                # Triangulate polygon (simple fan triangulation)
                for i in range(1, count - 1):
                    triangles.append([
                        face_indices[idx],
                        face_indices[idx + i],
                        face_indices[idx + i + 1]
                    ])
            idx += count
        
        return np.array(triangles) if triangles else np.array([]).reshape(0, 3)
    
    def _sample_points_from_triangles(self, vertices: np.ndarray, triangles: np.ndarray, num_points: int) -> np.ndarray:
        """Sample points uniformly from triangle surfaces"""
        if len(triangles) == 0:
            return np.array([])
        
        # Calculate triangle areas for weighted sampling
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        # Cross product for area calculation
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Normalize areas to get probabilities
        if np.sum(areas) == 0:
            return np.array([])
        
        probabilities = areas / np.sum(areas)
        
        # Sample triangles based on area
        sampled_triangle_indices = np.random.choice(
            len(triangles), size=num_points, p=probabilities
        )
        
        # Sample points within selected triangles using barycentric coordinates
        sampled_points = []
        for tri_idx in sampled_triangle_indices:
            # Generate random barycentric coordinates
            r1, r2 = np.random.random(2)
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            r3 = 1 - r1 - r2
            
            # Get triangle vertices
            tri = triangles[tri_idx]
            point = r1 * vertices[tri[0]] + r2 * vertices[tri[1]] + r3 * vertices[tri[2]]
            sampled_points.append(point)
        
        return np.array(sampled_points)
    
    def get_bounding_box(self, prim) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 3D bounding box, its center, and size
        Returns: (bbox_min, bbox_max, center)
        """
        # Get world transform
        world_transform = omni.usd.get_world_transform_matrix(prim)
        
        # Get local bounding box
        bbox_cache = UsdGeom.BBoxCache()
        bbox = bbox_cache.ComputeWorldBound(prim)
        
        if bbox.IsEmpty():
            return np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
        
        bbox_range = bbox.ComputeAlignedRange()
        bbox_min = np.array([bbox_range.GetMin()[0], bbox_range.GetMin()[1], bbox_range.GetMin()[2]])
        bbox_max = np.array([bbox_range.GetMax()[0], bbox_range.GetMax()[1], bbox_range.GetMax()[2]])
        center = (bbox_min + bbox_max) / 2.0
        
        return bbox_min, bbox_max, center
    
    def extract_all_pointclouds(self, usd_path: str, num_points_per_object: int = 1024) -> List[Dict[str, Any]]:
        """
        Extract point clouds from all mesh objects in the USD file
        """
        # Load USD file
        omni.usd.get_context().open_stage(usd_path)
        stage = omni.usd.get_context().get_stage()
        
        if not stage:
            raise RuntimeError(f"Failed to load USD file: {usd_path}")
        
        # Initialize world
        self.world.reset()
        
        pointcloud_data = []
        
        # Traverse all prims in the scene
        for prim in stage.TraverseAll():
            # Check if prim is a mesh
            if prim.IsA(UsdGeom.Mesh):
                prim_path = str(prim.GetPath())
                prim_name = prim.GetName()
                
                print(f"Processing mesh: {prim_path}")
                
                # Extract point cloud from mesh
                try:
                    points = self.sample_points_from_mesh(prim, num_points_per_object)
                    
                    if len(points) > 0:
                        # Get bounding box
                        bbox_min, bbox_max, bbox_center = self.get_bounding_box(prim)
                        
                        # Store data
                        obj_data = {
                            'prim_path': prim_path,
                            'prim_name': prim_name,
                            'point_cloud': points,
                            'bbox_min': bbox_min,
                            'bbox_max': bbox_max,
                            'bbox_center': bbox_center,
                            'bbox_size': bbox_max - bbox_min,
                            'num_points': len(points)
                        }
                        
                        pointcloud_data.append(obj_data)
                        print(f"  - Extracted {len(points)} points")
                        print(f"  - Bbox center: {bbox_center}")
                        print(f"  - Bbox size: {bbox_max - bbox_min}")
                    else:
                        print(f"  - No points extracted (empty mesh)")
                        
                except Exception as e:
                    print(f"  - Error processing {prim_path}: {str(e)}")
                    continue
        
        return pointcloud_data
    
    def save_pointclouds(self, pointcloud_data: List[Dict[str, Any]], output_path: str):
        """Save point cloud data to pickle file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(pointcloud_data, f)
        
        print(f"Saved point cloud data for {len(pointcloud_data)} objects to: {output_path}")


def main():
    """Main function to extract point clouds"""
    try:
        # Initialize extractor
        extractor = PointCloudExtractor()
        
        # Extract point clouds from all objects
        print(f"Loading USD file: {BACKGROUND_USD_PATH}")
        pointcloud_data = extractor.extract_all_pointclouds(
            BACKGROUND_USD_PATH, 
            num_points_per_object=2048  # Adjust as needed
        )
        
        # Save results
        output_path = "/workspace/isaaclab/IsaacSimData/Franka-Peg-In-Hole/extracted_pointclouds.pkl"
        extractor.save_pointclouds(pointcloud_data, output_path)
        
        # Print summary
        print(f"\n=== SUMMARY ===")
        print(f"Total objects processed: {len(pointcloud_data)}")
        for i, obj in enumerate(pointcloud_data):
            print(f"{i+1}. {obj['prim_name']} ({obj['prim_path']})")
            print(f"   Points: {obj['num_points']}")
            print(f"   Center: [{obj['bbox_center'][0]:.2f}, {obj['bbox_center'][1]:.2f}, {obj['bbox_center'][2]:.2f}]")
            print(f"   Size: [{obj['bbox_size'][0]:.2f}, {obj['bbox_size'][1]:.2f}, {obj['bbox_size'][2]:.2f}]")
        
        # Example of loading the saved data
        print(f"\n=== EXAMPLE: Loading saved data ===")
        with open(output_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"Loaded data for {len(loaded_data)} objects")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean shutdown
        simulation_app.close()


if __name__ == "__main__":
    main()