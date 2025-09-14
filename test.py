import json

OBJECT_CONFIG_PATH = "/workspace/isaaclab/SG/scene_items_extended.json"

with open(OBJECT_CONFIG_PATH, "r") as f:
    OBJ_CFG = json.load(f)["objects"]
    
    
grid_coords = []
for obj in OBJ_CFG:
    # slots (grid coords) if grid strategy
    print(obj["placement"]["strategy"], obj["placement"]["strategy"] == "grid")
    if obj["placement"]["strategy"] == "grid":
        grid_coords.extend(obj["placement"].get("grid_coordinates", []))
        
print(grid_coords)