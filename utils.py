import os, json, itertools

# SCENE_ITEMS_JSON = "/workspace/isaaclab/SG/scene_items.json"  # replace with your extended JSON version


# # Load config once
# with open(SCENE_ITEMS_JSON, "r") as f:
#     OBJ_CFG = json.load(f)["objects"]

# def _collect_counts_and_grids(obj_cfg_list):
#     data = {}
#     for obj in obj_cfg_list:
#         name = obj["name"]
#         count = obj["count"]
#         grid = []
#         for placement in obj.get("placement", []):
#             if placement.get("strategy") == "grid":
#                 grid.extend(placement.get("grid_coordinates", []))
#         data[name] = {
#             "count": count,
#             "grid": grid  # index -> position
#         }
#     return data

# ASSET_META = _collect_counts_and_grids(OBJ_CFG)

# def build_scene_signature(tables, chairs, cabinets, bowl_table_idx):
#     def encode(prefix, idx_list):
#         if len(idx_list) == 0:
#             return f"{prefix}0"
#         return prefix + ''.join(str(i) for i in idx_list)
#     tables_s   = encode("t", tables)
#     chairs_s   = encode("ch", chairs)
#     cabinets_s = encode("cb", cabinets)
#     bowl_s     = f"b{bowl_table_idx}"
#     return f"sc1_{tables_s}_{chairs_s}_{cabinets_s}_{bowl_s}"

# def generate_scene_combinations(
#     max_tables=None,
#     max_chairs=None,
#     max_cabinets=None,
#     limit_total=None
# ):
#     """
#     Generate all valid scenes:
#       - At least one table active
#       - Any subset (including empty) of chairs and cabinets
#       - Bowl placed on EACH active table -> expands scenes
#     Optional caps (max_*) restrict maximum number of active instances chosen (after subset picked).
#     """
#     T = ASSET_META.get("table", {}).get("count", 0)
#     C = ASSET_META.get("chair", {}).get("count", 0)
#     K = ASSET_META.get("cabinet", {}).get("count", 0)

#     if T == 0:
#         raise ValueError("No tables defined; cannot guarantee bowl placement.")

#     all_scenes = {}

#     table_index_sets = []
#     for r in range(1, T + 1):  # at least one table
#         for comb in itertools.combinations(range(T), r):
#             if max_tables is not None and len(comb) > max_tables:
#                 continue
#             table_index_sets.append(comb)

#     chair_index_sets = []
#     for r in range(0, C + 1):
#         if max_chairs is not None and r > max_chairs:
#             continue
#         for comb in itertools.combinations(range(C), r):
#             chair_index_sets.append(comb)

#     cabinet_index_sets = []
#     for r in range(0, K + 1):
#         if max_cabinets is not None and r > max_cabinets:
#             continue
#         for comb in itertools.combinations(range(K), r):
#             cabinet_index_sets.append(comb)

#     for tables in table_index_sets:
#         for chairs in chair_index_sets:
#             for cabinets in cabinet_index_sets:
#                 # For each active table, create a variant with bowl on that table
#                 for bowl_table in tables:
#                     sig = build_scene_signature(
#                         tables=list(tables),
#                         chairs=list(chairs),
#                         cabinets=list(cabinets),
#                         bowl_table_idx=bowl_table
#                     )
#                     all_scenes[sig] = {
#                         "tables": list(tables),
#                         "chairs": list(chairs),
#                         "cabinets": list(cabinets),
#                         "bowl": {"on_table": bowl_table}
#                     }
#                     if limit_total is not None and len(all_scenes) >= limit_total:
#                         return all_scenes
#     return all_scenes

# SCENES = generate_scene_combinations(
#     max_tables=None,      # or e.g. 2 to reduce
#     max_chairs=None,      # or e.g. 2
#     max_cabinets=None,    # or e.g. 2
#     limit_total=10      # or an int to truncate
# )


# for sig, cfg in SCENES.items():
#     # capture_scene(sig, cfg)
#     print(sig, cfg)
    

# print(f"Generated {len(SCENES)} scenes.")



def get_scene_obj_coords(json_path="/workspace/isaaclab/SG/eval_scenes_64.json"):
    # Step 1: read JSON
    with open(json_path, "r") as f:
        data = json.load(f)   # list of dicts

    # Step 2: build dict {row_string: raw_row}
    result_dict = {}

    for row in data:
        parts = []
        for obj, coords_list in row.items():
            for coords in coords_list:
                coords = coords[:2]  # only x,y
                coords_str = "_".join(str(c) for c in coords)
                parts.append(f"{obj}_{coords_str}")
        row_key = "_".join(parts)
        result_dict[row_key] = row

    return result_dict


