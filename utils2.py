import os, json, itertools

SCENE_ITEMS_JSON = "/workspace/isaaclab/SG/scene_items.json"

with open(SCENE_ITEMS_JSON, "r") as f:
    OBJ_CFG = json.load(f)["objects"]

def _collect_counts_and_grids(obj_cfg_list):
    data = {}
    for obj in obj_cfg_list:
        name = obj["name"]
        count = obj["count"]
        grids = []
        for placement in obj.get("placement", []):
            if placement.get("strategy") == "grid":
                grids.extend(placement.get("grid_coordinates", []))
        data[name] = {
            "count": count,
            "grid": grids,              # slot index -> position
            "num_slots": len(grids)
        }
    return data

ASSET_META = _collect_counts_and_grids(OBJ_CFG)

def build_scene_signature(tables, chairs, cabinet_slot_indices, bowl_table_idx):
    def encode(prefix, idx_list):
        if len(idx_list) == 0:
            return f"{prefix}0"
        return prefix + ''.join(str(i) for i in idx_list)
    tables_s = encode("t", tables)
    chairs_s = encode("ch", chairs)
    cabinets_s = encode("cbp", cabinet_slot_indices)  # cabinet placement slots
    bowl_s = f"b{bowl_table_idx}"
    return f"sc1_{tables_s}_{chairs_s}_{cabinets_s}_{bowl_s}"

def generate_scene_combinations(
    max_tables=None,
    max_chairs=None,
    max_cabinets=None,     # cap on number of cabinet INSTANCES (<= count)
    limit_total=None
):
    T_count = ASSET_META.get("table", {}).get("count", 0)
    C_count = ASSET_META.get("chair", {}).get("count", 0)
    Cab_inst_count = ASSET_META.get("cabinet", {}).get("count", 0)
    Cab_slots = ASSET_META.get("cabinet", {}).get("num_slots", 0)

    if T_count == 0:
        raise ValueError("No tables defined; bowl placement impossible.")

    # Table subsets (must be non-empty)
    table_subsets = []
    for r in range(1, T_count + 1):
        if max_tables is not None and r > max_tables:
            break
        for comb in itertools.combinations(range(T_count), r):
            if max_tables is not None and len(comb) > max_tables:
                continue
            table_subsets.append(comb)

    # Chair subsets
    chair_subsets = []
    for r in range(0, C_count + 1):
        if max_chairs is not None and r > max_chairs:
            break
        for comb in itertools.combinations(range(C_count), r):
            chair_subsets.append(comb)

    # Cabinet slot subsets (choose slot indices; limit by available instances & optional max_cabinets)
    cab_slot_subsets = []
    cab_cap = Cab_inst_count if max_cabinets is None else min(Cab_inst_count, max_cabinets)
    for r in range(0, min(cab_cap, Cab_slots) + 1):
        for comb in itertools.combinations(range(Cab_slots), r):
            cab_slot_subsets.append(comb)

    all_scenes = {}

    for tables in table_subsets:
        for chairs in chair_subsets:
            for cab_slots in cab_slot_subsets:
                # Bowl on each active table variant
                for bowl_table in tables:
                    sig = build_scene_signature(
                        tables=list(tables),
                        chairs=list(chairs),
                        cabinet_slot_indices=list(cab_slots),
                        bowl_table_idx=bowl_table
                    )
                    # cabinets: instance indices to activate (0..r-1)
                    cabinets_instances = list(range(len(cab_slots)))
                    all_scenes[sig] = {
                        "tables": list(tables),
                        "chairs": list(chairs),
                        "cabinets": cabinets_instances,          # instance indices
                        "cabinet_slots": list(cab_slots),        # slot indices (map 1‑to‑1 to instances)
                        "bowl": {"on_table": bowl_table}
                    }
                    if limit_total is not None and len(all_scenes) >= limit_total:
                        return all_scenes
    return all_scenes

SCENES = generate_scene_combinations(
    max_tables=None,
    max_chairs=None,
    max_cabinets=None,
    limit_total=None
)

for sig in list(SCENES.keys())[:50]:  # preview first 50
    print(sig, SCENES[sig])

print(f"Generated {len(SCENES)} scenes.")
