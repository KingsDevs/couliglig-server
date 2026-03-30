import random
import time
from fastapi import APIRouter, HTTPException
from services.redis_dock_runtime import redis_client_from_env, get_obs_builder_inputs, RL_CONSTANTS

router = APIRouter(prefix="/rl", tags=["rl"])

redis_client = redis_client_from_env()


@router.get("/builder-inputs")
def obs_builder_inputs(robot_id: str):
    """
    Returns all live runtime data from Redis needed to build observations:
    dock states, dock positions (x, y, yaw), and any other active runtime fields.
    """
    try:
        return get_obs_builder_inputs(redis_client, robot_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/builder-inputs/dummy")
def obs_builder_inputs_dummy(
    robot_id: str,
    num_items: int = 5,
    num_wz: int = 3,
    num_receivers: int = 3,
    num_pickers: int = 2,
    num_transporters: int = 2,
):
    """
    Returns randomly generated builder inputs for testing — no Redis required.
    All counts are capped by RL_CONSTANTS maximums.
    """
    rng = random.Random()

    MAX_ITEMS       = RL_CONSTANTS["MAX_ITEMS"]
    MAX_WZ          = RL_CONSTANTS["MAX_WZ"]
    MAX_PICKERS     = RL_CONSTANTS["MAX_PICKERS"]
    MAX_TRANSPORTERS = RL_CONSTANTS["MAX_TRANSPORTERS"]

    num_items        = min(num_items,        MAX_ITEMS)
    num_wz           = min(num_wz,           MAX_WZ)
    num_pickers      = min(num_pickers,      MAX_PICKERS)
    num_transporters = min(num_transporters, MAX_TRANSPORTERS)

    statuses = ["available", "reserved", "occupied"]

    def rand_pose():
        return round(rng.uniform(-10, 10), 3), round(rng.uniform(-10, 10), 3), round(rng.uniform(-3.14, 3.14), 4)

    # ---- robots (built first so dock robot_id draws from real namespaces) ----
    # Guarantee robot_id is always in the pool by placing it as the first picker.
    other_picker_count = num_pickers - 1
    picker_ns = [robot_id] + [
        ns for ns in (f"couliglig_bot_{i}" for i in range(MAX_PICKERS * 2))
        if ns != robot_id
    ][:other_picker_count]
    transporter_start = MAX_PICKERS * 2
    transporter_ns = [
        ns for ns in (f"couliglig_bot_{transporter_start + i}" for i in range(MAX_TRANSPORTERS * 2))
        if ns != robot_id
    ][:num_transporters]
    all_robot_ns = picker_ns + transporter_ns

    # ---- docks ----
    pickup_ids   = [f"pickup_{i}" for i in range(num_items)]
    wz_ids       = [f"wz_{i}"     for i in range(num_wz)]
    receiver_ids = [f"recv_{i}"   for i in range(num_receivers)]
    item_ids     = [f"item_{i}"   for i in range(num_items)]

    dock_states = []
    dock_positions = {}

    for i, did in enumerate(pickup_ids):
        x, y, yaw = rand_pose()
        has_item = rng.random() > 0.4
        status = rng.choice(statuses)
        dock_states.append({
            "dock_type": "pickup",
            "dock_id": did,
            "x": str(x), "y": str(y), "yaw": str(yaw),
            "status": status,
            "robot_id": rng.choice(all_robot_ns) if status != "available" and all_robot_ns else "",
            "item_id": item_ids[i] if has_item else "",
            "item_weight": str(rng.randint(1, 4)) if has_item else "",
            "receiver_dock_id": rng.choice(receiver_ids) if has_item else "",
            "ts": str(int(time.time())),
        })
        dock_positions[did] = (x, y, yaw)

    for did in wz_ids:
        x, y, yaw = rand_pose()
        status = rng.choice(statuses)
        dock_states.append({
            "dock_type": "waiting_zone",
            "dock_id": did,
            "x": str(x), "y": str(y), "yaw": str(yaw),
            "status": status,
            "robot_id": rng.choice(all_robot_ns) if status != "available" and all_robot_ns else "",
            "item_id": "", "item_weight": "", "receiver_dock_id": "",
            "ts": str(int(time.time())),
        })
        dock_positions[did] = (x, y, yaw)

    for did in receiver_ids:
        x, y, yaw = rand_pose()
        status = rng.choice(statuses)
        dock_states.append({
            "dock_type": "receiver",
            "dock_id": did,
            "x": str(x), "y": str(y), "yaw": str(yaw),
            "status": status,
            "robot_id": rng.choice(all_robot_ns) if status != "available" and all_robot_ns else "",
            "item_id": "", "item_weight": "", "receiver_dock_id": "",
            "ts": str(int(time.time())),
        })
        dock_positions[did] = (x, y, yaw)

    # ---- item weights & receiver docks ----
    item_weights = {}
    item_receiver_docks = {}
    for ds in dock_states:
        if ds["item_id"] and ds["item_weight"]:
            item_weights[ds["item_id"]] = float(ds["item_weight"])
        if ds["item_id"] and ds["receiver_dock_id"]:
            item_receiver_docks[ds["item_id"]] = ds["receiver_dock_id"]

    # ---- robots ----
    robot_positions = {}
    for ns in picker_ns:
        x, y, yaw = rand_pose()
        robot_positions[ns] = ("picker", x, y, yaw)
    for ns in transporter_ns:
        x, y, yaw = rand_pose()
        robot_positions[ns] = ("transporter", x, y, yaw)

    picker_has_item     = {ns: rng.random() > 0.5 for ns in picker_ns}
    transporter_loads   = {ns: (round(rng.uniform(0, 5), 2), 10.0) for ns in transporter_ns}
    transporter_carried = {ns: rng.sample(item_ids, k=min(rng.randint(0, 3), len(item_ids))) for ns in transporter_ns}
    transporter_in_wz   = {ns: rng.random() > 0.6 for ns in transporter_ns}

    # ---- waiting zones ----
    waiting_zones = []
    for did in wz_ids:
        x, y, yaw = dock_positions[did]
        status = next(ds["status"] for ds in dock_states if ds["dock_id"] == did)
        waiting_zones.append({
            "zone_id": did,
            "x": x, "y": y, "yaw": yaw,
            "status": status,
            "robot_id": next(ds["robot_id"] for ds in dock_states if ds["dock_id"] == did),
        })

    # ---- action map ----
    item_slots: list = []
    for i, ds in enumerate(d for d in dock_states if d["dock_type"] == "pickup"):
        item_slots.append({**ds, "index": i,
                           "available_for_pickup": bool(ds["item_id"]) and ds["status"] == "available"})
    while len(item_slots) < MAX_ITEMS:
        item_slots.append(None)

    wz_slots: list = []
    for i, ds in enumerate(d for d in dock_states if d["dock_type"] == "waiting_zone"):
        wz_slots.append({**ds, "index": i, "available_for_entry": ds["status"] == "available"})
    while len(wz_slots) < MAX_WZ:
        wz_slots.append(None)

    receiver_slots = [
        {**ds, "index": i}
        for i, ds in enumerate(d for d in dock_states if d["dock_type"] == "receiver")
    ]

    picker_slots: list = [{"namespace": ns, "agent_type": "picker", "index": i} for i, ns in enumerate(picker_ns)]
    while len(picker_slots) < MAX_PICKERS:
        picker_slots.append(None)

    transporter_slots: list = [{"namespace": ns, "agent_type": "transporter", "index": i} for i, ns in enumerate(transporter_ns)]
    while len(transporter_slots) < MAX_TRANSPORTERS:
        transporter_slots.append(None)

    action_map = {
        "item_slots":         item_slots,
        "wz_slots":           wz_slots,
        "receiver_slots":     receiver_slots,
        "picker_slots":       picker_slots,
        "transporter_slots":  transporter_slots,
    }

    rl_constants = {
        **RL_CONSTANTS,
        "PICKER_ACTION_DIM":      1 + MAX_ITEMS + num_transporters,
        "TRANSPORTER_ACTION_DIM": 1 + MAX_WZ + MAX_ITEMS,
        "NUM_PICKERS":      num_pickers,
        "NUM_TRANSPORTERS": num_transporters,
    }

    return {
        "dock_states":          dock_states,
        "dock_positions":       dock_positions,
        "item_weights":         item_weights,
        "item_receiver_docks":  item_receiver_docks,
        "robot_positions":      robot_positions,
        "picker_has_item":      picker_has_item,
        "transporter_loads":    transporter_loads,
        "transporter_carried":  transporter_carried,
        "transporter_in_wz":    transporter_in_wz,
        "waiting_zones":        waiting_zones,
        "action_map":           action_map,
        "rl_constants":         rl_constants,
    }
