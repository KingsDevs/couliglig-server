import json
import random
import time
import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from services.database import get_db_session
from schema import DockConfig, Dock
from definitions import DockType
from services.redis_dock_runtime import redis_client_from_env, get_obs_builder_inputs, RL_CONSTANTS

router = APIRouter(prefix="/rl", tags=["rl"])

redis_client = redis_client_from_env()


def _get_robot_ip(robot_id: str) -> str:
    """Look up a robot's IP from Redis by hostname or namespace."""
    raw = redis_client.get("robot_ips")
    if not raw:
        raise HTTPException(status_code=503, detail="No robots registered in Redis")
    try:
        robot_dict = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=503, detail="Malformed robot_ips in Redis")

    # Match by hostname (dict key) first, then by namespace field
    entry = robot_dict.get(robot_id)
    if entry is None:
        entry = next(
            (v for v in robot_dict.values() if isinstance(v, dict) and v.get("namespace") == robot_id),
            None,
        )
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Robot '{robot_id}' not found in Redis")

    ip = entry.get("ip") if isinstance(entry, dict) else entry
    if not ip:
        raise HTTPException(status_code=404, detail=f"No IP stored for robot '{robot_id}'")
    return ip


@router.patch("/robot-state/{robot_id}")
def update_robot_state(robot_id: str, body: dict):
    """
    Proxy a PATCH /rl request to the given robot.
    Looks up the robot's IP from Redis; no need to pass the IP manually.
    The body should contain the fields to update (UpdatePickerAgent or UpdateTransporterAgent).
    """
    ip = _get_robot_ip(robot_id)
    try:
        resp = requests.patch(f"http://{ip}:8000/rl", json=body, timeout=5)
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail=f"Could not connect to robot '{robot_id}' at {ip}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail=f"Timeout connecting to robot '{robot_id}' at {ip}")

    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


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
    dock_config_id: int,
    num_pickers: int = 2,
    num_transporters: int = 2,
    # --- scenario control flags ---
    all_items_available: bool = False,   # all pickup docks: has item + status=available, no robot
    no_docked_robots: bool = False,      # all docks: status=available, robot_id=""
    all_wz_available: bool = False,      # all waiting zones: status=available, robot_id=""
    all_receivers_available: bool = False,  # all receivers: status=available, robot_id=""
    pickers_have_item: bool = False,     # all pickers: has_item=True
    pickers_no_item: bool = False,       # all pickers: has_item=False
    transporters_empty: bool = False,    # all transporters: capacity=0, carried_items=[]
    transporters_in_wz: bool = False,    # all transporters: in_waiting_zone=True
    transporters_not_in_wz: bool = False,  # all transporters: in_waiting_zone=False
    db: Session = Depends(get_db_session),
):
    """
    Returns randomly generated builder inputs for testing — no live Redis required.
    Docks are loaded from the given dock_config_id in the database; positions come
    from the DB records (x, y, theta).  Robot state is randomised.

    Scenario flags (stackable):
    - all_items_available: every pickup dock has an item and is available to pick up
    - no_docked_robots: no dock has a robot assigned (all docks available)
    - all_wz_available: all waiting zones are free
    - all_receivers_available: all receiver docks are free
    - pickers_have_item / pickers_no_item: override picker item state
    - transporters_empty: transporters carry nothing
    - transporters_in_wz / transporters_not_in_wz: override transporter WZ state
    """
    # ---- load docks from DB ----
    config = db.query(DockConfig).filter(DockConfig.id == dock_config_id).first()
    if not config:
        raise HTTPException(status_code=404, detail=f"dock_config_id {dock_config_id} not found")

    db_docks = db.query(Dock).filter(Dock.config_id == dock_config_id).all()
    if not db_docks:
        raise HTTPException(status_code=404, detail=f"No docks found for dock_config_id {dock_config_id}")

    rng = random.Random()

    MAX_ITEMS        = RL_CONSTANTS["MAX_ITEMS"]
    MAX_WZ           = RL_CONSTANTS["MAX_WZ"]
    MAX_PICKERS      = RL_CONSTANTS["MAX_PICKERS"]
    MAX_TRANSPORTERS = RL_CONSTANTS["MAX_TRANSPORTERS"]

    num_pickers      = min(num_pickers,      MAX_PICKERS)
    num_transporters = min(num_transporters, MAX_TRANSPORTERS)

    statuses = ["available", "reserved", "occupied"]

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

    # ---- classify docks from DB ----
    pickup_docks   = [d for d in db_docks if d.dock_type == DockType.PICKUP]
    wz_docks       = [d for d in db_docks if d.dock_type == DockType.WAITING_ZONE]
    receiver_docks = [d for d in db_docks if d.dock_type == DockType.RECEIVER]
    handoff_docks  = [d for d in db_docks if d.dock_type == DockType.HANDOFF]

    receiver_ids = [d.dock_id for d in receiver_docks]
    item_ids     = [f"item_{i}" for i in range(len(pickup_docks))]

    dock_states    = []
    dock_positions = {}

    for i, d in enumerate(pickup_docks):
        x   = d.x     if d.x     is not None else 0.0
        y   = d.y     if d.y     is not None else 0.0
        yaw = d.theta if d.theta is not None else 0.0
        has_item = True if all_items_available else rng.random() > 0.4
        if all_items_available or no_docked_robots:
            status = "available"
            assigned_robot = ""
        else:
            status = rng.choice(statuses)
            assigned_robot = rng.choice(all_robot_ns) if status != "available" and all_robot_ns else ""
        dock_states.append({
            "dock_type": "pickup",
            "dock_id": d.dock_id,
            "x": str(x), "y": str(y), "yaw": str(yaw),
            "status": status,
            "robot_id": assigned_robot,
            "item_id": item_ids[i] if has_item else "",
            "item_weight": str(rng.randint(1, 4)) if has_item else "",
            "receiver_dock_id": rng.choice(receiver_ids) if has_item and receiver_ids else "",
            "ts": str(int(time.time())),
        })
        dock_positions[d.dock_id] = (x, y, yaw)

    for d in wz_docks:
        x   = d.x     if d.x     is not None else 0.0
        y   = d.y     if d.y     is not None else 0.0
        yaw = d.theta if d.theta is not None else 0.0
        if all_wz_available or no_docked_robots:
            status = "available"
            assigned_robot = ""
        else:
            status = rng.choice(statuses)
            assigned_robot = rng.choice(all_robot_ns) if status != "available" and all_robot_ns else ""
        dock_states.append({
            "dock_type": "waiting_zone",
            "dock_id": d.dock_id,
            "x": str(x), "y": str(y), "yaw": str(yaw),
            "status": status,
            "robot_id": assigned_robot,
            "item_id": "", "item_weight": "", "receiver_dock_id": "",
            "ts": str(int(time.time())),
        })
        dock_positions[d.dock_id] = (x, y, yaw)

    for d in receiver_docks:
        x   = d.x     if d.x     is not None else 0.0
        y   = d.y     if d.y     is not None else 0.0
        yaw = d.theta if d.theta is not None else 0.0
        if all_receivers_available or no_docked_robots:
            status = "available"
            assigned_robot = ""
        else:
            status = rng.choice(statuses)
            assigned_robot = rng.choice(all_robot_ns) if status != "available" and all_robot_ns else ""
        dock_states.append({
            "dock_type": "receiver",
            "dock_id": d.dock_id,
            "x": str(x), "y": str(y), "yaw": str(yaw),
            "status": status,
            "robot_id": assigned_robot,
            "item_id": "", "item_weight": "", "receiver_dock_id": "",
            "ts": str(int(time.time())),
        })
        dock_positions[d.dock_id] = (x, y, yaw)

    for d in handoff_docks:
        x   = d.x     if d.x     is not None else 0.0
        y   = d.y     if d.y     is not None else 0.0
        yaw = d.theta if d.theta is not None else 0.0
        if no_docked_robots:
            status = "available"
            assigned_robot = ""
        else:
            status = rng.choice(statuses)
            assigned_robot = rng.choice(all_robot_ns) if status != "available" and all_robot_ns else ""
        dock_states.append({
            "dock_type": "hand_off",
            "dock_id": d.dock_id,
            "x": str(x), "y": str(y), "yaw": str(yaw),
            "status": status,
            "robot_id": assigned_robot,
            "item_id": "", "item_weight": "", "receiver_dock_id": "",
            "ts": str(int(time.time())),
        })
        dock_positions[d.dock_id] = (x, y, yaw)

    # ---- item weights & receiver docks ----
    item_weights = {}
    item_receiver_docks = {}
    for ds in dock_states:
        if ds["item_id"] and ds["item_weight"]:
            item_weights[ds["item_id"]] = float(ds["item_weight"])
        if ds["item_id"] and ds["receiver_dock_id"]:
            item_receiver_docks[ds["item_id"]] = ds["receiver_dock_id"]

    # ---- robots ----
    def rand_pose():
        return round(rng.uniform(-10, 10), 3), round(rng.uniform(-10, 10), 3), round(rng.uniform(-3.14, 3.14), 4)

    robot_positions = {}
    for ns in picker_ns:
        x, y, yaw = rand_pose()
        robot_positions[ns] = ("picker", x, y, yaw)
    for ns in transporter_ns:
        x, y, yaw = rand_pose()
        robot_positions[ns] = ("transporter", x, y, yaw)

    if pickers_have_item:
        picker_has_item = {ns: True for ns in picker_ns}
    elif pickers_no_item:
        picker_has_item = {ns: False for ns in picker_ns}
    else:
        picker_has_item = {ns: rng.random() > 0.5 for ns in picker_ns}

    if transporters_empty:
        transporter_loads   = {ns: (0.0, 10.0) for ns in transporter_ns}
        transporter_carried = {ns: [] for ns in transporter_ns}
    else:
        transporter_loads   = {ns: (round(rng.uniform(0, 5), 2), 10.0) for ns in transporter_ns}
        transporter_carried = {ns: rng.sample(item_ids, k=min(rng.randint(0, 3), len(item_ids))) for ns in transporter_ns}

    if transporters_in_wz:
        transporter_in_wz = {ns: True for ns in transporter_ns}
    elif transporters_not_in_wz:
        transporter_in_wz = {ns: False for ns in transporter_ns}
    else:
        transporter_in_wz = {ns: rng.random() > 0.6 for ns in transporter_ns}

    # ---- transporter occupancy (active hand_off docks) ----
    active_statuses = {"reserved", "occupied"}
    transporter_occupany = [
        ds for ds in dock_states
        if ds["dock_type"] == "hand_off" and ds["status"] in active_statuses
    ]

    # ---- waiting zones ----
    waiting_zones = []
    for did in [d.dock_id for d in wz_docks]:
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
        "transporter_occupany": transporter_occupany,
        "action_map":           action_map,
        "rl_constants":         rl_constants,
    }
