import json
import math
import time
import numpy as np
import redis
import requests as _requests
from schema import Dock
from sqlalchemy.orm import Session
from definitions import DockType, DockStatus


def redis_client_from_env() -> redis.Redis:
    import os
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))

    return redis.Redis(host=host, port=port, db=db, decode_responses=True)

def _get_dock_type_by_id(r: redis.Redis, dock_id: str) -> DockType:
    dock_type = r.hgetall(f"dock_meta:{dock_id}").get("dock_type")
    if dock_type is None:
        raise ValueError(f"Invalid dock_id: {dock_id}")
    return DockType(dock_type)

def _check_if_dock_id_exists(r: redis.Redis, dock_id: str) -> bool:
    if not r.sismember("docks:all", dock_id):
        return False
    return True

def _dock_key(dock_type: DockType, dock_id: str) -> str:

    return f"dock:{dock_type.value}:{dock_id}"


def _dock_lock_key(dock_type: DockType, dock_id: str) -> str:
    return f"lock:dock:{dock_type.value}:{dock_id}"

def get_active_dock_config_id(r: redis.Redis) -> str:
    return r.get("active_dock_config")

# ---------------------------------------------------------
# Clear runtime
# ---------------------------------------------------------

def clear_all_dock_keys(r: redis.Redis) -> None:
    r.set("active_dock_config", "none")

    for pattern in ("dock:*", "dock_meta:*", "lock:dock:*"):
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break
    try:
        r.delete("docks:all")
        r.delete("docks:pickup")
        r.delete("docks:receiving")
    except Exception as e:
        print(f"Error clearing dock sets: {e}")

# ---------------------------------------------------------
# Activate dock configuration
# ---------------------------------------------------------

def activate_docks(r: redis.Redis, session: Session, dock_config_id: int) -> None:
    """
    Initialize Redis runtime from SQLite dock config.
    """

    clear_all_dock_keys(r)

    pipe = r.pipeline()
    now = int(time.time())

    # dock_config = session.query(DockConfig).filter_by(id=dock_config_id).first()
    docks = session.query(Dock).filter_by(config_id=dock_config_id).all()

    r.set("active_dock_config", dock_config_id)

    for d in docks:

        key = _dock_key(d.dock_type, d.dock_id)

        pipe.hset(
            key,
            mapping={
                "status": DockStatus.available.value,
                "robot_id": "",
                "item_id": "",
                "ts": str(now),
            },
        )

        pipe.hset(
            f"dock_meta:{d.dock_id}",
            mapping={
                "dock_type": d.dock_type.value,
            }
        )

        pipe.sadd("docks:all", d.dock_id)

    pipe.execute()


# ---------------------------------------------------------
# Add item to pickup dock
# ---------------------------------------------------------

def add_item_to_pickup_dock(
    r: redis.Redis,
    dock_id: str,
    item_id: str
) -> bool:
    
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")
    
    dock_type = _get_dock_type_by_id(r, dock_id)
    if dock_type != DockType.PICKUP:
        raise ValueError("Can only add items to pickup docks")

    key = _dock_key(dock_type, dock_id)

    data = r.hgetall(key)

    if not data:
        return False

    if data.get("item_id"):
        return False

    r.hset(
        key,
        mapping={
            "item_id": item_id,
            "ts": int(time.time()),
        },
    )

    return True


# ---------------------------------------------------------
# Remove item from pickup dock
# ---------------------------------------------------------

def remove_item_from_pickup_dock(
    r: redis.Redis,
    dock_id: str
) -> bool:
    
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")
    
    dock_type = _get_dock_type_by_id(r, dock_id)
    if dock_type != DockType.PICKUP:
        raise ValueError("Can only remove items from pickup docks")

    key = _dock_key(DockType.PICKUP, dock_id)

    data = r.hgetall(key)

    if not data:
        return False

    r.hset(
        key,
        mapping={
            "item_id": "",
            "ts": int(time.time()),
        },
    )

    return True


# ---------------------------------------------------------
# Reserve dock for robot (atomic lock)
# ---------------------------------------------------------

def reserve_dock(
    r: redis.Redis,
    dock_id: str,
    robot_id: str
) -> bool:
    
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")
    
    dock_type = _get_dock_type_by_id(r, dock_id)
    lock_key = _dock_lock_key(dock_type, dock_id)

    # atomic lock
    acquired = r.set(lock_key, robot_id, nx=True)

    if not acquired:
        return False

    dock_key = _dock_key(dock_type, dock_id)

    r.hset(
        dock_key,
        mapping={
            "status": DockStatus.reserved.value,
            "robot_id": robot_id,
            "ts": int(time.time()),
        },
    )

    return True


# ---------------------------------------------------------
# Mark dock occupied (robot arrived)
# ---------------------------------------------------------

def occupy_dock(
    r: redis.Redis,
    dock_id: str,
    robot_id: str,
):
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    dock_key = _dock_key(dock_type, dock_id)

    r.hset(
        dock_key,
        mapping={
            "status": DockStatus.occupied.value,
            "robot_id": robot_id,
            "ts": int(time.time()),
        },
    )


# ---------------------------------------------------------
# Release dock
# ---------------------------------------------------------

def release_dock(
    r: redis.Redis,
    dock_id: str,
):
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)

    lock_key = _dock_lock_key(dock_type, dock_id)
    dock_key = _dock_key(dock_type, dock_id)

    r.delete(lock_key)

    r.hset(
        dock_key,
        mapping={
            "status": DockStatus.available.value,
            "robot_id": "",
            "ts": int(time.time()),
        },
    )


# ---------------------------------------------------------
# Query dock state
# ---------------------------------------------------------

def get_dock_state(
    r: redis.Redis,
    dock_id: str
):
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    key = _dock_key(dock_type, dock_id)

    return r.hgetall(key)

# create function that will dock states for all activate docks
def get_all_dock_states(
    r: redis.Redis,
):

    active_config = r.get("active_dock_config")

    if not active_config or active_config == "none":
        return []

    cursor = 0
    dock_states = []
    while True:
        cursor, keys = r.scan(cursor=cursor, match=f"dock:*", count=500)

        for key in keys:
            data = r.hgetall(key)
            if data:
                dock_states.append({
                    "dock_type": key.split(":")[1],
                    "dock_id": key.split(":")[2],
                    "status": data.get("status"),
                    "robot_id": data.get("robot_id"),
                    "item_id": data.get("item_id"),
                    "ts": data.get("ts"),
                })

        if cursor == 0:
            break

    return dock_states


# =============================================================
# Runtime environment config
# =============================================================

def set_runtime_config(
    r: redis.Redis,
    map_w: int,
    map_h: int,
    max_distance: float,
    current_max_cap: float,
    max_items: int,
    max_pickers: int,
    max_transporters: int,
    max_wz: int,
) -> None:
    """Store environment-level constants used by observation builders."""
    r.hset("runtime:config", mapping={
        "map_w": map_w,
        "map_h": map_h,
        "max_distance": max_distance,
        "current_max_cap": current_max_cap,
        "max_items": max_items,
        "max_pickers": max_pickers,
        "max_transporters": max_transporters,
        "max_wz": max_wz,
    })


def _get_runtime_config(r: redis.Redis) -> dict:
    data = r.hgetall("runtime:config")
    return {
        "map_w": int(data["map_w"]),
        "map_h": int(data["map_h"]),
        "max_distance": float(data["max_distance"]),
        "current_max_cap": float(data["current_max_cap"]),
        "max_items": int(data["max_items"]),
        "max_pickers": int(data["max_pickers"]),
        "max_transporters": int(data["max_transporters"]),
        "max_wz": int(data["max_wz"]),
    }


# =============================================================
# Agent state management
# =============================================================

def update_picker_state(
    r: redis.Redis,
    robot_id: str,
    agent_index: int,
    x: float | None = None,
    y: float | None = None,
    held_item: str = "",
) -> None:
    """Upsert picker position and held-item state."""
    mapping: dict = {
        "type": "picker",
        "index": agent_index,
        "held_item": held_item,
    }
    if x is not None:
        mapping["x"] = x
    if y is not None:
        mapping["y"] = y
    r.hset(f"robot:{robot_id}", mapping=mapping)
    r.sadd("robots:pickers", robot_id)


def update_transporter_state(
    r: redis.Redis,
    robot_id: str,
    agent_index: int,
    x: float | None = None,
    y: float | None = None,
    capacity: int = 0,
    max_capacity: int = 1,
    in_waiting_zone: bool = False,
    carried_items: list = [],
) -> None:
    """Upsert transporter position and cargo state."""
    mapping: dict = {
        "type": "transporter",
        "index": agent_index,
        "capacity": capacity,
        "max_capacity": max_capacity,
        "in_waiting_zone": "1" if in_waiting_zone else "0",
        "carried_items": ",".join(str(i) for i in carried_items),
    }
    if x is not None:
        mapping["x"] = x
    if y is not None:
        mapping["y"] = y
    r.hset(f"robot:{robot_id}", mapping=mapping)
    r.sadd("robots:transporters", robot_id)


# =============================================================
# Item state management
# =============================================================

def update_item_state(
    r: redis.Redis,
    item_id: str,
    item_index: int,
    x: float,
    y: float,
    weight: float,
    pickup_status: bool,
    delivery_status: bool,
    receiver_index: int,
) -> None:
    """Upsert item position and delivery state."""
    r.hset(f"item:{item_id}", mapping={
        "index": item_index,
        "x": x,
        "y": y,
        "weight": weight,
        "pickup_status": "1" if pickup_status else "0",
        "delivery_status": "1" if delivery_status else "0",
        "receiver_index": receiver_index,
    })
    r.sadd("items:all", item_id)


# =============================================================
# Receiver location management
# =============================================================

def set_receiver_location(
    r: redis.Redis,
    receiver_index: int,
    x: float,
    y: float,
) -> None:
    """Store a receiver (delivery destination) position by its integer index."""
    r.hset(f"receiver:{receiver_index}", mapping={"x": x, "y": y})


# =============================================================
# Waiting zone state management
# =============================================================

def update_waiting_zone_state(
    r: redis.Redis,
    wz_id: str,
    wz_index: int,
    x: float,
    y: float,
    occupied: bool,
) -> None:
    """Upsert waiting-zone position and occupancy."""
    r.hset(f"wz:{wz_id}", mapping={
        "index": wz_index,
        "x": x,
        "y": y,
        "occupied": "1" if occupied else "0",
    })
    r.sadd("wz:all", wz_id)


# =============================================================
# Internal helpers
# =============================================================

def _euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _get_all_items_sorted(r: redis.Redis) -> list:
    items = []
    for iid in r.smembers("items:all"):
        data = r.hgetall(f"item:{iid}")
        if data:
            items.append(data)
    items.sort(key=lambda d: int(d["index"]))
    return items


def _get_all_robots_by_type_sorted(r: redis.Redis, robot_type: str) -> list:
    key = f"robots:{robot_type}s"
    robots = []
    for rid in r.smembers(key):
        data = r.hgetall(f"robot:{rid}")
        if data and data.get("type") == robot_type:
            robots.append(data)
    robots.sort(key=lambda d: int(d["index"]))
    return robots


def _get_all_waiting_zones_sorted(r: redis.Redis) -> list:
    zones = []
    for wid in r.smembers("wz:all"):
        data = r.hgetall(f"wz:{wid}")
        if data:
            zones.append(data)
    zones.sort(key=lambda d: int(d["index"]))
    return zones


# =============================================================
# Agent → robot hostname mapping
# =============================================================

def set_agent_robot_mapping(r: redis.Redis, agent_id: str, hostname: str) -> None:
    """Map a logical agent_id (e.g. 'picker_0') to a registered robot hostname."""
    r.hset("agent:robot_map", agent_id, hostname)


def get_all_agent_robot_mappings(r: redis.Redis) -> dict:
    """Return the full agent_id → hostname mapping dict."""
    return r.hgetall("agent:robot_map")


def _get_robot_ip(r: redis.Redis, hostname: str) -> str | None:
    """Look up the IP stored by the registration endpoint for a given hostname."""
    raw = r.get("robot_ips")
    if not raw:
        return None
    robot_ips = json.loads(raw)
    entry = robot_ips.get(hostname)
    if not entry:
        return None
    return entry if isinstance(entry, str) else entry.get("ip")


def fetch_robot_transform(ip: str, port: int = 8000) -> dict:
    """
    Call a robot's /transform endpoint and return
    {"x": float, "y": float, "yaw": float}.
    """
    resp = _requests.get(f"http://{ip}:{port}/transform", timeout=3)
    resp.raise_for_status()
    return resp.json()


def sync_all_robot_positions(r: redis.Redis, port: int = 8000) -> dict:
    """
    Fetch live transforms for every agent that has a hostname mapping and
    write x, y back into their Redis robot key.

    Returns a per-agent result dict:
        {"picker_0": {"status": "ok", "x": 1.2, "y": 3.4}, ...}
    """
    mappings = get_all_agent_robot_mappings(r)
    results: dict = {}

    for agent_id, hostname in mappings.items():
        ip = _get_robot_ip(r, hostname)
        if not ip:
            results[agent_id] = {"status": "error", "detail": f"No IP found for hostname '{hostname}'"}
            continue

        try:
            transform = fetch_robot_transform(ip, port)
            r.hset(f"robot:{agent_id}", mapping={"x": transform["x"], "y": transform["y"]})
            results[agent_id] = {"status": "ok", "x": transform["x"], "y": transform["y"], "yaw": transform.get("yaw")}
        except Exception as exc:
            results[agent_id] = {"status": "error", "detail": str(exc)}

    return results


# =============================================================
# Observation builders  –  match training-env get_observation()
# =============================================================

def get_picker_observation(r: redis.Redis, picker_id: str) -> dict:
    """
    Build the picker observation dict that matches the training environment exactly.

    Keys returned:
        obs_vec               float32 (1,)
        item_embeddings       float32 (MAX_ITEMS, 6)
        item_mask             int8    (MAX_ITEMS,)
        transporter_embeddings float32 (max_transporters, 4)
        transporter_mask      int8    (max_transporters,)
        action_mask           int8    (1 + MAX_ITEMS + max_transporters,)
    """
    cfg = _get_runtime_config(r)
    map_w = cfg["map_w"]
    map_h = cfg["map_h"]
    max_distance = cfg["max_distance"]
    current_max_cap = cfg["current_max_cap"]
    max_items = cfg["max_items"]
    max_transporters = cfg["max_transporters"]

    picker_data = r.hgetall(f"robot:{picker_id}")
    if not picker_data:
        raise ValueError(f"Picker {picker_id} not found in Redis")

    px = float(picker_data["x"])
    py = float(picker_data["y"])
    holds_item = 1.0 if picker_data.get("held_item", "") else 0.0

    picker_action_dim = 1 + max_items + max_transporters
    action_mask = np.zeros(picker_action_dim, dtype=np.int8)
    action_mask[0] = 1  # idle always allowed

    # ------------------------------------------------------------------
    # Item embeddings  [norm_dist, norm_weight, pickup_status,
    #                   delivery_status, dx_i, dy_i]
    # ------------------------------------------------------------------
    items = _get_all_items_sorted(r)
    item_emb_list = []
    item_mask = np.zeros(max_items, dtype=np.int8)

    for item_index in range(max_items):
        if item_index >= len(items):
            item_emb_list.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        item = items[item_index]
        ix = float(item["x"])
        iy = float(item["y"])
        pickup_status = item["pickup_status"] == "1"
        delivery_status = item["delivery_status"] == "1"
        weight = float(item["weight"])

        # Match training env: use dist=1.0 placeholder when already picked up
        dist = _euclidean_dist(px, py, ix, iy) if not pickup_status else 1.0
        norm_dist = min(dist / (max_distance + 1e-6), 1.0)
        norm_weight = weight / current_max_cap

        dx_i = (ix - px) / map_w
        dy_i = (iy - py) / map_h

        can_pick = (not pickup_status) and (not holds_item)
        item_mask[item_index] = 1 if can_pick else 0
        if can_pick:
            action_mask[1 + item_index] = 1

        item_emb_list.append([
            norm_dist, norm_weight,
            float(pickup_status), float(delivery_status),
            dx_i, dy_i,
        ])

    item_embeddings = np.array(item_emb_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # Transporter embeddings  [capacity_ratio, norm_dist, dx, dy]
    # ------------------------------------------------------------------
    transporters = _get_all_robots_by_type_sorted(r, "transporter")
    transporter_emb_list = []
    transporter_mask = np.zeros(max_transporters, dtype=np.int8)

    for i in range(max_transporters):
        if i >= len(transporters):
            transporter_emb_list.append([0.0, 0.0, 0.0, 0.0])
            continue

        t = transporters[i]
        tx = float(t["x"])
        ty = float(t["y"])
        capacity = int(t.get("capacity", 0))
        max_cap = int(t.get("max_capacity", 1))
        in_wz = t.get("in_waiting_zone", "0") == "1"

        dx = tx - px
        dy = ty - py
        dist = _euclidean_dist(px, py, tx, ty)
        norm_dist = min(dist / (max_distance + 1e-6), 1.0)

        if holds_item and capacity < max_cap and in_wz:
            transporter_mask[i] = 1
            action_mask[1 + max_items + i] = 1

        transporter_emb_list.append([
            min(capacity / max_cap, 1.0),
            norm_dist,
            dx / map_w,
            dy / map_h,
        ])

    transporter_embeddings = np.array(transporter_emb_list, dtype=np.float32)

    obs_vec = np.array([holds_item], dtype=np.float32)

    return {
        "obs_vec": obs_vec,
        "item_embeddings": item_embeddings,
        "item_mask": item_mask,
        "transporter_embeddings": transporter_embeddings,
        "transporter_mask": transporter_mask,
        "action_mask": action_mask,
    }


def get_transporter_observation(r: redis.Redis, transporter_id: str) -> dict:
    """
    Build the transporter observation dict that matches the training environment exactly.

    Keys returned:
        obs_vec                   float32 (1 + MAX_ITEMS,)
        picker_embeddings         float32 (max_pickers, 4)
        picker_mask               int8    (max_pickers,)
        item_embeddings           float32 (MAX_ITEMS, 6)
        item_mask                 int8    (MAX_ITEMS,)
        waiting_zone_embeddings   float32 (MAX_WZ, 3)
        waiting_zone_mask         int8    (MAX_WZ,)
        action_mask               int8    (1 + MAX_WZ + MAX_ITEMS,)
    """
    cfg = _get_runtime_config(r)
    map_w = cfg["map_w"]
    map_h = cfg["map_h"]
    max_distance = cfg["max_distance"]
    current_max_cap = cfg["current_max_cap"]
    max_items = cfg["max_items"]
    max_pickers = cfg["max_pickers"]
    max_wz = cfg["max_wz"]

    t_data = r.hgetall(f"robot:{transporter_id}")
    if not t_data:
        raise ValueError(f"Transporter {transporter_id} not found in Redis")

    tx = float(t_data["x"])
    ty = float(t_data["y"])
    capacity = int(t_data.get("capacity", 0))
    max_cap = int(t_data.get("max_capacity", 1))
    load_ratio = min(capacity / max_cap, 1.0)

    carried_raw = t_data.get("carried_items", "")
    carried_items = (
        {int(v) for v in carried_raw.split(",") if v.strip()}
        if carried_raw else set()
    )

    transporter_action_dim = 1 + max_wz + max_items
    action_mask = np.zeros(transporter_action_dim, dtype=np.int8)
    action_mask[0] = 1  # idle always allowed

    # ------------------------------------------------------------------
    # Picker embeddings  [norm_dist, holds, dx, dy]
    # ------------------------------------------------------------------
    pickers = _get_all_robots_by_type_sorted(r, "picker")
    picker_emb_list = []
    picker_mask = np.zeros(max_pickers, dtype=np.int8)

    for i in range(max_pickers):
        if i >= len(pickers):
            picker_emb_list.append([0.0, 0.0, 0.0, 0.0])
            continue

        p = pickers[i]
        px = float(p["x"])
        py = float(p["y"])
        holds = 1.0 if p.get("held_item", "") else 0.0

        dx = px - tx
        dy = py - ty
        dist = _euclidean_dist(tx, ty, px, py)
        norm_dist = min(dist / (max_distance + 1e-6), 1.0)

        picker_emb_list.append([norm_dist, holds, dx / map_w, dy / map_h])
        picker_mask[i] = 1  # all active pickers are visible

    picker_embeddings = np.array(picker_emb_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # Waiting-zone embeddings  [norm_dist, dx, dy]
    # ------------------------------------------------------------------
    waiting_zones = _get_all_waiting_zones_sorted(r)
    wz_emb_list = []
    wz_mask = np.zeros(max_wz, dtype=np.int8)

    for i in range(max_wz):
        if i >= len(waiting_zones):
            wz_emb_list.append([1.0, 0.0, 0.0])
            continue

        wz = waiting_zones[i]
        wx = float(wz["x"])
        wy = float(wz["y"])
        occupied = wz["occupied"] == "1"

        dx = wx - tx
        dy = wy - ty
        dist = _euclidean_dist(tx, ty, wx, wy)
        norm_dist = min(dist / (max_distance + 1e-6), 1.0)

        wz_emb_list.append([norm_dist, dx / map_w, dy / map_h])

        if not occupied:
            wz_mask[i] = 1
            action_mask[1 + i] = 1

    while len(wz_emb_list) < max_wz:
        wz_emb_list.append([1.0, 0.0, 0.0])

    wz_embeddings = np.array(wz_emb_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # Item embeddings  [norm_dist, norm_weight, pickup_status,
    #                   delivery_status, dx_r, dy_r]
    # (distances are to the receiver, not the item position)
    # ------------------------------------------------------------------
    items = _get_all_items_sorted(r)
    item_emb_list = []
    item_mask = np.zeros(max_items, dtype=np.int8)
    delivery_mask_scalar = []

    for item_index in range(max_items):
        if item_index >= len(items):
            item_emb_list.append([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            delivery_mask_scalar.append(0.0)
            continue

        item = items[item_index]
        pickup_status = item["pickup_status"] == "1"
        delivery_status = item["delivery_status"] == "1"
        weight = float(item["weight"])
        receiver_index = int(item["receiver_index"])

        recv_data = r.hgetall(f"receiver:{receiver_index}")
        if recv_data:
            rx = float(recv_data["x"])
            ry = float(recv_data["y"])
        else:
            rx, ry = tx, ty  # fallback: treat as co-located

        dist = _euclidean_dist(tx, ty, rx, ry)
        norm_dist = min(dist / (max_distance + 1e-6), 1.0)
        norm_weight = weight / current_max_cap

        dx_r = (rx - tx) / map_w
        dy_r = (ry - ty) / map_h

        can_deliver = (not delivery_status) and (item_index in carried_items)
        delivery_mask_scalar.append(1.0 if can_deliver else 0.0)

        item_emb_list.append([
            norm_dist, norm_weight,
            float(pickup_status), float(delivery_status),
            dx_r, dy_r,
        ])

        item_mask[item_index] = 1 if can_deliver else 0
        if can_deliver:
            action_mask[1 + max_wz + item_index] = 1

    while len(item_emb_list) < max_items:
        item_emb_list.append([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        delivery_mask_scalar.append(0.0)

    item_embeddings = np.array(item_emb_list, dtype=np.float32)

    obs_vec = np.concatenate([
        np.array([load_ratio], dtype=np.float32),
        np.array(delivery_mask_scalar, dtype=np.float32),
    ]).astype(np.float32)

    return {
        "obs_vec": obs_vec,
        "item_embeddings": item_embeddings,
        "item_mask": item_mask,
        "waiting_zone_embeddings": wz_embeddings,
        "waiting_zone_mask": wz_mask,
        "picker_embeddings": picker_embeddings,
        "picker_mask": picker_mask,
        "action_mask": action_mask,
    }


def get_observation(r: redis.Redis, agent_id: str) -> dict:
    """
    Dispatch to the correct observation builder based on agent prefix,
    mirroring the training environment's get_observation().
    """
    if agent_id.startswith("picker_"):
        return get_picker_observation(r, agent_id)
    elif agent_id.startswith("transporter_"):
        return get_transporter_observation(r, agent_id)
    else:
        raise ValueError(f"Unknown agent prefix for agent_id: {agent_id}")