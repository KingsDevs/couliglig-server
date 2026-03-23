import json
import math
import time
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
                "x": d.x,
                "y": d.y,
                "theta": d.theta,
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
            data["item_id"] = iid
            items.append(data)
    items.sort(key=lambda d: int(d["index"]))
    return items


def _get_all_robots_by_type_sorted(r: redis.Redis, robot_type: str) -> list:
    key = f"robots:{robot_type}s"
    robots = []
    for rid in r.smembers(key):
        data = r.hgetall(f"robot:{rid}")
        if data and data.get("type") == robot_type:
            data["robot_id"] = rid
            robots.append(data)
    robots.sort(key=lambda d: int(d["index"]))
    return robots


def _get_all_waiting_zones_sorted(r: redis.Redis) -> list:
    zones = []
    for wid in r.smembers("wz:all"):
        data = r.hgetall(f"wz:{wid}")
        if data:
            data["wz_id"] = wid
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
# All-docks state query
# =============================================================

def _get_all_docks_state(r: redis.Redis) -> list:
    """Return all active docks with their positions and runtime state."""
    docks = []
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="dock_meta:*", count=500)
        for key in keys:
            dock_id = key.split(":", 1)[1]
            meta = r.hgetall(key)
            if not meta:
                continue
            dock_type_str = meta.get("dock_type", "")
            try:
                dock_type = DockType(dock_type_str)
            except ValueError:
                continue
            runtime = r.hgetall(_dock_key(dock_type, dock_id))
            docks.append({
                "dock_id": dock_id,
                "dock_type": dock_type_str,
                "x": float(meta["x"]) if "x" in meta else None,
                "y": float(meta["y"]) if "y" in meta else None,
                "theta": float(meta["theta"]) if "theta" in meta else None,
                "status": runtime.get("status"),
                "robot_id": runtime.get("robot_id", ""),
                "item_id": runtime.get("item_id", ""),
            })
        if cursor == 0:
            break
    return docks


# =============================================================
# Agent state builders
# =============================================================

def get_picker_state(r: redis.Redis, picker_id: str) -> dict:
    """
    Return raw state for a picker agent:
      picker   – own id, x, y, held_item
      items    – id, index, x, y, weight, pickup_status, delivery_status, receiver_index
      transporters – id, index, x, y, capacity, max_capacity, in_waiting_zone
      docks    – dock_id, dock_type, x, y, theta, status, robot_id, item_id
    """
    picker_data = r.hgetall(f"robot:{picker_id}")
    if not picker_data:
        raise ValueError(f"Picker {picker_id} not found in Redis")

    items = _get_all_items_sorted(r)
    transporters = _get_all_robots_by_type_sorted(r, "transporter")
    docks = _get_all_docks_state(r)

    return {
        "picker": {
            "id": picker_id,
            "x": float(picker_data.get("x", 0)),
            "y": float(picker_data.get("y", 0)),
            "held_item": picker_data.get("held_item", ""),
        },
        "items": [
            {
                "id": item["item_id"],
                "index": int(item["index"]),
                "x": float(item["x"]),
                "y": float(item["y"]),
                "weight": float(item["weight"]),
                "pickup_status": item["pickup_status"] == "1",
                "delivery_status": item["delivery_status"] == "1",
                "receiver_index": int(item["receiver_index"]),
            }
            for item in items
        ],
        "transporters": [
            {
                "id": t["robot_id"],
                "index": int(t["index"]),
                "x": float(t.get("x", 0)),
                "y": float(t.get("y", 0)),
                "capacity": int(t.get("capacity", 0)),
                "max_capacity": int(t.get("max_capacity", 1)),
                "in_waiting_zone": t.get("in_waiting_zone", "0") == "1",
            }
            for t in transporters
        ],
        "docks": docks,
    }


def get_transporter_state(r: redis.Redis, transporter_id: str) -> dict:
    """
    Return raw state for a transporter agent:
      transporter  – own id, x, y, capacity, max_capacity, in_waiting_zone, carried_items
      pickers      – id, index, x, y, held_item
      items        – id, index, x, y, weight, statuses, receiver_x, receiver_y, carried
      waiting_zones – id, index, x, y, occupied
      docks        – dock_id, dock_type, x, y, theta, status, robot_id, item_id
    """
    t_data = r.hgetall(f"robot:{transporter_id}")
    if not t_data:
        raise ValueError(f"Transporter {transporter_id} not found in Redis")

    carried_raw = t_data.get("carried_items", "")
    carried_items = (
        {int(v) for v in carried_raw.split(",") if v.strip()}
        if carried_raw else set()
    )

    pickers = _get_all_robots_by_type_sorted(r, "picker")
    items = _get_all_items_sorted(r)
    waiting_zones = _get_all_waiting_zones_sorted(r)
    docks = _get_all_docks_state(r)

    items_out = []
    for item in items:
        receiver_index = int(item["receiver_index"])
        recv_data = r.hgetall(f"receiver:{receiver_index}")
        items_out.append({
            "id": item["item_id"],
            "index": int(item["index"]),
            "x": float(item["x"]),
            "y": float(item["y"]),
            "weight": float(item["weight"]),
            "pickup_status": item["pickup_status"] == "1",
            "delivery_status": item["delivery_status"] == "1",
            "receiver_index": receiver_index,
            "receiver_x": float(recv_data["x"]) if recv_data else None,
            "receiver_y": float(recv_data["y"]) if recv_data else None,
            "carried": int(item["index"]) in carried_items,
        })

    return {
        "transporter": {
            "id": transporter_id,
            "x": float(t_data.get("x", 0)),
            "y": float(t_data.get("y", 0)),
            "capacity": int(t_data.get("capacity", 0)),
            "max_capacity": int(t_data.get("max_capacity", 1)),
            "in_waiting_zone": t_data.get("in_waiting_zone", "0") == "1",
            "carried_items": list(carried_items),
        },
        "pickers": [
            {
                "id": p["robot_id"],
                "index": int(p["index"]),
                "x": float(p.get("x", 0)),
                "y": float(p.get("y", 0)),
                "held_item": p.get("held_item", ""),
            }
            for p in pickers
        ],
        "items": items_out,
        "waiting_zones": [
            {
                "id": wz["wz_id"],
                "index": int(wz["index"]),
                "x": float(wz["x"]),
                "y": float(wz["y"]),
                "occupied": wz["occupied"] == "1",
            }
            for wz in waiting_zones
        ],
        "docks": docks,
    }


def get_agent_state(r: redis.Redis, agent_id: str) -> dict:
    """
    Dispatch to the correct state builder based on agent prefix.
    """
    if agent_id.startswith("picker_"):
        return get_picker_state(r, agent_id)
    elif agent_id.startswith("transporter_"):
        return get_transporter_state(r, agent_id)
    else:
        raise ValueError(f"Unknown agent prefix for agent_id: {agent_id}")