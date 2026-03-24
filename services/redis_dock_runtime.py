import json
import time
import redis
import requests
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
    return bool(r.sismember("docks:all", dock_id))


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


def clear_all_runtime_keys(r: redis.Redis) -> None:
    """Clears dock keys AND all robot/wz runtime keys."""
    clear_all_dock_keys(r)
    for pattern in ("robot:pos:*", "robot:state:*", "wz:state:*", "dock_pos:*", "item:weight:*"):
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                r.delete(*keys)
            if cursor == 0:
                break


# ---------------------------------------------------------
# Activate dock configuration
# ---------------------------------------------------------

def activate_docks(r: redis.Redis, session: Session, dock_config_id: int) -> None:
    """Initialize Redis runtime from SQLite dock config."""
    clear_all_dock_keys(r)

    pipe = r.pipeline()
    now = int(time.time())

    docks = session.query(Dock).filter_by(config_id=dock_config_id).all()
    r.set("active_dock_config", dock_config_id)

    for d in docks:
        key = _dock_key(d.dock_type, d.dock_id)
        pipe.hset(
            key,
            mapping={
                "status": DockStatus.available.value,
                "x": d.x if d.x is not None else "",
                "y": d.y if d.y is not None else "",
                "yaw": d.theta if d.theta is not None else "",
                "robot_id": "",
                "item_id": "",
                "ts": str(now),
            },
        )
        pipe.hset(
            f"dock_meta:{d.dock_id}",
            mapping={"dock_type": d.dock_type.value},
        )
        pipe.sadd("docks:all", d.dock_id)

        # store dock (y, x) position if available on the model
        if hasattr(d, "y") and hasattr(d, "x"):
            pipe.hset(
                f"dock_pos:{d.dock_id}",
                mapping={"y": d.y, "x": d.x},
            )

    pipe.execute()


# ---------------------------------------------------------
# Dock positions
# ---------------------------------------------------------

def set_dock_position(r: redis.Redis, dock_id: str, y: int, x: int) -> None:
    r.hset(f"dock_pos:{dock_id}", mapping={"y": y, "x": x})


def get_dock_position(r: redis.Redis, dock_id: str) -> tuple[int, int] | None:
    data = r.hgetall(f"dock_pos:{dock_id}")
    if not data:
        return None
    return (int(data["y"]), int(data["x"]))


def get_all_dock_positions(r: redis.Redis) -> dict[str, tuple[float, float, float]]:
    """Returns { dock_id: (x, y, yaw) } for all docks in docks:all."""
    positions: dict[str, tuple[float, float, float]] = {}
    dock_ids = r.smembers("docks:all")
    for dock_id in dock_ids:
        dock_type_str = r.hget(f"dock_meta:{dock_id}", "dock_type")
        if dock_type_str is None:
            continue
        data = r.hgetall(f"dock:{dock_type_str}:{dock_id}")
        if not data:
            continue
        x = data.get("x")
        y = data.get("y")
        yaw = data.get("yaw")
        if x not in (None, "") and y not in (None, "") and yaw not in (None, ""):
            positions[dock_id] = (float(x), float(y), float(yaw))
    return positions


# ---------------------------------------------------------
# Item weights
# ---------------------------------------------------------


def get_all_item_weights(r: redis.Redis) -> dict[str, float]:
    """Returns { item_id: weight } for all docks in docks:all that have an item with a weight."""
    weights: dict[str, float] = {}
    dock_ids = r.smembers("docks:all")
    for dock_id in dock_ids:
        dock_type_str = r.hget(f"dock_meta:{dock_id}", "dock_type")
        if dock_type_str is None:
            continue
        data = r.hgetall(f"dock:{dock_type_str}:{dock_id}")
        if not data:
            continue
        item_id = data.get("item_id")
        item_weight = data.get("item_weight")
        if item_id and item_weight not in (None, ""):
            weights[item_id] = float(item_weight)
    return weights


# ---------------------------------------------------------
# Robot positions
# ---------------------------------------------------------

def set_robot_position(r: redis.Redis, agent_id: str, y: int, x: int) -> None:
    r.hset(f"robot:pos:{agent_id}", mapping={"x": x, "y": y})


def get_robot_position(r: redis.Redis, agent_id: str) -> tuple[int, int] | None:
    data = r.hgetall(f"robot:pos:{agent_id}")
    if not data:
        return None
    return (int(data["y"]), int(data["x"]))


def get_all_robot_positions(r: redis.Redis) -> dict[str, tuple[str, float, float, float]]:
    """Returns { namespace: (robot_type, x, y, yaw) } for all registered robots by querying each robot's /roslib/transform endpoint."""
    positions: dict[str, tuple[str, float, float, float]] = {}
    raw = r.get("robot_ips")
    if not raw:
        return positions
    try:
        robot_dict = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return positions
    for _, payload in robot_dict.items():
        print(f"Checking robot payload: {payload}")
        ip = payload.get("ip") if isinstance(payload, dict) else payload
        domain_id = payload.get("ros_domain_id") if isinstance(payload, dict) else None

        if not ip or not domain_id:
            continue
        try:
            resp = requests.get(f"http://{ip}:8000/roslib/transform", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                namespace = f"couliglig_bot_{domain_id}"
                positions[namespace] = (str(data["rl_robot_type"]), float(data["x"]), float(data["y"]), float(data["yaw"]))
        except Exception as e:
            print(f"Error fetching position for robot at {ip}: {e}")
            continue
    return positions


# ---------------------------------------------------------
# Robot state  (picker / transporter runtime)
# ---------------------------------------------------------

def set_picker_state(r: redis.Redis, agent_id: str, has_item: bool) -> None:
    r.hset(
        f"robot:state:{agent_id}",
        mapping={
            "agent_type": "picker",
            "has_item": int(has_item),
        },
    )


def set_transporter_state(
    r: redis.Redis,
    agent_id: str,
    capacity: float,
    max_capacity: float,
    carried_items: list[str],
    in_waiting_zone: bool,
) -> None:
    r.hset(
        f"robot:state:{agent_id}",
        mapping={
            "agent_type": "transporter",
            "capacity": capacity,
            "max_capacity": max_capacity,
            "carried_items": json.dumps(carried_items),
            "in_waiting_zone": int(in_waiting_zone),
        },
    )


def get_picker_state(r: redis.Redis, agent_id: str) -> dict | None:
    data = r.hgetall(f"robot:state:{agent_id}")
    if not data:
        return None
    return {
        "agent_id": agent_id,
        "agent_type": "picker",
        "has_item": bool(int(data.get("has_item", 0))),
    }


def get_transporter_state(r: redis.Redis, agent_id: str) -> dict | None:
    data = r.hgetall(f"robot:state:{agent_id}")
    if not data:
        return None
    return {
        "agent_id": agent_id,
        "agent_type": "transporter",
        "capacity": float(data.get("capacity", 0.0)),
        "max_capacity": float(data.get("max_capacity", 1.0)),
        "carried_items": json.loads(data.get("carried_items", "[]")),
        "in_waiting_zone": bool(int(data.get("in_waiting_zone", 0))),
    }


# ---------------------------------------------------------
# Waiting zone state
# ---------------------------------------------------------


def get_waiting_zone_state(r: redis.Redis, zone_id: str) -> dict | None:
    data = r.hgetall(f"wz:state:{zone_id}")
    if not data:
        return None
    return {
        "zone_id": zone_id,
        "x": int(data["x"]),
        "y": int(data["y"]),
        "occupied": bool(int(data.get("occupied", 0))),
    }


def get_all_waiting_zone_states(r: redis.Redis) -> list[dict]:
    """Returns all docks in docks:all whose dock_type is 'waiting_zone'."""
    zones: list[dict] = []
    dock_ids = r.smembers("docks:all")
    for dock_id in dock_ids:
        dock_type_str = r.hget(f"dock_meta:{dock_id}", "dock_type")
        if dock_type_str != "waiting_zone":
            continue
        data = r.hgetall(f"dock:{dock_type_str}:{dock_id}")
        if not data:
            continue
        zones.append({
            "zone_id":  dock_id,
            "x":        float(data["x"]) if data.get("x") not in (None, "") else None,
            "y":        float(data["y"]) if data.get("y") not in (None, "") else None,
            "yaw":      float(data["yaw"]) if data.get("yaw") not in (None, "") else None,
            "status":   data.get("status"),
            "robot_id": data.get("robot_id"),
        })
    return zones


# ---------------------------------------------------------
# Add item to pickup dock
# ---------------------------------------------------------

def add_item_to_pickup_dock(
    r: redis.Redis,
    dock_id: str,
    item_id: str,
    item_weight: float = 1.0
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

    r.hset(key, mapping={"item_id": item_id, "item_weight": item_weight, "ts": int(time.time())})

    return True


# ---------------------------------------------------------
# Remove item from pickup dock
# ---------------------------------------------------------

def remove_item_from_pickup_dock(r: redis.Redis, dock_id: str) -> bool:
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    if dock_type != DockType.PICKUP:
        raise ValueError("Can only remove items from pickup docks")

    key = _dock_key(DockType.PICKUP, dock_id)
    data = r.hgetall(key)

    if not data:
        return False

    r.hset(key, mapping={"item_id": "", "ts": int(time.time())})
    return True


# ---------------------------------------------------------
# Reserve dock for robot (atomic lock)
# ---------------------------------------------------------

def reserve_dock(r: redis.Redis, dock_id: str, robot_id: str) -> bool:
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    lock_key = _dock_lock_key(dock_type, dock_id)

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

def occupy_dock(r: redis.Redis, dock_id: str, robot_id: str) -> None:
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

def release_dock(r: redis.Redis, dock_id: str) -> None:
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

def get_dock_state(r: redis.Redis, dock_id: str) -> dict:
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    key = _dock_key(dock_type, dock_id)
    return r.hgetall(key)


def get_all_dock_states(r: redis.Redis) -> list[dict]:
    active_config = r.get("active_dock_config")
    if not active_config or active_config == "none":
        return []

    cursor = 0
    dock_states: list[dict] = []
    while True:
        cursor, keys = r.scan(cursor=cursor, match="dock:*", count=500)
        for key in keys:
            data = r.hgetall(key)
            if data:
                parts = key.split(":")          # dock : <type> : <id>
                dock_id = parts[2]
                dock_states.append({
                    "dock_type": parts[1],
                    "dock_id":   dock_id,
                    "x":         data.get("x"),
                    "y":         data.get("y"),
                    "yaw":       data.get("yaw"),
                    "status":    data.get("status"),
                    "robot_id":  data.get("robot_id"),
                    "item_id":   data.get("item_id"),
                    "item_weight": data.get("item_weight"),
                    "ts":        data.get("ts"),
                })
        if cursor == 0:
            break
    return dock_states


# ---------------------------------------------------------
# Convenience: single call to feed obs_builder
# ---------------------------------------------------------

def get_obs_builder_inputs(
    r: redis.Redis,
    picker_ids: list[str],
    transporter_ids: list[str],
) -> dict:
    """
    Fetches all live runtime data from Redis and returns a dict
    that maps directly to obs_context_from_server_state() parameters.

    Returns
    -------
    {
        "dock_states"        : list[dict],
        "dock_positions"     : dict[str, tuple[int,int]],
        "item_weights"       : dict[str, float],
        "robot_positions"    : dict[str, tuple[int,int]],
        "picker_has_item"    : dict[str, bool],
        "transporter_loads"  : dict[str, tuple[float,float]],
        "transporter_carried": dict[str, list[str]],
        "transporter_in_wz"  : dict[str, bool],
        "waiting_zones"      : list[dict],
    }
    """
    # picker_has_item: dict[str, bool] = {}
    # for pid in picker_ids:
    #     state = get_picker_state(r, pid)
    #     picker_has_item[pid] = state["has_item"] if state else False

    # transporter_loads:   dict[str, tuple[float, float]] = {}
    # transporter_carried: dict[str, list[str]]           = {}
    # transporter_in_wz:   dict[str, bool]                = {}
    # for tid in transporter_ids:
    #     state = get_transporter_state(r, tid)
    #     if state:
    #         transporter_loads[tid]   = (state["capacity"], state["max_capacity"])
    #         transporter_carried[tid] = state["carried_items"]
    #         transporter_in_wz[tid]   = state["in_waiting_zone"]
    #     else:
    #         transporter_loads[tid]   = (0.0, 1.0)
    #         transporter_carried[tid] = []
    #         transporter_in_wz[tid]   = False

    return {
        "dock_states":         get_all_dock_states(r),
        "dock_positions":      get_all_dock_positions(r),
        "item_weights":        get_all_item_weights(r),
        "robot_positions":     get_all_robot_positions(r),
        # "picker_has_item":     picker_has_item,
        # "transporter_loads":   transporter_loads,
        # "transporter_carried": transporter_carried,
        # "transporter_in_wz":   transporter_in_wz,
        "waiting_zones":       get_all_waiting_zone_states(r),
    }