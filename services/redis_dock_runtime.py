import time
import redis
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
    dock_id: str,
    item_id: str
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

    if data.get("item_id") != item_id:
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