import time
import redis
from schema import Dock
from sqlalchemy.orm import Session


def redis_client_from_env() -> redis.Redis:
    import os
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))

    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


def _dock_key(dock_type: str, dock_id: str) -> str:
    return f"dock:{dock_type}:{dock_id}"


def _dock_lock_key(dock_type: str, dock_id: str) -> str:
    return f"lock:dock:{dock_type}:{dock_id}"


# ---------------------------------------------------------
# Clear runtime
# ---------------------------------------------------------

def clear_all_dock_keys(r: redis.Redis) -> None:
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="dock:*", count=500)
        if keys:
            r.delete(*keys)

        if cursor == 0:
            break


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
    docks = session.query(Dock).filter_by(dock_config_id=dock_config_id).all()

    for d in docks:

        key = _dock_key(d.dock_type, d.dock_id)

        pipe.hset(
            key,
            mapping={
                "status": "available",
                "robot_id": "",
                "item_id": "",
                "ts": str(now),
            },
        )

    pipe.execute()


# ---------------------------------------------------------
# Add item to pickup dock
# ---------------------------------------------------------

def add_item_to_pickup_dock(
    r: redis.Redis,
    dock_id: str,
    item_id: str
) -> bool:

    key = _dock_key("pickup", dock_id)

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

    key = _dock_key("pickup", dock_id)

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
    dock_type: str,
    dock_id: str,
    robot_id: str
) -> bool:

    lock_key = _dock_lock_key(dock_type, dock_id)

    # atomic lock
    acquired = r.set(lock_key, robot_id, nx=True)

    if not acquired:
        return False

    dock_key = _dock_key(dock_type, dock_id)

    r.hset(
        dock_key,
        mapping={
            "status": "reserved",
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
    dock_type: str,
    dock_id: str,
    robot_id: str,
):

    dock_key = _dock_key(dock_type, dock_id)

    r.hset(
        dock_key,
        mapping={
            "status": "occupied",
            "robot_id": robot_id,
            "ts": int(time.time()),
        },
    )


# ---------------------------------------------------------
# Release dock
# ---------------------------------------------------------

def release_dock(
    r: redis.Redis,
    dock_type: str,
    dock_id: str,
):

    lock_key = _dock_lock_key(dock_type, dock_id)
    dock_key = _dock_key(dock_type, dock_id)

    r.delete(lock_key)

    r.hset(
        dock_key,
        mapping={
            "status": "available",
            "robot_id": "",
            "ts": int(time.time()),
        },
    )


# ---------------------------------------------------------
# Query dock state
# ---------------------------------------------------------

def get_dock_state(
    r: redis.Redis,
    dock_type: str,
    dock_id: str
):

    key = _dock_key(dock_type, dock_id)

    return r.hgetall(key)