import json
import os
import time
import redis
import requests
from schema import Dock
from sqlalchemy.orm import Session
from definitions import DockType, DockStatus


# ---------------------------------------------------------
# Dock lock lifetimes
# ---------------------------------------------------------
# A reservation expires unless its owner renews it (POST /dock/renew) or
# occupies the dock. This keeps a crashed or rebooted robot from holding a
# dock forever. A hand-off dock that has been occupied holds an item on the
# transporter's slot, so its lock persists until the item is delivered.

DOCK_LOCK_TTL_S = int(os.getenv("DOCK_LOCK_TTL_S", "300"))
# When a robot's WebSocket drops, its expiring locks are cut to this grace
# period. If the robot reconnects in time, they are restored to the full TTL.
DOCK_LOCK_DISCONNECT_GRACE_S = int(os.getenv("DOCK_LOCK_DISCONNECT_GRACE_S", "30"))


# ---------------------------------------------------------
# RL constants  (single source of truth for server + robots)
# ---------------------------------------------------------

RL_CONSTANTS = {
    "MAX_ITEMS": 20,
    "MAX_WZ": 8,
    "MAX_PICKERS": 10,
    "MAX_TRANSPORTERS": 5,
}


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

_DOCK_KEY_PATTERNS = ("dock:*", "dock_meta:*", "lock:dock:*")
_DOCK_SET_KEYS = ("docks:all", "docks:pickup", "docks:receiving")


def _scan_keys(r: redis.Redis, patterns) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=500)
            found.extend(keys)
            if cursor == 0:
                break
    return found


def clear_all_dock_keys(r: redis.Redis) -> None:
    keys = _scan_keys(r, _DOCK_KEY_PATTERNS)
    pipe = r.pipeline(transaction=True)
    pipe.set("active_dock_config", "none")
    if keys:
        pipe.delete(*keys)
    pipe.delete(*_DOCK_SET_KEYS)
    pipe.execute()


def clear_all_runtime_keys(r: redis.Redis) -> None:
    """Clears dock keys AND all robot/wz runtime keys."""
    clear_all_dock_keys(r)
    keys = _scan_keys(r, ("robot:pos:*", "robot:state:*", "wz:state:*", "dock_pos:*", "item:weight:*"))
    if keys:
        r.delete(*keys)


# ---------------------------------------------------------
# Activate dock configuration
# ---------------------------------------------------------

def activate_docks(r: redis.Redis, session: Session, dock_config_id: int) -> None:
    """Initialize Redis runtime from SQLite dock config.

    The old runtime is cleared and the new one written in a single MULTI/EXEC
    transaction, so a failure cannot leave Redis half-cleared with the new
    config already marked active.
    """
    docks = session.query(Dock).filter_by(config_id=dock_config_id).all()
    stale_keys = _scan_keys(r, _DOCK_KEY_PATTERNS)
    now = int(time.time())

    pipe = r.pipeline(transaction=True)
    if stale_keys:
        pipe.delete(*stale_keys)
    pipe.delete(*_DOCK_SET_KEYS)

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

        # store dock (y, x) position if the dock has one
        if d.x is not None and d.y is not None:
            pipe.hset(
                f"dock_pos:{d.dock_id}",
                mapping={"y": d.y, "x": d.x},
            )

    pipe.set("active_dock_config", dock_config_id)
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

def get_all_item_receiver_docks(r: redis.Redis) -> dict[str, str]:
    """Returns { item_id: receiver_dock_id } for all docks in docks:all that have an item with a receiver dock."""
    receivers: dict[str, str] = {}
    dock_ids = r.smembers("docks:all")
    for dock_id in dock_ids:
        dock_type_str = r.hget(f"dock_meta:{dock_id}", "dock_type")
        if dock_type_str is None:
            continue
        data = r.hgetall(f"dock:{dock_type_str}:{dock_id}")
        if not data:
            continue
        item_id = data.get("item_id")
        receiver_dock_id = data.get("receiver_dock_id")
        if item_id and receiver_dock_id not in (None, ""):
            receivers[item_id] = receiver_dock_id
    return receivers


# ---------------------------------------------------------
# Robot positions
# ---------------------------------------------------------


def get_all_robot_positions(r: redis.Redis, robot_id: str) -> dict[str, tuple[str, float, float, float]]:
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

        # domain_id 0 is valid, so test for None rather than falsiness
        if not ip or domain_id is None:
            continue
        try:
            resp = requests.get(f"http://{ip}:8000/roslib/transform", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                namespace = f"couliglig_bot_{domain_id}"
                # if namespace == robot_id:
                #     continue

                positions[namespace] = (str(data["rl_robot_type"]), float(data["x"]), float(data["y"]), float(data["yaw"]))
        except Exception as e:
            print(f"Error fetching position for robot at {ip}: {e}")
            continue
    return positions


# ---------------------------------------------------------
# Robot state  (picker / transporter runtime)
# ---------------------------------------------------------


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


def get_all_waiting_zone_states(r: redis.Redis) -> list[dict]:
    """Returns all docks in docks:all whose dock_type is 'waiting_zone'.

    Sorted by dock_id, the same order build_action_map uses for wz_slots, so
    observation row i and action slot i refer to the same zone.
    """
    reconcile_expired_locks(r)
    zones: list[dict] = []
    dock_ids = sorted(r.smembers("docks:all"))
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
    receiver_dock_id: str,
    item_weight: float = 1.0
) -> bool:
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")
    
    if not _check_if_dock_id_exists(r, receiver_dock_id):
        raise ValueError(f"Receiver Dock ID {receiver_dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    if dock_type != DockType.PICKUP:
        raise ValueError("Can only add items to pickup docks")
    
    receiver_dock_type = _get_dock_type_by_id(r, receiver_dock_id)
    if receiver_dock_type != DockType.RECEIVER:
        raise ValueError("Can only add items to pickup docks with a valid receiver dock")

    key = _dock_key(dock_type, dock_id)
    data = r.hgetall(key)

    if not data:
        return False
    if data.get("item_id"):
        return False

    r.hset(key, mapping={"item_id": item_id, "item_weight": item_weight, "receiver_dock_id": receiver_dock_id, "ts": int(time.time())})

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

    acquired = r.set(lock_key, robot_id, nx=True, ex=DOCK_LOCK_TTL_S)
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


def _check_owner(owner: str | None, dock_id: str, robot_id: str, missing_msg: str) -> None:
    if owner is None:
        raise ValueError(missing_msg)
    if owner != robot_id:
        raise ValueError(f"Dock {dock_id} is owned by '{owner}', not '{robot_id}'")


def _with_lock_watch(r: redis.Redis, lock_key: str, body):
    """Run body(pipe, owner, ttl) under WATCH on lock_key, retrying on WatchError.

    body reads nothing itself; it queues writes after calling pipe.multi().
    This makes "check owner, then write" atomic against the lock expiring or
    being taken by another robot in between.
    """
    with r.pipeline() as pipe:
        while True:
            try:
                pipe.watch(lock_key)
                owner = pipe.get(lock_key)
                ttl = pipe.ttl(lock_key)
                result = body(pipe, owner, ttl)
                pipe.execute()
                return result
            except redis.WatchError:
                continue


# ---------------------------------------------------------
# Mark dock occupied (robot arrived)
# ---------------------------------------------------------

def occupy_dock(r: redis.Redis, dock_id: str, robot_id: str, item_id: str | None = None) -> None:
    """Mark a reserved dock occupied by its owner.

    For an ordinary dock the lock TTL is refreshed. For a hand-off dock the
    lock is made persistent and item_id (if given) is recorded on the dock,
    because the item now sits on the transporter's slot until it is delivered.
    """
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    lock_key = _dock_lock_key(dock_type, dock_id)
    dock_key = _dock_key(dock_type, dock_id)

    def body(pipe, owner, _ttl):
        _check_owner(owner, dock_id, robot_id,
                     f"Dock {dock_id} has not been reserved — call reserve first")
        pipe.multi()
        if dock_type == DockType.HANDOFF:
            pipe.persist(lock_key)
        else:
            pipe.expire(lock_key, DOCK_LOCK_TTL_S)
        mapping = {
            "status": DockStatus.occupied.value,
            "robot_id": robot_id,
            "ts": int(time.time()),
        }
        if dock_type == DockType.HANDOFF and item_id:
            mapping["item_id"] = item_id
        pipe.hset(dock_key, mapping=mapping)

    _with_lock_watch(r, lock_key, body)


# ---------------------------------------------------------
# Renew a reservation (robot still on its way)
# ---------------------------------------------------------

def renew_dock(r: redis.Redis, dock_id: str, robot_id: str) -> int:
    """Extend the owner's lock to the full TTL. Returns the new TTL in seconds.

    A persistent lock (occupied hand-off dock) stays persistent and returns -1.
    Raises ValueError if the lock is gone or owned by someone else; the robot
    must then treat its reservation as lost.
    """
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    lock_key = _dock_lock_key(dock_type, dock_id)

    def body(pipe, owner, ttl):
        _check_owner(owner, dock_id, robot_id,
                     f"Dock {dock_id} has no active reservation to renew")
        pipe.multi()
        if ttl == -1:
            return -1
        pipe.expire(lock_key, DOCK_LOCK_TTL_S)
        return DOCK_LOCK_TTL_S

    return _with_lock_watch(r, lock_key, body)


# ---------------------------------------------------------
# Release dock
# ---------------------------------------------------------

def release_dock(r: redis.Redis, dock_id: str, robot_id: str, item_id: str | None = None) -> None:
    """Release a dock. Only its lock owner may release it, with one exception.

    A hand-off dock is locked by the picker that docked there, but it is the
    transporter that later delivers the item. So a hand-off dock may also be
    released by any robot that names the item recorded on it.
    """
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    dock_type = _get_dock_type_by_id(r, dock_id)
    lock_key = _dock_lock_key(dock_type, dock_id)
    dock_key = _dock_key(dock_type, dock_id)

    def body(pipe, owner, _ttl):
        released_by_item = (
            dock_type == DockType.HANDOFF
            and item_id
            and owner is not None
            and pipe.hget(dock_key, "item_id") == item_id
        )
        if not released_by_item:
            _check_owner(owner, dock_id, robot_id,
                         f"Dock {dock_id} has no active reservation to release")
        pipe.multi()
        pipe.delete(lock_key)
        mapping = {
            "status": DockStatus.available.value,
            "robot_id": "",
            "ts": int(time.time()),
        }
        if dock_type == DockType.HANDOFF:
            mapping["item_id"] = ""
        pipe.hset(dock_key, mapping=mapping)

    _with_lock_watch(r, lock_key, body)


# ---------------------------------------------------------
# Expired locks and disconnected robots
# ---------------------------------------------------------

def reconcile_expired_locks(r: redis.Redis) -> list[str]:
    """Mark docks available whose lock has expired. Returns the dock ids fixed.

    The lock key is the source of truth for ownership; the dock hash only
    mirrors it. When a lock expires, Redis deletes it but the hash still says
    reserved/occupied, so readers call this first to bring the hash in line.
    """
    dock_ids = sorted(r.smembers("docks:all"))
    if not dock_ids:
        return []

    pipe = r.pipeline(transaction=False)
    for dock_id in dock_ids:
        pipe.hget(f"dock_meta:{dock_id}", "dock_type")
    types = pipe.execute()

    candidates: list[tuple[str, DockType]] = []
    pipe = r.pipeline(transaction=False)
    for dock_id, type_str in zip(dock_ids, types):
        if type_str is None:
            continue
        dock_type = DockType(type_str)
        candidates.append((dock_id, dock_type))
        pipe.hget(_dock_key(dock_type, dock_id), "status")
        pipe.exists(_dock_lock_key(dock_type, dock_id))
    flat = pipe.execute()

    fixed: list[str] = []
    for i, (dock_id, dock_type) in enumerate(candidates):
        status, lock_exists = flat[2 * i], flat[2 * i + 1]
        if lock_exists or status in (None, DockStatus.available.value):
            continue
        lock_key = _dock_lock_key(dock_type, dock_id)
        dock_key = _dock_key(dock_type, dock_id)

        def body(p, owner, _ttl, dock_type=dock_type, dock_key=dock_key):
            if owner is not None:
                return False      # re-reserved meanwhile; the hash is current
            p.multi()
            mapping = {
                "status": DockStatus.available.value,
                "robot_id": "",
                "ts": int(time.time()),
            }
            if dock_type == DockType.HANDOFF:
                mapping["item_id"] = ""
            p.hset(dock_key, mapping=mapping)
            return True

        if _with_lock_watch(r, lock_key, body):
            fixed.append(dock_id)
    return fixed


def _locks_owned_by(r: redis.Redis, robot_id: str) -> list[str]:
    keys = _scan_keys(r, ("lock:dock:*",))
    if not keys:
        return []
    owners = r.mget(keys)
    return [k for k, owner in zip(keys, owners) if owner == robot_id]


def shorten_robot_locks(r: redis.Redis, robot_id: str,
                        grace_s: int = DOCK_LOCK_DISCONNECT_GRACE_S) -> list[str]:
    """Cut a disconnected robot's expiring locks to grace_s. Returns lock keys changed.

    Persistent locks (an occupied hand-off dock holding an item) are left
    alone: the item is physically still on that slot.
    """
    changed: list[str] = []
    for key in _locks_owned_by(r, robot_id):
        ttl = r.ttl(key)
        if ttl > grace_s:
            r.expire(key, grace_s)
            changed.append(key)
    return changed


def restore_robot_locks(r: redis.Redis, robot_id: str) -> list[str]:
    """Give a (re)connected robot's expiring locks the full TTL again."""
    changed: list[str] = []
    for key in _locks_owned_by(r, robot_id):
        if r.ttl(key) > 0:
            r.expire(key, DOCK_LOCK_TTL_S)
            changed.append(key)
    return changed


# ---------------------------------------------------------
# Query dock state
# ---------------------------------------------------------

def get_dock_state(r: redis.Redis, dock_id: str) -> dict:
    if not _check_if_dock_id_exists(r, dock_id):
        raise ValueError(f"Dock ID {dock_id} does not exist")

    reconcile_expired_locks(r)
    dock_type = _get_dock_type_by_id(r, dock_id)
    key = _dock_key(dock_type, dock_id)
    return r.hgetall(key)


def get_all_dock_states(r: redis.Redis) -> list[dict]:
    active_config = r.get("active_dock_config")
    if not active_config or active_config == "none":
        return []
    reconcile_expired_locks(r)

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
                    "receiver_dock_id": data.get("receiver_dock_id"),
                    "ts":        data.get("ts"),
                })
        if cursor == 0:
            break
    return dock_states


def get_active_docks(
    r: redis.Redis,
    dock_type: str | None = None,
) -> list[dict]:
    """Return all docks whose status is reserved or occupied.

    Parameters
    ----------
    dock_type : str | None
        Optional filter: ``"pickup"``, ``"waiting_zone"``, or ``"receiver"``.
        When ``None`` all dock types are returned.
    """
    active_config = r.get("active_dock_config")
    if not active_config or active_config == "none":
        return []
    reconcile_expired_locks(r)

    ACTIVE_STATUSES = {DockStatus.reserved.value, DockStatus.occupied.value}

    pattern = f"dock:{dock_type}:*" if dock_type else "dock:*"

    cursor = 0
    results: list[dict] = []
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=500)
        for key in keys:
            data = r.hgetall(key)
            if not data:
                continue
            if data.get("status") not in ACTIVE_STATUSES:
                continue
            parts = key.split(":")
            results.append({
                "dock_type":   parts[1],
                "dock_id":     parts[2],
                "x":           data.get("x"),
                "y":           data.get("y"),
                "yaw":         data.get("yaw"),
                "status":      data.get("status"),
                "robot_id":    data.get("robot_id"),
                "item_id":     data.get("item_id"),
                "item_weight": data.get("item_weight"),
                "ts":          data.get("ts"),
            })
        if cursor == 0:
            break
    return results


# ---------------------------------------------------------

def build_action_map(r: redis.Redis, agent_states: list[dict]) -> dict:
    """Build a deterministic, padded index→entity mapping for RL action decoding.

    The robot fetches this alongside observations so it can translate a discrete
    action integer back into a concrete dock/item/peer ID, then call the existing
    /dock/reserve, /dock/occupy, /dock/release endpoints directly.

    Slot ordering is alphabetically sorted by dock_id / namespace so indices are
    stable across calls as long as the dock configuration does not change.

    Parameters
    ----------
    r : redis.Redis
        Active Redis connection.
    agent_states : list[dict]
        Already-fetched list from ``_fetch_all_agent_states`` so we avoid a
        second HTTP round-trip to the robots.  Each dict must contain at least
        ``namespace`` and ``agent_type``.

    Returns
    -------
    dict with keys:
        item_slots        – list[dict|None], length MAX_ITEMS
        wz_slots          – list[dict|None], length MAX_WZ
        receiver_slots    – list[dict]       (unpadded, delivery lookup only)
        picker_slots      – list[dict|None], length MAX_PICKERS
        transporter_slots – list[dict|None], length MAX_TRANSPORTERS
    """
    MAX_ITEMS = RL_CONSTANTS["MAX_ITEMS"]
    MAX_WZ = RL_CONSTANTS["MAX_WZ"]
    MAX_PICKERS = RL_CONSTANTS["MAX_PICKERS"]
    MAX_TRANSPORTERS = RL_CONSTANTS["MAX_TRANSPORTERS"]

    reconcile_expired_locks(r)
    dock_ids = sorted(r.smembers("docks:all"))

    # ---- classify docks by type ----
    raw_items: list[dict] = []
    raw_wzs: list[dict] = []
    raw_receivers: list[dict] = []

    for dock_id in dock_ids:
        dock_type_str = r.hget(f"dock_meta:{dock_id}", "dock_type")
        if dock_type_str is None:
            continue
        data = r.hgetall(f"dock:{dock_type_str}:{dock_id}")
        if not data:
            continue

        entry = {
            "dock_id": dock_id,
            "dock_type": dock_type_str,
            "status": data.get("status", ""),
            "robot_id": data.get("robot_id", ""),
            "x": float(data["x"]) if data.get("x") not in (None, "") else None,
            "y": float(data["y"]) if data.get("y") not in (None, "") else None,
            "yaw": float(data["yaw"]) if data.get("yaw") not in (None, "") else None,
        }

        if dock_type_str == DockType.PICKUP.value:
            entry["item_id"] = data.get("item_id", "")
            entry["item_weight"] = (
                float(data["item_weight"])
                if data.get("item_weight") not in (None, "")
                else None
            )
            entry["available_for_pickup"] = (
                bool(entry["item_id"])
                and entry["status"] == DockStatus.available.value
            )
            raw_items.append(entry)

        elif dock_type_str == DockType.WAITING_ZONE.value:
            entry["available_for_entry"] = (
                entry["status"] == DockStatus.available.value
            )
            raw_wzs.append(entry)

        elif dock_type_str == DockType.RECEIVER.value:
            raw_receivers.append(entry)

    # ---- pad item_slots to MAX_ITEMS ----
    item_slots: list[dict | None] = []
    for i, entry in enumerate(raw_items[:MAX_ITEMS]):
        entry["index"] = i
        item_slots.append(entry)
    while len(item_slots) < MAX_ITEMS:
        item_slots.append(None)

    # ---- pad wz_slots to MAX_WZ ----
    wz_slots: list[dict | None] = []
    for i, entry in enumerate(raw_wzs[:MAX_WZ]):
        entry["index"] = i
        wz_slots.append(entry)
    while len(wz_slots) < MAX_WZ:
        wz_slots.append(None)

    # ---- receiver_slots (unpadded, for delivery target lookup) ----
    receiver_slots: list[dict] = []
    for i, entry in enumerate(raw_receivers):
        entry["index"] = i
        receiver_slots.append(entry)

    # ---- robot slots from agent_states ----
    pickers: list[dict] = []
    transporters: list[dict] = []
    for state in sorted(agent_states, key=lambda s: s.get("namespace", "")):
        ns = state.get("namespace", "")
        agent_type = state.get("agent_type", "")
        slot = {"namespace": ns, "agent_type": agent_type}

        if agent_type == "picker":
            pickers.append(slot)
        elif agent_type == "transporter":
            transporters.append(slot)

    picker_slots: list[dict | None] = []
    for i, s in enumerate(pickers[:MAX_PICKERS]):
        s["index"] = i
        picker_slots.append(s)
    while len(picker_slots) < MAX_PICKERS:
        picker_slots.append(None)

    transporter_slots: list[dict | None] = []
    for i, s in enumerate(transporters[:MAX_TRANSPORTERS]):
        s["index"] = i
        transporter_slots.append(s)
    while len(transporter_slots) < MAX_TRANSPORTERS:
        transporter_slots.append(None)

    return {
        "item_slots": item_slots,
        "wz_slots": wz_slots,
        "receiver_slots": receiver_slots,
        "picker_slots": picker_slots,
        "transporter_slots": transporter_slots,
    }


# ---------------------------------------------------------
# Convenience: single call to feed obs_builder
# ---------------------------------------------------------

def _fetch_all_agent_states(r: redis.Redis, robot_id: str) -> list[dict]:
    """Call GET /rl on every registered robot and return the list of responses."""
    raw = r.get("robot_ips")
    if not raw:
        return []
    try:
        robot_dict = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    states: list[dict] = []
    for _, payload in robot_dict.items():
        ip = payload.get("ip") if isinstance(payload, dict) else payload
        domain_id = payload.get("ros_domain_id") if isinstance(payload, dict) else None
        if not ip or domain_id is None:
            continue
        try:
            resp = requests.get(f"http://{ip}:8000/rl", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                data["namespace"] = f"couliglig_bot_{domain_id}"
                # if data["namespace"] == robot_id:
                #     continue
                
                states.append(data)
        except Exception as e:
            print(f"Error fetching /rl for robot at {ip}: {e}")
            continue
    return states


def get_obs_builder_inputs(
    r: redis.Redis,
    robot_id: str,
) -> dict:
    """
    Fetches all live runtime data from Redis (and robot endpoints) and
    returns a dict that maps directly to obs_context_from_server_state() parameters.

    Returns
    -------
    {
        "dock_states"        : list[dict],
        "dock_positions"     : dict[str, tuple[float,float,float]],
        "item_weights"       : dict[str, float],
        "robot_positions"    : dict[str, tuple[str,float,float,float]],
        "picker_has_item"    : dict[str, bool],
        "transporter_loads"  : dict[str, tuple[float,float]],
        "transporter_carried": dict[str, list[str]],
        "transporter_in_wz"  : dict[str, bool],
        "waiting_zones"      : list[dict],
        "action_map"         : dict  (from build_action_map),
        "rl_constants"       : dict  (MAX_ITEMS, MAX_WZ, dims, live counts),
    }
    """
    agent_states = _fetch_all_agent_states(r, robot_id)

    picker_has_item:     dict[str, bool]               = {}
    transporter_loads:   dict[str, tuple[float, float]] = {}
    transporter_carried: dict[str, list[str]]           = {}
    transporter_in_wz:   dict[str, bool]                = {}

    for state in agent_states:
        ns = state.get("namespace", state.get("agent_id", "unknown"))
        agent_type = state.get("agent_type")

        if agent_type == "picker":
            picker_has_item[ns] = bool(state.get("has_item", False))

        elif agent_type == "transporter":
            transporter_loads[ns]   = (float(state.get("capacity", 0.0)), float(state.get("max_capacity", 1.0)))
            transporter_carried[ns] = list(state.get("carried_items", []))
            transporter_in_wz[ns]   = bool(state.get("in_waiting_zone", False))

    # ---- action map (reuse agent_states, no extra HTTP calls) ----
    action_map = build_action_map(r, agent_states)

    num_pickers = sum(1 for s in action_map["picker_slots"] if s is not None)
    num_transporters = sum(1 for s in action_map["transporter_slots"] if s is not None)

    rl_constants = {
        **RL_CONSTANTS,
        "PICKER_ACTION_DIM": 1 + RL_CONSTANTS["MAX_ITEMS"] + num_transporters,
        "TRANSPORTER_ACTION_DIM": 1 + RL_CONSTANTS["MAX_WZ"] + RL_CONSTANTS["MAX_ITEMS"],
        "NUM_PICKERS": num_pickers,
        "NUM_TRANSPORTERS": num_transporters,
    }

    return {
        "dock_states":         get_all_dock_states(r),
        "dock_positions":      get_all_dock_positions(r),
        "item_weights":        get_all_item_weights(r),
        "item_receiver_docks": get_all_item_receiver_docks(r),
        "robot_positions":     get_all_robot_positions(r, robot_id),
        "picker_has_item":     picker_has_item,
        "transporter_loads":   transporter_loads,
        "transporter_carried": transporter_carried,
        "transporter_in_wz":   transporter_in_wz,
        "waiting_zones":       get_all_waiting_zone_states(r),
        "transporter_occupany": get_active_docks(r, dock_type=DockType.HANDOFF.value),
        "action_map":         action_map,
        "rl_constants":        rl_constants,
    }