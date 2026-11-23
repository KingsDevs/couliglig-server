import json
import uuid
import asyncio
import logging
import redis
import os
import requests
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from models import RobotRegistration, RobotStatus, RobotStatusesResponse, RobotOnline, RobotOnlineResponse
from models.robot_infos import RobotInfoDef, RobotInfoCreate, RobotInfoUpdate, RobotInfoBulkDelete
from schema import RobotInfo, MapConfig
from services import register_host
from services.redis_dock_runtime import (
    restore_robot_locks, shorten_robot_locks, DOCK_LOCK_DISCONNECT_GRACE_S,
)
from dotenv import load_dotenv
from datetime import datetime
from services.database import get_db_session

load_dotenv()

register_router = APIRouter(prefix="/robots", tags=["robots"])

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"), 
    port=os.getenv("REDIS_PORT", 6379),       
    db=0,           
    decode_responses=True
)

def is_couliglig_lan(name: str) -> bool:
    return name.startswith("couliglig")


# --- WebSocket robot helpers ---

_dashboard_subscribers: set[WebSocket] = set()


async def _broadcast_online():
    raw = redis_client.get("robot_ips")
    ip_dict = json.loads(raw) if raw else {}
    msg = json.dumps(list(ip_dict.values()))
    dead = set()
    for ws in _dashboard_subscribers:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _dashboard_subscribers.discard(ws)


def _robot_namespace(ros_domain_id) -> str:
    """The robot_id robots use for dock locks (their ROS namespace)."""
    return f"couliglig_bot_{ros_domain_id}"


def _mutate_robot_ips(fn):
    """Apply fn(ip_dict) to the robot_ips blob atomically (WATCH/MULTI).

    fn mutates the dict in place and returns a result; the blob is written
    back only if fn returns a truthy value. Concurrent handlers (the WebSocket
    loop and the threadpool-run /register) can no longer drop each other's
    updates.
    """
    with redis_client.pipeline() as pipe:
        while True:
            try:
                pipe.watch("robot_ips")
                raw = pipe.get("robot_ips")
                ip_dict = json.loads(raw) if raw else {}
                result = fn(ip_dict)
                pipe.multi()
                if result:
                    pipe.set("robot_ips", json.dumps(ip_dict))
                pipe.execute()
                return result
            except redis.WatchError:
                continue


def _register_robot_in_redis(hostname: str, ip: str, namespace: str, ros_domain_id: int,
                             battery: float | None = None, session: str | None = None):
    def fn(ip_dict):
        ip_dict[hostname] = {
            "hostname": hostname,
            "ip": ip,
            "namespace": namespace,
            "ros_domain_id": ros_domain_id,
            "battery": battery,
            "timestamp": datetime.now().isoformat() + "Z",
            "session": session,
        }
        return True
    _mutate_robot_ips(fn)


def _update_robot_timestamp(hostname: str, battery: float | None = None, session: str | None = None,
                            registration: dict | None = None):
    """Refresh a robot's heartbeat. Re-adds the entry if it has gone missing.

    Does nothing if a newer connection (different session) now owns the entry.
    """
    def fn(ip_dict):
        entry = ip_dict.get(hostname)
        if entry is None:
            if registration is None:
                return False
            entry = dict(registration)
            ip_dict[hostname] = entry
        elif session is not None and entry.get("session") not in (None, session):
            return False
        entry["timestamp"] = datetime.now().isoformat() + "Z"
        if battery is not None:
            entry["battery"] = battery
        return True
    _mutate_robot_ips(fn)


def _remove_robot_from_redis(hostname: str, session: str | None = None) -> bool:
    """Remove a robot's entry. With a session, only if that session still owns it.

    This stops the cleanup of an old socket from deleting the registration a
    reconnecting robot has just made.
    """
    def fn(ip_dict):
        entry = ip_dict.get(hostname)
        if entry is None:
            return False
        if session is not None and isinstance(entry, dict) and entry.get("session") != session:
            return False
        ip_dict.pop(hostname)
        return True
    removed = _mutate_robot_ips(fn)
    if removed:
        logging.info(f"[robot-ws] {hostname} removed from robot_ips")
    return bool(removed)


# A robot's entry is refreshed by every heartbeat on its /robots/ws socket (every 5 s;
# the socket is dropped after 15 s of silence). An entry older than this belongs to no
# live socket: e.g. the server restarted while the robot was connected, so that socket's
# cleanup never ran, and Redis kept the entry. The pruner removes such entries.
ROBOT_STALE_S = float(os.getenv("ROBOT_STALE_S", "30"))
ROBOT_PRUNE_INTERVAL_S = 10.0


def _entry_time(entry) -> datetime | None:
    if not isinstance(entry, dict):
        return None
    try:
        return datetime.fromisoformat(str(entry.get("timestamp", "")).rstrip("Z"))
    except ValueError:
        return None


def prune_stale_robots(now: datetime | None = None) -> list[dict]:
    """Remove robot_ips entries with no heartbeat for ROBOT_STALE_S (or no timestamp
    at all); shorten their dock locks as a disconnect does. Returns the removed entries."""
    now = now or datetime.now()          # timestamps are written with datetime.now() too
    removed: list[dict] = []

    def fn(ip_dict):
        removed.clear()                  # fn reruns when the WATCH fails
        for hostname, entry in list(ip_dict.items()):
            ts = _entry_time(entry)
            if ts is None or (now - ts).total_seconds() > ROBOT_STALE_S:
                ip_dict.pop(hostname)
                removed.append(entry if isinstance(entry, dict) else {"hostname": hostname, "ip": entry})
        return bool(removed)

    _mutate_robot_ips(fn)
    for entry in removed:
        logging.warning(f"[robot-ws] {entry.get('hostname')} removed from robot_ips: "
                        f"no heartbeat since {entry.get('timestamp') or 'ever'}")
        if entry.get("ros_domain_id") is not None:
            shorten_robot_locks(redis_client, _robot_namespace(entry["ros_domain_id"]))
    return removed


async def prune_stale_robots_forever(interval: float = ROBOT_PRUNE_INTERVAL_S):
    """Background task (main.py): prune now, then every interval; tell the dashboards."""
    while True:
        try:
            if prune_stale_robots():
                await _broadcast_online()
        except Exception as e:
            logging.warning(f"[robot-ws] pruning robot_ips failed: {e}")
        await asyncio.sleep(interval)


@register_router.websocket("/ws")
async def robot_websocket(websocket: WebSocket):
    await websocket.accept()
    hostname = None
    session = uuid.uuid4().hex
    registration = None
    try:
        data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        hostname = data["hostname"]
        ip = data["ip"]
        namespace = data.get("namespace", "couliglig")
        ros_domain_id = data.get("ros_domain_id", 0)
        battery = data.get("battery")

        _register_robot_in_redis(hostname, ip, namespace, ros_domain_id, battery, session=session)
        registration = {
            "hostname": hostname, "ip": ip, "namespace": namespace,
            "ros_domain_id": ros_domain_id, "battery": battery, "session": session,
        }
        restored = restore_robot_locks(redis_client, _robot_namespace(ros_domain_id))
        logging.info(f"[robot-ws] {hostname} ({ip}) connected"
                     + (f"; restored {len(restored)} dock lock(s)" if restored else ""))
        await _broadcast_online()

        while True:
            msg = await asyncio.wait_for(websocket.receive_json(), timeout=15)
            if msg.get("type") == "heartbeat":
                _update_robot_timestamp(hostname, msg.get("battery"), session=session,
                                        registration=registration)
                logging.debug(f"[robot-ws] {hostname} heartbeat")
                await _broadcast_online()

    except (WebSocketDisconnect, asyncio.TimeoutError) as e:
        logging.warning(f"[robot-ws] {hostname} disconnected ({type(e).__name__})")
    except Exception as e:
        logging.warning(f"[robot-ws] {hostname} error ({e})")
    finally:
        if hostname and _remove_robot_from_redis(hostname, session=session):
            # Give the robot a short grace period to reconnect before its
            # reservations expire, instead of holding them for the full TTL.
            if registration is not None:
                shortened = shorten_robot_locks(
                    redis_client, _robot_namespace(registration["ros_domain_id"]))
                if shortened:
                    logging.warning(f"[robot-ws] {hostname}: {len(shortened)} dock lock(s) "
                                    f"will expire in {DOCK_LOCK_DISCONNECT_GRACE_S}s unless it reconnects")
            await _broadcast_online()

@register_router.websocket("/online/ws")
async def robots_online_websocket(websocket: WebSocket):
    await websocket.accept()
    _dashboard_subscribers.add(websocket)
    try:
        # Send current state immediately on connect
        await _broadcast_online()
        # Keep connection open; dashboard just listens
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _dashboard_subscribers.discard(websocket)


@register_router.post("/register")
def register_robot(data: RobotRegistration):
    try:
        print(f"Registering host {data.hostname} with IP {data.ip}, namespace {data.namespace}, domain ID {data.ros_domain_id}")
        register_host(data.hostname, data.ip)

        if is_couliglig_lan(data.hostname):
            # Retrieve the existing dictionary from Redis
            existing_data = redis_client.get("robot_registrations")
            if existing_data:
                robot_dict = json.loads(existing_data)  # Deserialize JSON to a Python dictionary
            else:
                robot_dict = {}

            # Add or update the hostname with its IP
            robot_dict[data.hostname] = data.ip

            # Store the updated dictionary back in Redis
            redis_client.set("robot_registrations", json.dumps(robot_dict))  # Serialize dictionary to JSON

            def upsert(ip_dict):
                previous = ip_dict.get(data.hostname)
                ip_dict[data.hostname] = {
                    "hostname": data.hostname,
                    "ip": str(data.ip),
                    "namespace": data.namespace or "couliglig",
                    "ros_domain_id": data.ros_domain_id or 0,
                    "battery": data.battery,
                    "timestamp": datetime.now().isoformat() + "Z",
                    # keep the live WebSocket's ownership of this entry
                    "session": previous.get("session") if isinstance(previous, dict) else None,
                }
                return True

            _mutate_robot_ips(upsert)

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Requires root privileges"
        )
    except Exception as e:
        print(f"Error registering robot: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
    return {
        "status": "ok",
        "hostname": data.hostname,
        "ip": data.ip,
    }

@register_router.get("", response_model=RobotStatusesResponse)
def get_robot_statuses():
    try:
        existing_data = redis_client.get("robot_registrations")
        if existing_data:
            robot_dict = json.loads(existing_data)
        else:
            robot_dict = {}

        statuses = []

        for hostname, ip in robot_dict.items():
            try:
                url = f"http://{ip}:8000/status"
                try:
                    response = requests.get(url, timeout=5)
                except requests.ConnectionError:
                    # Remove the offline robot from the registrations
                    del robot_dict[hostname]
                    redis_client.set("robot_registrations", json.dumps(robot_dict))
                    continue

                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")

                try:
                    data = response.json()
                except ValueError:
                    raise RuntimeError("Invalid JSON")

                statuses.append(RobotStatus(
                    hostname=data.get("hostname", hostname),
                    robot_namespace=data.get("robot_namespace", ""),
                    domain_id=data.get("domain_id"),
                    ip=data.get("ip", ip),
                    uptime=data.get("uptime", ""),
                ))

            except Exception as e:
                statuses.append(RobotStatus(
                    hostname=hostname,
                    robot_namespace="",
                    domain_id=None,
                    ip=ip,
                    uptime="",
                    error=str(e)
                ))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RobotStatusesResponse(
        status="ok",
        registrations=statuses
    )

@register_router.get("/online", response_model=RobotOnlineResponse)
def get_online_robots():
    try:
        raw = redis_client.get("robot_ips")
        if not raw:
            return RobotOnlineResponse(data=[])

        robot_dict = json.loads(raw)
        online_robots = []

        for hostname, payload in robot_dict.items():
            # Backward compatibility if string IP still exists
            if isinstance(payload, str):
                online_robots.append(
                    RobotOnline(
                        hostname=hostname,
                        ip=payload
                    )
                )
                continue

            online_robots.append(
                RobotOnline(
                    hostname=payload.get("hostname", hostname),
                    ip=payload.get("ip"),
                    namespace=f"couliglig_bot_{payload.get('ros_domain_id', 0)}",
                    ros_domain_id=payload.get("ros_domain_id", 0),
                    timestamp=payload.get("timestamp", ""),
                    battery=payload.get("battery")
                )
            )

        return RobotOnlineResponse(data=online_robots)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@register_router.get("/infos", response_model=list[RobotInfoDef])
def get_robot_infos(map_id: int, db: Session = Depends(get_db_session)):
    try:
        infos = db.query(RobotInfo).filter(RobotInfo.map_id == map_id).all()
        return infos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@register_router.post("/infos", response_model=list[RobotInfoDef])
def create_robot_info(infos: list[RobotInfoCreate], db: Session = Depends(get_db_session)):
    try:
        created_infos = []
        for info in infos:
            map_cfg = db.query(MapConfig).filter(MapConfig.id == info.map_id).first()
            if not map_cfg:
                raise HTTPException(status_code=404, detail="Map config not found")

            existing = db.query(RobotInfo).filter(RobotInfo.map_id == info.map_id, RobotInfo.robot_name == info.robot_name).first()
            if existing:
                raise HTTPException(status_code=409, detail="Robot info with this map ID and name already exists")

            new_info = RobotInfo(
                robot_name=info.robot_name,
                initial_x=info.initial_x,
                initial_y=info.initial_y,
                initial_theta=info.initial_theta,
                map_id=info.map_id
            )
            db.add(new_info)
            created_infos.append(new_info)

        db.commit()
        for info in created_infos:
            db.refresh(info)

        return created_infos
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@register_router.post("/infos/upsert", response_model=list[RobotInfoDef])
def upsert_robot_infos(infos: list[RobotInfoCreate], db: Session = Depends(get_db_session)):
    try:
        result_infos = []
        for info in infos:
            map_cfg = db.query(MapConfig).filter(MapConfig.id == info.map_id).first()
            if not map_cfg:
                raise HTTPException(status_code=404, detail=f"Map config {info.map_id} not found")

            existing = db.query(RobotInfo).filter(
                RobotInfo.map_id == info.map_id,
                RobotInfo.robot_name == info.robot_name,
            ).first()

            if existing:
                existing.initial_x = info.initial_x
                existing.initial_y = info.initial_y
                existing.initial_theta = info.initial_theta
                result_infos.append(existing)
            else:
                new_info = RobotInfo(
                    robot_name=info.robot_name,
                    initial_x=info.initial_x,
                    initial_y=info.initial_y,
                    initial_theta=info.initial_theta,
                    map_id=info.map_id,
                )
                db.add(new_info)
                result_infos.append(new_info)

        db.commit()
        for info in result_infos:
            db.refresh(info)

        return result_infos
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@register_router.put("/infos", response_model=list[RobotInfoDef])
def update_robot_infos(infos: list[RobotInfoUpdate], db: Session = Depends(get_db_session)):
    try:
        updated_infos = []
        for info in infos:
            existing = db.query(RobotInfo).filter(RobotInfo.id == info.id).first()
            if not existing:
                raise HTTPException(status_code=404, detail=f"Robot info with id {info.id} not found")

            map_cfg = db.query(MapConfig).filter(MapConfig.id == info.map_id).first()
            if not map_cfg:
                raise HTTPException(status_code=404, detail=f"Map config {info.map_id} not found")

            conflict = db.query(RobotInfo).filter(
                RobotInfo.map_id == info.map_id,
                RobotInfo.robot_name == info.robot_name,
                RobotInfo.id != info.id
            ).first()
            if conflict:
                raise HTTPException(status_code=409, detail=f"Another robot info with name '{info.robot_name}' already exists on map {info.map_id}")

            existing.robot_name = info.robot_name
            existing.initial_x = info.initial_x
            existing.initial_y = info.initial_y
            existing.initial_theta = info.initial_theta
            existing.map_id = info.map_id
            updated_infos.append(existing)

        db.commit()
        for info in updated_infos:
            db.refresh(info)

        return updated_infos
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@register_router.delete("/infos/bulk")
def delete_robot_infos_bulk(payload: RobotInfoBulkDelete, db: Session = Depends(get_db_session)):
    try:
        to_delete = []
        for rid in payload.ids:
            existing = db.query(RobotInfo).filter(RobotInfo.id == rid).first()
            if not existing:
                raise HTTPException(status_code=404, detail=f"Robot info with id {rid} not found")
            to_delete.append(existing)

        for info in to_delete:
            db.delete(info)
        db.commit()

        return {"deleted": payload.ids}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@register_router.delete("/infos")
def delete_robot_info(map_id: int, db: Session = Depends(get_db_session)):
    try:
        existing = db.query(RobotInfo).filter(RobotInfo.map_id == map_id).first()
        if not existing:
            raise HTTPException(status_code=404, detail="Robot info with this map ID not found")

        db.delete(existing)
        db.commit()
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
