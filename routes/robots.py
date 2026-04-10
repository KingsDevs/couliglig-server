import json
import asyncio
import logging
import redis
import os
import requests
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from models import RobotRegistration, RobotStatus, RobotStatusesResponse, RobotOnline, RobotOnlineResponse
from models.robot_infos import RobotInfoDef
from schema import RobotInfo, MapConfig
from services import register_host
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
    _dashboard_subscribers -= dead


def _register_robot_in_redis(hostname: str, ip: str, namespace: str, ros_domain_id: int):
    existing = redis_client.get("robot_ips")
    ip_dict = json.loads(existing) if existing else {}
    ip_dict[hostname] = {
        "hostname": hostname,
        "ip": ip,
        "namespace": namespace,
        "ros_domain_id": ros_domain_id,
        "timestamp": datetime.now().isoformat() + "Z",
    }
    redis_client.set("robot_ips", json.dumps(ip_dict))


def _update_robot_timestamp(hostname: str):
    existing = redis_client.get("robot_ips")
    if not existing:
        return
    ip_dict = json.loads(existing)
    if hostname in ip_dict:
        ip_dict[hostname]["timestamp"] = datetime.now().isoformat() + "Z"
        redis_client.set("robot_ips", json.dumps(ip_dict))


def _remove_robot_from_redis(hostname: str):
    existing = redis_client.get("robot_ips")
    if not existing:
        return
    ip_dict = json.loads(existing)
    if hostname in ip_dict:
        ip_dict.pop(hostname)
        redis_client.set("robot_ips", json.dumps(ip_dict))
        logging.info(f"[robot-ws] {hostname} removed from robot_ips")


@register_router.websocket("/ws")
async def robot_websocket(websocket: WebSocket):
    await websocket.accept()
    hostname = None
    try:
        data = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        hostname = data["hostname"]
        ip = data["ip"]
        namespace = data.get("namespace", "couliglig")
        ros_domain_id = data.get("ros_domain_id", 0)

        _register_robot_in_redis(hostname, ip, namespace, ros_domain_id)
        logging.info(f"[robot-ws] {hostname} ({ip}) connected")
        await _broadcast_online()

        while True:
            msg = await asyncio.wait_for(websocket.receive_json(), timeout=15)
            if msg.get("type") == "heartbeat":
                _update_robot_timestamp(hostname)
                logging.debug(f"[robot-ws] {hostname} heartbeat")

    except (WebSocketDisconnect, asyncio.TimeoutError) as e:
        logging.warning(f"[robot-ws] {hostname} disconnected ({type(e).__name__})")
    except Exception as e:
        logging.warning(f"[robot-ws] {hostname} error ({e})")
    finally:
        if hostname:
            _remove_robot_from_redis(hostname)
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

            existing_ips = redis_client.get("robot_ips")
            if existing_ips:
                ip_dict = json.loads(existing_ips)
            else:
                ip_dict = {}

            ip_dict[data.hostname] = {
                "hostname": data.hostname,
                "ip": str(data.ip),
                "namespace": data.namespace or "couliglig",
                "ros_domain_id": data.ros_domain_id or 0,
                "timestamp": datetime.now().isoformat() + "Z"
            }

            redis_client.set("robot_ips", json.dumps(ip_dict))

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
                    timestamp=payload.get("timestamp", "")
                )
            )

        return RobotOnlineResponse(data=online_robots)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@register_router.get("/infos", response_model=list[RobotInfoDef])
def get_robot_infos(db: Session = Depends(get_db_session)):
    try:
        infos = db.query(RobotInfo).all()
        return infos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@register_router.post("/infos", response_model=list[RobotInfoDef])
def create_robot_info(infos: list[RobotInfoDef], db: Session = Depends(get_db_session)):
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
