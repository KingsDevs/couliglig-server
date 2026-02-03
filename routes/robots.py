import json
import redis
import os
import requests
from fastapi import APIRouter, HTTPException
from models import RobotRegistration, RobotStatus, RobotStatusesResponse, RobotOnline, RobotOnlineResponse
from services import register_host
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

register_router = APIRouter(prefix="/robots", tags=["robots"])

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"), 
    port=os.getenv("REDIS_PORT", 6379),       
    db=0,           
    decode_responses=True
)

redis_client.delete("robot_registrations")  # Clear the registrations on startup
# redis_client.delete("robot_ips")  # Clear the IPs on startup

def is_couliglig_lan(name: str) -> bool:
    return name.startswith("couliglig")

@register_router.post("/register")
def register_robot(data: RobotRegistration):
    try:
        print(f"Registering host {data.hostname} with IP {data.ip}")
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
                    namespace=payload.get("namespace", "couliglig"),
                    ros_domain_id=payload.get("ros_domain_id", 0),
                    timestamp=payload.get("timestamp", "")
                )
            )

        return RobotOnlineResponse(data=online_robots)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
