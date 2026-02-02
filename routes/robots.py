from fastapi import APIRouter, HTTPException
from models import RobotRegistration, RobotStatus, RobotStatusesResponse
from services import register_host
import redis
import json
from dotenv import load_dotenv
import os
import requests

load_dotenv()

register_router = APIRouter(prefix="/robots", tags=["robots"])

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"), 
    port=os.getenv("REDIS_PORT", 6379),       
    db=0,           
    decode_responses=True
)

redis_client.delete("robot_registrations")  # Clear the registrations on startup

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

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Requires root privileges"
        )
    except Exception as e:
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
        # Retrieve the dictionary of registrations from Redis
        existing_data = redis_client.get("robot_registrations")
        if existing_data:
            robot_dict = json.loads(existing_data)  # Deserialize JSON to a Python dictionary
        else:
            robot_dict = {}

        statuses = []

        # Loop through each hostname and IP in the dictionary
        for hostname, ip in robot_dict.items():
            try:
                url = f"http://{ip}:8000/status"  # Construct the URL for the robot's status endpoint
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    statuses.append(RobotStatus(hostname=hostname, ip=ip))
                else:
                    statuses.append(RobotStatus(
                        hostname=hostname,
                        ip=ip,
                        error=f"Failed to fetch status (HTTP {response.status_code})"
                    ))
            except requests.exceptions.RequestException as e:
                statuses.append(RobotStatus(
                    hostname=hostname,
                    ip=ip,
                    error=f"Could not connect to robot: {str(e)}"
                ))

        return RobotStatusesResponse(
            status="ok",
            registrations=statuses
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )