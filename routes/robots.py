
from fastapi import APIRouter, HTTPException
from models import RobotRegistration
from services import register_host
import redis
import json
from dotenv import load_dotenv
import os

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

@register_router.post("/register")
def register_robot(data: RobotRegistration):
    try:
        print(f"Registering host {data.hostname} with IP {data.ip}")
        register_host(data.hostname, data.ip)

        if is_couliglig_lan(data.hostname):
            print(f"Storing registration in Redis for {data.hostname}")
            existing_data = redis_client.get("robot_registrations")
            if existing_data:
                robot_list = json.loads(existing_data)
            else:
                robot_list = []

            robot_list.append((data.hostname, data.ip))
            redis_client.set("robot_registrations", json.dumps(robot_list)) 

        else:
            print(f"Hostname {data.hostname} is not a couliglig.lan address; skipping Redis storage.")

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

@register_router.get("/")
def get_robot_statuses():
    try:
        # Retrieve the list of tuples from Redis
        existing_data = redis_client.get("robot_registrations")
        print("Fetched from Redis:", existing_data)
        if existing_data:
            robot_list = json.loads(existing_data)  # Deserialize JSON to a Python list
        else:
            robot_list = []

        print(robot_list)

        return {
            "status": "ok",
            "registrations": robot_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )