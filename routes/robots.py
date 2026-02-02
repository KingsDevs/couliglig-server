
from fastapi import APIRouter, HTTPException
from models import RobotRegistration
from services import register_host
import redis
import json

register_router = APIRouter(prefix="/robots", tags=["robots"])

redis_client = redis.StrictRedis(
    host='localhost',  
    port=6379,        
    db=0,           
    decode_responses=True
)

def is_couliglig_lan(name: str) -> bool:
    return name.startswith("couliglig") and name.endswith(".lan")

@register_router.post("/register")
def register_robot(data: RobotRegistration):
    try:
        print(f"Registering host {data.hostname} with IP {data.ip}")
        register_host(data.hostname, data.ip)

        if is_couliglig_lan(data.hostname):
            existing_data = redis_client.get("robot_registrations")
            if existing_data:
                robot_list = json.loads(existing_data)
            else:
                robot_list = []

            robot_list.append((data.hostname, data.ip))
            redis_client.set("robot_registrations", json.dumps(robot_list)) 

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