
from fastapi import APIRouter, HTTPException
from models import RobotRegistration
from services import register_host

register_router = APIRouter(prefix="/robots", tags=["robots"])


@register_router.post("/register")
def register_robot(data: RobotRegistration):
    try:
        print(f"Registering host {data.hostname} with IP {data.ip}")
        register_host(data.hostname, data.ip)
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
    return {
        "status": "ok",
        "message": "Robot registration endpoint is active."
    }