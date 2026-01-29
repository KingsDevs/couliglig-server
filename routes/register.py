
from fastapi import APIRouter, HTTPException
from models import RobotRegistration
from services import register_host

router = APIRouter(prefix="/robots", tags=["robots"])


@router.post("/register")
def register_robot(data: RobotRegistration):
    try:
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