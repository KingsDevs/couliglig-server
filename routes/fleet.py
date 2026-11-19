"""Fleet coordinator operator calls (FLEET_MODE=bus): state, pause, e-stop, resume, policy.

Items and dock layouts go through the usual /dock/add-item, /dock/remove-item and
/dock/activate, which forward to the coordinator in bus mode.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import fleet_ops

router = APIRouter(prefix="/fleet", tags=["fleet"])


class EStopRequest(BaseModel):
    reason: str = ""


class PolicyRequest(BaseModel):
    policy: str          # stay | greedy | marl_onnx (GET /fleet/snapshot lists what is available)


def _call(op: str, **args) -> dict:
    if not fleet_ops.bus_mode():
        raise HTTPException(status_code=409, detail="FLEET_MODE=legacy: there is no fleet coordinator")
    try:
        return fleet_ops.call(op, **args)
    except fleet_ops.FleetRefused as e:
        raise HTTPException(status_code=400, detail=str(e))
    except fleet_ops.FleetUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/mode")
def mode():
    return {"fleet_mode": fleet_ops.fleet_mode()}


@router.get("/snapshot")
def snapshot():
    """The coordinator's world: robots, items, tasks, docks, policy, metrics."""
    return _call("snapshot")


@router.post("/pause")
def pause():
    """No new tasks; running ones finish."""
    return _call("pause")


@router.post("/resume")
def resume():
    """End a pause or an e-stop."""
    return _call("resume")


@router.post("/policy")
def set_policy(request: PolicyRequest):
    """Who decides in auto: a heuristic (stay, greedy) or the RL model. Kept across
    coordinator restarts; running tasks finish as they were planned."""
    return _call("set_policy", policy=request.policy)


@router.post("/estop")
def estop(request: EStopRequest):
    """Every robot stops now and cancels its task; rounds pause until /fleet/resume."""
    return _call("estop", reason=request.reason)
