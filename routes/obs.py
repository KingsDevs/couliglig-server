from fastapi import APIRouter, HTTPException
from services.redis_dock_runtime import redis_client_from_env, get_obs_builder_inputs

router = APIRouter(prefix="/obs", tags=["obs"])

redis_client = redis_client_from_env()


@router.get("/builder-inputs")
def obs_builder_inputs(robot_id: str):
    """
    Returns all live runtime data from Redis needed to build observations:
    dock states, dock positions (x, y, yaw), and any other active runtime fields.
    """
    try:
        return get_obs_builder_inputs(redis_client, robot_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
