from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.redis_dock_runtime import (
    redis_client_from_env,
    set_runtime_config,
    update_picker_state,
    update_transporter_state,
    update_item_state,
    set_receiver_location,
    update_waiting_zone_state,
    get_picker_observation,
    get_transporter_observation,
    get_observation,
    set_agent_robot_mapping,
    get_all_agent_robot_mappings,
    sync_all_robot_positions,
    fetch_robot_transform,
    _get_robot_ip,
)

router = APIRouter(prefix="/obs", tags=["observations"])

redis_client = redis_client_from_env()


# ---------------------------------------------------------
# Request bodies
# ---------------------------------------------------------

class RuntimeConfigPayload(BaseModel):
    map_w: int
    map_h: int
    max_distance: float
    current_max_cap: float
    max_items: int
    max_pickers: int
    max_transporters: int
    max_wz: int


class PickerStatePayload(BaseModel):
    robot_id: str
    agent_index: int
    x: Optional[float] = None
    y: Optional[float] = None
    held_item: Optional[str] = ""


class TransporterStatePayload(BaseModel):
    robot_id: str
    agent_index: int
    x: Optional[float] = None
    y: Optional[float] = None
    capacity: int
    max_capacity: int
    in_waiting_zone: bool
    carried_items: list[int] = []


class ItemStatePayload(BaseModel):
    item_id: str
    item_index: int
    x: float
    y: float
    weight: float
    pickup_status: bool
    delivery_status: bool
    receiver_index: int


class ReceiverLocationPayload(BaseModel):
    receiver_index: int
    x: float
    y: float


class WaitingZonePayload(BaseModel):
    wz_id: str
    wz_index: int
    x: float
    y: float
    occupied: bool


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

@router.post("/config")
def post_runtime_config(payload: RuntimeConfigPayload):
    """Store the environment constants needed to build observations."""
    set_runtime_config(
        redis_client,
        map_w=payload.map_w,
        map_h=payload.map_h,
        max_distance=payload.max_distance,
        current_max_cap=payload.current_max_cap,
        max_items=payload.max_items,
        max_pickers=payload.max_pickers,
        max_transporters=payload.max_transporters,
        max_wz=payload.max_wz,
    )
    return {"status": "ok"}


# ---------------------------------------------------------
# State updates
# ---------------------------------------------------------

@router.post("/state/picker")
def post_picker_state(payload: PickerStatePayload):
    """Upsert a picker's live position and held-item state."""
    update_picker_state(
        redis_client,
        robot_id=payload.robot_id,
        agent_index=payload.agent_index,
        x=payload.x,
        y=payload.y,
        held_item=payload.held_item or "",
    )
    return {"status": "ok"}


@router.post("/state/transporter")
def post_transporter_state(payload: TransporterStatePayload):
    """Upsert a transporter's live position and cargo state."""
    update_transporter_state(
        redis_client,
        robot_id=payload.robot_id,
        agent_index=payload.agent_index,
        x=payload.x,
        y=payload.y,
        capacity=payload.capacity,
        max_capacity=payload.max_capacity,
        in_waiting_zone=payload.in_waiting_zone,
        carried_items=payload.carried_items,
    )
    return {"status": "ok"}


@router.post("/state/item")
def post_item_state(payload: ItemStatePayload):
    """Upsert an item's live position and delivery state."""
    update_item_state(
        redis_client,
        item_id=payload.item_id,
        item_index=payload.item_index,
        x=payload.x,
        y=payload.y,
        weight=payload.weight,
        pickup_status=payload.pickup_status,
        delivery_status=payload.delivery_status,
        receiver_index=payload.receiver_index,
    )
    return {"status": "ok"}


@router.post("/state/receiver")
def post_receiver_location(payload: ReceiverLocationPayload):
    """Store a receiver (delivery destination) position by its integer index."""
    set_receiver_location(
        redis_client,
        receiver_index=payload.receiver_index,
        x=payload.x,
        y=payload.y,
    )
    return {"status": "ok"}


@router.post("/state/waiting-zone")
def post_waiting_zone_state(payload: WaitingZonePayload):
    """Upsert a waiting zone's position and occupancy."""
    update_waiting_zone_state(
        redis_client,
        wz_id=payload.wz_id,
        wz_index=payload.wz_index,
        x=payload.x,
        y=payload.y,
        occupied=payload.occupied,
    )
    return {"status": "ok"}


# ---------------------------------------------------------
# Batch state updates
# ---------------------------------------------------------

@router.post("/state/pickers/batch")
def post_picker_states_batch(payloads: list[PickerStatePayload]):
    """Upsert multiple picker states in one call."""
    for payload in payloads:
        update_picker_state(
            redis_client,
            robot_id=payload.robot_id,
            agent_index=payload.agent_index,
            x=payload.x,
            y=payload.y,
            held_item=payload.held_item or "",
        )
    return {"status": "ok", "updated": len(payloads)}


@router.post("/state/transporters/batch")
def post_transporter_states_batch(payloads: list[TransporterStatePayload]):
    """Upsert multiple transporter states in one call."""
    for payload in payloads:
        update_transporter_state(
            redis_client,
            robot_id=payload.robot_id,
            agent_index=payload.agent_index,
            x=payload.x,
            y=payload.y,
            capacity=payload.capacity,
            max_capacity=payload.max_capacity,
            in_waiting_zone=payload.in_waiting_zone,
            carried_items=payload.carried_items,
        )
    return {"status": "ok", "updated": len(payloads)}


@router.post("/state/items/batch")
def post_item_states_batch(payloads: list[ItemStatePayload]):
    """Upsert multiple item states in one call."""
    for payload in payloads:
        update_item_state(
            redis_client,
            item_id=payload.item_id,
            item_index=payload.item_index,
            x=payload.x,
            y=payload.y,
            weight=payload.weight,
            pickup_status=payload.pickup_status,
            delivery_status=payload.delivery_status,
            receiver_index=payload.receiver_index,
        )
    return {"status": "ok", "updated": len(payloads)}


@router.post("/state/waiting-zones/batch")
def post_waiting_zone_states_batch(payloads: list[WaitingZonePayload]):
    """Upsert multiple waiting zone states in one call."""
    for payload in payloads:
        update_waiting_zone_state(
            redis_client,
            wz_id=payload.wz_id,
            wz_index=payload.wz_index,
            x=payload.x,
            y=payload.y,
            occupied=payload.occupied,
        )
    return {"status": "ok", "updated": len(payloads)}


# ---------------------------------------------------------
# Observation getters
# ---------------------------------------------------------

@router.get("/picker/{picker_id}")
def get_picker_obs(picker_id: str):
    """
    Return the picker observation dict matching the training environment exactly.
    Arrays are serialised to nested lists for JSON transport.
    """
    try:
        obs = get_picker_observation(redis_client, picker_id)
        return {k: v.tolist() for k, v in obs.items()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=503, detail=f"Runtime config missing: {e}")


@router.get("/transporter/{transporter_id}")
def get_transporter_obs(transporter_id: str):
    """
    Return the transporter observation dict matching the training environment exactly.
    Arrays are serialised to nested lists for JSON transport.
    """
    try:
        obs = get_transporter_observation(redis_client, transporter_id)
        return {k: v.tolist() for k, v in obs.items()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=503, detail=f"Runtime config missing: {e}")


@router.get("/agent/{agent_id}")
def get_agent_obs(agent_id: str):
    """
    Dispatch by agent prefix (picker_* / transporter_*) and return the
    matching observation dict, mirroring env.get_observation().
    """
    try:
        obs = get_observation(redis_client, agent_id)
        return {k: v.tolist() for k, v in obs.items()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=503, detail=f"Runtime config missing: {e}")


# ---------------------------------------------------------
# Agent → robot hostname mapping
# ---------------------------------------------------------

class AgentRobotMappingPayload(BaseModel):
    agent_id: str
    hostname: str


@router.post("/agent-mapping")
def post_agent_mapping(payload: AgentRobotMappingPayload):
    """Map a logical agent_id (e.g. 'picker_0') to a registered robot hostname."""
    set_agent_robot_mapping(redis_client, payload.agent_id, payload.hostname)
    return {"status": "ok"}


@router.post("/agent-mappings/batch")
def post_agent_mappings_batch(payloads: list[AgentRobotMappingPayload]):
    """Map multiple agent_ids to hostnames in one call."""
    for p in payloads:
        set_agent_robot_mapping(redis_client, p.agent_id, p.hostname)
    return {"status": "ok", "mapped": len(payloads)}


@router.get("/agent-mappings")
def get_agent_mappings():
    """Return the full agent_id → hostname mapping dict."""
    return get_all_agent_robot_mappings(redis_client)


# ---------------------------------------------------------
# Position sync from live robot transforms
# ---------------------------------------------------------

@router.post("/sync-positions")
def sync_positions():
    """
    Fetch /transform from every mapped robot and write x, y back into Redis.
    Returns a per-agent result showing status and fetched coordinates.
    """
    results = sync_all_robot_positions(redis_client)
    any_error = any(v["status"] == "error" for v in results.values())
    return {"results": results, "all_ok": not any_error}


@router.get("/transform/{agent_id}")
def get_live_transform(agent_id: str):
    """
    Fetch and return the live /transform for a single agent without writing to Redis.
    Useful for debugging or one-off position checks.
    """
    mappings = get_all_agent_robot_mappings(redis_client)
    hostname = mappings.get(agent_id)
    if not hostname:
        raise HTTPException(status_code=404, detail=f"No robot mapped to agent '{agent_id}'")

    ip = _get_robot_ip(redis_client, hostname)
    if not ip:
        raise HTTPException(status_code=404, detail=f"No IP found for hostname '{hostname}'")

    try:
        transform = fetch_robot_transform(ip)
        return {"agent_id": agent_id, "hostname": hostname, "ip": ip, **transform}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
