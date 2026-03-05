from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.database import get_db_session  # adjust
from schema import DockConfig, Dock
from models.dock_schema import DockConfigCreate, DockConfigOut, DockOut, DockUpdate, DockCreate
from models.dock_actions import AddItemRequest, RemoveItemRequest, ReserveDockRequest, OccupyDockRequest, ReleaseDockRequest
from services import redis_client_from_env, clear_all_dock_keys, activate_docks, add_item_to_pickup_dock, remove_item_from_pickup_dock, release_dock, reserve_dock, occupy_dock, get_dock_state

router = APIRouter(prefix="/dock", tags=["dock"])

redis_client = redis_client_from_env()

@router.post("", response_model=DockConfigOut)
def create_config(payload: DockConfigCreate, db: Session = Depends(get_db_session)):
    exists = db.query(DockConfig).filter(DockConfig.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=409, detail="Config name already exists")

    cfg = DockConfig(name=payload.name, description=payload.description)
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg


@router.get("/configs", response_model=list[DockConfigOut])
def list_configs(db: Session = Depends(get_db_session)):
    return db.query(DockConfig).all()

@router.get("", response_model=list[DockOut])
def list_docks(config_id: int, db: Session = Depends(get_db_session)):
    return db.query(Dock).filter(Dock.config_id == config_id).all()

@router.post("/create_dock", response_model=list[DockOut])
def create_dock(payload: list[DockCreate], db: Session = Depends(get_db_session)):

    docks = []
    for p in payload:
        cfg = db.query(DockConfig).filter(DockConfig.id == p.config_id).first()
        if not cfg:
            raise HTTPException(status_code=404, detail="Config not found")

        exists = db.query(Dock).filter(Dock.dock_id == p.dock_id).first()
        if exists:
            raise HTTPException(status_code=409, detail="Dock ID already exists")

        dock = Dock(
            config_id=p.config_id,
            dock_id=p.dock_id,
            dock_type=p.dock_type,
            x=p.x,
            y=p.y,
            theta=p.theta,
        )

        db.add(dock)
        db.commit()
        db.refresh(dock)

        docks.append(dock)

    return docks

@router.put("/update_dock", response_model=DockOut)
def update_dock(dock_db_id: int, payload: DockUpdate, db: Session = Depends(get_db_session)):
    dock = db.query(Dock).filter(Dock.id == dock_db_id).first()
    if not dock:
        raise HTTPException(status_code=404, detail="Dock not found")

    dock.dock_type = payload.dock_type or dock.dock_type
    dock.x = payload.x if payload.x is not None else dock.x
    dock.y = payload.y if payload.y is not None else dock.y
    dock.theta = payload.theta if payload.theta is not None else dock.theta

    db.commit()
    db.refresh(dock)
    return dock

@router.delete("/delete_config")
def delete_config(config_id: int, db: Session = Depends(get_db_session)):
    cfg = db.query(DockConfig).filter(DockConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")

    db.delete(cfg)
    db.commit()

    return {"success": True}


@router.post("/activate")
def activate_config(config_id: int, db: Session = Depends(get_db_session)):
    cfg = db.query(DockConfig).filter(DockConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")

    activate_docks(redis_client, db, config_id)

    return {"success": True, "active_config_id": cfg.id}

@router.post("/add-item")
def add_item(request: AddItemRequest):

    success = add_item_to_pickup_dock(
        redis_client,
        request.dock_id,
        request.item_id,
    )

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Dock does not exist or already has an item",
        )

    return {"status": "ok"}


@router.post("/remove-item")
def remove_item(request: RemoveItemRequest):

    success = remove_item_from_pickup_dock(
        redis_client,
        request.dock_id,
        request.item_id,
    )

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Item mismatch or dock empty",
        )

    return {"status": "ok"}


@router.post("/reserve")
def reserve(request: ReserveDockRequest):

    success = reserve_dock(
        redis_client,
        request.dock_type,
        request.dock_id,
        request.robot_id,
    )

    if not success:
        raise HTTPException(
            status_code=409,
            detail="Dock already reserved",
        )

    return {"status": "ok"}


@router.post("/occupy")
def occupy(request: OccupyDockRequest):

    occupy_dock(
        redis_client,
        request.dock_type,
        request.dock_id,
        request.robot_id,
    )

    return {"status": "ok"}


@router.post("/release")
def release(request: ReleaseDockRequest):

    release_dock(
        redis_client,
        request.dock_type,
        request.dock_id,
    )

    return {"status": "ok"}

@router.get("/dock_state")
def dock_state(dock_type: str, dock_id: str):

    state = get_dock_state(redis_client, dock_type, dock_id)

    if not state:
        raise HTTPException(
            status_code=404,
            detail="Dock not found",
        )

    return state

@router.post("/clear")
def clear_docks():

    clear_all_dock_keys(redis_client)

    return {"status": "ok"}

