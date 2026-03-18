from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from services.database import get_db_session  # adjust
from schema import DockConfig, Dock
from models.dock_schema import DockConfigCreate, DockConfigUpdate, DockConfigOut, DockOut, DockUpdate, DockCreate
from models.dock_actions import AddItemRequest, RemoveItemRequest, ReserveDockRequest, OccupyDockRequest, ReleaseDockRequest
from services.redis_dock_runtime import redis_client_from_env, get_active_dock_config_id, clear_all_dock_keys, activate_docks, add_item_to_pickup_dock, remove_item_from_pickup_dock, release_dock, reserve_dock, occupy_dock, get_dock_state, get_all_dock_states

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
    active_config = get_active_dock_config_id(redis_client)
    configs = db.query(DockConfig).all()

    config_outs = []
    for cfg in configs:
        is_active = str(cfg.id) == active_config if active_config else False
        config_outs.append(DockConfigOut(
            id=cfg.id,
            name=cfg.name,
            description=cfg.description,
            is_active=is_active,
        ))
    
    return config_outs

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

        exists = db.query(Dock).filter(
            Dock.dock_id == p.dock_id,
            Dock.config_id == p.config_id
        ).first()

        if exists:
            raise HTTPException(
                status_code=409,
                detail="Dock ID already exists in the specified config"
            )
        
        exists_aruco = db.query(Dock).filter(
            Dock.aruco_id == p.aruco_id,
            Dock.config_id == p.config_id
        ).first()

        if exists_aruco:
            raise HTTPException(
                status_code=409,
                detail="Aruco ID already exists in the specified config"
            )

        dock = Dock(
            config_id=p.config_id,
            dock_id=p.dock_id,
            dock_type=p.dock_type,
            aruco_id=p.aruco_id,
            x=p.x,
            y=p.y,
            theta=p.theta,
        )

        db.add(dock)
        docks.append(dock)

    db.commit()

    for dock in docks:
        db.refresh(dock)

    return docks

@router.put("/update_config", response_model=DockConfigOut)
def update_config(config_id: int, payload: DockConfigUpdate, db: Session = Depends(get_db_session)):
    cfg = db.query(DockConfig).filter(DockConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")

    cfg.name = payload.name or cfg.name
    cfg.description = payload.description or cfg.description

    db.commit()
    db.refresh(cfg)
    return cfg

@router.put("/update_dock", response_model=DockOut)
def update_dock(dock_db_id: int, payload: DockUpdate, db: Session = Depends(get_db_session)):
    dock = db.query(Dock).filter(Dock.id == dock_db_id).first()
    if not dock:
        raise HTTPException(status_code=404, detail="Dock not found")

    if payload.dock_id:
        exists = db.query(Dock).filter(
            Dock.dock_id == payload.dock_id,
            Dock.config_id == dock.config_id,
            Dock.id != dock_db_id
        ).first()

        if exists:
            raise HTTPException(
                status_code=409,
                detail="Dock ID already exists in this config"
            )
        
    if payload.aruco_id is not None:
        exists_aruco = db.query(Dock).filter(
            Dock.aruco_id == payload.aruco_id,
            Dock.config_id == dock.config_id,
            Dock.id != dock_db_id
        ).first()

        if exists_aruco:
            raise HTTPException(
                status_code=409,
                detail="Aruco ID already exists in this config"
            )
        
    dock.dock_id = payload.dock_id or dock.dock_id
    dock.dock_type = payload.dock_type or dock.dock_type
    dock.aruco_id = payload.aruco_id if payload.aruco_id is not None else dock.aruco_id
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

@router.delete("/delete_dock")
def delete_dock(dock_db_id: int, db: Session = Depends(get_db_session)):
    dock = db.query(Dock).filter(Dock.id == dock_db_id).first()
    if not dock:
        raise HTTPException(status_code=404, detail="Dock not found")

    db.delete(dock)
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

    try:
        
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
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    
    return {"status": "ok"}


@router.post("/remove-item")
def remove_item(request: RemoveItemRequest):
    try:
        success = remove_item_from_pickup_dock(
            redis_client,
            request.dock_id
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Item mismatch or dock empty",
            )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    return {"status": "ok"}


@router.post("/reserve")
def reserve(request: ReserveDockRequest):
    try:
        success = reserve_dock(
            redis_client,
            request.dock_id,
            request.robot_id,
        )

        if not success:
            raise HTTPException(
                status_code=409,
                detail="Dock already reserved",
            )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    return {"status": "ok"}


@router.post("/occupy")
def occupy(request: OccupyDockRequest):
    try:
        occupy_dock(
            redis_client,
            request.dock_id,
            request.robot_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    return {"status": "ok"}


@router.post("/release")
def release(request: ReleaseDockRequest):
    try:
        release_dock(
            redis_client,
            request.dock_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    return {"status": "ok"}

@router.get("/dock_state")
def dock_state(dock_id: str):
    try:
        state = get_dock_state(redis_client, dock_id)
        if not state:
            raise HTTPException(
                status_code=404,
                detail="Dock not found",
            )

        return state
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@router.get("/all_dock_states")
def all_dock_states():
    states = get_all_dock_states(redis_client)

    return states

@router.post("/clear")
def clear_docks():

    clear_all_dock_keys(redis_client)

    return {"status": "ok"}

@router.get("/test/ui", response_class=HTMLResponse)
async def get_dock_form():
    with open("static/dock_test.html") as f:
        content = f.read()
    return HTMLResponse(content=content)