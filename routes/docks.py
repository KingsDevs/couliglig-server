from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from services.database import get_db_session  # adjust
from schema import DockConfig, Dock
from definitions import DockConfigCreate, DockConfigOut
from .redis_runtime import redis_client_from_env, clear_all_dock_keys, init_runtime_for_docks

router = APIRouter(prefix="/dock-configs", tags=["dock-configs"])


@router.post("", response_model=DockConfigOut)
def create_config(payload: DockConfigCreate, db: Session = Depends(get_db_session)):
    exists = db.query(DockConfig).filter(DockConfig.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=409, detail="Config name already exists")

    cfg = DockConfig(name=payload.name, description=payload.description, is_active=0)
    db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg


@router.get("", response_model=list[DockConfigOut])
def list_configs(db: Session = Depends(get_db_session)):
    return db.query(DockConfig).order_by(DockConfig.id.desc()).all()


@router.delete("/{config_id}")
def delete_config(config_id: int, db: Session = Depends(get_db_session)):
    cfg = db.query(DockConfig).filter(DockConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")

    # if deleting active config, you may want to clear redis too
    was_active = int(cfg.is_active) == 1

    db.delete(cfg)
    db.commit()

    if was_active:
        r = redis_client_from_env()
        clear_all_dock_keys(r)

    return {"success": True}


@router.post("/{config_id}/activate")
def activate_config(config_id: int, db: Session = Depends(get_db_session)):
    cfg = db.query(DockConfig).filter(DockConfig.id == config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Config not found")

    # 1) Make single active
    db.query(DockConfig).update({DockConfig.is_active: 0})
    db.commit()

    # 2) Reset redis runtime for active config
    r = redis_client_from_env()
    clear_all_dock_keys(r)

    docks = db.query(Dock).filter(Dock.config_id == cfg.id).all()
    init_runtime_for_docks(r, docks)

    return {"success": True, "active_config_id": cfg.id, "dock_count": len(docks)}


@router.get("/active", response_model=DockConfigOut)
def get_active_config(db: Session = Depends(get_db_session)):
    cfg = db.query(DockConfig).filter(DockConfig.is_active == 1).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="No active config")
    return cfg