from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pathlib import Path
from models.maps_schema import MapConfigOut
import shutil

from services.database import get_db_session
from schema import MapConfig, DockConfig

router = APIRouter(prefix="/map", tags=["map"])

MAP_STORAGE = Path("db/maps")
MAP_STORAGE.mkdir(exist_ok=True)


@router.get("", response_model=list[MapConfigOut])
def get_maps(db: Session = Depends(get_db_session)):
    return db.query(MapConfig).all()

@router.post("/upload", response_model=MapConfigOut)
def upload_map(
    dock_config_id: int = Form(...),
    map_yaml: UploadFile = File(...),
    map_image: UploadFile = File(...),
    db: Session = Depends(get_db_session)
):

    cfg = db.query(DockConfig).filter(DockConfig.id == dock_config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Dock config not found")
    
    dock_config_id_exists = db.query(MapConfig).filter(MapConfig.dock_config_id == dock_config_id).first()
    if dock_config_id_exists:
        raise HTTPException(status_code=409, detail="A map config for the specified dock config already exists")

    yaml_filename = map_yaml.filename
    image_filename = map_image.filename

    # ----------------------------
    # DB VALIDATION
    # ----------------------------

    existing_yaml = db.query(MapConfig).filter(
        MapConfig.map_yaml_filename == yaml_filename
    ).first()

    if existing_yaml:
        raise HTTPException(
            status_code=409,
            detail="YAML filename already exists"
        )

    existing_image = db.query(MapConfig).filter(
        MapConfig.map_image_filename == image_filename
    ).first()

    if existing_image:
        raise HTTPException(
            status_code=409,
            detail="Image filename already exists"
        )

    # ----------------------------
    # FILESYSTEM VALIDATION
    # ----------------------------

    yaml_path = MAP_STORAGE / yaml_filename
    image_path = MAP_STORAGE / image_filename

    if yaml_path.exists():
        raise HTTPException(
            status_code=409,
            detail="YAML file already exists on disk"
        )

    if image_path.exists():
        raise HTTPException(
            status_code=409,
            detail="Image file already exists on disk"
        )

    # ----------------------------
    # SAVE FILES
    # ----------------------------

    with open(yaml_path, "wb") as buffer:
        shutil.copyfileobj(map_yaml.file, buffer)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(map_image.file, buffer)

    # ----------------------------
    # SAVE DB RECORD
    # ----------------------------

    map_cfg = MapConfig(
        dock_config_id=dock_config_id,
        map_yaml_filename=yaml_filename,
        map_image_filename=image_filename
    )

    db.add(map_cfg)
    db.commit()
    db.refresh(map_cfg)

    return map_cfg

@router.delete("/delete_map")
def delete_map(map_db_id: int, db: Session = Depends(get_db_session)):
    map_cfg = db.query(MapConfig).filter(MapConfig.id == map_db_id).first()
    if not map_cfg:
        raise HTTPException(status_code=404, detail="Map config not found")

    yaml_path = MAP_STORAGE / map_cfg.map_yaml_filename
    image_path = MAP_STORAGE / map_cfg.map_image_filename

    if yaml_path.exists():
        yaml_path.unlink()

    if image_path.exists():
        image_path.unlink()

    db.delete(map_cfg)
    db.commit()

    return {"success": True}

@router.get('/ui/upload_test')
def upload_test_ui():
    with open("static/map_upload_test.html") as f:
        content = f.read()
    return HTMLResponse(content=content)