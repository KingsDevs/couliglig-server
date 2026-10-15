from fastapi import HTTPException
from sqlalchemy.orm import Session
from pathlib import Path

from schema import MapConfig, DockConfig

MAP_STORAGE = Path("db/maps")
MAP_STORAGE.mkdir(exist_ok=True)


def store_map_files(
    db: Session,
    dock_config_id: int,
    yaml_name: str,
    yaml_bytes: bytes,
    image_name: str,
    image_bytes: bytes,
    commit: bool = True,
    storage_dir: Path | None = None,
    written_files: list[Path] | None = None,
) -> MapConfig:
    """Validate, write the map files to disk and add the MapConfig row.

    With ``commit=False`` the row is only flushed (so ``id`` is populated) and
    the caller owns the transaction. Every file this call writes is appended to
    ``written_files`` so a caller can remove them if its transaction fails
    later; files written by a call that itself fails are removed here.
    """
    storage = storage_dir if storage_dir is not None else MAP_STORAGE
    if written_files is None:
        written_files = []

    cfg = db.query(DockConfig).filter(DockConfig.id == dock_config_id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Dock config not found")

    dock_config_id_exists = db.query(MapConfig).filter(MapConfig.dock_config_id == dock_config_id).first()
    if dock_config_id_exists:
        raise HTTPException(status_code=409, detail="A map config for the specified dock config already exists")

    # ----------------------------
    # DB VALIDATION
    # ----------------------------

    existing_yaml = db.query(MapConfig).filter(
        MapConfig.map_yaml_filename == yaml_name
    ).first()

    if existing_yaml:
        raise HTTPException(
            status_code=409,
            detail="YAML filename already exists"
        )

    existing_image = db.query(MapConfig).filter(
        MapConfig.map_image_filename == image_name
    ).first()

    if existing_image:
        raise HTTPException(
            status_code=409,
            detail="Image filename already exists"
        )

    # ----------------------------
    # FILESYSTEM VALIDATION
    # ----------------------------

    yaml_path = storage / yaml_name
    image_path = storage / image_name

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
    # SAVE FILES + DB RECORD
    # ----------------------------

    written_here: list[Path] = []
    try:
        for path, content in ((yaml_path, yaml_bytes), (image_path, image_bytes)):
            with open(path, "wb") as buffer:
                written_here.append(path)
                written_files.append(path)
                buffer.write(content)

        map_cfg = MapConfig(
            dock_config_id=dock_config_id,
            map_yaml_filename=yaml_name,
            map_image_filename=image_name
        )

        db.add(map_cfg)
        if commit:
            db.commit()
            db.refresh(map_cfg)
        else:
            db.flush()
    except Exception:
        if commit:
            db.rollback()
        remove_map_files(written_here)
        raise

    return map_cfg


def remove_map_files(paths: list[Path]) -> None:
    """Best-effort removal of map files written by ``store_map_files``."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
