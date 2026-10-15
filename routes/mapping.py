import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from services.database import get_db_session
from services.dock_service import create_docks
from services.map_storage import store_map_files, remove_map_files
from services.redis_dock_runtime import redis_client_from_env
from schema import DockConfig
from models.dock_schema import DockCreate
from models.mapping import FinishMappingRequest, FinishMappingResponse

router = APIRouter(prefix="/mapping", tags=["mapping"])

redis_client = redis_client_from_env()

ROBOT_PORT = 8000
SAVE_MAP_TIMEOUT = 60.0
DOCKS_YAML_NAME = "docks.yaml"


@dataclass
class MapSnapshot:
    yaml_name: str
    yaml_bytes: bytes
    image_name: str
    image_bytes: bytes
    # aruco id -> [x, y, theta] (robot pose facing the marker)
    poses: dict[int, list[float]] = field(default_factory=dict)


def _get_robot_ip(hostname: str) -> str:
    raw = redis_client.get("robot_ips")
    try:
        ip_dict = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        ip_dict = {}

    entry = ip_dict.get(hostname) if isinstance(ip_dict, dict) else None
    ip = entry.get("ip") if isinstance(entry, dict) else None
    if not ip:
        raise HTTPException(status_code=404, detail=f"Robot '{hostname}' is not online")
    return ip


async def _request_save_map(robot_ip: str) -> httpx.Response:
    try:
        async with httpx.AsyncClient(timeout=SAVE_MAP_TIMEOUT) as client:
            response = await client.post(f"http://{robot_ip}:{ROBOT_PORT}/slam/save-map")
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach robot at {robot_ip}: {type(e).__name__}: {e}",
        )

    if not response.is_success:
        raise HTTPException(
            status_code=502,
            detail=f"Robot save-map failed with status {response.status_code}: {response.text[:500]}",
        )

    return response


def _parse_docks_yaml(content: bytes) -> dict[int, list[float]]:
    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=502, detail=f"Robot returned malformed docks.yaml: {e}")

    docks = data.get("docks") if isinstance(data, dict) else None
    if docks is None:
        docks = {}
    if not isinstance(docks, dict):
        raise HTTPException(status_code=502, detail="Robot returned malformed docks.yaml: 'docks' is not a mapping")

    poses: dict[int, list[float]] = {}
    for name, entry in docks.items():
        try:
            aruco_id = int(entry["aruco_markers"]["center"])
            pose = [float(v) for v in entry["pose"]]
            if len(pose) != 3:
                raise ValueError("pose must be [x, y, theta]")
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=502,
                detail=f"Robot returned malformed docks.yaml entry '{name}': {e}",
            )
        # Keep the first entry if the robot reports the same marker twice
        poses.setdefault(aruco_id, pose)

    return poses


def _parse_snapshot(content: bytes) -> MapSnapshot:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            files = {}
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Flatten any directories inside the archive (and drop path traversal)
                files[PurePosixPath(info.filename).name] = zf.read(info)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError) as e:
        raise HTTPException(status_code=502, detail=f"Robot returned an invalid map archive: {e}")

    map_yamls = [n for n in files if n.lower().endswith(".yaml") and n != DOCKS_YAML_NAME]
    images = [n for n in files if n.lower().endswith(".pgm")]

    if len(map_yamls) != 1:
        raise HTTPException(
            status_code=502,
            detail=f"Robot map archive must contain exactly one map yaml, found {sorted(map_yamls)}",
        )
    if len(images) != 1:
        raise HTTPException(
            status_code=502,
            detail=f"Robot map archive must contain exactly one .pgm image, found {sorted(images)}",
        )

    docks_yaml = files.get(DOCKS_YAML_NAME)

    return MapSnapshot(
        yaml_name=map_yamls[0],
        yaml_bytes=files[map_yamls[0]],
        image_name=images[0],
        image_bytes=files[images[0]],
        poses=_parse_docks_yaml(docks_yaml) if docks_yaml is not None else {},
    )


@router.post("/finish", response_model=FinishMappingResponse)
async def finish_mapping(payload: FinishMappingRequest, db: Session = Depends(get_db_session)):
    """Pull the map + detected docks from the robot and persist them as a new
    DockConfig (with its Docks) and MapConfig in a single transaction."""

    # ----------------------------
    # REQUEST VALIDATION
    # ----------------------------

    exists = db.query(DockConfig).filter(DockConfig.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=409, detail="Config name already exists")

    dock_ids = [d.dock_id for d in payload.docks]
    duplicate_dock_ids = sorted({d for d in dock_ids if dock_ids.count(d) > 1})
    if duplicate_dock_ids:
        raise HTTPException(status_code=422, detail=f"Duplicate dock_id in request: {duplicate_dock_ids}")

    aruco_ids = [d.aruco_id for d in payload.docks]
    duplicate_aruco_ids = sorted({a for a in aruco_ids if aruco_ids.count(a) > 1})
    if duplicate_aruco_ids:
        raise HTTPException(status_code=422, detail=f"Duplicate aruco_id in request: {duplicate_aruco_ids}")

    robot_ip = _get_robot_ip(payload.hostname)

    # End the read transaction so no DB connection is held during the robot call
    db.rollback()

    # ----------------------------
    # FETCH SNAPSHOT FROM ROBOT
    # ----------------------------

    response = await _request_save_map(robot_ip)
    snapshot = _parse_snapshot(response.content)

    missing = [a for a in aruco_ids if a not in snapshot.poses]
    if missing:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Requested aruco ids were not detected by the robot",
                "missing_aruco_ids": missing,
                "detected_aruco_ids": sorted(snapshot.poses),
                "docks_status": response.headers.get("X-Docks-Status"),
            },
        )

    unreviewed = sorted(set(snapshot.poses) - set(aruco_ids))

    # ----------------------------
    # SAVE (single transaction)
    # ----------------------------

    written_files: list[Path] = []
    try:
        cfg = DockConfig(name=payload.name, description=payload.description)
        db.add(cfg)
        db.flush()

        docks = create_docks(
            db,
            [
                DockCreate(
                    config_id=cfg.id,
                    dock_id=d.dock_id,
                    dock_type=d.dock_type,
                    aruco_id=d.aruco_id,
                    x=snapshot.poses[d.aruco_id][0],
                    y=snapshot.poses[d.aruco_id][1],
                    theta=snapshot.poses[d.aruco_id][2],
                )
                for d in payload.docks
            ],
            commit=False,
        )

        map_cfg = store_map_files(
            db,
            cfg.id,
            snapshot.yaml_name,
            snapshot.yaml_bytes,
            snapshot.image_name,
            snapshot.image_bytes,
            commit=False,
            written_files=written_files,
        )

        db.commit()
    except Exception as e:
        db.rollback()
        remove_map_files(written_files)
        if isinstance(e, IntegrityError):
            raise HTTPException(status_code=409, detail=f"Conflict while saving mapping result: {e.orig}")
        raise

    return FinishMappingResponse(
        dock_config_id=cfg.id,
        map_id=map_cfg.id,
        docks_created=len(docks),
        unreviewed=unreviewed,
    )
