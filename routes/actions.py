from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from services.database import get_db_session
from services.generate_dock_yaml import generate_dock_yaml
from models.actions import NavigationActionRequest
from schema import DockConfig, Dock, RobotInfo, MapConfig
import asyncio
import httpx
import os

router = APIRouter(prefix="/actions", tags=["actions"])

async def call_robot_nav(robot_ip, robot_name, robot_info: RobotInfo, map: MapConfig, dock_buffer):
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:

            with open(map.map_yaml_path, "rb") as map_yaml, \
                 open(map.map_image_path, "rb") as map_pgm:

                files = {
                    "map_yaml": (os.path.basename(map.map_yaml_path), map_yaml, "application/x-yaml"),
                    "map_pgm": (os.path.basename(map.map_image_path), map_pgm, "image/x-portable-graymap"),
                    "dock_database": ("dock.yaml", dock_buffer, "application/x-yaml"),
                }
                
                data = {
                    "command": "start",
                    "enable_commander": True,
                    "x_val": robot_info.initial_x,
                    "y_val": robot_info.initial_y,
                }

                response = await client.post(
                    f"http://{robot_ip}:8000/roslib/nav",
                    data=data,
                    files=files
                )

        return {
            "robot": robot_name,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else None
        }

    except Exception as e:
        return {
            "robot": robot_name,
            "success": False,
            "error": str(e)
        }
    
async def call_robot_nav_stop(robot_ip, robot_name):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:

            response = await client.post(
                f"http://{robot_ip}:8000/roslib/nav",
                data={
                    "command": "stop",
                    "enable_commander": True,
                    "x_val": 0.0,
                    "y_val": 0.0
                }
            )

        return {
            "robot": robot_name,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else None
        }

    except Exception as e:
        return {
            "robot": robot_name,
            "success": False,
            "error": str(e)
        }
    
@router.post("/stop_nav")
async def stop_navigation(request: NavigationActionRequest):

    tasks = []

    for robot_data in request.robot_data:
        tasks.append(
            call_robot_nav_stop(
                robot_data.robot_ip,
                robot_data.namespace
            )
        )

    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed
    }


@router.post("/run_nav")
async def run_navigation(request: NavigationActionRequest, db: Session = Depends(get_db_session)):
    dock_config = db.query(DockConfig).filter(DockConfig.id == request.config_id).first()
    if not dock_config:
        raise HTTPException(status_code=404, detail="Dock configuration not found")
    
    docks = db.query(Dock).filter(Dock.config_id == request.config_id).all()
    if not docks:
        raise HTTPException(status_code=404, detail="No docks found for the specified configuration")
    
    map = db.query(MapConfig).filter(MapConfig.dock_config_id == request.config_id).one()
    if not map:
        raise HTTPException(status_code=404, detail="No map found for the specified dock configuration")
    
    robot_infos = db.query(RobotInfo).filter(RobotInfo.map_id == map.id).all()
    if not robot_infos:
        raise HTTPException(status_code=404, detail="No robot info found for the specified map")
    
    robot_info_dict = {info.robot_name: info for info in robot_infos}
    dock_buffer = generate_dock_yaml(docks, f"dock_database_{request.config_id}.yaml")

    tasks = []
    for robot_data in request.robot_data:
        robot_info = robot_info_dict.get(robot_data.namespace)
        if not robot_info:
            raise HTTPException(status_code=404, detail=f"No robot info found for namespace {robot_data.namespace}")

        tasks.append(call_robot_nav(robot_data.robot_ip, robot_data.namespace, robot_info, map, dock_buffer))

    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed
    }

