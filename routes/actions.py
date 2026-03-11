from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from services.database import get_db_session
from services.generate_dock_yaml import generate_dock_yaml
from models.actions import NavigationActionRequest
from schema import DockConfig, Dock, RobotInfo, MapConfig

router = APIRouter(prefix="/actions", tags=["actions"])

@router.post("/run_nav")
def run_navigation(request: NavigationActionRequest, db: Session = Depends(get_db_session)):
    dock_config = db.query(DockConfig).filter(DockConfig.id == request.config_id).first()
    if not dock_config:
        raise HTTPException(status_code=404, detail="Dock configuration not found")
    
    docks = db.query(Dock).filter(Dock.config_id == request.config_id).all()
    if not docks:
        raise HTTPException(status_code=404, detail="No docks found for the specified configuration")
    
    generate_dock_yaml(docks, f"dock_database_{request.config_id}.yaml")

    return {"message": "Navigation action triggered"}