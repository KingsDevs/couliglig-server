from pydantic import BaseModel, Field
from typing import Optional, List
from definitions import DockType


class FinishMappingDock(BaseModel):
    aruco_id: int
    dock_id: str = Field(min_length=1)
    dock_type: DockType


class FinishMappingRequest(BaseModel):
    hostname: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: Optional[str] = None
    docks: List[FinishMappingDock]


class FinishMappingResponse(BaseModel):
    dock_config_id: int
    map_id: int
    docks_created: int
    unreviewed: List[int] = []
