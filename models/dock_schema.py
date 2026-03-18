from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from definitions import DockType

class DockConfigCreate(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = None

class DockConfigUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class DockConfigOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    is_active: bool = False

    class Config:
        from_attributes = True


class DockCreate(BaseModel):
    config_id: int
    dock_id: str = Field(min_length=1)
    dock_type: DockType
    aruco_id: int = 1
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class DockUpdate(BaseModel):
    dock_id: Optional[str] = None
    dock_type: Optional[DockType] = None
    aruco_id: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    theta: Optional[float] = None

class DockOut(BaseModel):
    id: int
    config_id: int
    dock_id: str
    dock_type: DockType
    aruco_id: int
    x: float
    y: float
    theta: float

    class Config:
        from_attributes = True