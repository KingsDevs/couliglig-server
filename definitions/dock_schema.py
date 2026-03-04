from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from .dock_types import DockType


class DockConfigCreate(BaseModel):
    name: str = Field(min_length=1)
    description: Optional[str] = None


class DockConfigOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


class DockCreate(BaseModel):
    config_id: int
    dock_id: str = Field(min_length=1)
    dock_type: DockType
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class DockUpdate(BaseModel):
    dock_type: Optional[DockType] = None
    x: Optional[float] = None
    y: Optional[float] = None
    theta: Optional[float] = None
    frame_id: Optional[str] = None
    enabled: Optional[int] = None


class DockOut(BaseModel):
    id: int
    config_id: int
    dock_id: str
    dock_type: DockType
    x: float
    y: float
    theta: float

    class Config:
        from_attributes = True