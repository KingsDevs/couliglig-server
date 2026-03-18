from pydantic import BaseModel
from definitions import DockType

class AddItemRequest(BaseModel):
    dock_id: str
    item_id: str


class RemoveItemRequest(BaseModel):
    dock_id: str


class ReserveDockRequest(BaseModel):
    dock_id: str
    robot_id: str


class OccupyDockRequest(BaseModel):
    dock_id: str
    robot_id: str


class ReleaseDockRequest(BaseModel):
    dock_id: str