from pydantic import BaseModel
from definitions import DockType

class AddItemRequest(BaseModel):
    dock_id: str
    item_id: str
    receiver_dock_id: str
    item_weight: float | None = None  # Optional weight for the item being added


class RemoveItemRequest(BaseModel):
    dock_id: str


class ReserveDockRequest(BaseModel):
    dock_id: str
    robot_id: str


class OccupyDockRequest(BaseModel):
    dock_id: str
    robot_id: str
    # Hand-off docks only: the item now sitting on the transporter's slot
    item_id: str | None = None


class RenewDockRequest(BaseModel):
    dock_id: str
    robot_id: str


class ReleaseDockRequest(BaseModel):
    dock_id: str
    robot_id: str
    # Hand-off docks only: lets the transporter that delivered this item
    # release the slot even though the picker holds the lock
    item_id: str | None = None
