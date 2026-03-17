from enum import Enum

class DockType(Enum):
    PICKUP = "pickup"
    RECEIVER = "receiver"
    WAITING_ZONE = "waiting_zone"

class DockStatus(Enum):
    available = "available"
    reserved = "reserved"
    occupied = "occupied"