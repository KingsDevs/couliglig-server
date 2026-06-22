from pydantic import BaseModel, IPvAnyAddress
from typing import Optional

class RobotOnline(BaseModel):
    ip: IPvAnyAddress
    hostname: str
    namespace: str = "couliglig"
    ros_domain_id: int = 0
    timestamp: str = ""
    battery: Optional[float] = None

class RobotOnlineResponse(BaseModel):
    data: list[RobotOnline]