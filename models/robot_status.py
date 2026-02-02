from pydantic import BaseModel
from typing import List, Optional

class RobotStatus(BaseModel):
    hostname: str
    robot_namespace: str
    domain_id: Optional[int]
    ip: str
    uptime: str
    error: Optional[str] = None

class RobotStatusesResponse(BaseModel):
    status: str
    registrations: List[RobotStatus]