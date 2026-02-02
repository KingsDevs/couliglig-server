from pydantic import BaseModel
from typing import List, Optional

class RobotStatus(BaseModel):
    hostname: str
    ip: str
    error: Optional[str] = None

class RobotStatusesResponse(BaseModel):
    status: str
    registrations: List[RobotStatus]