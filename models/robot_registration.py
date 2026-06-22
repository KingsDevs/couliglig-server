from pydantic import BaseModel
from typing import Optional

class RobotRegistration(BaseModel):
    hostname: str
    ip: str
    namespace: Optional[str] = None
    ros_domain_id: Optional[int] = None
    battery: Optional[float] = None
