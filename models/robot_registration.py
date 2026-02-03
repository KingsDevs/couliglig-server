from pydantic import BaseModel, IPvAnyAddress
from typing import Optional

class RobotRegistration(BaseModel):
    hostname: str
    ip: IPvAnyAddress
    namespace: Optional[str] = None
    ros_domain_id: Optional[int] = None
