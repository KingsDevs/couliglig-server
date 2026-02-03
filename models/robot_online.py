from pydantic import BaseModel, IPvAnyAddress

class RobotOnline(BaseModel):
    ip: IPvAnyAddress
    hostname: str
    namespace: str = "couliglig"
    ros_domain_id: int = 0
    timestamp: str = ""

class RobotOnlineResponse(BaseModel):
    data: list[RobotOnline]