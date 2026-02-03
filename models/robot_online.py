from pydantic import BaseModel, IPvAnyAddress

class RobotOnline(BaseModel):
    ip: IPvAnyAddress
    hostname: str

class RobotOnlineResponse(BaseModel):
    data: list[RobotOnline]