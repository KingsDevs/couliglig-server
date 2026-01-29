from pydantic import BaseModel

class RobotRegistration(BaseModel):
    hostname: str
    ip: str