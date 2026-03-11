from pydantic import BaseModel

class RobotDataNavActionRequest(BaseModel):
    robot_ip: str
    namespace: str

class NavigationActionRequest(BaseModel):
    config_id: int
    robot_data: list[RobotDataNavActionRequest]
