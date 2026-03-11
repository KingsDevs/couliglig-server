from pydantic import BaseModel

class RobotDataNavActionRequest(BaseModel):
    robot_ip: str
    robot_name: str

class NavigationActionRequest(BaseModel):
    config_id: int
    robot_data: list[RobotDataNavActionRequest]
