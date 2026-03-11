from pydantic import BaseModel

class NavigationActionRequest(BaseModel):
    config_id: int
    robot_ips: list[str]
