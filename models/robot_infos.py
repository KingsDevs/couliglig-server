from pydantic import BaseModel


class RobotInfoDef(BaseModel):
    robot_name: str
    initial_x: float
    initial_y: float
    initial_theta: float  # radians
    map_id: int

    class Config:
        from_attributes = True

