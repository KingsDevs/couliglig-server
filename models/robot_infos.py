from pydantic import BaseModel


class RobotInfoDef(BaseModel):
    id: int
    robot_name: str
    initial_x: float
    initial_y: float
    initial_theta: float  # radians
    map_id: int

    class Config:
        from_attributes = True


class RobotInfoCreate(BaseModel):
    robot_name: str
    initial_x: float
    initial_y: float
    initial_theta: float  # radians
    map_id: int

    class Config:
        from_attributes = True


class RobotInfoUpdate(BaseModel):
    id: int
    robot_name: str
    initial_x: float
    initial_y: float
    initial_theta: float  # radians
    map_id: int

    class Config:
        from_attributes = True


class RobotInfoBulkDelete(BaseModel):
    ids: list[int]

