from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base


class RobotInfo(Base):
    __tablename__ = "robot_info"

    id = Column(Integer, primary_key=True)
    robot_name = Column(String, unique=True, nullable=False)
    initial_x = Column(Float, nullable=False)
    initial_y = Column(Float, nullable=False)
    initial_theta = Column(Float, default=0.0, nullable=False) # radians
    map_id = Column(Integer, ForeignKey("map_configs.id"), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())