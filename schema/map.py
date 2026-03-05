from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base

class MapConfig(Base):
    __tablename__ = "map_configs"

    id = Column(Integer, primary_key=True)
    dock_config_id = Column(Integer, ForeignKey("dock_configs.id"), nullable=False, unique=True)
    map_yaml_filename = Column(String, unique=True, nullable=False)
    map_image_filename = Column(String, unique=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    @property
    def map_yaml_path(self):
        return f"db/maps/{self.map_yaml_filename}"

    @property
    def map_image_path(self):
        return f"db/maps/{self.map_image_filename}"