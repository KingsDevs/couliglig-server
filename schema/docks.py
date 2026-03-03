from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base import Base
from sqlalchemy import Enum

from definitions import DockType


class DockConfig(Base):
    __tablename__ = "dock_configs"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    docks = relationship("Dock", back_populates="config", cascade="all, delete-orphan")


class Dock(Base):
    __tablename__ = "docks"

    id = Column(Integer, primary_key=True)
    config_id = Column(Integer, ForeignKey("dock_configs.id"), nullable=False)

    dock_id = Column(String, nullable=False)
    dock_type = Column(Enum(DockType, name="dock_type_enum"), nullable=False)

    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    theta = Column(Float, default=0.0, nullable=True)

    config = relationship("DockConfig", back_populates="docks")

    __table_args__ = (
        UniqueConstraint("config_id", "dock_id", name="uq_docks_configid_dockid"),
    )