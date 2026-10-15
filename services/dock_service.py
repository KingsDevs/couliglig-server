from fastapi import HTTPException
from sqlalchemy.orm import Session

from schema import DockConfig, Dock
from models.dock_schema import DockCreate


def create_docks(db: Session, docks: list[DockCreate], commit: bool = True) -> list[Dock]:
    """Validate and add Dock rows.

    With ``commit=False`` the rows are only flushed (so ``id`` is populated)
    and the caller owns the transaction.
    """
    created = []

    for p in docks:

        cfg = db.query(DockConfig).filter(DockConfig.id == p.config_id).first()
        if not cfg:
            raise HTTPException(status_code=404, detail="Config not found")

        exists = db.query(Dock).filter(
            Dock.dock_id == p.dock_id,
            Dock.config_id == p.config_id
        ).first()

        if exists:
            raise HTTPException(
                status_code=409,
                detail="Dock ID already exists in the specified config"
            )

        exists_aruco = db.query(Dock).filter(
            Dock.aruco_id == p.aruco_id,
            Dock.config_id == p.config_id
        ).first()

        if exists_aruco:
            raise HTTPException(
                status_code=409,
                detail="Aruco ID already exists in the specified config"
            )

        dock = Dock(
            config_id=p.config_id,
            dock_id=p.dock_id,
            dock_type=p.dock_type,
            aruco_id=p.aruco_id,
            x=p.x,
            y=p.y,
            theta=p.theta,
        )

        db.add(dock)
        created.append(dock)

    if commit:
        db.commit()
        for dock in created:
            db.refresh(dock)
    else:
        db.flush()

    return created
