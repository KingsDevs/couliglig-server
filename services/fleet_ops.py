"""FLEET_MODE on the server: who decides what the robots do.

    legacy  (default) each robot's rl_executioner decides; docks live in Redis
            (services/redis_dock_runtime.py) and robots reserve/occupy/release them
    bus     the fleet coordinator (coordinator/) decides; the operator routes
            (activate, add-item, remove-item, /fleet/*) forward to it over the
            fleet.ops RPC, and the robots' dock lock routes are refused, so a robot
            still running the legacy stack cannot act beside the coordinator

The whole fleet switches together: set FLEET_MODE=bus here and on every robot.
"""
from __future__ import annotations

import asyncio
import os

from sqlalchemy.orm import Session

from definitions import DockType
from schema import Dock

LEGACY, BUS = "legacy", "bus"


class FleetUnavailable(RuntimeError):
    """The coordinator could not be reached or did not answer in time."""


class FleetRefused(ValueError):
    """The coordinator answered with ok: false."""


def fleet_mode() -> str:
    mode = os.getenv("FLEET_MODE", LEGACY).strip().lower()
    if mode not in (LEGACY, BUS):
        raise RuntimeError(f"FLEET_MODE must be {LEGACY!r} or {BUS!r}, not {mode!r}")
    return mode


def bus_mode() -> bool:
    return fleet_mode() == BUS


def amqp_url() -> str:
    return os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")


def call(op: str, timeout_s: float = 5.0, **args) -> dict:
    """One operator call to the coordinator, from a sync route (FastAPI's threadpool).

    Returns the coordinator's reply; raises FleetRefused for ok: false and
    FleetUnavailable when there is no answer within timeout_s (the request
    expires in the queue, so it never runs late).
    """
    from coordinator.service import OpsClient   # aio-pika only when bus mode is used

    async def run():
        async with OpsClient(amqp_url(), timeout_s=timeout_s) as ops:
            return await ops.call(op, **args)

    try:
        reply = asyncio.run(run())
    except asyncio.TimeoutError as e:
        raise FleetUnavailable(f"the coordinator did not answer {op!r} within {timeout_s:.0f} s") from e
    except (OSError, ConnectionError) as e:
        raise FleetUnavailable(f"cannot reach the broker: {e}") from e
    except Exception as e:  # noqa: BLE001 - aio-pika connection errors have their own classes
        if type(e).__module__.startswith(("aiormq", "aio_pika")):
            raise FleetUnavailable(f"broker error: {e}") from e
        raise
    if not reply.get("ok"):
        raise FleetRefused(reply.get("error") or f"{op} refused")
    return reply


def coordinator_docks(db: Session, config_id: int) -> list[dict]:
    """A dock config from SQLite, as the coordinator's activate op takes it.

    Hand-off docks sit on a transporter and move with it: their stored pose is
    unused (the commander finds them by TF), so a missing one is 0, 0.
    """
    out = []
    for d in db.query(Dock).filter_by(config_id=config_id).order_by(Dock.dock_id).all():
        if (d.x is None or d.y is None) and d.dock_type != DockType.HANDOFF:
            raise ValueError(f"dock {d.dock_id} has no position")
        out.append({"id": d.dock_id, "type": d.dock_type.value, "x": float(d.x or 0.0),
                    "y": float(d.y or 0.0), "yaw": float(d.theta or 0.0)})
    if not out:
        raise ValueError(f"dock config {config_id} has no docks")
    return out
