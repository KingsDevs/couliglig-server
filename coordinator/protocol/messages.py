"""Message payloads between the coordinator and the robots' fleet_agent.

Every message travels in an Envelope (envelope.py). This module is the
contract both sides compile against; fleet_agent keeps a copy, and
PROTOCOL_HASH (envelope.py) tells the coordinator when the copies differ.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(str, Enum):
    PICKER = "picker"
    TRANSPORTER = "transporter"


class ExecState(str, Enum):
    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    HOLD = "HOLD"              # parked; waiting for the coordinator
    SAFE_STOP = "SAFE_STOP"    # stopped where it was, finishing the leg was unsafe
    ESTOP = "ESTOP"
    MANUAL = "MANUAL"


class TaskStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    STARTED = "STARTED"
    ARRIVED = "ARRIVED"
    LIFT_UP = "LIFT_UP"
    LIFT_DOWN = "LIFT_DOWN"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


TERMINAL = {TaskStatus.REJECTED, TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}


class TaskKind(str, Enum):
    PICK = "PICK"
    HANDOFF_PLACE = "HANDOFF_PLACE"
    PARK_ZONE = "PARK_ZONE"
    DELIVER = "DELIVER"
    GO_HOME = "GO_HOME"
    HOLD = "HOLD"


class Lift(str, Enum):
    NONE = "NONE"
    UP = "UP"
    DOWN = "DOWN"


class LiftState(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class CancelMode(str, Enum):
    STOP_NOW = "STOP_NOW"
    FINISH_LEG = "FINISH_LEG"


class Pose(_Msg):
    x: float
    y: float
    yaw: float = 0.0
    frame: str = "map"
    stamp: float = 0.0          # robot clock, seconds
    loc_ok: bool = True


class CurrentTask(_Msg):
    task_id: str
    attempt: int
    leg: int = 0
    status: TaskStatus


# ---------------- robot -> coordinator ----------------

class Hello(_Msg):
    robot_type: str                     # small, big, aloha, cart
    role: Role
    capacity: float = 4.0               # transporter slots; ignored for pickers
    map_hash: str = ""
    dock_db_hash: str = ""
    protocol_hash: str
    current_task: Optional[CurrentTask] = None
    lift_state: LiftState = LiftState.UNKNOWN
    pose: Optional[Pose] = None
    software: str = ""


class Heartbeat(_Msg):
    pose: Pose
    exec_state: ExecState
    task_id: Optional[str] = None
    leg: int = 0
    leg_progress_m: float = 0.0
    battery_pct: Optional[float] = None
    nav2_active: bool = True


class Event(_Msg):
    event_id: str
    task_id: str
    attempt: int
    leg: int = 0
    status: TaskStatus
    reason: str = ""
    pose: Optional[Pose] = None


# ---------------- coordinator -> robot ----------------

class Welcome(_Msg):
    epoch: int
    accepted: bool
    hello_id: str = ""                 # msg_id of the hello this answers; robots ignore others
    reason: str = ""
    task_to_resume: Optional[str] = None


class Leg(_Msg):
    dock_id: str
    pose: Pose
    lift: Lift = Lift.NONE


class TaskCommand(_Msg):
    task_id: str
    attempt: int
    kind: TaskKind
    legs: list[Leg] = Field(min_length=1)
    valid_until: float                 # coordinator clock, seconds; reject if past
    leg_timeout_s: float


class CancelCommand(_Msg):
    task_id: str
    mode: CancelMode = CancelMode.STOP_NOW


class CoordinatorHeartbeat(_Msg):
    epoch: int
    policy: str
    paused: bool = False


class EStop(_Msg):
    reason: str = ""
    issued_by: str = ""


class Resume(_Msg):
    issued_by: str = ""


# message type -> payload model
PAYLOADS: dict[str, type[_Msg]] = {
    "robot.hello": Hello,
    "robot.heartbeat": Heartbeat,
    "robot.event": Event,
    "coordinator.welcome": Welcome,
    "command.task": TaskCommand,
    "command.cancel": CancelCommand,
    "coordinator.heartbeat": CoordinatorHeartbeat,
    "coordinator.estop": EStop,
    "coordinator.resume": Resume,
}
