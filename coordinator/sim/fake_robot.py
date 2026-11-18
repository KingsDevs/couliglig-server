"""A simulated robot that speaks the fleet protocol (the fleet_agent contract).

Transport-free: receive(envelope) and step(dt) return the envelopes the robot
sends. SimFleet (fleet.py) wires robots to a Coordinator in-process; amqp.py
runs the same robot over RabbitMQ.

Behaviour the coordinator relies on, as fleet_agent will implement it:
  - hello on start and whenever the coordinator's epoch changes
  - heartbeat at 1 Hz and on every state change; seq monotonic per session
  - a command.task is refused (REJECTED) when it is expired, from an older
    epoch, or the robot is busy; a repeated (task_id, attempt) is ignored
  - events: ACCEPTED, STARTED, ARRIVED per leg, LIFT_UP / LIFT_DOWN, SUCCEEDED
    or FAILED / CANCELED; every event keeps its event_id until it is sent
  - watchdog: no coordinator heartbeat for hold_after_s -> finish the current
    leg, then HOLD (no new legs) until the coordinator is heard again
  - estop: stop now (ESTOP) until resume
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from coordinator.grid import Grid, GridDistance
from coordinator.protocol import PROTOCOL_HASH, Envelope, make
from coordinator.protocol import messages as m


@dataclass
class _Task:
    cmd: m.TaskCommand
    leg: int = 0
    path: list = field(default_factory=list)
    progress: float = 0.0          # cells travelled along path


class FakeRobot:
    def __init__(self, robot_id: str, role: str, cell, grid: Grid, clock: Callable[[], float],
                 distance: GridDistance | None = None, speed_cells_s: float = 2.0, capacity: float = 4.0,
                 robot_type: str = "small", hold_after_s: float = 3.0, heartbeat_period_s: float = 1.0):
        self.id, self.role, self.grid, self.clock = robot_id, role, grid, clock
        self.distance = distance or GridDistance(grid)
        self.speed, self.capacity, self.robot_type = speed_cells_s, capacity, robot_type
        self.hold_after_s, self.heartbeat_period_s = hold_after_s, heartbeat_period_s
        self.cell = (float(cell[0]), float(cell[1]))
        self.session = uuid.uuid4().hex
        self.seq = 0
        self.epoch = 0
        self.welcomed = False
        self.hello_id = ""
        self.refused_reason = ""
        self.exec_state = m.ExecState.IDLE
        self.lift = m.LiftState.DOWN
        self.task: Optional[_Task] = None
        self.last_finished: Optional[m.CurrentTask] = None
        self.seen: set[tuple[str, int]] = set()
        self.last_coord_heartbeat = clock()
        self._last_heartbeat = -math.inf
        self.fail_next: Optional[str] = None   # test hook: fail the next leg with this reason
        self.fail_count = 0                     # ... and this many legs after it
        self.log: list[tuple[str, str]] = []   # (task_id, status) sent

    # ---------------- outbound

    def _env(self, type_: str, payload) -> Envelope:
        self.seq += 1
        return make(type_, payload, robot_id=self.id, session=self.session, seq=self.seq, epoch=self.epoch,
                    sent_at=self.clock())

    def pose(self) -> m.Pose:
        x, y = self.grid.position_of(self.cell)
        return m.Pose(x=x, y=y, stamp=self.clock())

    def hello(self) -> Envelope:
        env = self._hello()
        self.hello_id = env.msg_id
        return env

    def _hello(self) -> Envelope:
        current = None
        if self.task is not None:
            current = m.CurrentTask(task_id=self.task.cmd.task_id, attempt=self.task.cmd.attempt,
                                    leg=self.task.leg, status=m.TaskStatus.STARTED)
        elif self.last_finished is not None:
            current = self.last_finished
        return self._env("robot.hello", m.Hello(robot_type=self.robot_type, role=self.role, capacity=self.capacity,
                                                 protocol_hash=PROTOCOL_HASH, current_task=current,
                                                 lift_state=self.lift, pose=self.pose()))

    def heartbeat(self) -> Envelope:
        self._last_heartbeat = self.clock()
        return self._env("robot.heartbeat", m.Heartbeat(
            pose=self.pose(), exec_state=self.exec_state,
            task_id=self.task.cmd.task_id if self.task else None, leg=self.task.leg if self.task else 0))

    def _event(self, cmd: m.TaskCommand, status: m.TaskStatus, reason: str = "", leg: int = 0) -> Envelope:
        self.log.append((cmd.task_id, status.value))
        return self._env("robot.event", m.Event(event_id=uuid.uuid4().hex, task_id=cmd.task_id, attempt=cmd.attempt,
                                                 leg=leg, status=status, reason=reason, pose=self.pose()))

    # ---------------- inbound

    def receive(self, env: Envelope) -> list[Envelope]:
        out = []
        if env.type == "coordinator.welcome":
            w: m.Welcome = env.body()
            if w.hello_id != self.hello_id:
                return []                 # answers an earlier hello (duplicate or late)
            self.epoch, self.welcomed = w.epoch, w.accepted
            self.refused_reason = w.reason
            self.last_coord_heartbeat = self.clock()
            if self.task is not None and w.task_to_resume != self.task.cmd.task_id:
                # the coordinator has rolled it back: stop, and say so (never drop a task silently)
                cmd = self.task.cmd
                self.task = None
                self.last_finished = m.CurrentTask(task_id=cmd.task_id, attempt=cmd.attempt,
                                                   status=m.TaskStatus.CANCELED)
                self._set_state(m.ExecState.IDLE)
                out.append(self._event(cmd, m.TaskStatus.CANCELED, "not resumed by coordinator"))
            out.append(self.heartbeat())
        elif env.type == "coordinator.heartbeat":
            hb: m.CoordinatorHeartbeat = env.body()
            self.last_coord_heartbeat = self.clock()
            if hb.epoch != self.epoch:
                out.append(self.hello())   # a new coordinator incarnation: reconcile
            elif self.exec_state == m.ExecState.HOLD:
                if self.task is not None:
                    self._plan_leg()       # resume with the leg it held before
                self._set_state(m.ExecState.EXECUTING if self.task else m.ExecState.IDLE)
                out.append(self.heartbeat())
        elif env.type == "command.task":
            out += self._on_task(env)
        elif env.type == "command.cancel":
            c: m.CancelCommand = env.body()
            if self.task is not None and self.task.cmd.task_id == c.task_id and env.epoch >= self.epoch:
                cmd = self.task.cmd
                self.task = None
                self.last_finished = m.CurrentTask(task_id=cmd.task_id, attempt=cmd.attempt,
                                                   status=m.TaskStatus.CANCELED)
                self._set_state(m.ExecState.IDLE)
                out += [self._event(cmd, m.TaskStatus.CANCELED, "cancelled"), self.heartbeat()]
        elif env.type == "coordinator.estop":
            self._set_state(m.ExecState.ESTOP)
            out.append(self.heartbeat())
        elif env.type == "coordinator.resume":
            if self.exec_state == m.ExecState.ESTOP:
                self._set_state(m.ExecState.EXECUTING if self.task else m.ExecState.IDLE)
                out.append(self.heartbeat())
        return out

    def _on_task(self, env: Envelope) -> list[Envelope]:
        cmd: m.TaskCommand = env.body()
        key = (cmd.task_id, cmd.attempt)
        if key in self.seen:
            return []                      # duplicate delivery
        self.seen.add(key)
        if env.epoch < self.epoch:
            return [self._event(cmd, m.TaskStatus.REJECTED, "stale epoch")]
        if cmd.valid_until < self.clock():
            return [self._event(cmd, m.TaskStatus.REJECTED, "stale")]
        if not self.welcomed:
            return [self._event(cmd, m.TaskStatus.REJECTED, "not welcomed")]
        if self.task is not None or self.exec_state not in (m.ExecState.IDLE,):
            return [self._event(cmd, m.TaskStatus.REJECTED, f"busy ({self.exec_state.value})")]
        self.task = _Task(cmd=cmd)
        self._plan_leg()
        self._set_state(m.ExecState.EXECUTING)
        return [self._event(cmd, m.TaskStatus.ACCEPTED), self._event(cmd, m.TaskStatus.STARTED), self.heartbeat()]

    # ---------------- simulation

    def _plan_leg(self):
        t = self.task
        leg = t.cmd.legs[t.leg]
        target = self.grid.cell_of(leg.pose.x, leg.pose.y)
        t.path = self.distance.path(self.cell, target) or [self.cell, target]
        t.progress = 0.0

    def _set_state(self, state: m.ExecState):
        self.exec_state = state

    def step(self, dt: float) -> list[Envelope]:
        out = []
        now = self.clock()
        coordinator_lost = now - self.last_coord_heartbeat > self.hold_after_s
        if self.task is not None and self.exec_state == m.ExecState.EXECUTING:
            out += self._advance(dt)
        if coordinator_lost and self.exec_state == m.ExecState.IDLE:
            self._set_state(m.ExecState.HOLD)
            out.append(self.heartbeat())
        if now - self._last_heartbeat >= self.heartbeat_period_s:
            out.append(self.heartbeat())
        return out

    def _advance(self, dt: float) -> list[Envelope]:
        t = self.task
        t.progress = min(t.progress + self.speed * dt, len(t.path) - 1)
        i = int(t.progress)
        self.cell = t.path[min(i, len(t.path) - 1)]
        if t.progress < len(t.path) - 1:
            return []
        cmd = t.cmd
        cmd_leg = cmd.legs[t.leg]
        out = [self._event(cmd, m.TaskStatus.ARRIVED, leg=t.leg)]
        if self.fail_next:
            reason = self.fail_next
            if self.fail_count > 0:
                self.fail_count -= 1
            else:
                self.fail_next = None
            return out + self._finish(m.TaskStatus.FAILED, reason)
        if cmd_leg.lift == m.Lift.UP:
            self.lift = m.LiftState.UP
            out.append(self._event(cmd, m.TaskStatus.LIFT_UP, leg=t.leg))
        elif cmd_leg.lift == m.Lift.DOWN:
            self.lift = m.LiftState.DOWN
            out.append(self._event(cmd, m.TaskStatus.LIFT_DOWN, leg=t.leg))
        if t.leg + 1 < len(cmd.legs):
            t.leg += 1
            if self.clock() - self.last_coord_heartbeat > self.hold_after_s:
                self._set_state(m.ExecState.HOLD)   # finish the leg, start no new one
                return out + [self.heartbeat()]
            self._plan_leg()
            return out
        return out + self._finish(m.TaskStatus.SUCCEEDED)

    def _finish(self, status: m.TaskStatus, reason: str = "") -> list[Envelope]:
        cmd = self.task.cmd
        self.task = None
        self.last_finished = m.CurrentTask(task_id=cmd.task_id, attempt=cmd.attempt, status=status)
        self._set_state(m.ExecState.IDLE)
        return [self._event(cmd, status, reason), self.heartbeat()]
