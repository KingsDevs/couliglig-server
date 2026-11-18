"""The fleet coordinator's state machine: the single writer of fleet state.

Synchronous and I/O-free: messages and clock ticks go in, messages to send come
out. coordinator/service.py connects it to RabbitMQ; tests drive it directly.

    handle(envelope)  robot.hello / robot.heartbeat / robot.event
    handle_op(op)     operator calls (activate, add_item, pause, estop, ...)
    tick()            liveness, command expiry, task deadlines, decision rounds,
                      the 1 Hz coordinator heartbeat

Decision round (the training env's tick): deciders are online, idle robots
with a fresh pose; each is observed (coordinator.spec), the policy picks one
action per robot, conflicts are resolved as the env's _tick does (picks, then
hand-offs, then zone claims, then deliveries; the closest claimant wins, ties
by numeric robot id), and the winners' commitments are applied at decision
time, as in training. Losers stay idle and are reconsidered next round.

Item lifecycle:
    AVAILABLE -> CLAIMED (pick sent) -> HELD (picked) -> HANDING (hand-off sent)
    -> ON_TRANSPORTER -> DELIVERING (delivery sent) -> DELIVERED
A failed task rolls its commitment back; the robot's physical report (lift
events, lift state in hello) decides where an interrupted item really is.
"""
from __future__ import annotations

import logging
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from coordinator.grid import Grid, GridDistance
from coordinator.policies.base import Decider, DecisionPolicy, DecisionRequest, decode, is_legal
from coordinator.protocol import PROTOCOL_HASH, Envelope, make
from coordinator.protocol import messages as m
from coordinator.spec import Item, Picker, Transporter, World, Zone, observe
from coordinator.spec.policy_spec import PolicySpec
from coordinator.store import Store

log = logging.getLogger(__name__)

# item status
AVAILABLE, CLAIMED, HELD, HANDING, ON_TRANSPORTER, DELIVERING, DELIVERED = (
    "AVAILABLE", "CLAIMED", "HELD", "HANDING", "ON_TRANSPORTER", "DELIVERING", "DELIVERED")
# task status (coordinator side)
SENT, ACCEPTED, STARTED = "SENT", "ACCEPTED", "STARTED"
SUCCEEDED, FAILED, CANCELED, REJECTED, EXPIRED = "SUCCEEDED", "FAILED", "CANCELED", "REJECTED", "EXPIRED"
ACTIVE = {SENT, ACCEPTED, STARTED}
_RANK = {SENT: 0, ACCEPTED: 1, STARTED: 2}
# robot liveness
ONLINE, STALE, OFFLINE, REFUSED = "ONLINE", "STALE", "OFFLINE", "REFUSED"


@dataclass
class Config:
    stale_s: float = 3.0              # no heartbeat this long: excluded from deciding
    offline_s: float = 30.0           # no heartbeat this long: unstarted tasks reverted
    pose_fresh_s: float = 2.0
    heartbeat_period_s: float = 1.0
    round_debounce_s: float = 0.2     # triggers within this window share one round
    round_period_s: float = 1.0       # periodic round (parked transporters are re-asked)
    command_ttl_s: float = 10.0       # valid_until = sent + this
    accept_grace_s: float = 2.0       # unaccepted after valid_until + this: expired
    robot_speed_mps: float = 0.26
    leg_overhead_s: float = 90.0      # docking, lift, undock per leg
    leg_timeout_factor: float = 2.0
    blacklist_after: int = 2          # failures of one robot at one target
    blacklist_s: float = 300.0
    lost_task_s: float = 15.0         # robot idle without our task this long: task lost
    remaining_frac: float = 1.0       # continuous operation (no order batches yet)
    time_frac: float = 0.0
    map_hash: str = ""                # expected in hello when set
    dock_db_hash: str = ""
    handoff_slots_default: int = 2    # slots per transporter without hand_off docks


@dataclass
class Out:
    kind: str                          # "command" | "broadcast" | "declare_robot"
    robot_id: str
    envelope: Optional[Envelope] = None


@dataclass
class Dock:
    id: str
    type: str                          # pickup | receiver | waiting_zone | hand_off
    x: float
    y: float
    yaw: float = 0.0
    cell: tuple[int, int] = (0, 0)


@dataclass
class RobotRec:
    id: str
    role: str = ""
    robot_type: str = ""
    capacity: float = 4.0
    zone: Optional[str] = None         # transporters: waiting zone claimed or held
    parked: bool = False               # transporters: arrived in that zone
    carried: list[str] = field(default_factory=list)
    held_item: Optional[str] = None    # pickers: item claimed or on board
    # volatile
    session: str = ""
    last_seq: int = -1
    last_seen: float = -math.inf
    pose: Optional[m.Pose] = None
    pose_at: float = -math.inf
    exec_state: str = m.ExecState.IDLE.value
    status: str = OFFLINE
    task_id: Optional[str] = None
    reported_task: Optional[str] = None   # the task the robot's heartbeats say it runs
    reported_since: float = 0.0

    PERSISTENT = ("id", "role", "robot_type", "capacity", "zone", "parked", "carried", "held_item")

    def persistent(self) -> dict:
        return {k: getattr(self, k) for k in self.PERSISTENT}


@dataclass
class ItemRec:
    id: str
    pickup_dock: str
    receiver_dock: str
    weight: float = 1.0
    status: str = AVAILABLE
    holder: Optional[str] = None       # picker
    transporter: Optional[str] = None
    created_at: float = 0.0


@dataclass
class TaskRec:
    id: str
    robot_id: str
    kind: str
    attempt: int = 1
    item_id: Optional[str] = None
    target: Optional[str] = None       # pickup dock | transporter id | zone dock | receiver dock
    slot: Optional[str] = None         # hand-off dock used
    legs: list[dict] = field(default_factory=list)
    dest_cell: tuple[int, int] = (0, 0)
    status: str = SENT
    reason: str = ""
    epoch: int = 0
    created_at: float = 0.0
    valid_until: float = 0.0
    deadline: float = 0.0
    lift_up: bool = False
    lift_down: bool = False
    cancel_sent: bool = False


def numeric_id(robot_id: str):
    """Sort key for tie-breaks: the robot's number, so couliglig_bot_2 < couliglig_bot_10.

    (The training env compares ids as text, picker_10 < picker_2: a known env
    defect; the trajectory tests treat those ties as expected differences.)"""
    match = re.search(r"(\d+)$", robot_id)
    return (int(match.group(1)), robot_id) if match else (math.inf, robot_id)


class CoordinatorFeasibility:
    """What the fleet can physically do now, on top of the env's rules."""

    def __init__(self, coord: "Coordinator"):
        self.c = coord

    def can_pick(self, picker, item):
        dock = self.c.items[item.id].pickup_dock
        return not self.c.blacklisted(picker.id, dock)

    def can_hand_to(self, picker, transporter):
        t = self.c.robots[transporter.id]
        return (t.zone is not None and t.parked and t.task_id is None
                and self.c.free_slot(t.id) is not None and not self.c.blacklisted(picker.id, t.id))

    def can_enter_zone(self, transporter, zone):
        return not self.c.blacklisted(transporter.id, zone.id)

    def can_deliver(self, transporter, item):
        return not self.c.blacklisted(transporter.id, self.c.items[item.id].receiver_dock)


class Coordinator:
    def __init__(self, store: Store, policy: DecisionPolicy, spec: PolicySpec, grid: Grid,
                 distance: Callable | None = None, config: Config | None = None,
                 clock: Callable[[], float] = time.time):
        self.store, self.policy, self.spec, self.grid = store, policy, spec, grid
        self.distance = distance or GridDistance(grid)
        self.config = config or Config()
        self.clock = clock
        self.session = uuid.uuid4().hex
        self.epoch = 0
        self.seq = 0
        self.paused = False
        self.docks: dict[str, Dock] = {}
        self.robots: dict[str, RobotRec] = {}
        self.items: dict[str, ItemRec] = {}
        self.tasks: dict[str, TaskRec] = {}
        self.failures: dict[tuple[str, str], list[float]] = {}
        self.slots: dict[str, str] = {}            # hand-off slot -> picker using it
        self._round_due: Optional[float] = None
        self._last_round = -math.inf
        self._last_heartbeat = -math.inf
        self.metrics = {"rounds": 0, "commands": 0, "events": 0, "duplicate_events": 0,
                        "stale_telemetry": 0, "ignored_events": 0, "last_round_ms": 0.0,
                        "max_round_ms": 0.0, "deciders": 0}
        self.feasibility = CoordinatorFeasibility(self)

    # ================================================================ lifecycle

    def start(self) -> int:
        """Load persisted state and start a new epoch (fences off older commands)."""
        self.epoch = self.store.next_epoch()
        for d in self.store.get_meta("docks", []):
            self._add_dock(d)
        for row in self.store.load_items():
            self.items[row["id"]] = ItemRec(**row)
        for row in self.store.load_robots():
            self.robots[row["id"]] = RobotRec(**row)
        for row in self.store.load_tasks(active_only=True):
            row["dest_cell"] = tuple(row["dest_cell"])
            t = TaskRec(**row)
            self.tasks[t.id] = t
            if t.robot_id in self.robots:
                self.robots[t.robot_id].task_id = t.id
            if t.slot:
                self.slots[t.slot] = t.robot_id
        log.info("coordinator epoch %d: %d docks, %d items, %d robots, %d active tasks",
                 self.epoch, len(self.docks), len(self.items), len(self.robots), len(self.tasks))
        self._trigger_round()
        return self.epoch

    def now(self) -> float:
        return self.clock()

    # ================================================================ inbound messages

    def handle(self, env: Envelope) -> list[Out]:
        if env.type == "robot.hello":
            return self._on_hello(env)
        if env.type == "robot.heartbeat":
            return self._on_heartbeat(env)
        if env.type == "robot.event":
            return self._on_event(env)
        log.warning("ignoring %s from %s", env.type, env.robot_id)
        return []

    def _on_hello(self, env: Envelope) -> list[Out]:
        hello: m.Hello = env.body()
        now = self.now()
        r = self.robots.get(env.robot_id) or RobotRec(id=env.robot_id)
        self.robots[r.id] = r
        # Same session = the robot did not restart (it re-says hello on a new
        # epoch). A hello older than messages already seen is a late copy.
        same_session = bool(r.session) and env.session == r.session and r.status != REFUSED
        if same_session and env.seq <= r.last_seq:
            self.metrics["stale_telemetry"] += 1
            return []
        reason = ""
        if hello.protocol_hash != PROTOCOL_HASH:
            reason = f"protocol_hash {hello.protocol_hash} != {PROTOCOL_HASH}"
        elif self.config.map_hash and hello.map_hash != self.config.map_hash:
            reason = f"map_hash {hello.map_hash} != {self.config.map_hash}"
        elif self.config.dock_db_hash and hello.dock_db_hash != self.config.dock_db_hash:
            reason = f"dock_db_hash {hello.dock_db_hash} != {self.config.dock_db_hash}"
        if r.role and r.role != hello.role.value:
            reason = reason or f"role {hello.role.value} != registered {r.role}"
        outs = [Out("declare_robot", r.id)]
        if reason:
            r.status = REFUSED
            log.warning("refusing %s: %s", r.id, reason)
            return outs + [self._command(r.id, "coordinator.welcome",
                                         m.Welcome(epoch=self.epoch, accepted=False, reason=reason,
                                                   hello_id=env.msg_id))]
        r.role, r.robot_type, r.capacity = hello.role.value, hello.robot_type, float(hello.capacity)
        r.session, r.last_seq, r.last_seen, r.status = env.session, env.seq, now, ONLINE
        if hello.pose is not None:
            r.pose, r.pose_at = hello.pose, now
        resume, extra = self._reconcile(r, hello, same_session)
        self._save(robots=[r])
        self._trigger_round()
        r.reported_task, r.reported_since = (hello.current_task.task_id if hello.current_task else None), now
        welcome = self._command(r.id, "coordinator.welcome",
                                m.Welcome(epoch=self.epoch, accepted=True, task_to_resume=resume,
                                          hello_id=env.msg_id))
        return outs + [welcome] + extra

    def _on_heartbeat(self, env: Envelope) -> list[Out]:
        r = self.robots.get(env.robot_id)
        if r is None or r.status == REFUSED or env.session != r.session:
            return []  # unknown session: the robot must say hello first
        if env.seq <= r.last_seq:
            self.metrics["stale_telemetry"] += 1
            return []
        hb: m.Heartbeat = env.body()
        now = self.now()
        r.last_seq, r.last_seen = env.seq, now
        r.pose, r.pose_at, r.exec_state = hb.pose, now, hb.exec_state.value
        if hb.task_id != r.reported_task:
            r.reported_task, r.reported_since = hb.task_id, now
        if r.status != ONLINE:
            log.info("%s back online", r.id)
            r.status = ONLINE
            self._trigger_round()
        return []

    def _on_event(self, env: Envelope) -> list[Out]:
        ev: m.Event = env.body()
        if self.store.seen_event(ev.event_id):
            self.metrics["duplicate_events"] += 1
            return []
        record = {"event_id": ev.event_id, "robot_id": env.robot_id, "task_id": ev.task_id,
                  "attempt": ev.attempt, "status": ev.status.value, "reason": ev.reason}
        r = self.robots.get(env.robot_id)
        if r is not None and env.session == r.session:
            r.last_seen = self.now()
        return self._apply_task_event(env.robot_id, ev.task_id, ev.attempt, ev.status, ev.reason, record)

    # ================================================================ task state machine

    def _apply_task_event(self, robot_id, task_id, attempt, status: m.TaskStatus, reason, record) -> list[Out]:
        t = self.tasks.get(task_id)
        if t is None or t.robot_id != robot_id or t.attempt != attempt or t.status not in ACTIVE:
            # superseded attempt, a finished task, or not ours: log it, change nothing
            self.metrics["ignored_events"] += 1
            self.store.commit(event=record)
            return []
        self.metrics["events"] += 1
        changed_items, changed_robots = [], []
        if status == m.TaskStatus.LIFT_UP:
            t.lift_up = True
            if t.kind == m.TaskKind.PICK.value and t.item_id:
                item = self.items[t.item_id]
                if item.status == CLAIMED:
                    item.status = HELD  # it is on the picker now, whatever happens next
                    changed_items.append(item)
        elif status == m.TaskStatus.LIFT_DOWN:
            t.lift_down = True
        elif status in (m.TaskStatus.ACCEPTED, m.TaskStatus.STARTED, m.TaskStatus.ARRIVED):
            new = {m.TaskStatus.ACCEPTED: ACCEPTED}.get(status, STARTED)
            if _RANK[new] > _RANK[t.status]:
                t.status = new
        elif status == m.TaskStatus.SUCCEEDED:
            changed_items, changed_robots = self._succeed(t)
        else:  # REJECTED, FAILED, CANCELED
            final = {m.TaskStatus.REJECTED: REJECTED, m.TaskStatus.FAILED: FAILED}.get(status, CANCELED)
            changed_items, changed_robots = self._finish_failed(t, final, reason or status.value)
        self._save(items=changed_items, tasks=[t], robots=changed_robots, event=record)
        if t.status not in ACTIVE:
            self._trigger_round()
        return []

    def _succeed(self, t: TaskRec):
        r = self.robots[t.robot_id]
        items, robots = [], [r]
        kind = m.TaskKind(t.kind)
        if kind == m.TaskKind.PICK:
            item = self.items[t.item_id]
            item.status, item.holder = HELD, r.id
            items.append(item)
        elif kind == m.TaskKind.HANDOFF_PLACE:
            items, robots = self._handoff_done(t)
        elif kind == m.TaskKind.PARK_ZONE:
            r.parked = r.zone == t.target
        elif kind == m.TaskKind.DELIVER:
            item = self.items[t.item_id]
            item.status = DELIVERED
            if item.id in r.carried:
                r.carried.remove(item.id)
            items.append(item)
        self._close(t, SUCCEEDED, "")
        return items, robots

    def _handoff_done(self, t: TaskRec):
        picker = self.robots[t.robot_id]
        transporter = self.robots[t.target]
        item = self.items[t.item_id]
        item.status, item.holder, item.transporter = ON_TRANSPORTER, None, transporter.id
        picker.held_item = None
        if item.id not in transporter.carried:
            transporter.carried.append(item.id)
        return [item], [picker, transporter]

    def _finish_failed(self, t: TaskRec, final: str, reason: str):
        """Roll the task's commitment back, keeping what physically happened."""
        r = self.robots[t.robot_id]
        items, robots = [], [r]
        kind = m.TaskKind(t.kind)
        if kind == m.TaskKind.PICK:
            item = self.items[t.item_id]
            if t.lift_up or item.status == HELD:
                item.status, item.holder = HELD, r.id        # it is on the picker
            else:
                item.status, item.holder = AVAILABLE, None
                r.held_item = None
            items.append(item)
        elif kind == m.TaskKind.HANDOFF_PLACE:
            if t.lift_down:   # the box was put down on the transporter's slot
                items, robots = self._handoff_done(t)
            else:
                item = self.items[t.item_id]
                item.status, item.holder, item.transporter = HELD, r.id, None
                items.append(item)
        elif kind == m.TaskKind.PARK_ZONE:
            if r.zone == t.target:
                r.zone, r.parked = None, False
        elif kind == m.TaskKind.DELIVER:
            item = self.items[t.item_id]
            item.status = ON_TRANSPORTER     # still on board: load it back
            if item.id not in r.carried:
                r.carried.append(item.id)
            items.append(item)
        if final in (FAILED, REJECTED) and t.target and reason not in ("offline", "expired"):
            self._record_failure(r.id, t.target)
        self._close(t, final, reason)
        log.info("task %s %s %s for %s: %s", t.id, t.kind, final, t.robot_id, reason)
        return items, robots

    def _close(self, t: TaskRec, status: str, reason: str):
        t.status, t.reason = status, reason
        if t.slot and self.slots.get(t.slot) == t.robot_id:
            del self.slots[t.slot]
        r = self.robots.get(t.robot_id)
        if r is not None and r.task_id == t.id:
            r.task_id = None

    # ================================================================ reconciliation

    def _reconcile(self, r: RobotRec, hello: m.Hello, same_session: bool) -> tuple[Optional[str], list[Out]]:
        """Line the task table up with what the (re)connecting robot reports.

        Only a restarted robot (new session) can have lost a task; within a
        session, every task ends with an event, so a mismatch there is just
        a hello that crossed a command in flight and changes nothing.
        """
        cur = hello.current_task
        active = self.tasks.get(r.task_id) if r.task_id else None
        outs: list[Out] = []
        if active is not None and active.status in ACTIVE:
            if cur is not None and cur.task_id == active.id and cur.attempt == active.attempt:
                if cur.status in m.TERMINAL:
                    # finished while we could not hear it; the event itself is in the robot's outbox
                    self._apply_task_event(r.id, active.id, active.attempt, cur.status, "reported in hello",
                                           {"event_id": f"{active.id}:{active.attempt}:{cur.status.value}:hello",
                                            "robot_id": r.id, "task_id": active.id, "attempt": active.attempt,
                                            "status": cur.status.value, "reason": "hello"})
                    return None, outs
                return active.id, outs
            if same_session:
                return active.id, outs
            # a restarted robot no longer knows the task: its physical state decides
            if r.role == m.Role.PICKER.value and active.kind == m.TaskKind.PICK.value:
                active.lift_up = active.lift_up or hello.lift_state == m.LiftState.UP
            if r.role == m.Role.PICKER.value and active.kind == m.TaskKind.HANDOFF_PLACE.value:
                active.lift_down = active.lift_down or hello.lift_state == m.LiftState.DOWN
            items, robots = self._finish_failed(active, FAILED, "robot_restarted")
            self._save(items=items, tasks=[active], robots=robots)
        if cur is not None and cur.status not in m.TERMINAL and (active is None or cur.task_id != active.id):
            outs.append(self._command(r.id, "command.cancel", m.CancelCommand(task_id=cur.task_id)))
        return None, outs

    # ================================================================ operator calls

    def handle_op(self, op: dict) -> tuple[dict, list[Out]]:
        name = op.get("op")
        try:
            handler = getattr(self, f"_op_{name}")
        except AttributeError:
            return {"ok": False, "error": f"unknown op {name!r}"}, []
        try:
            return handler(op)
        except (KeyError, ValueError) as e:
            return {"ok": False, "error": str(e)}, []

    def _op_activate(self, op):
        """Replace the dock layout. Refused while tasks are running."""
        if any(t.status in ACTIVE for t in self.tasks.values()):
            raise ValueError("tasks are active; cancel or finish them first")
        self.docks = {}
        for d in op["docks"]:
            self._add_dock(d)
        self.store.set_meta("docks", [self._dock_row(d) for d in self.docks.values()])
        self._trigger_round()
        return {"ok": True, "docks": len(self.docks)}, []

    def _op_add_item(self, op):
        item_id = str(op["item_id"])
        if item_id in self.items and self.items[item_id].status != DELIVERED:
            raise ValueError(f"item {item_id} already exists")
        for key, kind in (("pickup_dock", "pickup"), ("receiver_dock", "receiver")):
            dock = self.docks.get(op[key])
            if dock is None or dock.type != kind:
                raise ValueError(f"{key} {op[key]!r} is not a {kind} dock")
        if any(i.pickup_dock == op["pickup_dock"] and i.status == AVAILABLE for i in self.items.values()):
            raise ValueError(f"pickup dock {op['pickup_dock']} already holds an item")
        item = ItemRec(id=item_id, pickup_dock=op["pickup_dock"], receiver_dock=op["receiver_dock"],
                       weight=float(op.get("weight", 1.0)), created_at=self.now())
        self.items[item_id] = item
        self._save(items=[item])
        self._trigger_round()
        return {"ok": True, "item_id": item_id}, []

    def _op_remove_item(self, op):
        item = self.items[str(op["item_id"])]
        if item.status != AVAILABLE:
            raise ValueError(f"item {item.id} is {item.status}; only AVAILABLE items can be removed")
        del self.items[item.id]
        self.store.commit(deleted_items=[item.id])
        return {"ok": True}, []

    def _op_pause(self, op):
        self.paused = True
        return {"ok": True}, [self._heartbeat_out()]

    def _op_resume(self, op):
        self.paused = False
        self._trigger_round()
        return {"ok": True}, [self._broadcast("coordinator.resume", m.Resume(issued_by=op.get("issued_by", ""))),
                              self._heartbeat_out()]

    def _op_estop(self, op):
        self.paused = True
        return {"ok": True}, [self._broadcast("coordinator.estop",
                                              m.EStop(reason=op.get("reason", ""), issued_by=op.get("issued_by", "")))]

    def _op_snapshot(self, op):
        return {"ok": True, "epoch": self.epoch, "paused": self.paused, "metrics": dict(self.metrics),
                "robots": {r.id: {"role": r.role, "status": r.status, "exec_state": r.exec_state,
                                  "task": r.task_id, "zone": r.zone, "carried": list(r.carried),
                                  "held_item": r.held_item} for r in self.robots.values()},
                "items": {i.id: {"status": i.status, "holder": i.holder, "transporter": i.transporter}
                          for i in self.items.values()},
                "tasks": {t.id: {"robot": t.robot_id, "kind": t.kind, "status": t.status, "target": t.target}
                          for t in self.tasks.values() if t.status in ACTIVE}}, []

    # ================================================================ clock

    def tick(self) -> list[Out]:
        now = self.now()
        outs: list[Out] = []
        if now - self._last_heartbeat >= self.config.heartbeat_period_s:
            self._last_heartbeat = now
            outs.append(self._heartbeat_out())
        outs += self._liveness(now)
        outs += self._task_timeouts(now)
        self._prune(now)
        due = self._round_due is not None and now >= self._round_due
        periodic = now - self._last_round >= self.config.round_period_s
        if not self.paused and (due or periodic):
            outs += self.run_round()
        return outs

    def _liveness(self, now) -> list[Out]:
        outs = []
        for r in self.robots.values():
            if r.status in (OFFLINE, REFUSED):
                continue
            silent = now - r.last_seen
            if silent > self.config.offline_s:
                r.status = OFFLINE
                log.warning("%s offline (%.0f s silent)", r.id, silent)
                t = self.tasks.get(r.task_id) if r.task_id else None
                if t is not None and t.status in (SENT, ACCEPTED):
                    items, robots = self._finish_failed(t, CANCELED, "offline")
                    self._save(items=items, tasks=[t], robots=robots)
                    outs.append(self._command(r.id, "command.cancel", m.CancelCommand(task_id=t.id)))
                    self._trigger_round()
            elif silent > self.config.stale_s and r.status == ONLINE:
                r.status = STALE
                log.info("%s stale (%.1f s silent)", r.id, silent)
        return outs

    def _task_timeouts(self, now) -> list[Out]:
        outs = []
        for t in list(self.tasks.values()):
            if t.status == SENT and now > t.valid_until + self.config.accept_grace_s:
                items, robots = self._finish_failed(t, EXPIRED, "expired")
                self._save(items=items, tasks=[t], robots=robots)
                outs.append(self._command(t.robot_id, "command.cancel", m.CancelCommand(task_id=t.id)))
                self._trigger_round()
            elif t.status in (ACCEPTED, STARTED) and self._robot_lost_task(t, now):
                # safety net: fleet_agent always ends a task with an event, so this means a bug
                log.error("robot %s is idle without task %s for %.0f s; failing it",
                          t.robot_id, t.id, now - self.robots[t.robot_id].reported_since)
                items, robots = self._finish_failed(t, FAILED, "lost by robot")
                self._save(items=items, tasks=[t], robots=robots)
                self._trigger_round()
            elif t.status in (ACCEPTED, STARTED) and now > t.deadline and not t.cancel_sent:
                # the robot's own leg timeout should have fired; ask it to stop, it reports the outcome
                t.cancel_sent = True
                log.warning("task %s past its deadline; cancelling", t.id)
                outs.append(self._command(t.robot_id, "command.cancel", m.CancelCommand(task_id=t.id)))
        return outs

    def _prune(self, now, keep_s: float = 600.0):
        """Finished tasks leave memory after a while (they stay in SQLite)."""
        for tid in [tid for tid, t in self.tasks.items()
                    if t.status not in ACTIVE and now - t.created_at > keep_s]:
            del self.tasks[tid]

    def _robot_lost_task(self, t: TaskRec, now: float) -> bool:
        r = self.robots.get(t.robot_id)
        return (r is not None and r.status == ONLINE and r.exec_state == m.ExecState.IDLE.value
                and r.reported_task != t.id and now - max(r.reported_since, t.created_at) > self.config.lost_task_s)

    # ================================================================ decision round

    def run_round(self) -> list[Out]:
        start = time.perf_counter()
        now = self.now()
        self._round_due = None
        self._last_round = now
        world, pickers, transporters, zones = self._snapshot()
        deciders = []
        for idx, rid in enumerate(pickers):
            if self._can_decide(rid, now):
                deciders.append(Decider(rid, observe(world, self.spec, True, idx)))
        for idx, rid in enumerate(transporters):
            if self._can_decide(rid, now):
                deciders.append(Decider(rid, observe(world, self.spec, False, idx)))
        deciders = [d for d in deciders if d.observation.choice]
        outs: list[Out] = []
        if deciders:
            actions = self.policy.decide(DecisionRequest(world=world, deciders=deciders))
            choices = []
            for d in deciders:
                action = actions.get(d.robot_id, 0)
                if not is_legal(d.observation, action):
                    log.warning("policy gave %s illegal action %s; idle", d.robot_id, action)
                    continue
                choice = decode(self.spec, d.observation, action)
                if choice.verb != "idle":
                    choices.append((d.robot_id, choice))
            outs = self._resolve_and_commit(choices, transporters, zones, now)
        elapsed = (time.perf_counter() - start) * 1000
        self.metrics["rounds"] += 1
        self.metrics["deciders"] = len(deciders)
        self.metrics["last_round_ms"] = elapsed
        self.metrics["max_round_ms"] = max(self.metrics["max_round_ms"], elapsed)
        return outs

    def _can_decide(self, rid: str, now: float) -> bool:
        r = self.robots[rid]
        return (r.status == ONLINE and r.task_id is None and r.exec_state == m.ExecState.IDLE.value
                and r.pose is not None and r.pose.loc_ok and now - r.pose_at <= self.config.pose_fresh_s
                and not self._incoming(rid))

    def _incoming(self, transporter_id: str) -> list[TaskRec]:
        """Hand-offs on their way to this transporter (it is held until they arrive)."""
        return [t for t in self.tasks.values() if t.status in ACTIVE
                and t.kind == m.TaskKind.HANDOFF_PLACE.value and t.target == transporter_id]

    def _resolve_and_commit(self, choices, transporters, zones, now) -> list[Out]:
        """The env's _tick order: picks, hand-offs, zone claims, deliveries."""
        outs = []
        by_verb: dict[str, list] = {"pick": [], "handoff": [], "zone": [], "deliver": []}
        for rid, choice in choices:
            by_verb[choice.verb].append((rid, choice.target))

        def closest_first(entries, target_cell):
            return sorted(entries, key=lambda e: (self.distance(self._cell(e[0]), target_cell(e)), numeric_id(e[0])))

        taken = set()
        for rid, item_id in closest_first(by_verb["pick"], lambda e: self.docks[self.items[e[1]].pickup_dock].cell):
            item = self.items[item_id]
            if item_id in taken or item.status != AVAILABLE:
                continue
            taken.add(item_id)
            outs.append(self._commit_pick(rid, item, now))

        # one hand-off per transporter per round (the env settles each
        # transporter's claimants to a single winner), by distance to where
        # the transporter is heading
        served = set()
        for rid, t_idx in closest_first(by_verb["handoff"],
                                        lambda e: self._dest_cell(transporters[e[1]])):
            tid = transporters[t_idx]
            if tid in served:
                continue
            served.add(tid)
            t = self.robots[tid]
            picker = self.robots[rid]
            item = self.items.get(picker.held_item) if picker.held_item else None
            slot = self.free_slot(tid)
            if item is None or item.status != HELD or slot is None or self._room(t) < item.weight:
                continue
            outs.append(self._commit_handoff(rid, t, item, slot, now))

        claimed = set()
        for rid, z_idx in closest_first(by_verb["zone"], lambda e: self.docks[zones[e[1]]].cell):
            zone_id = zones[z_idx]
            if zone_id in claimed or self._zone_holder(zone_id) is not None:
                continue
            claimed.add(zone_id)
            outs.append(self._commit_zone(rid, zone_id, now))

        for rid, item_id in by_verb["deliver"]:
            item = self.items.get(item_id)
            if item is None or item.status != ON_TRANSPORTER or item.transporter != rid:
                continue
            outs.append(self._commit_deliver(rid, item, now))
        return outs

    # ---------------- commitments (applied at decision time, as in training)

    def _commit_pick(self, rid, item: ItemRec, now) -> Out:
        dock = self.docks[item.pickup_dock]
        item.status, item.holder = CLAIMED, rid
        self.robots[rid].held_item = item.id
        task = self._new_task(rid, m.TaskKind.PICK, item.id, dock.id, [self._leg(dock, m.Lift.UP)], dock.cell, now)
        self._save(items=[item], tasks=[task], robots=[self.robots[rid]])
        return self._task_out(task)

    def _commit_handoff(self, rid, t: RobotRec, item: ItemRec, slot: str, now) -> Out:
        item.status, item.transporter = HANDING, t.id
        self.slots[slot] = rid
        pose = t.pose or m.Pose(x=0.0, y=0.0)
        slot_dock = self.docks.get(slot)
        leg_pose = m.Pose(x=slot_dock.x, y=slot_dock.y, yaw=slot_dock.yaw) if slot_dock else \
            m.Pose(x=pose.x, y=pose.y, yaw=pose.yaw)
        legs = [{"dock_id": slot, "pose": leg_pose.model_dump(), "lift": m.Lift.DOWN.value}]
        task = self._new_task(rid, m.TaskKind.HANDOFF_PLACE, item.id, t.id, legs, self._dest_cell(t.id), now,
                              slot=slot)
        self._save(items=[item], tasks=[task])
        return self._task_out(task)

    def _commit_zone(self, rid, zone_id: str, now) -> Out:
        r = self.robots[rid]
        r.zone, r.parked = zone_id, False   # an earlier zone is freed by moving on
        dock = self.docks[zone_id]
        task = self._new_task(rid, m.TaskKind.PARK_ZONE, None, zone_id, [self._leg(dock, m.Lift.NONE)], dock.cell, now)
        self._save(tasks=[task], robots=[r])
        return self._task_out(task)

    def _commit_deliver(self, rid, item: ItemRec, now) -> Out:
        r = self.robots[rid]
        r.zone, r.parked = None, False      # leaving the zone frees it (env: free_waiting_zone)
        item.status = DELIVERING
        if item.id in r.carried:            # unloaded at decision time, as in the env
            r.carried.remove(item.id)
        dock = self.docks[item.receiver_dock]
        task = self._new_task(rid, m.TaskKind.DELIVER, item.id, dock.id, [self._leg(dock, m.Lift.NONE)], dock.cell, now)
        self._save(items=[item], tasks=[task], robots=[r])
        return self._task_out(task)

    def _new_task(self, rid, kind: m.TaskKind, item_id, target, legs, dest_cell, now, slot=None) -> TaskRec:
        cells = self.distance(self._cell(rid), dest_cell)
        travel_s = (cells if math.isfinite(cells) else 50.0) * self.grid.cell_size / self.config.robot_speed_mps
        leg_timeout = self.config.leg_timeout_factor * travel_s + self.config.leg_overhead_s
        task = TaskRec(id=uuid.uuid4().hex[:12], robot_id=rid, kind=kind.value, item_id=item_id, target=target,
                       slot=slot, legs=legs, dest_cell=tuple(dest_cell), epoch=self.epoch, created_at=now,
                       valid_until=now + self.config.command_ttl_s,
                       deadline=now + self.config.command_ttl_s + leg_timeout * len(legs))
        self.tasks[task.id] = task
        self.robots[rid].task_id = task.id
        return task

    def _leg(self, dock: Dock, lift: m.Lift) -> dict:
        return {"dock_id": dock.id, "pose": m.Pose(x=dock.x, y=dock.y, yaw=dock.yaw).model_dump(), "lift": lift.value}

    def _task_out(self, task: TaskRec) -> Out:
        self.metrics["commands"] += 1
        cells = self.distance(self._cell(task.robot_id), task.dest_cell)
        travel_s = (cells if math.isfinite(cells) else 50.0) * self.grid.cell_size / self.config.robot_speed_mps
        cmd = m.TaskCommand(task_id=task.id, attempt=task.attempt, kind=task.kind,
                            legs=[m.Leg(**leg) for leg in task.legs], valid_until=task.valid_until,
                            leg_timeout_s=self.config.leg_timeout_factor * travel_s + self.config.leg_overhead_s)
        return self._command(task.robot_id, "command.task", cmd)

    # ================================================================ world snapshot

    def _snapshot(self):
        """The fleet in the training env's terms (coordinator.spec.World)."""
        live = [r for r in self.robots.values() if r.status in (ONLINE, STALE) and r.pose is not None]
        pickers = sorted((r.id for r in live if r.role == m.Role.PICKER.value), key=numeric_id)
        transporters = sorted((r.id for r in live if r.role == m.Role.TRANSPORTER.value), key=numeric_id)
        zones = sorted((d.id for d in self.docks.values() if d.type == "waiting_zone"))
        p_index = {rid: i for i, rid in enumerate(pickers)}
        t_index = {rid: i for i, rid in enumerate(transporters)}

        items = [it for it in self.items.values() if it.status != DELIVERED]
        items.sort(key=lambda it: (it.created_at, it.id))
        i_index = {it.id: n for n, it in enumerate(items)}
        world_items = []
        for it in items:
            holder = p_index.get(it.holder, -1) if it.status in (CLAIMED, HELD) else -1
            world_items.append(Item(id=it.id, cell=self.docks[it.pickup_dock].cell, w=it.weight,
                                    recv_cell=self.docks[it.receiver_dock].cell, picked=it.status != AVAILABLE,
                                    handed=it.status in (HANDING, ON_TRANSPORTER, DELIVERING), holder=holder))

        def leg_steps(rid):
            r = self.robots[rid]
            t = self.tasks.get(r.task_id) if r.task_id else None
            busy = t is not None or r.exec_state != m.ExecState.IDLE.value
            dest = t.dest_cell if t is not None else self._cell(rid)
            d = self.distance(self._cell(rid), dest)
            steps = (0.0 if d < 0.5 else max(1.0, math.ceil(d / self.spec.agent_speed))) if math.isfinite(d) else 1.0
            return busy, dest, (steps if busy else 0.0), t

        world_pickers = []
        for rid in pickers:
            r = self.robots[rid]
            busy, dest, free_in, t = leg_steps(rid)
            item = self.items.get(r.held_item) if r.held_item else None
            hold = item is not None and item.status in (CLAIMED, HELD) and item.holder == rid
            handoff_to = t_index.get(t.target, -1) if t is not None and t.kind == m.TaskKind.HANDOFF_PLACE.value else -1
            world_pickers.append(Picker(id=rid, cell=self._cell(rid), dest=dest, busy=busy, hold=hold,
                                        w=item.weight if hold else 0.0, free_in=free_in, handoff_to=handoff_to))

        world_transporters = []
        for rid in transporters:
            r = self.robots[rid]
            busy, dest, free_in, _ = leg_steps(rid)
            if r.zone and r.task_id is None:
                dest = self.docks[r.zone].cell if r.parked else dest
            for k, pid in enumerate(pickers):
                if world_pickers[k].handoff_to == t_index[rid]:
                    free_in = max(free_in, world_pickers[k].free_in)   # held until they arrive
            carried = [i_index[i] for i in r.carried if i in i_index]
            world_transporters.append(Transporter(id=rid, cell=self._cell(rid), dest=dest, cap=r.capacity,
                                                  load=r.capacity - self._room(r), zone=r.zone is not None,
                                                  free_in=free_in, carried=carried))

        world_zones = [Zone(id=z, cell=self.docks[z].cell, occ=self._zone_holder(z) is not None) for z in zones]
        world = World(items=world_items, pickers=world_pickers, transporters=world_transporters, zones=world_zones,
                      distance=self.distance, agent_speed=self.spec.agent_speed,
                      remaining_frac=self.config.remaining_frac, time_frac=self.config.time_frac,
                      feasibility=self.feasibility)
        return world, pickers, transporters, zones

    # ================================================================ helpers

    def _cell(self, rid: str) -> tuple[int, int]:
        r = self.robots[rid]
        if r.pose is None:
            return (0, 0)
        return self.grid.cell_of(r.pose.x, r.pose.y)

    def _dest_cell(self, rid: str) -> tuple[int, int]:
        """Where a robot will be once its current task ends (its cell if idle)."""
        r = self.robots[rid]
        t = self.tasks.get(r.task_id) if r.task_id else None
        if t is not None and t.status in ACTIVE:
            return t.dest_cell
        return self._cell(rid)

    def _room(self, t: RobotRec) -> float:
        on_board = sum(self.items[i].weight for i in t.carried if i in self.items)
        incoming = sum(self.items[x.item_id].weight for x in self._incoming(t.id))
        return t.capacity - on_board - incoming

    def _zone_holder(self, zone_id: str) -> Optional[str]:
        for r in self.robots.values():
            if r.zone == zone_id:
                return r.id
        return None

    def slots_of(self, transporter_id: str) -> list[str]:
        configured = sorted(d.id for d in self.docks.values()
                            if d.type == "hand_off" and d.id.startswith(f"{transporter_id}_"))
        return configured or [f"{transporter_id}_slot{i}" for i in range(self.config.handoff_slots_default)]

    def free_slot(self, transporter_id: str) -> Optional[str]:
        for slot in self.slots_of(transporter_id):
            if slot not in self.slots:
                return slot
        return None

    def blacklisted(self, robot_id: str, target: str) -> bool:
        times = self.failures.get((robot_id, target), [])
        recent = [t for t in times if self.now() - t < self.config.blacklist_s]
        return len(recent) >= self.config.blacklist_after

    def _record_failure(self, robot_id: str, target: str):
        self.failures.setdefault((robot_id, target), []).append(self.now())

    def _add_dock(self, d: dict):
        dock = Dock(id=str(d["id"]), type=str(d["type"]), x=float(d["x"]), y=float(d["y"]), yaw=float(d.get("yaw", 0.0)))
        dock.cell = self.grid.cell_of(dock.x, dock.y)
        self.docks[dock.id] = dock

    @staticmethod
    def _dock_row(d: Dock) -> dict:
        return {"id": d.id, "type": d.type, "x": d.x, "y": d.y, "yaw": d.yaw}

    def _trigger_round(self):
        due = self.now() + self.config.round_debounce_s
        if self._round_due is None or due < self._round_due:
            self._round_due = due

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _command(self, robot_id: str, type_: str, payload) -> Out:
        env = make(type_, payload, robot_id=robot_id, session=self.session, seq=self._next_seq(),
                   epoch=self.epoch, sent_at=self.now())
        return Out("command", robot_id, env)

    def _broadcast(self, type_: str, payload) -> Out:
        env = make(type_, payload, session=self.session, seq=self._next_seq(), epoch=self.epoch, sent_at=self.now())
        return Out("broadcast", "", env)

    def _heartbeat_out(self) -> Out:
        return self._broadcast("coordinator.heartbeat",
                               m.CoordinatorHeartbeat(epoch=self.epoch, policy=getattr(self.policy, "name", "?"),
                                                      paused=self.paused))

    def _save(self, *, items=(), tasks=(), robots=(), event=None):
        self.store.commit(items=[asdict(i) for i in items], tasks=[asdict(t) for t in tasks],
                          robots=[r.persistent() for r in robots], event=event)
