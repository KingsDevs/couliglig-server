"""Coordinator plus FakeRobots in one process, on a simulated clock.

Messages pass through a channel that can drop, duplicate and delay them, so
the protocol's guarantees can be tested deterministically without a broker:

    telemetry  may be lost (heartbeats)
    events     at least once: may be duplicated or delayed, never lost
               (fleet_agent keeps an outbox; the broker queue is durable)
    commands   may be duplicated or delayed; a lost command expires
    broadcast  may be lost

Kill/restart helpers simulate a robot reboot, a coordinator restart and a
network outage.
"""
from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field

from coordinator.core import Config, Coordinator, DELIVERED
from coordinator.grid import Grid, GridDistance
from coordinator.policies import GreedyPolicy
from coordinator.protocol import Envelope, parse
from coordinator.sim.fake_robot import FakeRobot
from coordinator.spec.policy_spec import PolicySpec
from coordinator.store import Store


class SimClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@dataclass
class Chaos:
    drop_telemetry: float = 0.0
    drop_commands: float = 0.0
    drop_broadcast: float = 0.0
    duplicate: float = 0.0
    max_delay_s: float = 0.0
    seed: int = 0


def _fake_robot(fleet: "SimFleet", robot_id: str, role: str, cell, previous=None, **kw) -> FakeRobot:
    if previous is not None:
        robot = FakeRobot(robot_id, previous.role, previous.cell, fleet.grid, fleet.clock, distance=fleet.distance,
                          speed_cells_s=previous.speed, capacity=previous.capacity)
        robot.lift = previous.lift
        return robot
    return FakeRobot(robot_id, role, cell, fleet.grid, fleet.clock, distance=fleet.distance,
                     speed_cells_s=kw.pop("speed_cells_s", fleet.speed), **kw)


@dataclass(order=True)
class _Msg:
    due: float
    order: int
    to: str = field(compare=False)        # "coordinator" or a robot id
    env: Envelope = field(compare=False)


class SimFleet:
    def __init__(self, spec: PolicySpec, grid: Grid, docks: list[dict], policy=None, config: Config | None = None,
                 store: Store | None = None, chaos: Chaos | None = None, speed_cells_s: float = 2.0,
                 dt: float = 0.1, robot_factory=None):
        self.spec, self.grid, self.dt = spec, grid, dt
        self.clock = SimClock()
        self.distance = GridDistance(grid)
        self.config = config or Config()
        self.store = store or Store()
        self.policy = policy or GreedyPolicy(spec)
        self.chaos = chaos or Chaos()
        self.rng = random.Random(self.chaos.seed)
        self.speed = speed_cells_s
        # robot_factory(fleet, robot_id, role, cell, previous=None, **kw) -> a robot; previous is the
        # robot being rebooted (its physical state survives). Default: FakeRobot.
        self.robot_factory = robot_factory or _fake_robot
        self.robots: dict[str, FakeRobot] = {}
        self.down: set[str] = set()          # robots that are off (no messages either way)
        self.partitioned: set[str] = set()   # robots cut off from the network (keep running)
        self.coordinator_down = False
        self._queue: list[_Msg] = []
        self._held: list[_Msg] = []
        self._order = itertools.count()
        self.coord = self._new_coordinator()
        self.coord.start()
        reply, _ = self.coord.handle_op({"op": "activate", "docks": docks})
        assert reply["ok"], reply

    def _new_coordinator(self) -> Coordinator:
        return Coordinator(self.store, self.policy, self.spec, self.grid, distance=self.distance,
                           config=self.config, clock=self.clock)

    # ---------------- fleet management

    def add_robot(self, robot_id: str, role: str, cell, **kw) -> FakeRobot:
        robot = self.robot_factory(self, robot_id, role, cell, None, **kw)
        self.robots[robot_id] = robot
        self._boot(robot_id, robot)
        return robot

    def _boot(self, robot_id: str, robot):
        # a robot with an outbox (fleet_agent) sends what it kept before its hello
        self._robot_outputs(robot_id, robot.boot() if hasattr(robot, "boot") else [robot.hello()])

    def op(self, op: dict) -> dict:
        reply, outs = self.coord.handle_op(op)
        self._route(outs)
        return reply

    def reboot_robot(self, robot_id: str) -> FakeRobot:
        """Power cycle: a new fleet_agent session; the physical state (cell, lift) survives."""
        old = self.robots[robot_id]
        robot = self.robot_factory(self, robot_id, old.role, old.cell, old)
        self.robots[robot_id] = robot
        self.down.discard(robot_id)
        self._boot(robot_id, robot)
        return robot

    def restart_coordinator(self):
        """Crash and restart: in-memory state is lost, SQLite survives, epoch += 1."""
        self.coord = self._new_coordinator()
        self.coord.start()
        self.coordinator_down = False
        for msg in self._held:
            msg.due = self.clock.t
            heapq.heappush(self._queue, msg)
        self._held = []

    # ---------------- messaging

    def _send(self, src: str, to: str, env: Envelope, kind: str):
        if to != "coordinator" and to != "*" and (to in self.down or to in self.partitioned):
            return
        if src in self.down or src in self.partitioned:
            if kind == "event" and src in self.partitioned:
                self.robots[src]._outbox = getattr(self.robots[src], "_outbox", []) + [env]
            return
        drop = {"telemetry": self.chaos.drop_telemetry, "command": self.chaos.drop_commands,
                "broadcast": self.chaos.drop_broadcast}.get(kind, 0.0)
        if self.rng.random() < drop:
            return
        copies = 2 if kind != "telemetry" and self.rng.random() < self.chaos.duplicate else 1
        for _ in range(copies):
            delay = self.rng.uniform(0, self.chaos.max_delay_s) if self.chaos.max_delay_s else 0.0
            # serialize like the wire does
            heapq.heappush(self._queue, _Msg(self.clock.t + delay, next(self._order), to, parse(env.to_bytes())))

    def _route(self, outs):
        for out in outs:
            if out.kind == "command":
                self._send("coordinator", out.robot_id, out.envelope, kind="command")
            elif out.kind == "broadcast":
                for rid in self.robots:
                    self._send("coordinator", rid, out.envelope, kind="broadcast")

    def _robot_outputs(self, rid: str, envs):
        for env in envs:
            kind = "telemetry" if env.type == "robot.heartbeat" else "event"
            self._send(rid, "coordinator", env, kind=kind)

    def heal(self, robot_id: str):
        """End a network partition; the robot's outbox (events) is flushed."""
        self.partitioned.discard(robot_id)
        robot = self.robots[robot_id]
        for env in getattr(robot, "_outbox", []):
            self._send(robot_id, "coordinator", env, kind="event")
        robot._outbox = []

    # ---------------- time

    def step(self):
        self.clock.t += self.dt
        while self._queue and self._queue[0].due <= self.clock.t:
            msg = heapq.heappop(self._queue)
            if msg.to == "coordinator":
                if not self.coordinator_down:
                    self._route(self.coord.handle(msg.env))
                elif msg.env.type != "robot.heartbeat":
                    self._held.append(msg)   # durable queue: waits for the coordinator
            elif msg.to not in self.down and msg.to not in self.partitioned:
                self._robot_outputs(msg.to, self.robots[msg.to].receive(msg.env))
        for rid, robot in self.robots.items():
            if rid not in self.down:
                self._robot_outputs(rid, robot.step(self.dt))
        if not self.coordinator_down:
            self._route(self.coord.tick())

    def run(self, seconds: float, until=None) -> bool:
        """Advance; True as soon as until() holds (if given)."""
        end = self.clock.t + seconds
        while self.clock.t < end:
            self.step()
            if until is not None and until():
                return True
        return until() if until is not None else True

    # ---------------- checks

    def delivered(self) -> set[str]:
        return {i.id for i in self.coord.items.values() if i.status == DELIVERED}

    def all_delivered(self) -> bool:
        return bool(self.coord.items) and all(i.status == DELIVERED for i in self.coord.items.values())
