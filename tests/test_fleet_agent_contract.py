"""
Phase 3: the robots' real fleet_agent core against the coordinator.

couliglig-main's FleetAgent (src/couliglig_task_manager/couliglig_task_manager/
fleet_agent/agent.py) replaces FakeRobot in SimFleet. A simulated executor
moves the robot on the grid and works the lift; everything else -- command
checks, events, the on-disk outbox, watchdog, restart recovery -- is the code
the robots run. Messages cross as bytes, between the server's protocol and
fleet_agent's copy, as on the wire.

Skipped when no couliglig-main checkout is found (COULIGLIG_MAIN, or
../couliglig-main next to this repo).

Run with:
    pytest tests/test_fleet_agent_contract.py -v
"""
import importlib
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from coordinator.core import ACTIVE, AVAILABLE, DELIVERED, HELD
from coordinator.policies import StayPolicy
from coordinator.protocol import PROTOCOL_HASH, make, parse
from coordinator.protocol import messages as sm
from coordinator.sim.fleet import Chaos, SimFleet

from tests.test_coordinator_core import SPEC, layout

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path(os.environ.get("COULIGLIG_MAIN", ROOT.parent / "couliglig-main"))
PKG = MAIN / "src" / "couliglig_task_manager"

if not (PKG / "couliglig_task_manager" / "fleet_agent" / "agent.py").exists():
    pytest.skip(f"no fleet_agent under {MAIN}", allow_module_level=True)
sys.path.insert(0, str(PKG))
agent_mod = importlib.import_module("couliglig_task_manager.fleet_agent.agent")
agent_protocol = importlib.import_module("couliglig_task_manager.fleet_agent.protocol")
am = agent_protocol.messages
FleetAgent, Action = agent_mod.FleetAgent, agent_mod.Action
START_LEG, CANCEL, STOP = agent_mod.START_LEG, agent_mod.CANCEL, agent_mod.STOP


class AgentRobot:
    """FleetAgent plus a simulated executor, shaped like FakeRobot for SimFleet."""

    def __init__(self, fleet: SimFleet, robot_id: str, role: str, cell, state_path, lift="DOWN",
                 speed_cells_s=None, capacity=4.0):
        self.fleet, self.id, self.role, self.capacity = fleet, robot_id, role, capacity
        self.state_path = state_path
        self.speed = speed_cells_s or fleet.speed
        self.cell = (float(cell[0]), float(cell[1]))
        self.agent = FleetAgent(robot_id, role, capacity=capacity, state_path=state_path, clock=fleet.clock)
        self.agent.set_lift(lift)
        self._sync_pose()
        self.leg = None              # (token, Leg, path, progress)
        self.stopping = False
        self.fail_next = None        # fail the next leg with this reason
        self.stuck = False           # legs never arrive (leg timeout)
        self._outbox = []            # SimFleet's own partition buffer (unused: the agent keeps its events)

    # ---- the FakeRobot surface SimFleet and the tests use
    @property
    def lift(self):
        return sm.LiftState(self.agent.lift.value)

    @property
    def exec_state(self):
        return sm.ExecState(self.agent.exec_state.value)

    @property
    def task(self):
        return self.agent.task

    @property
    def welcomed(self):
        return self.agent.welcomed

    @property
    def epoch(self):
        return self.agent.epoch

    @property
    def log(self):
        return list(self.agent.sent)

    def boot(self):
        self._act(self.agent.start())
        return self._drain()

    def receive(self, env):
        self._act(self.agent.receive(env.to_bytes()))
        return self._drain()

    def step(self, dt):
        if self.stopping:
            self.stopping = False
            self._act(self.agent.stopped())
        if self.leg is not None and not self.stuck:
            self._advance(dt)
        self._act(self.agent.step())
        return self._drain()

    # ---- executor
    def _act(self, actions):
        for a in actions:
            if a.kind == START_LEG:
                target = self.fleet.grid.cell_of(a.leg.pose.x, a.leg.pose.y)
                path = self.fleet.distance.path(self.cell, target) or [self.cell, target]
                self.leg = [a.token, a.leg, path, 0.0]
            elif a.kind in (CANCEL, STOP):
                self.leg = None
                self.stopping = True      # stopped() on the next step, as a real cancel takes a moment

    def _advance(self, dt):
        token, leg, path, progress = self.leg
        progress = min(progress + self.speed * dt, len(path) - 1)
        self.leg[3] = progress
        self.cell = path[min(int(progress), len(path) - 1)]
        self._sync_pose()
        if progress < len(path) - 1:
            return
        self.leg = None
        if self.fail_next:
            reason, self.fail_next = self.fail_next, None
            self._act(self.agent.leg_done(token, False, reason))
            return
        lift = {"UP": "UP", "DOWN": "DOWN"}.get(leg.lift.value)
        self._act(self.agent.leg_done(token, True, lift=lift))

    def _sync_pose(self):
        x, y = self.fleet.grid.position_of(self.cell)
        self.agent.set_pose(am.Pose(x=x, y=y, stamp=self.fleet.clock()))

    def _drain(self):
        """What the bus link would publish: outbox first, in order, then hello/heartbeat."""
        if self.id in self.fleet.partitioned or self.id in self.fleet.down:
            self.agent.take_transient()          # lost; events stay in the agent's outbox
            return []
        out = []
        for env in self.agent.outbox():
            out.append(parse(env.to_bytes()))
            self.agent.confirm(env.msg_id)
        out += [parse(env.to_bytes()) for env in self.agent.take_transient()]
        return out


def agent_factory(tmp_path):
    def make_robot(fleet, robot_id, role, cell, previous=None, **kw):
        if previous is not None:      # reboot: same disk, same physical state
            return AgentRobot(fleet, robot_id, previous.role, previous.cell, previous.state_path,
                              lift=previous.lift.value, speed_cells_s=previous.speed, capacity=previous.capacity)
        return AgentRobot(fleet, robot_id, role, cell, tmp_path / f"{robot_id}.json", **kw)
    return make_robot


def fleet(tmp_path, pickers=2, transporters=2, items=4, policy=None, chaos=None):
    grid, docks = layout()
    f = SimFleet(SPEC, grid, docks, policy=policy, chaos=chaos, robot_factory=agent_factory(tmp_path))
    for k in range(pickers):
        f.add_robot(f"couliglig_bot_{k + 1}", "picker", (2, 1 + k * (grid.width - 3) // max(1, pickers - 1)))
    for k in range(transporters):
        f.add_robot(f"couliglig_bot_{pickers + k + 1}", "transporter",
                    (grid.height - 3, 1 + k * (grid.width - 3) // max(1, transporters - 1)))
    f.run(1.0)
    for k in range(items):
        reply = f.op({"op": "add_item", "item_id": f"i{k}", "pickup_dock": f"P{k % 6}",
                      "receiver_dock": f"R{k % 2}"})
        assert reply["ok"], reply
    return f


def delivered_once(f: SimFleet) -> bool:
    """Every DELIVER task SUCCEEDED at most once per item, as the robots reported it."""
    items = Counter()
    for robot in f.robots.values():
        for task_id, status in robot.log:
            t = f.coord.tasks.get(task_id)
            if status == "SUCCEEDED" and t is not None and t.kind == "DELIVER":
                items[t.item_id] += 1
    return all(n == 1 for n in items.values())


def first_task(f, robot_id, kind=None):
    def found():
        t = f.robots[robot_id].task
        return t is not None and (kind is None or t.cmd.kind.value == kind)
    return found


# ===========================================================================

class TestProtocolCopy:

    def test_hash_matches(self):
        assert agent_protocol.PROTOCOL_HASH == PROTOCOL_HASH

    def test_copy_is_in_sync(self):
        script = MAIN / "scripts" / "sync_protocol.py"
        result = subprocess.run([sys.executable, str(script), "--check", "--server", str(ROOT)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_every_message_crosses_the_wire(self):
        samples = {
            "coordinator.welcome": sm.Welcome(epoch=3, accepted=True, hello_id="h", task_to_resume="t"),
            "command.task": sm.TaskCommand(task_id="t", attempt=1, kind=sm.TaskKind.PICK, valid_until=1.0,
                                           leg_timeout_s=60.0, legs=[sm.Leg(dock_id="P0", pose=sm.Pose(x=1, y=2),
                                                                           lift=sm.Lift.UP)]),
            "command.cancel": sm.CancelCommand(task_id="t", mode=sm.CancelMode.FINISH_LEG),
            "coordinator.heartbeat": sm.CoordinatorHeartbeat(epoch=3, policy="stay"),
            "coordinator.estop": sm.EStop(reason="r"),
            "coordinator.resume": sm.Resume(),
        }
        for type_, payload in samples.items():
            env = make(type_, payload, robot_id="couliglig_bot_1", epoch=3)
            theirs = agent_protocol.parse(env.to_bytes())
            assert theirs.body().model_dump(mode="json") == payload.model_dump(mode="json")
            assert parse(theirs.to_bytes()) == env


class TestEndToEnd:

    @pytest.mark.parametrize("policy", [None, "stay"])
    def test_delivers_everything(self, tmp_path, policy):
        f = fleet(tmp_path, items=6, policy=StayPolicy(SPEC) if policy else None)
        assert f.run(900, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)
        assert all(not r.agent.outbox() for r in f.robots.values())

    @pytest.mark.parametrize("seed", range(4))
    def test_duplicates_delays_and_loss(self, tmp_path, seed):
        chaos = Chaos(drop_telemetry=0.2, drop_commands=0.15, drop_broadcast=0.2, duplicate=0.3,
                      max_delay_s=0.8, seed=seed)
        f = fleet(tmp_path, items=6, chaos=chaos, policy=StayPolicy(SPEC) if seed % 2 else None)
        assert f.run(1800, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)

    @pytest.mark.parametrize("seed", range(3))
    def test_coordinator_restart_under_chaos(self, tmp_path, seed):
        chaos = Chaos(drop_telemetry=0.2, drop_commands=0.15, drop_broadcast=0.2, duplicate=0.3,
                      max_delay_s=0.8, seed=seed)
        f = fleet(tmp_path, items=6, chaos=chaos)
        f.run(60, until=lambda: sum(i.status == DELIVERED for i in f.coord.items.values()) >= 2)
        f.coordinator_down = True
        f.run(4.0)
        f.restart_coordinator()
        assert f.run(1800, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)
        assert all(r.epoch == f.coord.epoch for r in f.robots.values())


class TestFailures:

    def test_idle_robots_hold_and_come_back(self, tmp_path):
        f = fleet(tmp_path, items=0, policy=StayPolicy(SPEC))
        assert f.run(60, until=lambda: all(r.task is None and r.welcomed for r in f.robots.values())
                     and all(t.status not in ACTIVE for t in f.coord.tasks.values()))
        f.coordinator_down = True
        f.run(3.5)
        assert all(r.exec_state == sm.ExecState.HOLD for r in f.robots.values())
        f.restart_coordinator()
        f.run(3.0)
        assert all(r.exec_state != sm.ExecState.HOLD for r in f.robots.values())

    def test_failed_pick_rolls_back(self, tmp_path):
        f = fleet(tmp_path, pickers=1, transporters=1, items=1)
        picker = f.robots["couliglig_bot_1"]
        picker.fail_next = "dock blocked"
        assert f.run(120, until=lambda: any(t.status == "FAILED" for t in f.coord.tasks.values()))
        failed = next(t for t in f.coord.tasks.values() if t.status == "FAILED")
        assert (failed.kind, failed.reason) == ("PICK", "dock blocked")
        assert f.run(900, until=f.all_delivered)

    def test_leg_timeout_safe_stops_then_recovers(self, tmp_path):
        f = fleet(tmp_path, pickers=1, transporters=1, items=1)
        picker = f.robots["couliglig_bot_1"]
        assert f.run(30, until=first_task(f, "couliglig_bot_1", "PICK"))
        task_id = picker.task.cmd.task_id
        picker.stuck = True
        timeout = picker.task.cmd.leg_timeout_s
        states = set()
        assert f.run(timeout + 5, until=lambda: (states.add(picker.exec_state.value) or
                                                  f.coord.tasks[task_id].status == "FAILED"))
        assert f.coord.tasks[task_id].reason == "timeout"
        assert f.coord.items["i0"].status == AVAILABLE
        assert "SAFE_STOP" in states
        picker.stuck = False
        assert f.run(900, until=f.all_delivered)

    def test_picker_rebooted_while_picking(self, tmp_path):
        f = fleet(tmp_path, pickers=1, transporters=1, items=1)
        assert f.run(30, until=first_task(f, "couliglig_bot_1", "PICK"))
        task_id = f.robots["couliglig_bot_1"].task.cmd.task_id
        f.reboot_robot("couliglig_bot_1")        # lift still down: it never picked
        f.run(0.5)
        first = f.coord.tasks[task_id]
        assert first.status == "FAILED"
        assert first.reason == "agent_restarted"
        assert f.coord.items["i0"].status in (AVAILABLE, "CLAIMED")   # rolled back (and maybe re-planned)
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)

    def test_picker_rebooted_after_lifting_before_reporting_it(self, tmp_path):
        """The lift went up, then the agent died before its LIFT_UP event was written."""
        f = fleet(tmp_path, pickers=1, transporters=1, items=1)
        picker = f.robots["couliglig_bot_1"]
        assert f.run(30, until=first_task(f, "couliglig_bot_1", "PICK"))
        task_id = picker.task.cmd.task_id
        picker.agent.lift = am.LiftState.UP        # physically lifted; the agent never heard about it
        rebooted = f.reboot_robot("couliglig_bot_1")
        assert rebooted.lift == sm.LiftState.UP
        f.run(2.0)
        assert f.coord.tasks[task_id].status == "FAILED"
        assert f.coord.items["i0"].status == HELD   # the box is on the picker, not back on the shelf
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)

    def test_picker_rebooted_holding_an_item(self, tmp_path):
        f = fleet(tmp_path, pickers=1, transporters=1, items=1)
        assert f.run(120, until=lambda: f.coord.items["i0"].status == HELD)
        picker = f.reboot_robot("couliglig_bot_1")
        assert picker.lift == sm.LiftState.UP
        f.run(2.0)
        assert f.coord.items["i0"].status != AVAILABLE
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)

    def test_duplicate_command_after_reboot_is_not_run(self, tmp_path):
        f = fleet(tmp_path, pickers=1, transporters=1, items=1)
        assert f.run(30, until=first_task(f, "couliglig_bot_1", "PICK"))
        cmd = f.robots["couliglig_bot_1"].task.cmd
        replay = make("command.task", sm.TaskCommand.model_validate(cmd.model_dump(mode="json")),
                      robot_id="couliglig_bot_1", epoch=f.coord.epoch, sent_at=f.clock())
        robot = f.reboot_robot("couliglig_bot_1")
        f.run(1.0)
        before = len(robot.log)
        assert robot.receive(replay) == []
        assert len(robot.log) == before                    # not even a REJECTED
        assert robot.task is None or robot.task.cmd.task_id != cmd.task_id

    def test_partitioned_transporter_keeps_its_events_on_disk(self, tmp_path):
        f = fleet(tmp_path, pickers=1, transporters=1, items=2)
        transporter = f.robots["couliglig_bot_2"]
        assert f.run(300, until=first_task(f, "couliglig_bot_2", "DELIVER"))
        f.partitioned.add("couliglig_bot_2")
        assert f.run(60, until=lambda: transporter.task is None)   # finished the delivery offline
        assert transporter.agent.outbox()                            # SUCCEEDED waits on disk
        assert transporter.exec_state == sm.ExecState.HOLD
        f.heal("couliglig_bot_2")
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)

    def test_estop_and_resume(self, tmp_path):
        f = fleet(tmp_path, items=4)
        f.run(5.0)
        f.op({"op": "estop", "reason": "test"})
        f.run(1.0)
        assert all(r.exec_state == sm.ExecState.ESTOP and r.task is None for r in f.robots.values())
        commands = f.coord.metrics["commands"]
        f.run(5.0)
        assert f.coord.metrics["commands"] == commands
        f.op({"op": "resume"})
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)
        assert not f.coord.blacklisted("couliglig_bot_1", "P0")   # an e-stop is not the target's fault
