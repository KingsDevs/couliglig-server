"""
Phase 2c: the coordinator core with simulated robots, in-process.

SimFleet runs the Coordinator and FakeRobots on a simulated clock and can
drop, duplicate and delay messages, restart the coordinator, reboot or
power off robots, and cut a robot off the network.

Run with:
    pytest tests/test_coordinator_core.py -v
"""
from collections import Counter
from pathlib import Path

import pytest

from coordinator.core import (
    ACTIVE, AVAILABLE, CLAIMED, DELIVERED, HELD, OFFLINE, REFUSED, Config, Coordinator,
)
from coordinator.grid import Grid
from coordinator.policies import AssignPolicy, FallbackPolicy, GreedyPolicy, MarlOnnxPolicy, StayPolicy
from coordinator.protocol import make
from coordinator.protocol import messages as m
from coordinator.sim.fake_robot import FakeRobot
from coordinator.sim.fleet import Chaos, SimFleet
from coordinator.spec import load_spec
from coordinator.store import Store

MODEL_DIR = Path(__file__).resolve().parent.parent / "policy_models" / "imit_pointer_obs_tc-round_2"
SPEC = load_spec(MODEL_DIR)


def layout(width=24, height=16, pickups=6, receivers=2, zones=3):
    grid = Grid.open(width, height)

    def dock(r, c, kind, name):
        x, y = grid.position_of((r, c))
        return {"id": name, "type": kind, "x": x, "y": y}

    docks = [dock(0, 2 + k * (width - 4) // max(1, pickups - 1), "pickup", f"P{k}") for k in range(pickups)]
    docks += [dock(height - 1, 3 + k * (width - 6) // max(1, receivers - 1), "receiver", f"R{k}")
              for k in range(receivers)]
    docks += [dock(height // 2, 3 + k * (width - 6) // max(1, zones - 1), "waiting_zone", f"Z{k}")
              for k in range(zones)]
    return grid, docks


def fleet(pickers=2, transporters=2, items=4, policy=None, chaos=None, config=None, store=None, **kw):
    grid, docks = layout(**kw)
    f = SimFleet(SPEC, grid, docks, policy=policy, chaos=chaos, config=config, store=store)
    for k in range(pickers):
        f.add_robot(f"couliglig_bot_{k + 1}", "picker", (2, 1 + k * (grid.width - 3) // max(1, pickers - 1)))
    for k in range(transporters):
        f.add_robot(f"couliglig_bot_{pickers + k + 1}", "transporter",
                    (grid.height - 3, 1 + k * (grid.width - 3) // max(1, transporters - 1)))
    f.run(1.0)  # hello / welcome
    for k in range(items):
        reply = f.op({"op": "add_item", "item_id": f"i{k}", "pickup_dock": f"P{k % 6}",
                      "receiver_dock": f"R{k % 2}"})
        assert reply["ok"], reply
    return f


def delivered_once(f: SimFleet) -> bool:
    """No item was delivered twice (SUCCEEDED of DELIVER tasks, as the robots sent them)."""
    deliveries = Counter()
    for robot in f.robots.values():
        for task_id, status in robot.log:
            if status == "SUCCEEDED":
                t = f.store._db.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()
                if t and '"DELIVER"' in t[0]:
                    deliveries[task_id] += 1
    items = Counter()
    for task_id in deliveries:
        import json
        data = json.loads(f.store._db.execute("SELECT data FROM tasks WHERE id=?", (task_id,)).fetchone()[0])
        items[data["item_id"]] += 1
    return all(n == 1 for n in items.values())


# ===========================================================================
# End to end
# ===========================================================================

class TestEndToEnd:

    @pytest.mark.parametrize("policy_cls", [GreedyPolicy, StayPolicy, AssignPolicy])
    def test_heuristics_deliver_everything(self, policy_cls):
        f = fleet(policy=policy_cls(SPEC), items=6)
        assert f.run(900, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)
        assert f.coord.metrics["ignored_events"] == 0

    def test_trained_policy_delivers_everything(self):
        f = fleet(policy=FallbackPolicy(MarlOnnxPolicy(MODEL_DIR), GreedyPolicy(SPEC)), items=6)
        assert f.run(900, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)

    def test_on_the_real_map(self):
        grid = Grid.from_map_yaml(Path(__file__).resolve().parent.parent / "db" / "maps" / "map.yaml")
        free = sorted(c for c in ((r, cc) for r in range(grid.height) for cc in range(grid.width)) if grid.free(c))

        def dock(cell, kind, name):
            x, y = grid.position_of(cell)
            return {"id": name, "type": kind, "x": x, "y": y}

        docks = [dock(free[0], "pickup", "P0"), dock(free[5], "pickup", "P1"), dock(free[-1], "receiver", "R0"),
                 dock(free[len(free) // 2], "waiting_zone", "Z0")]
        f = SimFleet(SPEC, grid, docks)
        f.add_robot("couliglig_bot_1", "picker", free[10])
        f.add_robot("couliglig_bot_2", "transporter", free[60])
        f.run(1.0)
        for k in range(2):
            assert f.op({"op": "add_item", "item_id": f"i{k}", "pickup_dock": f"P{k}", "receiver_dock": "R0"})["ok"]
        assert f.run(900, until=f.all_delivered), f.op({"op": "snapshot"})


# ===========================================================================
# Conflict resolution (the env's _tick rules)
# ===========================================================================

class TestConflicts:

    def _pick_tasks(self, f):
        return [t for t in f.coord.tasks.values() if t.kind == "PICK"]

    def test_one_item_two_pickers_tie_goes_to_lower_id(self):
        grid, docks = layout(pickups=1, receivers=1, zones=1)
        f = SimFleet(SPEC, grid, docks)
        p = next(d for d in docks if d["id"] == "P0")
        r, c = grid.cell_of(p["x"], p["y"])
        f.add_robot("couliglig_bot_12", "picker", (r + 3, c + 2))
        f.add_robot("couliglig_bot_2", "picker", (r + 3, c - 2))   # same distance, lower id
        f.run(1.0)
        f.op({"op": "add_item", "item_id": "i0", "pickup_dock": "P0", "receiver_dock": "R0"})
        f.run(2.0)
        picks = self._pick_tasks(f)
        assert [t.robot_id for t in picks] == ["couliglig_bot_2"]

    def test_closer_picker_wins(self):
        grid, docks = layout(pickups=1, receivers=1, zones=1)
        f = SimFleet(SPEC, grid, docks)
        p = next(d for d in docks if d["id"] == "P0")
        r, c = grid.cell_of(p["x"], p["y"])
        f.add_robot("couliglig_bot_1", "picker", (r + 6, c))
        f.add_robot("couliglig_bot_2", "picker", (r + 2, c))
        f.run(1.0)
        f.op({"op": "add_item", "item_id": "i0", "pickup_dock": "P0", "receiver_dock": "R0"})
        f.run(2.0)
        assert [t.robot_id for t in self._pick_tasks(f)] == ["couliglig_bot_2"]

    def test_one_zone_two_transporters(self):
        grid, docks = layout(pickups=1, receivers=1, zones=1)
        f = SimFleet(SPEC, grid, docks)
        z = next(d for d in docks if d["type"] == "waiting_zone")
        r, c = grid.cell_of(z["x"], z["y"])
        f.add_robot("couliglig_bot_3", "transporter", (r + 2, c + 3))
        f.add_robot("couliglig_bot_4", "transporter", (r + 2, c - 3))
        f.run(3.0)
        zones = [t for t in f.coord.tasks.values() if t.kind == "PARK_ZONE"]
        assert [t.robot_id for t in zones] == ["couliglig_bot_3"]

    def test_transporter_never_over_capacity(self):
        f = fleet(pickers=3, transporters=1, items=5)
        t = f.coord.robots["couliglig_bot_4"]
        t.capacity = 2.0   # (the robot's hello said 4; shrink it to force contention)
        worst = 0.0
        for _ in range(9000):
            f.step()
            worst = min(worst, f.coord._room(t))
            if f.all_delivered():
                break
        assert f.all_delivered()
        assert worst >= 0.0


# ===========================================================================
# Imperfect delivery
# ===========================================================================

class TestImperfectDelivery:

    @pytest.mark.parametrize("seed", range(6))
    def test_duplicates_delays_and_loss(self, seed):
        chaos = Chaos(drop_telemetry=0.2, drop_commands=0.15, drop_broadcast=0.2, duplicate=0.3,
                      max_delay_s=0.8, seed=seed)
        f = fleet(items=6, chaos=chaos, policy=StayPolicy(SPEC) if seed % 2 else None)
        assert f.run(1800, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)
        assert f.coord.metrics["duplicate_events"] > 0

    @pytest.mark.parametrize("seed", range(6))
    def test_coordinator_restart_under_chaos(self, seed):
        """Seed 1 once delivered an item twice: a late hello made the coordinator
        roll back a delivery the robot then completed."""
        chaos = Chaos(drop_telemetry=0.2, drop_commands=0.15, drop_broadcast=0.2, duplicate=0.3,
                      max_delay_s=0.8, seed=seed)
        f = fleet(items=6, chaos=chaos, policy=StayPolicy(SPEC) if seed % 2 else None)
        f.run(60, until=lambda: sum(i.status == DELIVERED for i in f.coord.items.values()) >= 2)
        f.coordinator_down = True
        f.run(4.0)
        f.restart_coordinator()
        assert f.run(1800, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)

    def test_old_heartbeat_dropped(self):
        f = fleet(items=0)
        robot = f.robots["couliglig_bot_1"]
        newer = robot.heartbeat()
        older = make("robot.heartbeat", newer.body(), robot_id=robot.id, session=robot.session, seq=newer.seq - 5)
        f.coord.handle(newer)
        before = f.coord.metrics["stale_telemetry"]
        f.coord.handle(older)
        assert f.coord.metrics["stale_telemetry"] == before + 1

    def test_lost_command_expires_and_is_reassigned(self):
        f = fleet(pickers=2, transporters=1, items=1)
        first = next(t for t in f.coord.tasks.values() if t.kind == "PICK") if f.run(
            3, until=lambda: any(t.kind == "PICK" for t in f.coord.tasks.values())) else None
        assert first is not None
        # pretend the command never arrived: the robot never accepts
        f.robots[first.robot_id].seen.add((first.id, first.attempt))
        f.robots[first.robot_id].task = None
        f.robots[first.robot_id].exec_state = m.ExecState.IDLE
        f.coord.tasks[first.id].status = "SENT"
        assert f.run(f.config.command_ttl_s + f.config.accept_grace_s + 2,
                     until=lambda: f.coord.tasks[first.id].status == "EXPIRED")
        assert f.run(900, until=f.all_delivered)


class TestRobotChecks:

    def _robot(self):
        grid, _ = layout()
        clock = lambda: 100.0  # noqa: E731
        robot = FakeRobot("couliglig_bot_1", "picker", (2, 2), grid, clock)
        robot.receive(make("coordinator.welcome", m.Welcome(epoch=5, accepted=True), epoch=5))
        return robot

    def _task(self, valid_until, epoch):
        cmd = m.TaskCommand(task_id="t1", attempt=1, kind=m.TaskKind.PICK,
                            legs=[m.Leg(dock_id="P0", pose=m.Pose(x=1.0, y=1.0), lift=m.Lift.UP)],
                            valid_until=valid_until, leg_timeout_s=60)
        return make("command.task", cmd, robot_id="couliglig_bot_1", epoch=epoch)

    def test_expired_command_rejected(self):
        events = self._robot().receive(self._task(valid_until=99.0, epoch=5))
        assert [e.body().status for e in events] == [m.TaskStatus.REJECTED]
        assert events[0].body().reason == "stale"

    def test_old_epoch_rejected(self):
        events = self._robot().receive(self._task(valid_until=200.0, epoch=4))
        assert events[0].body().reason == "stale epoch"

    def test_duplicate_command_ignored(self):
        robot = self._robot()
        assert robot.receive(self._task(valid_until=200.0, epoch=5))
        assert robot.receive(self._task(valid_until=200.0, epoch=5)) == []


# ===========================================================================
# Failures and recovery
# ===========================================================================

class TestFailures:

    def test_failed_pick_rolls_back_and_blacklists(self):
        f = fleet(pickers=1, transporters=1, items=0, policy=StayPolicy(SPEC))
        picker = f.robots["couliglig_bot_1"]
        picker.fail_next, picker.fail_count = "dock blocked", 1      # the next two legs fail
        f.op({"op": "add_item", "item_id": "i0", "pickup_dock": "P0", "receiver_dock": "R0"})
        assert f.run(120, until=lambda: picker.fail_next is None and picker.task is None)
        f.run(1.0)   # let the last FAILED event arrive
        failed = [t for t in f.coord.tasks.values() if t.kind == "PICK" and t.status == "FAILED"]
        assert len(failed) == 2
        assert f.coord.items["i0"].status == AVAILABLE
        assert f.coord.blacklisted("couliglig_bot_1", "P0")
        f.run(5)
        assert f.coord.items["i0"].status == AVAILABLE   # not offered again while blacklisted

    def test_coordinator_restart_mid_run(self):
        f = fleet(items=6)
        assert f.run(60, until=lambda: sum(i.status == DELIVERED for i in f.coord.items.values()) >= 2)
        assert any(t.status in ACTIVE for t in f.coord.tasks.values())
        old_epoch = f.coord.epoch
        f.coordinator_down = True
        f.run(5.0)
        f.restart_coordinator()
        assert f.coord.epoch == old_epoch + 1
        assert f.run(900, until=f.all_delivered), f.op({"op": "snapshot"})
        assert delivered_once(f)
        assert all(r.epoch == f.coord.epoch for r in f.robots.values())

    def test_idle_robots_hold_within_3s_of_losing_the_coordinator(self):
        f = fleet(items=0, policy=StayPolicy(SPEC))
        assert f.run(60, until=lambda: all(r.task is None and r.welcomed for r in f.robots.values())
                     and all(t.status not in ACTIVE for t in f.coord.tasks.values()))
        f.coordinator_down = True
        f.run(3.5)
        assert all(r.exec_state == m.ExecState.HOLD for r in f.robots.values())
        f.restart_coordinator()
        f.run(3.0)
        assert all(r.exec_state != m.ExecState.HOLD for r in f.robots.values())

    def test_picker_rebooted_while_picking(self):
        f = fleet(pickers=1, transporters=1, items=1)
        assert f.run(30, until=lambda: f.robots["couliglig_bot_1"].task is not None)
        task_id = f.robots["couliglig_bot_1"].task.cmd.task_id
        f.reboot_robot("couliglig_bot_1")      # lift still down: it never picked
        f.run(0.5)
        first = f.coord.tasks[task_id]
        assert (first.status, first.reason) == ("FAILED", "robot_restarted")
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)

    def test_picker_rebooted_holding_an_item(self):
        f = fleet(pickers=1, transporters=1, items=1)
        assert f.run(120, until=lambda: f.coord.items["i0"].status == HELD)
        picker = f.reboot_robot("couliglig_bot_1")
        assert picker.lift == m.LiftState.UP
        f.run(2.0)
        assert f.coord.items["i0"].status != AVAILABLE   # never handed to someone else
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)

    def test_powered_off_picker_goes_offline_and_work_moves(self):
        f = fleet(pickers=2, transporters=1, items=3)
        f.down.add("couliglig_bot_1")
        f.run(f.config.offline_s + 2)
        assert f.coord.robots["couliglig_bot_1"].status == OFFLINE
        assert f.run(900, until=f.all_delivered)

    def test_partitioned_transporter_keeps_its_events(self):
        f = fleet(pickers=1, transporters=1, items=2)
        assert f.run(300, until=lambda: f.robots["couliglig_bot_2"].task is not None
                     and f.robots["couliglig_bot_2"].task.cmd.kind == m.TaskKind.DELIVER)
        f.partitioned.add("couliglig_bot_2")
        f.run(10.0)
        f.heal("couliglig_bot_2")
        assert f.run(900, until=f.all_delivered)
        assert delivered_once(f)

    def test_estop_and_resume(self):
        f = fleet(items=4)
        f.run(5.0)
        f.op({"op": "estop", "reason": "test"})
        f.run(1.0)
        assert all(r.exec_state == m.ExecState.ESTOP for r in f.robots.values())
        commands = f.coord.metrics["commands"]
        f.run(5.0)
        assert f.coord.metrics["commands"] == commands     # paused: no new work
        f.op({"op": "resume"})
        assert f.run(900, until=f.all_delivered)


# ===========================================================================
# Registration, operator calls, persistence
# ===========================================================================

class TestRegistration:

    def test_wrong_protocol_hash_refused(self):
        f = fleet(items=0)
        robot = f.robots["couliglig_bot_1"]
        hello = robot.hello()
        hello.payload["protocol_hash"] = "0" * 16
        outs = f.coord.handle(hello)
        welcome = next(o.envelope.body() for o in outs if o.envelope and o.envelope.type == "coordinator.welcome")
        assert not welcome.accepted and "protocol_hash" in welcome.reason
        assert f.coord.robots["couliglig_bot_1"].status == REFUSED

    def test_add_item_validation(self):
        f = fleet(items=0)
        assert not f.op({"op": "add_item", "item_id": "x", "pickup_dock": "R0", "receiver_dock": "R1"})["ok"]
        assert not f.op({"op": "add_item", "item_id": "x", "pickup_dock": "P0", "receiver_dock": "nope"})["ok"]
        assert f.op({"op": "add_item", "item_id": "x", "pickup_dock": "P0", "receiver_dock": "R0"})["ok"]
        assert not f.op({"op": "add_item", "item_id": "y", "pickup_dock": "P0", "receiver_dock": "R0"})["ok"]
        assert not f.op({"op": "add_item", "item_id": "x", "pickup_dock": "P1", "receiver_dock": "R0"})["ok"]
        assert not f.op({"op": "nonsense"})["ok"]

    def test_manual_fleet_gets_no_tasks_until_auto(self, tmp_path):
        store = Store(tmp_path / "coordinator.db")
        store.set_meta("paused", True)               # as a manual coordinator left it
        f = fleet(items=2, store=store)
        f.run(20.0)
        assert f.coord.metrics["commands"] == 0
        assert f.op({"op": "snapshot"})["control"] == "manual"
        f.op({"op": "resume"})
        assert f.op({"op": "snapshot"})["control"] == "auto"
        assert f.run(900, until=f.all_delivered)

    def test_manual_or_auto_is_kept_across_restarts(self, tmp_path):
        grid, _ = layout()

        def start(**kw):
            c = Coordinator(Store(tmp_path / "coordinator.db"), StayPolicy(SPEC), SPEC, grid, **kw)
            c.start()
            return c

        c = start(manual=True)                       # a fresh store takes the default
        assert c.paused
        c.handle_op({"op": "resume"})
        c.store.close()
        c = start(manual=True)                       # then the last setting wins
        assert not c.paused
        c.handle_op({"op": "estop", "reason": "test"})
        c.store.close()
        assert start().paused                        # e-stop leaves it manual

    def test_switching_to_manual_lets_running_tasks_finish(self):
        f = fleet(items=4)
        assert f.run(60.0, until=lambda: any(t.status in ACTIVE for t in f.coord.tasks.values()))
        running = [t.id for t in f.coord.tasks.values() if t.status in ACTIVE]
        f.op({"op": "pause"})
        commands = f.coord.metrics["commands"]
        assert f.run(500.0, until=lambda: not any(t.status in ACTIVE for t in f.coord.tasks.values()))
        f.run(30.0)
        final = {t["id"]: t["status"] for t in f.coord.store.load_tasks()}
        assert all(final[t] == "SUCCEEDED" for t in running)
        assert f.coord.metrics["commands"] == commands      # nothing new was sent

    def test_operator_picks_the_policy(self, tmp_path):
        from coordinator.policies import make_policy_set
        config = {"primary": "stay", "fallback": "stay", "marl_onnx": {"model_dir": str(MODEL_DIR)}}
        grid, _ = layout()

        def start():
            choices, unavailable, default, spec = make_policy_set(config)
            c = Coordinator(Store(tmp_path / "coordinator.db"), choices[default], spec, grid,
                            choices=choices, unavailable=unavailable, policy_name=default)
            c.start()
            return c

        c = start()
        snap = c.handle_op({"op": "snapshot"})[0]
        assert snap["policy"] == "stay"
        assert [(p["name"], p["kind"], p["available"]) for p in snap["policies"]] == [
            ("stay", "heuristic", True), ("greedy", "heuristic", True), ("assign", "heuristic", True),
            ("marl_onnx", "rl", True)]
        reply, outs = c.handle_op({"op": "set_policy", "policy": "marl_onnx"})
        assert reply == {"ok": True, "policy": "marl_onnx"}
        assert outs[0].envelope.body().policy == "marl_onnx"          # robots see it in the heartbeat
        assert c.policy is c.choices["marl_onnx"] and c.policy.fallback is c.choices["stay"]
        assert not c.handle_op({"op": "set_policy", "policy": "random"})[0]["ok"]
        c.store.close()
        assert start().policy_name == "marl_onnx"                    # kept across restarts

    def test_a_broken_model_is_unavailable_not_fatal(self, tmp_path):
        from coordinator.policies import make_policy_set
        (tmp_path / "model").mkdir()
        for f in ("policy.json", "fixtures.json"):
            (tmp_path / "model" / f).write_bytes((MODEL_DIR / f).read_bytes())   # no .onnx files
        choices, unavailable, default, spec = make_policy_set(
            {"primary": "marl_onnx", "fallback": "greedy", "marl_onnx": {"model_dir": str(tmp_path / "model")}})
        assert sorted(choices) == ["assign", "greedy", "stay"] and "marl_onnx" in unavailable
        assert default == "greedy"                                   # primary unavailable: the fallback
        c = Coordinator(Store(), choices[default], spec, layout()[0], choices=choices,
                        unavailable=unavailable, policy_name=default)
        c.start()
        reply = c.handle_op({"op": "set_policy", "policy": "marl_onnx"})[0]
        assert not reply["ok"] and "not available" in reply["error"]

    def test_activate_brings_the_configs_map(self, tmp_path):
        # a 6 m x 3 m free map whose origin is at (-2, -1)
        (tmp_path / "m.pgm").write_bytes(b"P5\n120 60\n255\n" + bytes([254]) * 120 * 60)
        (tmp_path / "m.yaml").write_text("image: m.pgm\nresolution: 0.05\norigin: [-2.0, -1.0, 0]\n"
                                         "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
        docks = [{"id": "P0", "type": "pickup", "x": -1.5, "y": 1.5}, {"id": "R0", "type": "receiver", "x": 3.5, "y": 0.0}]

        def start():
            c = Coordinator(Store(tmp_path / "coordinator.db"), StayPolicy(SPEC), SPEC, Grid.open(40, 40))
            c.start()
            return c

        c = start()
        reply, _ = c.handle_op({"op": "activate", "docks": docks, "map_yaml": str(tmp_path / "m.yaml")})
        assert reply["ok"] and reply["grid"] == [4, 8]
        assert c.docks["P0"].cell == (0, 0) and c.grid.free(c.docks["R0"].cell)
        c.store.close()
        again = start()                              # the map is kept across restarts
        assert (again.grid.height, again.grid.width) == (4, 8)
        assert again.docks["R0"].cell == c.docks["R0"].cell
        reply, _ = again.handle_op({"op": "activate", "docks": docks, "map_yaml": str(tmp_path / "nope.yaml")})
        assert not reply["ok"] and "cannot load map" in reply["error"]

    def test_state_survives_restart_from_disk(self, tmp_path):
        store = Store(tmp_path / "coordinator.db")
        f = fleet(items=3, store=store)
        f.run(20.0)
        statuses = {i.id: i.status for i in f.coord.items.values()}
        epoch = f.coord.epoch
        store.close()
        grid, _ = layout()
        again = Coordinator(Store(tmp_path / "coordinator.db"), GreedyPolicy(SPEC), SPEC, grid)
        assert again.start() == epoch + 1
        assert {i.id: i.status for i in again.items.values()} == statuses
        assert len(again.docks) == len(f.coord.docks)


# ===========================================================================
# Performance
# ===========================================================================

class TestRoundCost:

    @pytest.mark.parametrize("policy", ["greedy", "assign", "marl"])
    def test_round_for_15_robots_under_100ms(self, policy):
        chosen = {"greedy": GreedyPolicy(SPEC), "assign": AssignPolicy(SPEC)}.get(policy) \
            or MarlOnnxPolicy(MODEL_DIR, verify=False)
        f = fleet(pickers=10, transporters=5, items=6, policy=chosen, width=40, height=24, pickups=6, zones=5)
        f.coord.metrics["max_round_ms"] = 0.0
        f.run(60.0)
        assert f.coord.metrics["rounds"] > 10
        print(f"\n{policy}: max round {f.coord.metrics['max_round_ms']:.1f} ms over {f.coord.metrics['rounds']} rounds")
        assert f.coord.metrics["max_round_ms"] < 100.0
