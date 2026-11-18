"""
Phase 3: fleet_agent (agent + driver + bus) against the coordinator over a real RabbitMQ.

Each robot is couliglig-main's FleetAgent behind an AgentDriver, with its Bus
in its own thread, as on the robot; only the executor is simulated (it moves
along grid paths in real time). Needs Docker (a throwaway broker, as in
test_coordinator_amqp.py) and a couliglig-main checkout.

Run with:
    pytest tests/test_fleet_agent_amqp.py -v -s
"""
import asyncio
import importlib
import subprocess
import time
from collections import Counter

import pytest

aio_pika = pytest.importorskip("aio_pika")

from coordinator.grid import GridDistance  # noqa: E402
from coordinator.protocol import parse, topology as server_topology  # noqa: E402
from coordinator.service import OpsClient  # noqa: E402

from tests.test_coordinator_amqp import Harness, _all_delivered, _wait_for_broker, broker  # noqa: E402,F401
from tests.test_fleet_agent_contract import MAIN  # noqa: E402,F401  (skips without couliglig-main)
from tests.test_dock_locks import client, db_session, fake_redis  # noqa: E402,F401

agent_mod = importlib.import_module("couliglig_task_manager.fleet_agent.agent")
driver_mod = importlib.import_module("couliglig_task_manager.fleet_agent.driver")
bus_mod = importlib.import_module("couliglig_task_manager.fleet_agent.bus")
am = importlib.import_module("couliglig_task_manager.fleet_agent.protocol").messages


class SimExecutor:
    """Drives along grid paths in real time; calls the driver back like the ROS executor will."""

    def __init__(self, grid, cell, speed_cells_s=8.0):
        self.grid, self.distance = grid, GridDistance(grid)
        self.cell, self.speed = (float(cell[0]), float(cell[1])), speed_cells_s
        self.driver = None
        self.leg = None
        self.stopping = False

    def start_leg(self, action):
        target = self.grid.cell_of(action.leg.pose.x, action.leg.pose.y)
        path = self.distance.path(self.cell, target) or [self.cell, target]
        self.leg = [action, path, 0.0]

    def cancel(self, action):
        self.leg, self.stopping = None, True

    stop = cancel

    def advance(self, dt):
        if self.stopping:
            self.stopping = False
            self.driver.stopped()
        if self.leg is None:
            return
        action, path, progress = self.leg
        progress = min(progress + self.speed * dt, len(path) - 1)
        self.leg[2] = progress
        self.cell = path[min(int(progress), len(path) - 1)]
        if progress >= len(path) - 1:
            self.leg = None
            lift = {"UP": am.LiftState.UP, "DOWN": am.LiftState.DOWN}.get(action.leg.lift.value)
            self.driver.leg_done(action.token, True, lift=lift)

    def pose(self):
        x, y = self.grid.position_of(self.cell)
        return am.Pose(x=x, y=y, stamp=time.time())


class BusRobot:
    """One robot as fleet_agent will run it: agent + driver + bus thread + (simulated) executor."""

    def __init__(self, url, robot_id, role, cell, grid, state_path):
        self.id = robot_id
        self.executor = SimExecutor(grid, cell)
        self.agent = agent_mod.FleetAgent(robot_id, role, state_path=state_path)
        self.driver = driver_mod.AgentDriver(self.agent, self.executor)
        self.executor.driver = self.driver
        self.bus = bus_mod.Bus(url, robot_id, self.driver, backoff_max_s=2.0)
        self._stop = False

    async def run(self, dt=0.05):
        self.driver.set_lift(am.LiftState.DOWN)
        self.driver.set_pose(self.executor.pose())
        self.driver.start()
        self.bus.start_thread()
        last = time.monotonic()
        try:
            while not self._stop:
                now = time.monotonic()
                self.executor.advance(now - last)
                last = now
                self.driver.set_pose(self.executor.pose())
                self.driver.tick()
                await asyncio.sleep(dt)
        finally:
            await asyncio.get_running_loop().run_in_executor(None, self.bus.stop)

    def stop(self):
        self._stop = True


class AgentHarness(Harness):
    def __init__(self, url, db, tmp_path, **kw):
        super().__init__(url, db, **kw)
        self.tmp_path = tmp_path

    def start_robots(self):
        for rid, role, cell in self.plan:
            robot = BusRobot(self.url, rid, role, cell, self.grid, self.tmp_path / f"{rid}.json")
            self.robots[rid] = robot
            self.tasks.append(asyncio.create_task(robot.run()))

    def deliveries(self) -> Counter:
        out = Counter()
        tasks = self.service.coord.tasks
        for robot in self.robots.values():
            for task_id, status in robot.agent.sent:
                t = tasks.get(task_id)
                if status == "SUCCEEDED" and t is not None and t.kind == "DELIVER":
                    out[t.item_id] += 1
        return out


async def _online(h, ops, n=4, timeout_s=30):
    await h.wait(ops, lambda s: sum(r["status"] == "ONLINE" for r in s["robots"].values()) == n, timeout_s)


def test_agents_deliver_over_rabbitmq(broker, tmp_path):
    async def scenario():
        h = AgentHarness(broker["url"], tmp_path / "coordinator.db", tmp_path)
        h.start_coordinator()
        await h.coordinator_ready()
        h.start_robots()
        async with OpsClient(broker["url"]) as ops:
            assert (await ops.call("activate", docks=h.docks))["ok"]
            await _online(h, ops)
            for k in range(4):
                assert (await ops.call("add_item", item_id=f"i{k}", pickup_dock=f"P{k}",
                                       receiver_dock=f"R{k % 2}"))["ok"]
            await h.wait(ops, _all_delivered(4), 90)
        deliveries = h.deliveries()
        await h.stop_robots()
        await h.stop_coordinator()
        assert sorted(deliveries) == ["i0", "i1", "i2", "i3"] and set(deliveries.values()) == {1}
        assert all(not r.agent.outbox() for r in h.robots.values())

    asyncio.run(scenario())


def test_agents_survive_a_broker_restart(broker, tmp_path):
    async def scenario():
        h = AgentHarness(broker["url"], tmp_path / "coordinator.db", tmp_path)
        h.start_coordinator()
        await h.coordinator_ready()
        h.start_robots()
        async with OpsClient(broker["url"]) as ops:
            assert (await ops.call("activate", docks=h.docks))["ok"]
            await _online(h, ops)
            for k in range(4):
                assert (await ops.call("add_item", item_id=f"i{k}", pickup_dock=f"P{k}",
                                       receiver_dock=f"R{k % 2}"))["ok"]
        await asyncio.sleep(2.0)
        restart = await asyncio.create_subprocess_exec("docker", "restart", broker["name"],
                                                       stdout=asyncio.subprocess.DEVNULL,
                                                       stderr=asyncio.subprocess.DEVNULL)
        await restart.wait()
        await _wait_for_broker(broker["url"])
        async with OpsClient(broker["url"]) as ops:
            await h.wait(ops, _all_delivered(4), 150)
        deliveries = h.deliveries()
        await h.stop_robots()
        await h.stop_coordinator()
        assert set(deliveries.values()) == {1}
        assert all(r.bus.connects >= 2 for r in h.robots.values())     # each one reconnected

    asyncio.run(scenario())


def test_events_wait_on_disk_until_the_coordinator_queue_exists(broker, tmp_path):
    """No coordinator has ever declared coord.events: the event is unroutable, so it is
    kept, and published (before the hello) once the queue appears."""
    vhost = f"fresh{int(time.time() * 1000) % 100000}"
    for cmd in (["add_vhost", vhost], ["set_permissions", "-p", vhost, "guest", ".*", ".*", ".*"]):
        subprocess.run(["docker", "exec", broker["name"], "rabbitmqctl", *cmd], check=True, capture_output=True)
    url = broker["url"].rstrip("/") + "/" + vhost

    async def scenario():
        agent = agent_mod.FleetAgent("couliglig_bot_9", "picker", state_path=tmp_path / "bot9.json")
        executor = SimExecutor(Harness(url, tmp_path / "x.db").grid, (2, 2))
        driver = driver_mod.AgentDriver(agent, executor)
        executor.driver = driver
        driver.set_lift(am.LiftState.DOWN)
        driver.start()
        # a command before any welcome: the agent answers REJECTED, an event for the outbox
        cmd = am.TaskCommand(task_id="t1", attempt=1, kind=am.TaskKind.PICK, valid_until=time.time() + 30,
                             leg_timeout_s=30, legs=[am.Leg(dock_id="P0", pose=am.Pose(x=0, y=0))])
        driver.deliver(agent_mod.make("command.task", cmd, epoch=1).to_bytes())
        driver.tick()
        assert [e.body().status.value for e in agent.outbox()] == ["REJECTED"]
        bus = bus_mod.Bus(url, agent.id, driver)
        bus.start_thread()
        try:
            for _ in range(60):
                if bus.connected:
                    break
                await asyncio.sleep(0.1)
            assert bus.connected
            await asyncio.sleep(1.0)
            assert len(agent.outbox()) == 1                     # unroutable: still on disk
            # with nothing to send, heartbeats flow to a broker without a coordinator:
            # they are dropped, and must not cost the link
            kept = agent.outbox()
            agent.confirm(kept[0].msg_id)
            for _ in range(30):
                driver.tick()
                await asyncio.sleep(0.1)
            assert bus.connected and bus.connects == 1
            agent._outbox = kept                                 # put the event back
            connection = await aio_pika.connect(url)
            channel = await connection.channel()
            _, queues = await server_topology.declare_coordinator(channel)
            for _ in range(100):
                driver.tick()
                if not agent.outbox():
                    break
                await asyncio.sleep(0.1)
            assert not agent.outbox()
            got = []
            for _ in range(80):                                  # until the hello retry (5 s) lands
                driver.tick()
                while (message := await queues[server_topology.Q_EVENTS].get(fail=False)) is not None:
                    got.append(parse(message.body).type)
                    await message.ack()
                if "robot.hello" in got:
                    break
                await asyncio.sleep(0.1)
            await connection.close()
            assert got[0] == "robot.event" and "robot.hello" in got
        finally:
            await asyncio.get_running_loop().run_in_executor(None, bus.stop)

    asyncio.run(scenario())


def test_http_routes_drive_the_fleet(broker, tmp_path, monkeypatch, client, db_session):  # noqa: F811
    """FLEET_MODE=bus end to end: an operator activates docks and adds items over HTTP,
    the coordinator plans, and fleet_agent robots deliver."""
    from definitions import DockType
    from schema.docks import Dock, DockConfig

    monkeypatch.setenv("FLEET_MODE", "bus")
    monkeypatch.setenv("AMQP_URL", broker["url"])
    h = AgentHarness(broker["url"], tmp_path / "coordinator.db", tmp_path)
    cfg = DockConfig(name="bus e2e", description="")
    db_session.add(cfg)
    db_session.commit()
    db_session.refresh(cfg)
    for d in h.docks:
        db_session.add(Dock(config_id=cfg.id, dock_id=d["id"], dock_type=DockType(d["type"]),
                            x=d["x"], y=d["y"], theta=0.0))
    db_session.commit()

    async def http(method, url, **kw):
        return await asyncio.get_running_loop().run_in_executor(None, lambda: getattr(client, method)(url, **kw))

    async def scenario():
        h.start_coordinator()
        await h.coordinator_ready()
        h.start_robots()
        r = await http("post", f"/dock/activate?config_id={cfg.id}")
        assert r.status_code == 200 and r.json()["docks"] == len(h.docks), r.text
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            robots = (await http("get", "/fleet/snapshot")).json()["robots"]
            if sum(x["status"] == "ONLINE" for x in robots.values()) == 4:
                break
            await asyncio.sleep(0.5)
        for k in range(3):
            r = await http("post", "/dock/add-item", json={"dock_id": f"P{k}", "item_id": f"box{k}",
                                                           "receiver_dock_id": f"R{k % 2}"})
            assert r.status_code == 200, r.text
        r = await http("post", "/dock/add-item", json={"dock_id": "P0", "item_id": "dup",
                                                       "receiver_dock_id": "R0"})
        assert r.status_code == 400                     # P0 already holds box0 (the coordinator said so)
        assert (await http("post", "/dock/reserve", json={"dock_id": "P0", "robot_id": "x"})).status_code == 409
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            items = (await http("get", "/fleet/snapshot")).json()["items"]
            if len(items) == 3 and all(i["status"] == "DELIVERED" for i in items.values()):
                break
            await asyncio.sleep(0.5)
        else:
            raise AssertionError(f"not delivered: {items}")
        deliveries = h.deliveries()
        await h.stop_robots()
        await h.stop_coordinator()
        assert set(deliveries) == {"box0", "box1", "box2"} and set(deliveries.values()) == {1}

    asyncio.run(scenario())
