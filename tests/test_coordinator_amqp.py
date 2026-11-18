"""
Phase 2d: coordinator and simulated robots over a real RabbitMQ.

Starts a throwaway rabbitmq:3-management container (skipped without Docker),
then runs CoordinatorService with AmqpRobots in real time.

Run with:
    pytest tests/test_coordinator_amqp.py -v -s
"""
import asyncio
import json
import shutil
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

aio_pika = pytest.importorskip("aio_pika")

from coordinator.core import Config, Coordinator  # noqa: E402
from coordinator.policies import StayPolicy  # noqa: E402
from coordinator.service import CoordinatorService, OpsClient  # noqa: E402
from coordinator.sim.amqp import AmqpRobot  # noqa: E402
from coordinator.sim.fake_robot import FakeRobot  # noqa: E402
from coordinator.spec import load_spec  # noqa: E402
from coordinator.store import Store  # noqa: E402

from tests.test_coordinator_core import layout  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "policy_models" / "imit_pointer_obs_tc-round_2"
SPEC = load_spec(MODEL_DIR)
IMAGE = "rabbitmq:3-management"


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    probe = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _docker_ok(), reason=f"needs Docker and the {IMAGE} image")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_broker(url: str, timeout_s: float = 90.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            connection = await aio_pika.connect(url, timeout=3)
            await connection.close()
            return
        except Exception:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(1.0)


@pytest.fixture(scope="module")
def broker():
    port = _free_port()
    name = f"couliglig-test-rabbit-{port}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5672", IMAGE],
                   check=True, capture_output=True)
    url = f"amqp://guest:guest@127.0.0.1:{port}/"
    try:
        asyncio.run(_wait_for_broker(url))
        yield {"url": url, "name": name}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


class Harness:
    """Coordinator service + robots + ops client, all on one event loop."""

    def __init__(self, url: str, db: Path, robots=(("picker", 2), ("transporter", 2))):
        self.url, self.db = url, db
        self.grid, self.docks = layout()
        self.config = Config(stale_s=3.0, offline_s=30.0, round_period_s=0.5, robot_speed_mps=6.0)
        self.robots: dict[str, AmqpRobot] = {}
        self.plan = []
        n = 0
        for role, count in robots:
            for k in range(count):
                n += 1
                cell = (2, 1 + k * 10) if role == "picker" else (self.grid.height - 3, 1 + k * 10)
                self.plan.append((f"couliglig_bot_{n}", role, cell))
        self.tasks = []

    def start_coordinator(self):
        coord = Coordinator(Store(self.db), StayPolicy(SPEC), SPEC, self.grid, config=self.config)
        coord.start()
        self.service = CoordinatorService(coord, self.url, tick_s=0.05)
        self.service_task = asyncio.create_task(self.service.run())
        return coord

    async def coordinator_ready(self):
        await asyncio.wait_for(self.service.ready.wait(), timeout=30)

    async def stop_coordinator(self):
        self.service.stop()
        await self.service_task
        self.service.coord.store.close()

    def start_robots(self):
        for rid, role, cell in self.plan:
            robot = FakeRobot(rid, role, cell, self.grid, time.time, speed_cells_s=8.0)
            amqp = AmqpRobot(robot, self.url)
            self.robots[rid] = amqp
            self.tasks.append(asyncio.create_task(amqp.run()))

    async def stop_robots(self):
        for amqp in self.robots.values():
            amqp.stop()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def wait(self, ops: OpsClient, condition, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                snap = await ops.call("snapshot")
            except asyncio.TimeoutError:
                snap = None
            if snap and snap.get("ok") and condition(snap):
                return snap
            await asyncio.sleep(0.5)
        raise AssertionError(f"timed out; last snapshot: {json.dumps(snap)[:2000] if snap else None}")

    def deliveries(self) -> Counter:
        """Item -> number of DELIVER tasks the robots completed (from the coordinator's task table)."""
        store = Store(self.db)
        try:
            tasks = {t["id"]: t for t in store.load_tasks()}
        finally:
            store.close()
        out = Counter()
        for amqp in self.robots.values():
            for task_id, status in amqp.robot.log:
                t = tasks.get(task_id)
                if status == "SUCCEEDED" and t and t["kind"] == "DELIVER":
                    out[t["item_id"]] += 1
        return out


def _all_delivered(n):
    return lambda snap: len(snap["items"]) >= n and all(i["status"] == "DELIVERED" for i in snap["items"].values())


def _pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def test_end_to_end_over_rabbitmq(broker, tmp_path):
    async def scenario():
        h = Harness(broker["url"], tmp_path / "coordinator.db")
        h.start_coordinator()
        await h.coordinator_ready()
        h.start_robots()
        async with OpsClient(broker["url"]) as ops:
            assert (await ops.call("activate", docks=h.docks))["ok"]
            await h.wait(ops, lambda s: sum(r["status"] == "ONLINE" for r in s["robots"].values()) == 4, 30)
            for k in range(4):
                reply = await ops.call("add_item", item_id=f"i{k}", pickup_dock=f"P{k}", receiver_dock=f"R{k % 2}")
                assert reply["ok"], reply
            await h.wait(ops, _all_delivered(4), 90)
        command_ms = [x for a in h.robots.values() for x in a.command_latency_ms]
        event_ms = h.service.latency_ms["events"]
        print(f"\ncommand latency p50 {_pct(command_ms, 50):.1f} ms, p99 {_pct(command_ms, 99):.1f} ms; "
              f"event latency p50 {_pct(event_ms, 50):.1f} ms, p99 {_pct(event_ms, 99):.1f} ms; "
              f"max round {h.service.coord.metrics['max_round_ms']:.1f} ms")
        assert _pct(command_ms, 99) < 200 and _pct(event_ms, 99) < 200
        await h.stop_robots()
        await h.stop_coordinator()
        assert all(n == 1 for n in h.deliveries().values())

    asyncio.run(scenario())


def test_coordinator_restart_over_rabbitmq(broker, tmp_path):
    async def scenario():
        h = Harness(broker["url"], tmp_path / "coordinator.db")
        first = h.start_coordinator()
        await h.coordinator_ready()
        h.start_robots()
        async with OpsClient(broker["url"]) as ops:
            assert (await ops.call("activate", docks=h.docks))["ok"]
            await h.wait(ops, lambda s: sum(r["status"] == "ONLINE" for r in s["robots"].values()) == 4, 30)
            for k in range(6):
                assert (await ops.call("add_item", item_id=f"i{k}", pickup_dock=f"P{k}",
                                       receiver_dock=f"R{k % 2}"))["ok"]
            await h.wait(ops, lambda s: sum(i["status"] == "DELIVERED" for i in s["items"].values()) >= 1, 90)
            await h.stop_coordinator()
            await asyncio.sleep(4.0)       # robots lose the coordinator's heartbeat
            assert all(a.robot.exec_state.value in ("HOLD", "EXECUTING") for a in h.robots.values())
            second = h.start_coordinator()
            assert second.epoch == first.epoch + 1
            await h.wait(ops, _all_delivered(6), 120)
        await h.stop_robots()
        await h.stop_coordinator()
        assert all(a.robot.epoch == second.epoch for a in h.robots.values())
        assert all(n == 1 for n in h.deliveries().values())

    asyncio.run(scenario())


def test_broker_restart(broker, tmp_path):
    async def scenario():
        h = Harness(broker["url"], tmp_path / "coordinator.db")
        h.start_coordinator()
        await h.coordinator_ready()
        h.start_robots()
        async with OpsClient(broker["url"]) as ops:
            assert (await ops.call("activate", docks=h.docks))["ok"]
            await h.wait(ops, lambda s: sum(r["status"] == "ONLINE" for r in s["robots"].values()) == 4, 30)
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
        await h.stop_robots()
        await h.stop_coordinator()
        assert all(n == 1 for n in h.deliveries().values())

    asyncio.run(scenario())
