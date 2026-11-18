"""A FakeRobot on RabbitMQ: what fleet_agent's bus client will do.

    - its command queue robot.<id>.commands (prefetch 1, ack after validation)
    - an exclusive queue on fleet.broadcast (coordinator heartbeat, e-stop)
    - heartbeats to fleet.telemetry, events to fleet.events (persistent,
      publisher confirms); events that cannot be sent wait in an outbox
"""
from __future__ import annotations

import asyncio
import logging
import time

import aio_pika

from coordinator.protocol import ProtocolError, parse, topology
from coordinator.sim.fake_robot import FakeRobot

log = logging.getLogger(__name__)


class AmqpRobot:
    def __init__(self, robot: FakeRobot, amqp_url: str, dt: float = 0.05):
        self.robot, self.url, self.dt = robot, amqp_url, dt
        self.outbox: list = []
        self.command_latency_ms: list[float] = []
        self._stopping = asyncio.Event()

    async def run(self):
        connection = await aio_pika.connect_robust(self.url, heartbeat=10)
        try:
            channel = await connection.channel(publisher_confirms=True)
            await channel.set_qos(prefetch_count=topology.ROBOT_PREFETCH)
            self.exchanges = await topology.declare_exchanges(channel)
            commands = await topology.declare_robot_commands(channel, self.robot.id, self.exchanges)
            broadcast = await channel.declare_queue(exclusive=True, auto_delete=True,
                                                    arguments=topology.BROADCAST_ARGS)
            await broadcast.bind(self.exchanges[topology.BROADCAST])
            await commands.consume(self._on_command)
            await broadcast.consume(self._on_broadcast, no_ack=True)
            await self._send([self.robot.hello()])
            last = time.monotonic()
            while not self._stopping.is_set():
                now = time.monotonic()
                await self._send(self.robot.step(now - last))
                last = now
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.dt)
                except asyncio.TimeoutError:
                    pass
        finally:
            await connection.close()

    def stop(self):
        self._stopping.set()

    async def _on_command(self, message):
        try:
            env = parse(message.body)
        except ProtocolError as e:
            log.warning("%s: bad command: %s", self.robot.id, e)
            await message.reject(requeue=False)
            return
        self.command_latency_ms.append((time.time() - env.sent_at) * 1000)
        out = self.robot.receive(env)
        await message.ack()          # validated (accepted or rejected): done with the message
        await self._send(out)

    async def _on_broadcast(self, message):
        try:
            env = parse(message.body)
        except ProtocolError:
            return
        await self._send(self.robot.receive(env))

    async def _send(self, envs):
        pending, self.outbox = self.outbox + list(envs), []
        for env in pending:
            try:
                if env.type == "robot.heartbeat":
                    await self.exchanges[topology.TELEMETRY].publish(
                        aio_pika.Message(env.to_bytes()), routing_key=topology.heartbeat_key(self.robot.id))
                else:
                    kind = env.payload.get("status", "hello").lower() if env.type == "robot.event" else "hello"
                    await self.exchanges[topology.EVENTS].publish(
                        aio_pika.Message(env.to_bytes(), delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                                         message_id=env.msg_id),
                        routing_key=topology.event_key(self.robot.id, kind))
            except Exception as e:  # noqa: BLE001 - broker gone: keep events, drop telemetry
                if env.type != "robot.heartbeat":
                    self.outbox.append(env)
                log.debug("%s: publish failed (%s)", self.robot.id, e)
