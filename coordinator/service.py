"""The coordinator on RabbitMQ: the Coordinator core plus aio-pika plumbing.

    coord.telemetry  heartbeats, auto-ack (latest wins; old seq dropped by the core)
    coord.events     hello and task events, manual ack after the core has
                     committed the change to SQLite; a message that keeps
                     failing is dead-lettered after the queue's delivery limit
    coord.ops        operator RPC (reply_to + correlation_id)
    tick             every tick_s: liveness, expiry, rounds, 1 Hz heartbeat

The core is synchronous and runs on the event loop thread, so its state is
only ever touched by one task at a time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import aio_pika

from coordinator.core import Coordinator, Out
from coordinator.protocol import ProtocolError, parse, topology

log = logging.getLogger(__name__)


class CoordinatorService:
    def __init__(self, coord: Coordinator, amqp_url: str, tick_s: float = 0.05):
        self.coord, self.url, self.tick_s = coord, amqp_url, tick_s
        self.connection = None
        self.channel = None
        self.exchanges = {}
        self._declared: set[str] = set()
        self._stopping = asyncio.Event()
        self.ready = asyncio.Event()       # set once the queues are consumed
        self.latency_ms: dict[str, list[float]] = {"telemetry": [], "events": []}

    async def run(self):
        self.connection = await aio_pika.connect_robust(self.url, heartbeat=10)
        self.channel = await self.connection.channel(publisher_confirms=True)
        await self.channel.set_qos(prefetch_count=topology.COORDINATOR_PREFETCH)
        self.exchanges, queues = await topology.declare_coordinator(self.channel)
        await queues[topology.Q_TELEMETRY].consume(self._on_telemetry, no_ack=True)
        await queues[topology.Q_EVENTS].consume(self._on_event)
        await queues[topology.Q_OPS].consume(self._on_op)
        self.ready.set()
        log.info("coordinator epoch %d on %s", self.coord.epoch, self.url.split("@")[-1])
        try:
            while not self._stopping.is_set():
                try:
                    await self._publish(self.coord.tick())
                except Exception:  # noqa: BLE001 - the coordinator must keep ticking
                    log.exception("tick failed")
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.tick_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.connection.close()

    def stop(self):
        self._stopping.set()

    # ---------------- inbound

    async def _on_telemetry(self, message: aio_pika.abc.AbstractIncomingMessage):
        try:
            env = parse(message.body)
        except ProtocolError as e:
            log.warning("bad telemetry: %s", e)
            return
        self._latency("telemetry", env.sent_at)
        await self._publish(self.coord.handle(env))

    async def _on_event(self, message: aio_pika.abc.AbstractIncomingMessage):
        try:
            env = parse(message.body)
        except ProtocolError as e:
            log.warning("bad event, dead-lettered: %s", e)
            await message.reject(requeue=False)
            return
        self._latency("events", env.sent_at)
        try:
            outs = self.coord.handle(env)   # commits to SQLite before returning
        except Exception:
            log.exception("event %s from %s failed; requeued", env.type, env.robot_id)
            await message.nack(requeue=True)
            return
        await message.ack()
        await self._publish(outs)

    async def _on_op(self, message: aio_pika.abc.AbstractIncomingMessage):
        async with message.process(requeue=False):
            try:
                op = json.loads(message.body)
                reply, outs = self.coord.handle_op(op)
            except Exception as e:  # noqa: BLE001 - report every failure to the caller
                log.exception("op failed")
                reply, outs = {"ok": False, "error": f"{type(e).__name__}: {e}"}, []
            if message.reply_to:
                await self.channel.default_exchange.publish(
                    aio_pika.Message(json.dumps(reply).encode(), correlation_id=message.correlation_id,
                                     content_type="application/json"),
                    routing_key=message.reply_to)
            await self._publish(outs)

    def _latency(self, kind: str, sent_at: float):
        samples = self.latency_ms[kind]
        samples.append((time.time() - sent_at) * 1000)
        if len(samples) > 10000:
            del samples[:5000]

    # ---------------- outbound

    async def _publish(self, outs: list[Out]):
        """Send what the core produced. During a broker outage a send fails:
        that is logged and skipped, never fatal. A lost command expires and is
        rolled back by the core; a lost broadcast is replaced by the next one."""
        for out in outs:
            try:
                await self._publish_one(out)
            except Exception as e:  # noqa: BLE001 - the connection is recovering
                log.warning("could not send %s to %s: %s", out.kind,
                            out.robot_id or "all", e)
                if out.kind == "declare_robot":
                    self._declared.discard(out.robot_id)

    async def _publish_one(self, out: Out):
        if out.kind == "declare_robot":
            if out.robot_id not in self._declared:
                await topology.declare_robot_commands(self.channel, out.robot_id, self.exchanges)
                self._declared.add(out.robot_id)
        elif out.kind == "command":
            await self.exchanges[topology.COMMANDS].publish(
                aio_pika.Message(out.envelope.to_bytes(), delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                                 message_id=out.envelope.msg_id, content_type="application/json"),
                routing_key=out.robot_id)
        elif out.kind == "broadcast":
            await self.exchanges[topology.BROADCAST].publish(
                aio_pika.Message(out.envelope.to_bytes(), content_type="application/json"), routing_key="")


class OpsClient:
    """Operator RPC to the coordinator (FastAPI will use this)."""

    def __init__(self, amqp_url: str, timeout_s: float = 5.0):
        self.url, self.timeout_s = amqp_url, timeout_s
        self._pending: dict[str, asyncio.Future] = {}

    async def __aenter__(self):
        self.connection = await aio_pika.connect_robust(self.url)
        self.channel = await self.connection.channel()
        self.exchanges = await topology.declare_exchanges(self.channel)
        # the ops queue is durable: a call made before the coordinator is up waits there
        ops = await self.channel.declare_queue(topology.Q_OPS, durable=True)
        await ops.bind(self.exchanges[topology.OPS], routing_key="ops")
        self.callback = await self.channel.declare_queue(exclusive=True, auto_delete=True)
        await self.callback.consume(self._on_reply, no_ack=True)
        return self

    async def __aexit__(self, *exc):
        await self.connection.close()

    async def _on_reply(self, message):
        future = self._pending.pop(message.correlation_id, None)
        if future is not None and not future.done():
            future.set_result(json.loads(message.body))

    async def call(self, op: str, **args) -> dict:
        cid = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[cid] = future
        await self.exchanges[topology.OPS].publish(
            aio_pika.Message(json.dumps({"op": op, **args}).encode(), correlation_id=cid,
                             reply_to=self.callback.name, content_type="application/json",
                             expiration=self.timeout_s),
            routing_key="ops")
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_s)
        finally:
            self._pending.pop(cid, None)
