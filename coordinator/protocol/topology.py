"""RabbitMQ exchanges and queues (the plan's section 5), declared from code.

    fleet.telemetry  topic   robot.<id>.heartbeat   -> coord.telemetry   latest-wins, may be lost
    fleet.events     topic   robot.<id>.event.<k>   -> coord.events      at least once, deduplicated
    fleet.commands   direct  <robot_id>             -> robot.<id>.commands  expire, epoch-fenced
    fleet.broadcast  fanout                         -> one queue per robot  coordinator heartbeat, e-stop
    fleet.ops        direct  ops                    -> coord.ops         operator RPC
    fleet.dlx        fanout                         -> fleet.deadletter  debugging
"""
from __future__ import annotations

TELEMETRY = "fleet.telemetry"
EVENTS = "fleet.events"
COMMANDS = "fleet.commands"
BROADCAST = "fleet.broadcast"
OPS = "fleet.ops"
DLX = "fleet.dlx"

Q_TELEMETRY = "coord.telemetry"
Q_EVENTS = "coord.events"
Q_OPS = "coord.ops"
Q_DEADLETTER = "fleet.deadletter"

TELEMETRY_ARGS = {"x-max-length": 500, "x-overflow": "drop-head", "x-message-ttl": 3000}
EVENTS_ARGS = {"x-queue-type": "quorum", "x-delivery-limit": 5, "x-dead-letter-exchange": DLX}
COMMANDS_ARGS = {"x-max-length": 20, "x-message-ttl": 15000, "x-dead-letter-exchange": DLX}
BROADCAST_ARGS = {"x-message-ttl": 2000}
DEADLETTER_ARGS = {"x-max-length": 10000}

COORDINATOR_PREFETCH = 50
ROBOT_PREFETCH = 1


def heartbeat_key(robot_id: str) -> str:
    return f"robot.{robot_id}.heartbeat"


def event_key(robot_id: str, kind: str) -> str:
    return f"robot.{robot_id}.event.{kind}"


def commands_queue(robot_id: str) -> str:
    return f"robot.{robot_id}.commands"


async def declare_exchanges(channel):
    import aio_pika

    T = aio_pika.ExchangeType
    return {
        TELEMETRY: await channel.declare_exchange(TELEMETRY, T.TOPIC, durable=False),
        EVENTS: await channel.declare_exchange(EVENTS, T.TOPIC, durable=True),
        COMMANDS: await channel.declare_exchange(COMMANDS, T.DIRECT, durable=True),
        BROADCAST: await channel.declare_exchange(BROADCAST, T.FANOUT, durable=False),
        OPS: await channel.declare_exchange(OPS, T.DIRECT, durable=True),
        DLX: await channel.declare_exchange(DLX, T.FANOUT, durable=True),
    }


async def declare_coordinator(channel):
    """Exchanges plus the coordinator's own queues. Returns (exchanges, queues)."""
    ex = await declare_exchanges(channel)
    dead = await channel.declare_queue(Q_DEADLETTER, durable=True, arguments=DEADLETTER_ARGS)
    await dead.bind(ex[DLX])
    telemetry = await channel.declare_queue(Q_TELEMETRY, durable=False, arguments=TELEMETRY_ARGS)
    await telemetry.bind(ex[TELEMETRY], routing_key="robot.*.heartbeat")
    events = await channel.declare_queue(Q_EVENTS, durable=True, arguments=EVENTS_ARGS)
    await events.bind(ex[EVENTS], routing_key="robot.*.event.*")
    ops = await channel.declare_queue(Q_OPS, durable=True)
    await ops.bind(ex[OPS], routing_key="ops")
    return ex, {Q_TELEMETRY: telemetry, Q_EVENTS: events, Q_OPS: ops}


async def declare_robot_commands(channel, robot_id: str, exchanges=None):
    """A robot's command queue (both sides declare it; the arguments must match)."""
    ex = exchanges or await declare_exchanges(channel)
    queue = await channel.declare_queue(commands_queue(robot_id), durable=True, arguments=COMMANDS_ARGS)
    await queue.bind(ex[COMMANDS], routing_key=robot_id)
    return queue
