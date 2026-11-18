"""Fleet protocol: envelope, message payloads, RabbitMQ topology.

couliglig-main's fleet_agent keeps a copy of this package (Phase 3); both
sides compute PROTOCOL_HASH from the message schemas, robot.hello carries
it, and the coordinator refuses a robot whose copy differs.
"""
from coordinator.protocol.envelope import PROTOCOL_HASH, SCHEMA, Envelope, ProtocolError, make, parse
from coordinator.protocol import messages, topology

__all__ = ["PROTOCOL_HASH", "SCHEMA", "Envelope", "ProtocolError", "make", "parse", "messages", "topology"]
