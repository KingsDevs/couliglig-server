"""The envelope every fleet message travels in, and the protocol hash.

    {"schema": "couliglig.fleet/1", "type": "robot.event", "msg_id": "...",
     "robot_id": "couliglig_bot_3", "session": "...", "seq": 1842, "epoch": 7,
     "sent_at": 1794890000.123, "payload": {...}}

    session   new on every fleet_agent / coordinator start
    seq       monotonic per (robot_id, session); stale telemetry is dropped
    epoch     the coordinator incarnation; commands from an older one are refused
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from coordinator.protocol.messages import PAYLOADS

SCHEMA = "couliglig.fleet/1"


def _protocol_hash() -> str:
    """Hash of every message's JSON schema: equal on both sides iff the copies agree."""
    schemas = {name: model.model_json_schema() for name, model in sorted(PAYLOADS.items())}
    blob = json.dumps({"schema": SCHEMA, "messages": schemas}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


PROTOCOL_HASH = _protocol_hash()


class ProtocolError(ValueError):
    pass


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: str = SCHEMA
    type: str
    msg_id: str
    robot_id: str = ""
    session: str = ""
    seq: int = 0
    epoch: int = 0
    sent_at: float = 0.0
    payload: dict[str, Any]

    def body(self):
        """The payload as its typed model."""
        return PAYLOADS[self.type].model_validate(self.payload)

    def to_bytes(self) -> bytes:
        data = self.model_dump()
        data["schema"] = data.pop("schema_")
        return json.dumps(data, separators=(",", ":")).encode()


def make(type_: str, payload: BaseModel, *, robot_id: str = "", session: str = "", seq: int = 0,
         epoch: int = 0, sent_at: float | None = None) -> Envelope:
    if type_ not in PAYLOADS or not isinstance(payload, PAYLOADS[type_]):
        raise ProtocolError(f"{type_!r} does not carry {type(payload).__name__}")
    return Envelope(type=type_, msg_id=uuid.uuid4().hex, robot_id=robot_id, session=session, seq=seq,
                    epoch=epoch, sent_at=time.time() if sent_at is None else sent_at,
                    payload=payload.model_dump(mode="json"))


def parse(raw: bytes | str | dict) -> Envelope:
    """Bytes -> validated Envelope. Unknown schema versions and types are refused."""
    try:
        data = json.loads(raw) if isinstance(raw, (bytes, str)) else dict(raw)
    except ValueError as e:
        raise ProtocolError(f"not JSON: {e}") from e
    schema = data.pop("schema", None)
    if schema != SCHEMA:
        raise ProtocolError(f"unsupported schema {schema!r}, expected {SCHEMA!r}")
    if data.get("type") not in PAYLOADS:
        raise ProtocolError(f"unknown message type {data.get('type')!r}")
    try:
        env = Envelope(schema_=schema, **data)
        env.body()  # validate the payload too
    except ValueError as e:
        raise ProtocolError(str(e)) from e
    return env
