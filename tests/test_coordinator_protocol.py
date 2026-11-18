"""
Phase 2b: the fleet protocol (envelope, messages, topology names).

Run with:
    pytest tests/test_coordinator_protocol.py -v
"""
import json

import pytest

from coordinator.protocol import PROTOCOL_HASH, SCHEMA, ProtocolError, make, parse, topology
from coordinator.protocol import messages as m
from coordinator.protocol.envelope import _protocol_hash

POSE = m.Pose(x=1.0, y=2.0, yaw=0.5)

SAMPLES = {
    "robot.hello": m.Hello(robot_type="small", role=m.Role.PICKER, protocol_hash=PROTOCOL_HASH,
                           current_task=m.CurrentTask(task_id="t1", attempt=1, status=m.TaskStatus.STARTED)),
    "robot.heartbeat": m.Heartbeat(pose=POSE, exec_state=m.ExecState.EXECUTING, task_id="t1", battery_pct=80),
    "robot.event": m.Event(event_id="e1", task_id="t1", attempt=1, status=m.TaskStatus.LIFT_UP, pose=POSE),
    "coordinator.welcome": m.Welcome(epoch=3, accepted=True, task_to_resume="t1"),
    "command.task": m.TaskCommand(task_id="t1", attempt=1, kind=m.TaskKind.PICK,
                                  legs=[m.Leg(dock_id="P1", pose=POSE, lift=m.Lift.UP)],
                                  valid_until=1e9, leg_timeout_s=120),
    "command.cancel": m.CancelCommand(task_id="t1", mode=m.CancelMode.FINISH_LEG),
    "coordinator.heartbeat": m.CoordinatorHeartbeat(epoch=3, policy="greedy"),
    "coordinator.estop": m.EStop(reason="button", issued_by="op"),
    "coordinator.resume": m.Resume(issued_by="op"),
}


class TestEnvelope:

    @pytest.mark.parametrize("type_", sorted(SAMPLES))
    def test_round_trip(self, type_):
        env = make(type_, SAMPLES[type_], robot_id="couliglig_bot_1", session="s", seq=7, epoch=3)
        back = parse(env.to_bytes())
        assert back.type == type_ and back.seq == 7 and back.epoch == 3 and back.robot_id == "couliglig_bot_1"
        assert back.body() == SAMPLES[type_]

    def test_wire_format_has_schema(self):
        data = json.loads(make("command.cancel", m.CancelCommand(task_id="t")).to_bytes())
        assert data["schema"] == SCHEMA == "couliglig.fleet/1"
        assert set(data) == {"schema", "type", "msg_id", "robot_id", "session", "seq", "epoch", "sent_at", "payload"}

    def test_unknown_schema_version_rejected(self):
        data = json.loads(make("command.cancel", m.CancelCommand(task_id="t")).to_bytes())
        data["schema"] = "couliglig.fleet/2"
        with pytest.raises(ProtocolError, match="schema"):
            parse(json.dumps(data))

    def test_unknown_type_rejected(self):
        data = json.loads(make("command.cancel", m.CancelCommand(task_id="t")).to_bytes())
        data["type"] = "command.dance"
        with pytest.raises(ProtocolError, match="unknown message type"):
            parse(json.dumps(data))

    def test_invalid_payload_rejected(self):
        data = json.loads(make("command.cancel", m.CancelCommand(task_id="t")).to_bytes())
        data["payload"] = {"task_id": "t", "mode": "EXPLODE"}
        with pytest.raises(ProtocolError):
            parse(json.dumps(data))

    def test_unknown_fields_rejected(self):
        data = json.loads(make("command.cancel", m.CancelCommand(task_id="t")).to_bytes())
        data["payload"]["extra"] = 1
        with pytest.raises(ProtocolError):
            parse(json.dumps(data))

    def test_task_needs_a_leg(self):
        with pytest.raises(ValueError):
            m.TaskCommand(task_id="t", attempt=1, kind=m.TaskKind.PICK, legs=[], valid_until=0, leg_timeout_s=1)

    def test_not_json_rejected(self):
        with pytest.raises(ProtocolError):
            parse(b"\x00not json")

    def test_type_and_payload_must_agree(self):
        with pytest.raises(ProtocolError):
            make("command.task", m.CancelCommand(task_id="t"))


class TestProtocolHash:

    def test_stable(self):
        assert PROTOCOL_HASH == _protocol_hash()
        assert len(PROTOCOL_HASH) == 16

    def test_changes_with_the_schema(self, monkeypatch):
        class Extra(m.CancelCommand):
            reason: str = ""

        monkeypatch.setitem(m.PAYLOADS, "command.cancel", Extra)
        assert _protocol_hash() != PROTOCOL_HASH


class TestTopology:

    def test_names(self):
        assert topology.heartbeat_key("couliglig_bot_3") == "robot.couliglig_bot_3.heartbeat"
        assert topology.event_key("couliglig_bot_3", "succeeded") == "robot.couliglig_bot_3.event.succeeded"
        assert topology.commands_queue("couliglig_bot_3") == "robot.couliglig_bot_3.commands"

    def test_bounded_queues(self):
        assert topology.TELEMETRY_ARGS["x-overflow"] == "drop-head"
        assert topology.COMMANDS_ARGS["x-message-ttl"] == 15000
        assert topology.EVENTS_ARGS["x-queue-type"] == "quorum"
        assert topology.EVENTS_ARGS["x-dead-letter-exchange"] == topology.DLX
