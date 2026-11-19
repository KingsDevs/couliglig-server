"""
Phase 3 cut-over: FLEET_MODE on the server's HTTP routes.

legacy (default): nothing changes. bus: /dock/activate, /dock/add-item and
/dock/remove-item forward to the coordinator (fleet_ops.call, stubbed here),
the robots' dock lock routes answer 409, and /fleet/* reaches the coordinator.
tests/test_fleet_agent_amqp.py runs the same routes against a real coordinator.

Run with:
    pytest tests/test_fleet_routes.py -v
"""
import pytest

from definitions import DockType
from schema.docks import Dock, DockConfig
from services import fleet_ops

from tests.test_dock_locks import HANDOFF, client, db_session, fake_redis  # noqa: F401


@pytest.fixture()
def config_id(db_session):  # noqa: F811
    cfg = DockConfig(name="bus", description="phase 3")
    db_session.add(cfg)
    db_session.commit()
    db_session.refresh(cfg)
    for d in (dict(dock_id="P0", dock_type=DockType.PICKUP, aruco_id=1, x=1.0, y=2.0, theta=0.5),
              dict(dock_id="R0", dock_type=DockType.RECEIVER, aruco_id=2, x=3.0, y=4.0, theta=0.0),
              dict(dock_id=HANDOFF, dock_type=DockType.HANDOFF, aruco_id=3, x=None, y=None, theta=None)):
        db_session.add(Dock(config_id=cfg.id, **d))
    db_session.commit()
    return cfg.id


@pytest.fixture()
def coordinator(monkeypatch):
    """FLEET_MODE=bus with a recorded, scripted coordinator."""
    monkeypatch.setenv("FLEET_MODE", "bus")
    calls, replies = [], {}

    def call(op, timeout_s=5.0, **args):
        calls.append((op, args))
        reply = replies.get(op, {"ok": True})
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(fleet_ops, "call", call)
    return calls, replies


class TestLegacy:

    def test_mode_defaults_to_legacy(self, client, monkeypatch):  # noqa: F811
        monkeypatch.delenv("FLEET_MODE", raising=False)
        assert client.get("/fleet/mode").json() == {"fleet_mode": "legacy"}
        assert client.get("/fleet/snapshot").status_code == 409

    def test_activate_still_uses_redis(self, client, config_id, fake_redis, monkeypatch):  # noqa: F811
        monkeypatch.delenv("FLEET_MODE", raising=False)
        assert client.post(f"/dock/activate?config_id={config_id}").json()["success"]
        assert fake_redis.sismember("docks:all", "P0")


class TestBus:

    def test_activate_sends_the_config_to_the_coordinator(self, client, config_id, coordinator,  # noqa: F811
                                                           fake_redis):
        calls, replies = coordinator
        replies["activate"] = {"ok": True, "docks": 3}
        r = client.post(f"/dock/activate?config_id={config_id}")
        assert r.status_code == 200 and r.json()["fleet_mode"] == "bus"
        op, args = calls[0]
        assert op == "activate"
        assert args["docks"] == [
            {"id": "P0", "type": "pickup", "x": 1.0, "y": 2.0, "yaw": 0.5},
            {"id": "R0", "type": "receiver", "x": 3.0, "y": 4.0, "yaw": 0.0},
            {"id": HANDOFF, "type": "hand_off", "x": 0.0, "y": 0.0, "yaw": 0.0},
        ]
        assert not fake_redis.exists("docks:all")          # Redis is not the runtime any more
        assert "map_yaml" not in args                      # this config has no map

    def test_activate_sends_the_configs_map(self, client, config_id, coordinator, db_session):  # noqa: F811
        from schema import MapConfig
        db_session.add(MapConfig(dock_config_id=config_id, map_yaml_filename="floor.yaml",
                                 map_image_filename="floor.pgm"))
        db_session.commit()
        calls, replies = coordinator
        replies["activate"] = {"ok": True, "docks": 3}
        assert client.post(f"/dock/activate?config_id={config_id}").status_code == 200
        assert calls[0][1]["map_yaml"] == "db/maps/floor.yaml"

    def test_add_item(self, client, coordinator):  # noqa: F811
        calls, _ = coordinator
        r = client.post("/dock/add-item", json={"dock_id": "P0", "item_id": "box1", "receiver_dock_id": "R0"})
        assert r.status_code == 200
        assert calls == [("add_item", {"item_id": "box1", "pickup_dock": "P0", "receiver_dock": "R0",
                                       "weight": 1.0})]

    def test_refusal_is_400_and_no_coordinator_is_503(self, client, coordinator):  # noqa: F811
        _, replies = coordinator
        replies["add_item"] = fleet_ops.FleetRefused("pickup dock P0 already holds an item")
        r = client.post("/dock/add-item", json={"dock_id": "P0", "item_id": "b", "receiver_dock_id": "R0"})
        assert (r.status_code, r.json()["detail"]) == (400, "pickup dock P0 already holds an item")
        replies["add_item"] = fleet_ops.FleetUnavailable("the coordinator did not answer")
        r = client.post("/dock/add-item", json={"dock_id": "P0", "item_id": "b", "receiver_dock_id": "R0"})
        assert r.status_code == 503

    def test_remove_item_finds_the_waiting_item(self, client, coordinator):  # noqa: F811
        calls, replies = coordinator
        replies["snapshot"] = {"ok": True, "items": {
            "a": {"pickup_dock": "P0", "status": "HELD"},
            "b": {"pickup_dock": "P0", "status": "AVAILABLE"}}}
        assert client.post("/dock/remove-item", json={"dock_id": "P0"}).status_code == 200
        assert calls[-1] == ("remove_item", {"item_id": "b"})
        assert client.post("/dock/remove-item", json={"dock_id": "P9"}).status_code == 400

    @pytest.mark.parametrize("route, body", [
        ("/dock/reserve", {"dock_id": "P0", "robot_id": "couliglig_bot_1"}),
        ("/dock/occupy", {"dock_id": "P0", "robot_id": "couliglig_bot_1"}),
        ("/dock/release", {"dock_id": "P0", "robot_id": "couliglig_bot_1"}),
        ("/dock/renew", {"dock_id": "P0", "robot_id": "couliglig_bot_1"}),
    ])
    def test_legacy_robot_lock_routes_are_refused(self, client, coordinator, route, body):  # noqa: F811
        r = client.post(route, json=body)
        assert r.status_code == 409 and "FLEET_MODE=bus" in r.json()["detail"]

    def test_fleet_routes(self, client, coordinator):  # noqa: F811
        calls, replies = coordinator
        replies["snapshot"] = {"ok": True, "robots": {}, "items": {}}
        assert client.get("/fleet/mode").json() == {"fleet_mode": "bus"}
        assert client.get("/fleet/snapshot").json()["robots"] == {}
        assert client.post("/fleet/estop", json={"reason": "door open"}).status_code == 200
        assert client.post("/fleet/resume").status_code == 200
        assert client.post("/fleet/pause").status_code == 200
        assert [c[0] for c in calls] == ["snapshot", "estop", "resume", "pause"]
        assert calls[1][1] == {"reason": "door open"}


@pytest.fixture()
def real_coordinator(monkeypatch):
    """FLEET_MODE=bus with an in-process Coordinator behind fleet_ops.call, so the routes
    see the coordinator's real replies (no broker, no robots)."""
    from coordinator.core import Coordinator
    from coordinator.grid import Grid
    from coordinator.policies import StayPolicy
    from coordinator.store import Store
    from tests.test_coordinator_core import SPEC

    monkeypatch.setenv("FLEET_MODE", "bus")
    coord = Coordinator(Store(), StayPolicy(SPEC), SPEC, Grid.open(12, 12), manual=True)
    coord.start()

    def call(op, timeout_s=5.0, **args):
        reply, _ = coord.handle_op({"op": op, **args})
        if not reply.get("ok"):
            raise fleet_ops.FleetRefused(reply.get("error") or f"{op} refused")
        return reply

    monkeypatch.setattr(fleet_ops, "call", call)
    return coord


class TestBusAgainstTheCoordinator:

    def test_items_through_the_dock_routes(self, client, config_id, real_coordinator):  # noqa: F811
        assert client.post(f"/dock/activate?config_id={config_id}").status_code == 200
        r = client.post("/dock/add-item", json={"dock_id": "P0", "item_id": "box1",
                                                "receiver_dock_id": "R0", "item_weight": 2.0})
        assert r.status_code == 200
        states = {d["dock_id"]: d for d in client.get("/dock/all_dock_states").json()}
        assert set(states) == {"P0", "R0", HANDOFF}
        assert (states["P0"]["dock_type"], states["P0"]["status"], states["P0"]["item_id"],
                states["P0"]["item_weight"], states["P0"]["receiver_dock_id"]) == \
            ("pickup", "available", "box1", 2.0, "R0")
        assert states["R0"]["item_id"] is None
        assert client.post("/dock/remove-item", json={"dock_id": "P0"}).status_code == 200
        assert "box1" not in real_coordinator.items

    def test_manual_by_default_and_auto_on_resume(self, client, real_coordinator):  # noqa: F811
        snap = client.get("/fleet/snapshot").json()
        assert (snap["control"], snap["paused"]) == ("manual", True)
        assert client.post("/fleet/resume").status_code == 200
        assert client.get("/fleet/snapshot").json()["control"] == "auto"
        assert client.post("/fleet/pause").status_code == 200
        assert client.get("/fleet/snapshot").json()["control"] == "manual"


def test_dock_states_from_a_snapshot():
    snapshot = {
        "docks": {"P0": {"type": "pickup", "x": 1.0, "y": 2.0, "yaw": 0.0},
                  "R0": {"type": "receiver", "x": 3.0, "y": 4.0, "yaw": 0.0},
                  "Z0": {"type": "waiting_zone", "x": 5.0, "y": 6.0, "yaw": 0.0},
                  "Z1": {"type": "waiting_zone", "x": 7.0, "y": 8.0, "yaw": 0.0}},
        "items": {"a": {"status": "CLAIMED", "pickup_dock": "P0", "receiver_dock": "R0", "weight": 1.0,
                        "holder": "couliglig_bot_1", "transporter": None},
                  "b": {"status": "DELIVERING", "pickup_dock": "P9", "receiver_dock": "R0", "weight": 1.0,
                        "holder": None, "transporter": "couliglig_bot_3"}},
        "robots": {"couliglig_bot_3": {"zone": "Z0", "parked": True},
                   "couliglig_bot_4": {"zone": "Z1", "parked": False}},
    }
    rows = {r["dock_id"]: r for r in fleet_ops.dock_states(snapshot)}
    assert (rows["P0"]["status"], rows["P0"]["robot_id"], rows["P0"]["item_id"]) == ("reserved", "couliglig_bot_1", "a")
    assert (rows["R0"]["status"], rows["R0"]["robot_id"]) == ("reserved", "couliglig_bot_3")
    assert (rows["Z0"]["status"], rows["Z0"]["robot_id"]) == ("occupied", "couliglig_bot_3")
    assert (rows["Z1"]["status"], rows["Z1"]["robot_id"]) == ("reserved", "couliglig_bot_4")


def test_bad_fleet_mode_is_an_error(monkeypatch):
    monkeypatch.setenv("FLEET_MODE", "both")
    with pytest.raises(RuntimeError, match="FLEET_MODE"):
        fleet_ops.fleet_mode()


def test_dock_without_position_is_refused(db_session):  # noqa: F811
    cfg = DockConfig(name="broken", description="")
    db_session.add(cfg)
    db_session.commit()
    db_session.add(Dock(config_id=cfg.id, dock_id="P0", dock_type=DockType.PICKUP, aruco_id=1))
    db_session.commit()
    with pytest.raises(ValueError, match="no position"):
        fleet_ops.coordinator_docks(db_session, cfg.id)
