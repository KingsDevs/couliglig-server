"""
Phase 0 safety fixes for the legacy Redis runtime:

  - dock locks expire (TTL), can be renewed, and expired locks are reconciled
  - hand-off docks persist while holding an item and can be released by item
  - a disconnected robot's locks get a short grace period, restored on reconnect
  - robot_ips updates are atomic and a stale socket cannot remove a newer one
  - activation tolerates docks without a position
  - ROS_DOMAIN_ID 0 is treated as a real domain
  - waiting zones are listed in the same order as action-map slots

Run with:
    pytest tests/test_dock_locks.py -v
"""

import json
import time
from unittest.mock import patch, MagicMock

import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from definitions import DockType
from main import app
from schema.base import Base
from schema.docks import DockConfig, Dock
from services.database import get_db_session
import services.redis_dock_runtime as rdr
import routes.docks as docks_route
import routes.robots as robots_route


PICKER = "couliglig_bot_1"
PICKER_2 = "couliglig_bot_2"
TRANSPORTER = "couliglig_bot_3"
HANDOFF = "couliglig_bot_3_front_left"


@pytest.fixture()
def fake_redis():
    r = fakeredis.FakeRedis(decode_responses=True)
    yield r
    r.flushall()


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(fake_redis, db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    docks_route.redis_client = fake_redis
    robots_route.redis_client = fake_redis
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _seed(db_session, r, extra=()):
    cfg = DockConfig(name="locks", description="phase 0")
    db_session.add(cfg)
    db_session.commit()
    db_session.refresh(cfg)
    docks = [
        dict(dock_id="pickup_1", dock_type=DockType.PICKUP, aruco_id=1, x=1.0, y=2.0, theta=0.0),
        dict(dock_id="receiver_1", dock_type=DockType.RECEIVER, aruco_id=2, x=3.0, y=4.0, theta=0.0),
        dict(dock_id="wz_b", dock_type=DockType.WAITING_ZONE, aruco_id=3, x=5.0, y=6.0, theta=0.0),
        dict(dock_id="wz_a", dock_type=DockType.WAITING_ZONE, aruco_id=4, x=7.0, y=8.0, theta=0.0),
        dict(dock_id=HANDOFF, dock_type=DockType.HANDOFF, aruco_id=5, x=9.0, y=1.0, theta=0.0),
        *extra,
    ]
    for d in docks:
        db_session.add(Dock(config_id=cfg.id, **d))
    db_session.commit()
    rdr.activate_docks(r, db_session, cfg.id)
    return cfg.id


def _lock_key(dock_type: DockType, dock_id: str) -> str:
    return f"lock:dock:{dock_type.value}:{dock_id}"


def _expire_now(r, key):
    """Simulate a lock reaching its TTL."""
    r.pexpire(key, 1)
    time.sleep(0.01)


# ===========================================================================
# TTL, renew, reconcile
# ===========================================================================

class TestLockTtl:

    def test_reserve_sets_ttl(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        assert rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        ttl = fake_redis.ttl(_lock_key(DockType.PICKUP, "pickup_1"))
        assert 0 < ttl <= rdr.DOCK_LOCK_TTL_S

    def test_renew_extends_ttl(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        key = _lock_key(DockType.PICKUP, "pickup_1")
        fake_redis.expire(key, 5)
        assert rdr.renew_dock(fake_redis, "pickup_1", PICKER) == rdr.DOCK_LOCK_TTL_S
        assert fake_redis.ttl(key) > 5

    def test_renew_by_other_robot_fails(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        with pytest.raises(ValueError, match="owned by"):
            rdr.renew_dock(fake_redis, "pickup_1", PICKER_2)

    def test_renew_after_expiry_fails(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        _expire_now(fake_redis, _lock_key(DockType.PICKUP, "pickup_1"))
        with pytest.raises(ValueError, match="no active reservation"):
            rdr.renew_dock(fake_redis, "pickup_1", PICKER)

    def test_expired_lock_frees_dock(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        _expire_now(fake_redis, _lock_key(DockType.PICKUP, "pickup_1"))

        state = rdr.get_dock_state(fake_redis, "pickup_1")
        assert state["status"] == "available"
        assert state["robot_id"] == ""
        assert rdr.reserve_dock(fake_redis, "pickup_1", PICKER_2)

    def test_expired_lock_hidden_from_readers(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "wz_a", PICKER)
        _expire_now(fake_redis, _lock_key(DockType.WAITING_ZONE, "wz_a"))

        assert rdr.get_active_docks(fake_redis) == []
        statuses = {d["dock_id"]: d["status"] for d in rdr.get_all_dock_states(fake_redis)}
        assert statuses["wz_a"] == "available"
        action_map = rdr.build_action_map(fake_redis, [])
        wz = {s["dock_id"]: s for s in action_map["wz_slots"] if s}
        assert wz["wz_a"]["available_for_entry"] is True

    def test_reconcile_leaves_live_locks_alone(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        assert rdr.reconcile_expired_locks(fake_redis) == []
        assert rdr.get_dock_state(fake_redis, "pickup_1")["status"] == "reserved"

    def test_occupy_refreshes_ttl(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        key = _lock_key(DockType.PICKUP, "pickup_1")
        fake_redis.expire(key, 5)
        rdr.occupy_dock(fake_redis, "pickup_1", PICKER)
        assert fake_redis.ttl(key) > 5

    def test_release_by_non_owner_rejected(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        with pytest.raises(ValueError, match="owned by"):
            rdr.release_dock(fake_redis, "pickup_1", PICKER_2)

    def test_item_id_does_not_bypass_owner_on_ordinary_dock(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.add_item_to_pickup_dock(fake_redis, "pickup_1", "itm_1", "receiver_1")
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        with pytest.raises(ValueError, match="owned by"):
            rdr.release_dock(fake_redis, "pickup_1", PICKER_2, item_id="itm_1")


# ===========================================================================
# Hand-off docks
# ===========================================================================

class TestHandoffDock:

    def _hand_off(self, r, item_id="itm_1"):
        assert rdr.reserve_dock(r, HANDOFF, PICKER)
        rdr.occupy_dock(r, HANDOFF, PICKER, item_id=item_id)

    def test_occupied_handoff_lock_persists_and_records_item(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        self._hand_off(fake_redis)
        assert fake_redis.ttl(_lock_key(DockType.HANDOFF, HANDOFF)) == -1
        state = rdr.get_dock_state(fake_redis, HANDOFF)
        assert state["status"] == "occupied"
        assert state["item_id"] == "itm_1"

    def test_transporter_releases_by_item(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        self._hand_off(fake_redis)
        rdr.release_dock(fake_redis, HANDOFF, TRANSPORTER, item_id="itm_1")

        state = rdr.get_dock_state(fake_redis, HANDOFF)
        assert state["status"] == "available"
        assert state["item_id"] == ""
        assert fake_redis.get(_lock_key(DockType.HANDOFF, HANDOFF)) is None

    def test_wrong_item_rejected(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        self._hand_off(fake_redis)
        with pytest.raises(ValueError, match="owned by"):
            rdr.release_dock(fake_redis, HANDOFF, TRANSPORTER, item_id="itm_other")

    def test_free_handoff_can_be_reused(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        self._hand_off(fake_redis)
        rdr.release_dock(fake_redis, HANDOFF, TRANSPORTER, item_id="itm_1")
        assert rdr.reserve_dock(fake_redis, HANDOFF, PICKER_2)

    def test_release_endpoint_accepts_item_id(self, client, db_session, fake_redis):
        _seed(db_session, fake_redis)
        assert client.post("/dock/reserve", json={"dock_id": HANDOFF, "robot_id": PICKER}).status_code == 200
        assert client.post("/dock/occupy", json={"dock_id": HANDOFF, "robot_id": PICKER,
                                                 "item_id": "itm_1"}).status_code == 200
        resp = client.post("/dock/release", json={"dock_id": HANDOFF, "robot_id": TRANSPORTER,
                                                  "item_id": "itm_1"})
        assert resp.status_code == 200
        assert rdr.get_dock_state(fake_redis, HANDOFF)["status"] == "available"


# ===========================================================================
# /dock/renew endpoint
# ===========================================================================

class TestRenewEndpoint:

    def test_renew_ok(self, client, db_session, fake_redis):
        _seed(db_session, fake_redis)
        client.post("/dock/reserve", json={"dock_id": "pickup_1", "robot_id": PICKER})
        resp = client.post("/dock/renew", json={"dock_id": "pickup_1", "robot_id": PICKER})
        assert resp.status_code == 200
        assert resp.json()["ttl_s"] == rdr.DOCK_LOCK_TTL_S

    def test_renew_lost_reservation_is_409(self, client, db_session, fake_redis):
        _seed(db_session, fake_redis)
        client.post("/dock/reserve", json={"dock_id": "pickup_1", "robot_id": PICKER})
        resp = client.post("/dock/renew", json={"dock_id": "pickup_1", "robot_id": PICKER_2})
        assert resp.status_code == 409

    def test_renew_unknown_dock_is_400(self, client, db_session, fake_redis):
        _seed(db_session, fake_redis)
        resp = client.post("/dock/renew", json={"dock_id": "nope", "robot_id": PICKER})
        assert resp.status_code == 400


# ===========================================================================
# Disconnect grace period
# ===========================================================================

class TestDisconnectGrace:

    def test_shorten_and_restore(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        rdr.reserve_dock(fake_redis, "wz_a", PICKER_2)
        mine = _lock_key(DockType.PICKUP, "pickup_1")
        other = _lock_key(DockType.WAITING_ZONE, "wz_a")

        assert rdr.shorten_robot_locks(fake_redis, PICKER, grace_s=10) == [mine]
        assert fake_redis.ttl(mine) <= 10
        assert fake_redis.ttl(other) > 10

        assert rdr.restore_robot_locks(fake_redis, PICKER) == [mine]
        assert fake_redis.ttl(mine) > 10

    def test_occupied_handoff_not_shortened(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, HANDOFF, PICKER)
        rdr.occupy_dock(fake_redis, HANDOFF, PICKER, item_id="itm_1")
        assert rdr.shorten_robot_locks(fake_redis, PICKER, grace_s=10) == []
        assert fake_redis.ttl(_lock_key(DockType.HANDOFF, HANDOFF)) == -1


# ===========================================================================
# Robot WebSocket registry
# ===========================================================================

def _hello(hostname="couliglig-1", domain=1):
    return {"hostname": hostname, "ip": "10.0.0.5", "namespace": "couliglig",
            "ros_domain_id": domain, "battery": 80.0}


class TestRobotWebSocket:

    def test_connect_registers_with_session(self, client, fake_redis):
        with client.websocket_connect("/robots/ws") as ws:
            ws.send_json(_hello())
            ws.send_json({"type": "heartbeat", "battery": 79.0})
            ws.send_json({"type": "heartbeat"})   # round-trip so the server has processed both
            entry = json.loads(fake_redis.get("robot_ips"))["couliglig-1"]
            assert entry["session"]
            assert entry["battery"] == 79.0
        assert json.loads(fake_redis.get("robot_ips")) == {}

    def test_disconnect_shortens_locks(self, client, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        key = _lock_key(DockType.PICKUP, "pickup_1")
        with client.websocket_connect("/robots/ws") as ws:
            ws.send_json(_hello(domain=1))
            ws.send_json({"type": "heartbeat"})
        assert fake_redis.ttl(key) <= rdr.DOCK_LOCK_DISCONNECT_GRACE_S

    def test_reconnect_restores_locks(self, client, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        key = _lock_key(DockType.PICKUP, "pickup_1")
        fake_redis.expire(key, 10)
        with client.websocket_connect("/robots/ws") as ws:
            ws.send_json(_hello(domain=1))
            ws.send_json({"type": "heartbeat"})
            assert fake_redis.ttl(key) > rdr.DOCK_LOCK_DISCONNECT_GRACE_S

    def test_stale_socket_cleanup_keeps_new_registration(self, client, fake_redis):
        with client.websocket_connect("/robots/ws") as old:
            old.send_json(_hello())
            old.send_json({"type": "heartbeat"})
            with client.websocket_connect("/robots/ws") as new:
                new.send_json(_hello())
                new.send_json({"type": "heartbeat"})
                new_session = json.loads(fake_redis.get("robot_ips"))["couliglig-1"]["session"]
                old.close()
                new.send_json({"type": "heartbeat"})
                entry = json.loads(fake_redis.get("robot_ips")).get("couliglig-1")
                assert entry is not None
                assert entry["session"] == new_session

    def test_remove_with_stale_session_is_noop(self, fake_redis):
        robots_route.redis_client = fake_redis
        robots_route._register_robot_in_redis("couliglig-1", "10.0.0.5", "couliglig", 1, session="new")
        assert robots_route._remove_robot_from_redis("couliglig-1", session="old") is False
        assert "couliglig-1" in json.loads(fake_redis.get("robot_ips"))
        assert robots_route._remove_robot_from_redis("couliglig-1", session="new") is True
        assert json.loads(fake_redis.get("robot_ips")) == {}

    def test_heartbeat_readds_missing_entry(self, client, fake_redis):
        with client.websocket_connect("/robots/ws") as ws:
            ws.send_json(_hello())
            ws.send_json({"type": "heartbeat"})
            fake_redis.set("robot_ips", json.dumps({}))
            ws.send_json({"type": "heartbeat"})
            ws.send_json({"type": "heartbeat"})
            assert "couliglig-1" in json.loads(fake_redis.get("robot_ips"))

    def test_register_endpoint_keeps_ws_session(self, client, fake_redis):
        with client.websocket_connect("/robots/ws") as ws:
            ws.send_json(_hello())
            ws.send_json({"type": "heartbeat"})
            session = json.loads(fake_redis.get("robot_ips"))["couliglig-1"]["session"]
            with patch.object(robots_route, "register_host"):
                resp = client.post("/robots/register", json={
                    "hostname": "couliglig-1", "ip": "10.0.0.5",
                    "namespace": "couliglig", "ros_domain_id": 1,
                })
            assert resp.status_code == 200
            assert json.loads(fake_redis.get("robot_ips"))["couliglig-1"]["session"] == session
        assert "couliglig-1" not in json.loads(fake_redis.get("robot_ips"))


# ===========================================================================
# Smaller runtime fixes
# ===========================================================================

class TestRuntimeFixes:

    def test_activate_with_dock_missing_position(self, db_session, fake_redis):
        cfg_id = _seed(db_session, fake_redis, extra=[
            dict(dock_id="pickup_nopos", dock_type=DockType.PICKUP, aruco_id=9, x=None, y=None, theta=None),
        ])
        assert fake_redis.get("active_dock_config") == str(cfg_id)
        assert fake_redis.sismember("docks:all", "pickup_nopos")
        assert fake_redis.hgetall("dock_pos:pickup_nopos") == {}
        assert fake_redis.hgetall("dock_pos:pickup_1") == {"y": "2.0", "x": "1.0"}

    def test_activate_replaces_previous_runtime(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "pickup_1", PICKER)
        cfg = DockConfig(name="second", description="")
        db_session.add(cfg)
        db_session.commit()
        db_session.refresh(cfg)
        db_session.add(Dock(config_id=cfg.id, dock_id="only", dock_type=DockType.PICKUP,
                            aruco_id=1, x=0.0, y=0.0, theta=0.0))
        db_session.commit()

        rdr.activate_docks(fake_redis, db_session, cfg.id)
        assert fake_redis.smembers("docks:all") == {"only"}
        assert fake_redis.get(_lock_key(DockType.PICKUP, "pickup_1")) is None
        assert fake_redis.get("active_dock_config") == str(cfg.id)

    def test_robot_positions_include_domain_zero(self, fake_redis):
        fake_redis.set("robot_ips", json.dumps({"bot_0": {"ip": "10.0.0.9", "ros_domain_id": 0}}))
        resp = MagicMock(status_code=200, json=MagicMock(
            return_value={"rl_robot_type": "picker", "x": 1.0, "y": 2.0, "yaw": 0.0}))
        with patch("services.redis_dock_runtime.requests.get", return_value=resp):
            positions = rdr.get_all_robot_positions(fake_redis, "couliglig_bot_0")
        assert "couliglig_bot_0" in positions

    def test_waiting_zones_match_action_map_order(self, db_session, fake_redis):
        _seed(db_session, fake_redis)
        zone_ids = [z["zone_id"] for z in rdr.get_all_waiting_zone_states(fake_redis)]
        slot_ids = [s["dock_id"] for s in rdr.build_action_map(fake_redis, [])["wz_slots"] if s]
        assert zone_ids == ["wz_a", "wz_b"]
        assert zone_ids == slot_ids
