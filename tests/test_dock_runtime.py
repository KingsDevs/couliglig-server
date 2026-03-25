"""
Pytest suite for all endpoints that use redis_dock_runtime.

Isolation strategy:
  - SQLite in-memory database for all DB-backed endpoints.
  - fakeredis (in-memory Redis drop-in) for all Redis state.
  - requests_mock / unittest.mock for external HTTP calls made to robots.

Run with:
    pytest tests/test_dock_runtime.py -v
"""

import json
import pytest
import fakeredis
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from schema.base import Base
from schema.docks import DockConfig, Dock
from definitions import DockType
from main import app
from services.database import get_db_session
import services.redis_dock_runtime as rdr
import routes.docks as docks_route
import routes.rl as obs_route


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_redis():
    """Return a fresh fakeredis instance for each test."""
    r = fakeredis.FakeRedis(decode_responses=True)
    yield r
    r.flushall()


@pytest.fixture()
def db_session():
    """In-memory SQLite session that is torn down after each test."""
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
    """
    TestClient with:
      - DB dependency overridden to in-memory SQLite
      - Redis client on both routers patched to fakeredis
    """
    def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db

    # Patch the module-level redis_client used by each router
    docks_route.redis_client = fake_redis
    obs_route.redis_client = fake_redis

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_config_and_docks(db_session, fake_redis) -> tuple[int, list[str]]:
    """Create a DockConfig with three docks in SQLite and return (config_id, dock_ids)."""
    cfg = DockConfig(name="test_config", description="pytest config")
    db_session.add(cfg)
    db_session.commit()
    db_session.refresh(cfg)

    docks_data = [
        dict(dock_id="dock_pickup_1",   dock_type=DockType.PICKUP,       aruco_id=1, x=1.0, y=2.0, theta=0.0),
        dict(dock_id="dock_receiver_1", dock_type=DockType.RECEIVER,     aruco_id=2, x=3.0, y=4.0, theta=1.57),
        dict(dock_id="dock_wz_1",       dock_type=DockType.WAITING_ZONE, aruco_id=3, x=5.0, y=6.0, theta=3.14),
    ]
    dock_ids = []
    for d in docks_data:
        dock = Dock(config_id=cfg.id, **d)
        db_session.add(dock)
        dock_ids.append(d["dock_id"])

    db_session.commit()

    # Activate via the service directly so Redis is seeded
    rdr.activate_docks(fake_redis, db_session, cfg.id)

    return cfg.id, dock_ids


# ===========================================================================
# /dock  –  config management
# ===========================================================================

class TestDockConfig:

    def test_create_config(self, client):
        resp = client.post("/dock", json={"name": "cfg_1", "description": "first"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "cfg_1"
        assert "id" in data

    def test_create_config_duplicate_name(self, client):
        client.post("/dock", json={"name": "dup", "description": ""})
        resp = client.post("/dock", json={"name": "dup", "description": ""})
        assert resp.status_code == 409

    def test_list_configs_empty(self, client):
        resp = client.get("/dock/configs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_configs_shows_active(self, client, fake_redis, db_session):
        config_id, _ = _seed_config_and_docks(db_session, fake_redis)
        resp = client.get("/dock/configs")
        assert resp.status_code == 200
        configs = resp.json()
        assert len(configs) == 1
        assert configs[0]["is_active"] is True

    def test_update_config(self, client):
        resp = client.post("/dock", json={"name": "old_name"})
        config_id = resp.json()["id"]
        resp = client.put(f"/dock/update_config?config_id={config_id}", json={"name": "new_name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "new_name"

    def test_delete_config(self, client):
        resp = client.post("/dock", json={"name": "to_delete"})
        config_id = resp.json()["id"]
        resp = client.delete(f"/dock/delete_config?config_id={config_id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_delete_config_not_found(self, client):
        resp = client.delete("/dock/delete_config?config_id=9999")
        assert resp.status_code == 404


# ===========================================================================
# /dock  –  dock management
# ===========================================================================

class TestDockManagement:

    def test_create_docks(self, client, db_session):
        cfg_resp = client.post("/dock", json={"name": "c1"})
        config_id = cfg_resp.json()["id"]

        payload = [
            {"config_id": config_id, "dock_id": "d1", "dock_type": "pickup",
             "aruco_id": 1, "x": 1.0, "y": 2.0, "theta": 0.0},
        ]
        resp = client.post("/dock/create_dock", json=payload)
        assert resp.status_code == 200
        assert resp.json()[0]["dock_id"] == "d1"

    def test_create_dock_duplicate_id(self, client):
        cfg_resp = client.post("/dock", json={"name": "c2"})
        config_id = cfg_resp.json()["id"]
        payload = [{"config_id": config_id, "dock_id": "d2", "dock_type": "pickup",
                    "aruco_id": 2, "x": 0.0, "y": 0.0, "theta": 0.0}]
        client.post("/dock/create_dock", json=payload)
        resp = client.post("/dock/create_dock", json=payload)
        assert resp.status_code == 409

    def test_list_docks(self, client, db_session, fake_redis):
        config_id, dock_ids = _seed_config_and_docks(db_session, fake_redis)
        resp = client.get(f"/dock?config_id={config_id}")
        assert resp.status_code == 200
        returned_ids = [d["dock_id"] for d in resp.json()]
        for did in dock_ids:
            assert did in returned_ids

    def test_update_dock(self, client, db_session, fake_redis):
        config_id, _ = _seed_config_and_docks(db_session, fake_redis)
        from schema.docks import Dock as DockModel
        dock = db_session.query(DockModel).filter_by(dock_id="dock_pickup_1").first()
        resp = client.put(f"/dock/update_dock?dock_db_id={dock.id}", json={"x": 9.9, "y": 8.8})
        assert resp.status_code == 200
        assert float(resp.json()["x"]) == pytest.approx(9.9)

    def test_delete_dock(self, client, db_session, fake_redis):
        config_id, _ = _seed_config_and_docks(db_session, fake_redis)
        from schema.docks import Dock as DockModel
        dock = db_session.query(DockModel).filter_by(dock_id="dock_pickup_1").first()
        resp = client.delete(f"/dock/delete_dock?dock_db_id={dock.id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ===========================================================================
# /dock  –  activation & clear
# ===========================================================================

class TestActivateAndClear:

    def test_activate_config(self, client, db_session, fake_redis):
        config_id, _ = _seed_config_and_docks(db_session, fake_redis)
        # Clear then re-activate via the HTTP endpoint
        fake_redis.set("active_dock_config", "none")
        resp = client.post(f"/dock/activate?config_id={config_id}")
        assert resp.status_code == 200
        assert resp.json()["active_config_id"] == config_id

    def test_activate_config_not_found(self, client):
        resp = client.post("/dock/activate?config_id=9999")
        assert resp.status_code == 404

    def test_clear_docks(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        assert fake_redis.get("active_dock_config") is not None
        resp = client.post("/dock/clear")
        assert resp.status_code == 200
        assert fake_redis.get("active_dock_config") == "none"


# ===========================================================================
# /dock  –  runtime actions
# ===========================================================================

class TestDockActions:

    def test_add_item(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.post("/dock/add-item", json={
            "dock_id": "dock_pickup_1",
            "item_id": "item_001",
            "item_weight": 2.5,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_add_item_to_non_pickup_dock(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.post("/dock/add-item", json={
            "dock_id": "dock_receiver_1",
            "item_id": "item_002",
        })
        assert resp.status_code == 400

    def test_add_item_already_has_item(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        client.post("/dock/add-item", json={"dock_id": "dock_pickup_1", "item_id": "item_001"})
        resp = client.post("/dock/add-item", json={"dock_id": "dock_pickup_1", "item_id": "item_002"})
        assert resp.status_code == 400

    def test_remove_item(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        client.post("/dock/add-item", json={"dock_id": "dock_pickup_1", "item_id": "item_001"})
        resp = client.post("/dock/remove-item", json={"dock_id": "dock_pickup_1"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_remove_item_invalid_dock(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.post("/dock/remove-item", json={"dock_id": "ghost_dock"})
        assert resp.status_code == 400

    def test_reserve_dock(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.post("/dock/reserve", json={"dock_id": "dock_pickup_1", "robot_id": "robot_1"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_reserve_dock_already_reserved(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        client.post("/dock/reserve", json={"dock_id": "dock_pickup_1", "robot_id": "robot_1"})
        resp = client.post("/dock/reserve", json={"dock_id": "dock_pickup_1", "robot_id": "robot_2"})
        assert resp.status_code == 409

    def test_occupy_dock(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        client.post("/dock/reserve", json={"dock_id": "dock_pickup_1", "robot_id": "robot_1"})
        resp = client.post("/dock/occupy", json={"dock_id": "dock_pickup_1", "robot_id": "robot_1"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_occupy_dock_without_reserve_returns_400(self, client, db_session, fake_redis):
        """Cannot occupy a dock that has not been reserved first."""
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.post("/dock/occupy", json={"dock_id": "dock_pickup_1", "robot_id": "robot_1"})
        assert resp.status_code == 400
        assert "reserved" in resp.json()["detail"].lower()

    def test_occupy_dock_wrong_robot_returns_400(self, client, db_session, fake_redis):
        """Cannot occupy a dock reserved by a different robot."""
        _seed_config_and_docks(db_session, fake_redis)
        client.post("/dock/reserve", json={"dock_id": "dock_pickup_1", "robot_id": "robot_1"})
        resp = client.post("/dock/occupy", json={"dock_id": "dock_pickup_1", "robot_id": "robot_2"})
        assert resp.status_code == 400
        assert "robot_1" in resp.json()["detail"]

    def test_release_dock(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        client.post("/dock/reserve", json={"dock_id": "dock_pickup_1", "robot_id": "robot_1"})
        resp = client.post("/dock/release", json={"dock_id": "dock_pickup_1"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        # Verify status reset to available
        state = client.get("/dock/dock_state?dock_id=dock_pickup_1").json()
        assert state["status"] == "available"
        assert state["robot_id"] == ""

    def test_release_invalid_dock(self, client):
        resp = client.post("/dock/release", json={"dock_id": "no_such_dock"})
        assert resp.status_code == 400


# ===========================================================================
# /dock  –  state queries
# ===========================================================================

class TestDockStateQueries:

    def test_get_dock_state(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.get("/dock/dock_state?dock_id=dock_pickup_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "available"
        assert float(data["x"]) == pytest.approx(1.0)
        assert float(data["y"]) == pytest.approx(2.0)

    def test_get_dock_state_not_found(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.get("/dock/dock_state?dock_id=ghost_dock")
        assert resp.status_code == 400

    def test_get_all_dock_states(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        resp = client.get("/dock/all_dock_states")
        assert resp.status_code == 200
        states = resp.json()
        assert len(states) == 3
        dock_ids = [s["dock_id"] for s in states]
        assert "dock_pickup_1" in dock_ids
        assert "dock_receiver_1" in dock_ids
        assert "dock_wz_1" in dock_ids

    def test_get_all_dock_states_no_active_config(self, client, fake_redis):
        resp = client.get("/dock/all_dock_states")
        assert resp.status_code == 200
        assert resp.json() == []


# ===========================================================================
# /rl/builder-inputs
# ===========================================================================

class TestObsBuilderInputs:

    def _mock_picker_rl_response(self):
        return {
            "agent_id": "couliglig_bot_1",
            "agent_type": "picker",
            "has_item": False,
        }

    def _mock_transporter_rl_response(self):
        return {
            "agent_id": "couliglig_bot_2",
            "agent_type": "transporter",
            "capacity": 1.0,
            "max_capacity": 4.0,
            "carried_items": ["item_001"],
            "in_waiting_zone": False,
        }

    def _mock_transform_response(self, robot_type="picker"):
        return {
            "rl_robot_type": robot_type,
            "frame_id": "map",
            "child_frame_id": "base_link",
            "x": 1.5,
            "y": -0.5,
            "yaw": 0.1,
        }

    def test_obs_builder_inputs_no_active_config(self, client, fake_redis):
        """With no active config, dock_states should be empty but endpoint returns 200."""
        with patch("services.redis_dock_runtime.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dock_states"] == []
        assert data["dock_positions"] == {}

    def test_obs_builder_inputs_with_active_config(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        client.post("/dock/add-item", json={
            "dock_id": "dock_pickup_1",
            "item_id": "item_001",
            "item_weight": 3.0,
        })

        with patch("services.redis_dock_runtime.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_1")

        assert resp.status_code == 200
        data = resp.json()

        # dock_states: all three docks present
        assert len(data["dock_states"]) == 3

        # dock_positions: all three docks with x, y, yaw
        assert len(data["dock_positions"]) == 3
        pos = data["dock_positions"]["dock_pickup_1"]
        assert len(pos) == 3  # (x, y, yaw)

        # item_weights: item_001 present
        assert "item_001" in data["item_weights"]
        assert data["item_weights"]["item_001"] == pytest.approx(3.0)

        # waiting_zones: dock_wz_1 is the only waiting_zone
        zone_ids = [z["zone_id"] for z in data["waiting_zones"]]
        assert "dock_wz_1" in zone_ids
        assert "dock_pickup_1" not in zone_ids

    def test_obs_builder_inputs_robot_positions(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)

        fake_redis.set("robot_ips", json.dumps({
            "couliglig_bot_1": {"ip": "192.168.1.10", "ros_domain_id": 1, "namespace": "couliglig_bot_1"},
        }))

        transform_mock = MagicMock(
            status_code=200,
            json=MagicMock(return_value=self._mock_transform_response("picker")),
        )
        rl_mock = MagicMock(
            status_code=200,
            json=MagicMock(return_value=self._mock_picker_rl_response()),
        )

        def side_effect(url, timeout=3):
            if "/roslib/transform" in url:
                return transform_mock
            if url.endswith("/rl"):
                return rl_mock
            return MagicMock(status_code=404)

        with patch("services.redis_dock_runtime.requests.get", side_effect=side_effect):
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_1")

        assert resp.status_code == 200
        data = resp.json()

        # robot_positions keyed by namespace couliglig_bot_1
        assert "couliglig_bot_1" in data["robot_positions"]
        pos = data["robot_positions"]["couliglig_bot_1"]
        assert len(pos) == 4  # (robot_type, x, y, yaw)
        assert pos[0] == "picker"

        # picker_has_item keyed by namespace
        assert "couliglig_bot_1" in data["picker_has_item"]
        assert data["picker_has_item"]["couliglig_bot_1"] is False

    def test_obs_builder_inputs_transporter_state(self, client, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)

        fake_redis.set("robot_ips", json.dumps({
            "couliglig_bot_2": {"ip": "192.168.1.11", "ros_domain_id": 2, "namespace": "couliglig_bot_2"},
        }))

        transform_mock = MagicMock(
            status_code=200,
            json=MagicMock(return_value=self._mock_transform_response("transporter")),
        )
        rl_mock = MagicMock(
            status_code=200,
            json=MagicMock(return_value=self._mock_transporter_rl_response()),
        )

        def side_effect(url, timeout=3):
            if "/roslib/transform" in url:
                return transform_mock
            if url.endswith("/rl"):
                return rl_mock
            return MagicMock(status_code=404)

        with patch("services.redis_dock_runtime.requests.get", side_effect=side_effect):
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_2")

        assert resp.status_code == 200
        data = resp.json()

        ns = "couliglig_bot_2"
        assert ns in data["transporter_loads"]
        assert data["transporter_loads"][ns][0] == pytest.approx(1.0)   # capacity
        assert data["transporter_loads"][ns][1] == pytest.approx(4.0)   # max_capacity
        assert data["transporter_carried"][ns] == ["item_001"]
        assert data["transporter_in_wz"][ns] is False

    def test_obs_builder_inputs_robot_offline(self, client, db_session, fake_redis):
        """Offline robots should be silently skipped, not raise 500."""
        _seed_config_and_docks(db_session, fake_redis)
        fake_redis.set("robot_ips", json.dumps({
            "offline_bot": {"ip": "10.0.0.99", "ros_domain_id": 9, "namespace": "couliglig_bot_9"},
        }))

        with patch(
            "services.redis_dock_runtime.requests.get",
            side_effect=Exception("Connection refused"),
        ):
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_9")

        assert resp.status_code == 200
        data = resp.json()
        assert data["robot_positions"] == {}
        assert data["picker_has_item"] == {}


# ===========================================================================
# redis_dock_runtime unit tests (service-layer, no HTTP)
# ===========================================================================

class TestRedisDockRuntimeUnit:

    def test_get_all_dock_positions_includes_yaw(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        positions = rdr.get_all_dock_positions(fake_redis)
        assert "dock_pickup_1" in positions
        x, y, yaw = positions["dock_pickup_1"]
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert yaw == pytest.approx(0.0)

    def test_get_all_dock_positions_skips_missing_yaw(self, fake_redis):
        """Docks with empty/missing yaw should not appear in positions."""
        fake_redis.sadd("docks:all", "no_yaw_dock")
        fake_redis.hset("dock_meta:no_yaw_dock", mapping={"dock_type": "pickup"})
        fake_redis.hset("dock:pickup:no_yaw_dock", mapping={"x": "1.0", "y": "2.0", "yaw": ""})
        positions = rdr.get_all_dock_positions(fake_redis)
        assert "no_yaw_dock" not in positions

    def test_get_all_item_weights(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        rdr.add_item_to_pickup_dock(fake_redis, "dock_pickup_1", "item_abc", item_weight=5.0)
        weights = rdr.get_all_item_weights(fake_redis)
        assert weights["item_abc"] == pytest.approx(5.0)

    def test_get_all_item_weights_empty_when_no_item(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        weights = rdr.get_all_item_weights(fake_redis)
        assert weights == {}

    def test_get_all_waiting_zone_states(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        zones = rdr.get_all_waiting_zone_states(fake_redis)
        assert len(zones) == 1
        assert zones[0]["zone_id"] == "dock_wz_1"
        assert zones[0]["x"] == pytest.approx(5.0)
        assert zones[0]["y"] == pytest.approx(6.0)

    def test_get_all_robot_positions_empty_when_no_registrations(self, fake_redis):
        positions = rdr.get_all_robot_positions(fake_redis, "couliglig_bot_1")
        assert positions == {}

    def test_get_all_robot_positions_calls_transform(self, fake_redis):
        fake_redis.set("robot_ips", json.dumps({
            "bot_1": {"ip": "192.168.1.5", "ros_domain_id": 1},
        }))
        mock_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"rl_robot_type": "picker", "x": 0.5, "y": 1.5, "yaw": 0.2}),
        )
        with patch("services.redis_dock_runtime.requests.get", return_value=mock_resp):
            positions = rdr.get_all_robot_positions(fake_redis, "couliglig_bot_1")

        assert "couliglig_bot_1" in positions
        robot_type, x, y, yaw = positions["couliglig_bot_1"]
        assert robot_type == "picker"
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(1.5)
        assert yaw == pytest.approx(0.2)

    def test_reserve_dock_prevents_double_reserve(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        assert rdr.reserve_dock(fake_redis, "dock_pickup_1", "robot_A") is True
        assert rdr.reserve_dock(fake_redis, "dock_pickup_1", "robot_B") is False

    def test_reserve_release_cycle(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "dock_pickup_1", "robot_A")
        rdr.release_dock(fake_redis, "dock_pickup_1")
        # After release, another robot should be able to reserve
        assert rdr.reserve_dock(fake_redis, "dock_pickup_1", "robot_B") is True

    def test_add_and_remove_item(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        assert rdr.add_item_to_pickup_dock(fake_redis, "dock_pickup_1", "itm_1", item_weight=1.5)
        state = rdr.get_dock_state(fake_redis, "dock_pickup_1")
        assert state["item_id"] == "itm_1"
        assert float(state["item_weight"]) == pytest.approx(1.5)

        assert rdr.remove_item_from_pickup_dock(fake_redis, "dock_pickup_1")
        state = rdr.get_dock_state(fake_redis, "dock_pickup_1")
        assert state["item_id"] == ""

    def test_clear_all_dock_keys(self, db_session, fake_redis):
        _seed_config_and_docks(db_session, fake_redis)
        rdr.clear_all_dock_keys(fake_redis)
        assert fake_redis.get("active_dock_config") == "none"
        assert fake_redis.scard("docks:all") == 0


# ===========================================================================
# build_action_map  (unit tests)
# ===========================================================================

class TestBuildActionMap:

    def test_action_map_item_slots_length(self, db_session, fake_redis):
        """item_slots is always padded to MAX_ITEMS regardless of dock count."""
        _seed_config_and_docks(db_session, fake_redis)
        action_map = rdr.build_action_map(fake_redis, [])
        assert len(action_map["item_slots"]) == rdr.RL_CONSTANTS["MAX_ITEMS"]

    def test_action_map_wz_slots_length(self, db_session, fake_redis):
        """wz_slots is always padded to MAX_WZ regardless of dock count."""
        _seed_config_and_docks(db_session, fake_redis)
        action_map = rdr.build_action_map(fake_redis, [])
        assert len(action_map["wz_slots"]) == rdr.RL_CONSTANTS["MAX_WZ"]

    def test_action_map_deterministic_ordering(self, db_session, fake_redis):
        """Calling build_action_map twice gives the same slot order."""
        _seed_config_and_docks(db_session, fake_redis)
        map1 = rdr.build_action_map(fake_redis, [])
        map2 = rdr.build_action_map(fake_redis, [])
        assert map1 == map2

    def test_action_map_item_slot_content(self, db_session, fake_redis):
        """First item_slot matches the (only) pickup dock with correct fields."""
        _seed_config_and_docks(db_session, fake_redis)
        rdr.add_item_to_pickup_dock(fake_redis, "dock_pickup_1", "item_A", item_weight=2.5)
        action_map = rdr.build_action_map(fake_redis, [])

        slot = action_map["item_slots"][0]
        assert slot is not None
        assert slot["dock_id"] == "dock_pickup_1"
        assert slot["item_id"] == "item_A"
        assert slot["item_weight"] == pytest.approx(2.5)
        assert slot["index"] == 0
        assert slot["available_for_pickup"] is True

    def test_action_map_empty_item_slot(self, db_session, fake_redis):
        """Pickup dock without an item still appears with item_id='' and available_for_pickup=False."""
        _seed_config_and_docks(db_session, fake_redis)
        action_map = rdr.build_action_map(fake_redis, [])

        slot = action_map["item_slots"][0]
        assert slot is not None
        assert slot["item_id"] == ""
        assert slot["available_for_pickup"] is False

    def test_action_map_padding_is_none(self, db_session, fake_redis):
        """Slots beyond real dock count are None (padding)."""
        _seed_config_and_docks(db_session, fake_redis)
        action_map = rdr.build_action_map(fake_redis, [])
        # Only 1 pickup dock, so index 1..MAX_ITEMS-1 should be None
        assert action_map["item_slots"][1] is None
        # Only 1 wz dock, so index 1..MAX_WZ-1 should be None
        assert action_map["wz_slots"][1] is None

    def test_action_map_wz_available_for_entry(self, db_session, fake_redis):
        """WZ slot shows available_for_entry based on dock status."""
        _seed_config_and_docks(db_session, fake_redis)
        action_map = rdr.build_action_map(fake_redis, [])
        wz = action_map["wz_slots"][0]
        assert wz is not None
        assert wz["dock_id"] == "dock_wz_1"
        assert wz["available_for_entry"] is True

    def test_action_map_wz_reserved_not_available(self, db_session, fake_redis):
        """Reserving a WZ dock makes available_for_entry False."""
        _seed_config_and_docks(db_session, fake_redis)
        rdr.reserve_dock(fake_redis, "dock_wz_1", "robot_X")
        action_map = rdr.build_action_map(fake_redis, [])
        wz = action_map["wz_slots"][0]
        assert wz["available_for_entry"] is False

    def test_action_map_receiver_slots(self, db_session, fake_redis):
        """Receiver docks appear in receiver_slots (unpadded)."""
        _seed_config_and_docks(db_session, fake_redis)
        action_map = rdr.build_action_map(fake_redis, [])
        assert len(action_map["receiver_slots"]) == 1
        assert action_map["receiver_slots"][0]["dock_id"] == "dock_receiver_1"
        assert action_map["receiver_slots"][0]["index"] == 0

    def test_action_map_robot_slots_from_agent_states(self, fake_redis):
        """Robot slots are built from agent_states, sorted by namespace."""
        agent_states = [
            {"namespace": "couliglig_bot_2", "agent_type": "transporter"},
            {"namespace": "couliglig_bot_1", "agent_type": "picker"},
        ]
        action_map = rdr.build_action_map(fake_redis, agent_states)
        # picker
        assert action_map["picker_slots"][0] is not None
        assert action_map["picker_slots"][0]["namespace"] == "couliglig_bot_1"
        assert action_map["picker_slots"][0]["index"] == 0
        # transporter
        assert action_map["transporter_slots"][0] is not None
        assert action_map["transporter_slots"][0]["namespace"] == "couliglig_bot_2"
        assert action_map["transporter_slots"][0]["index"] == 0

    def test_action_map_robot_slots_padded(self, fake_redis):
        """Robot slots are padded to MAX_PICKERS / MAX_TRANSPORTERS."""
        action_map = rdr.build_action_map(fake_redis, [])
        assert len(action_map["picker_slots"]) == rdr.RL_CONSTANTS["MAX_PICKERS"]
        assert len(action_map["transporter_slots"]) == rdr.RL_CONSTANTS["MAX_TRANSPORTERS"]
        assert all(s is None for s in action_map["picker_slots"])
        assert all(s is None for s in action_map["transporter_slots"])

    def test_action_map_empty_redis(self, fake_redis):
        """With no docks at all, all slots are None / empty."""
        action_map = rdr.build_action_map(fake_redis, [])
        assert all(s is None for s in action_map["item_slots"])
        assert all(s is None for s in action_map["wz_slots"])
        assert action_map["receiver_slots"] == []


# ===========================================================================
# /rl/builder-inputs  –  action_map & rl_constants in response
# ===========================================================================

class TestBuilderInputsActionMap:

    def test_response_contains_action_map(self, client, db_session, fake_redis):
        """GET /rl/builder-inputs returns action_map key."""
        _seed_config_and_docks(db_session, fake_redis)
        with patch("services.redis_dock_runtime.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_1")
        assert resp.status_code == 200
        data = resp.json()
        assert "action_map" in data
        assert "item_slots" in data["action_map"]
        assert "wz_slots" in data["action_map"]
        assert "receiver_slots" in data["action_map"]
        assert "picker_slots" in data["action_map"]
        assert "transporter_slots" in data["action_map"]

    def test_response_contains_rl_constants(self, client, db_session, fake_redis):
        """GET /rl/builder-inputs returns rl_constants with expected keys."""
        _seed_config_and_docks(db_session, fake_redis)
        with patch("services.redis_dock_runtime.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_1")
        assert resp.status_code == 200
        consts = resp.json()["rl_constants"]
        assert consts["MAX_ITEMS"] == 20
        assert consts["MAX_WZ"] == 8
        assert consts["MAX_PICKERS"] == 10
        assert consts["MAX_TRANSPORTERS"] == 5
        assert "PICKER_ACTION_DIM" in consts
        assert "TRANSPORTER_ACTION_DIM" in consts
        assert "NUM_PICKERS" in consts
        assert "NUM_TRANSPORTERS" in consts

    def test_rl_constants_dynamic_counts(self, client, db_session, fake_redis):
        """NUM_PICKERS/NUM_TRANSPORTERS reflect live robot count."""
        _seed_config_and_docks(db_session, fake_redis)
        fake_redis.set("robot_ips", json.dumps({
            "bot1": {"ip": "10.0.0.1", "ros_domain_id": 1, "namespace": "couliglig_bot_1"},
            "bot2": {"ip": "10.0.0.2", "ros_domain_id": 2, "namespace": "couliglig_bot_2"},
        }))

        picker_rl = {"agent_id": "couliglig_bot_1", "agent_type": "picker", "has_item": False}
        transporter_rl = {
            "agent_id": "couliglig_bot_2", "agent_type": "transporter",
            "capacity": 0, "max_capacity": 4, "carried_items": [], "in_waiting_zone": False,
        }

        def side_effect(url, timeout=3):
            if url.endswith("/rl") and "10.0.0.1" in url:
                return MagicMock(status_code=200, json=MagicMock(return_value=picker_rl))
            if url.endswith("/rl") and "10.0.0.2" in url:
                return MagicMock(status_code=200, json=MagicMock(return_value=transporter_rl))
            return MagicMock(status_code=404)

        with patch("services.redis_dock_runtime.requests.get", side_effect=side_effect):
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_1")

        consts = resp.json()["rl_constants"]
        assert consts["NUM_PICKERS"] == 1
        assert consts["NUM_TRANSPORTERS"] == 1
        # PICKER_ACTION_DIM = 1 + MAX_ITEMS + num_transporters
        assert consts["PICKER_ACTION_DIM"] == 1 + 20 + 1

    def test_action_map_item_slot_matches_dock(self, client, db_session, fake_redis):
        """item_slots[0] dock_id corresponds to the pickup dock in Redis."""
        _seed_config_and_docks(db_session, fake_redis)
        rdr.add_item_to_pickup_dock(fake_redis, "dock_pickup_1", "item_99", item_weight=7.0)

        with patch("services.redis_dock_runtime.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            resp = client.get("/rl/builder-inputs?robot_id=couliglig_bot_1")

        am = resp.json()["action_map"]
        slot = am["item_slots"][0]
        assert slot["dock_id"] == "dock_pickup_1"
        assert slot["item_id"] == "item_99"
        assert slot["item_weight"] == pytest.approx(7.0)
        assert slot["available_for_pickup"] is True
