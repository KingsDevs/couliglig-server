"""
Tests for GET /rl/builder-inputs/dummy

Isolation strategy:
  - In-memory SQLite seeded with a real DockConfig + Docks.
  - No Redis needed — the endpoint is fully self-contained.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from schema.base import Base
from schema.docks import DockConfig, Dock
from definitions import DockType
from main import app
from services.database import get_db_session
from services.redis_dock_runtime import RL_CONSTANTS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    session = Session()

    cfg = DockConfig(name="test_cfg", description="dummy test config")
    session.add(cfg)
    session.flush()

    docks = []
    for i in range(5):
        docks.append(Dock(config_id=cfg.id, dock_id=f"pickup_{i}",
                          dock_type=DockType.PICKUP, x=float(i), y=float(i), theta=0.0))
    for i in range(3):
        docks.append(Dock(config_id=cfg.id, dock_id=f"wz_{i}",
                          dock_type=DockType.WAITING_ZONE, x=float(i+10), y=float(i+10), theta=0.0))
    for i in range(3):
        docks.append(Dock(config_id=cfg.id, dock_id=f"recv_{i}",
                          dock_type=DockType.RECEIVER, x=float(i+20), y=float(i+20), theta=0.0))
    session.add_all(docks)
    session.commit()

    yield session, cfg.id

    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def test_client(db_session):
    session, config_id = db_session

    def override_db():
        yield session

    app.dependency_overrides[get_db_session] = override_db
    with TestClient(app) as c:
        yield c, config_id
    app.dependency_overrides.clear()


ROBOT_ID = "couliglig_bot_99"
BASE_URL = "/rl/builder-inputs/dummy"


def get_dummy(client, config_id, extra: dict | None = None) -> dict:
    params = {"robot_id": ROBOT_ID, "dock_config_id": config_id, **(extra or {})}
    resp = client.get(BASE_URL, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestStructure:
    def test_top_level_keys_present(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        expected_keys = {
            "dock_states", "dock_positions", "item_weights", "item_receiver_docks",
            "robot_positions", "picker_has_item", "transporter_loads",
            "transporter_carried", "transporter_in_wz", "waiting_zones",
            "action_map", "rl_constants",
        }
        assert expected_keys.issubset(data.keys())

    def test_action_map_keys_present(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        am = data["action_map"]
        assert set(am.keys()) == {
            "item_slots", "wz_slots", "receiver_slots", "picker_slots", "transporter_slots"
        }

    def test_rl_constants_keys_present(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        rc = data["rl_constants"]
        for key in ("MAX_ITEMS", "MAX_WZ", "MAX_PICKERS", "MAX_TRANSPORTERS",
                    "PICKER_ACTION_DIM", "TRANSPORTER_ACTION_DIM",
                    "NUM_PICKERS", "NUM_TRANSPORTERS"):
            assert key in rc


# ---------------------------------------------------------------------------
# Dock counts & values come from DB
# ---------------------------------------------------------------------------

class TestCounts:
    def test_dock_counts_match_db(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        types = [d["dock_type"] for d in data["dock_states"]]
        assert types.count("pickup") == 5
        assert types.count("waiting_zone") == 3
        assert types.count("receiver") == 3

    def test_dock_ids_match_db(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        dock_ids = {d["dock_id"] for d in data["dock_states"]}
        assert {f"pickup_{i}" for i in range(5)}.issubset(dock_ids)
        assert {f"wz_{i}" for i in range(3)}.issubset(dock_ids)
        assert {f"recv_{i}" for i in range(3)}.issubset(dock_ids)

    def test_dock_positions_use_db_values(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        # pickup_0 was seeded with x=0.0, y=0.0, theta=0.0
        pos = data["dock_positions"]["pickup_0"]
        assert pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 0.0

    def test_action_map_slots_padded_to_max(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        am = data["action_map"]
        assert len(am["item_slots"]) == RL_CONSTANTS["MAX_ITEMS"]
        assert len(am["wz_slots"]) == RL_CONSTANTS["MAX_WZ"]
        assert len(am["picker_slots"]) == RL_CONSTANTS["MAX_PICKERS"]
        assert len(am["transporter_slots"]) == RL_CONSTANTS["MAX_TRANSPORTERS"]

    def test_robot_counts_match_params(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"num_pickers": 3, "num_transporters": 2})
        assert len(data["picker_has_item"]) == 3
        assert len(data["transporter_loads"]) == 2

    def test_robot_counts_capped_at_max(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"num_pickers": 999, "num_transporters": 999})
        assert len(data["picker_has_item"]) <= RL_CONSTANTS["MAX_PICKERS"]
        assert len(data["transporter_loads"]) <= RL_CONSTANTS["MAX_TRANSPORTERS"]


# ---------------------------------------------------------------------------
# 404 for bad dock_config_id
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_missing_config_returns_404(self, test_client):
        client, _ = test_client
        resp = client.get(BASE_URL, params={"robot_id": ROBOT_ID, "dock_config_id": 99999})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# robot_id always present
# ---------------------------------------------------------------------------

class TestRobotIdPresent:
    def test_robot_id_in_robot_positions(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        assert ROBOT_ID in data["robot_positions"]

    def test_robot_id_in_picker_has_item(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        assert ROBOT_ID in data["picker_has_item"]

    def test_robot_id_is_first_picker_slot(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        first = data["action_map"]["picker_slots"][0]
        assert first is not None
        assert first["namespace"] == ROBOT_ID

    def test_robot_id_not_in_transporter_slots(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        namespaces = [s["namespace"] for s in data["action_map"]["transporter_slots"] if s]
        assert ROBOT_ID not in namespaces

    def test_dock_robot_ids_are_known_robots(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id)
        known = set(data["robot_positions"].keys())
        for dock in data["dock_states"]:
            rid = dock.get("robot_id", "")
            if rid:
                assert rid in known, f"Unknown robot_id '{rid}' on dock {dock['dock_id']}"


# ---------------------------------------------------------------------------
# Scenario flags
# ---------------------------------------------------------------------------

class TestAllItemsAvailable:
    def test_all_pickup_docks_available(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"all_items_available": "true"})
        for dock in data["dock_states"]:
            if dock["dock_type"] == "pickup":
                assert dock["status"] == "available"
                assert dock["robot_id"] == ""
                assert dock["item_id"] != ""

    def test_action_map_all_available_for_pickup(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"all_items_available": "true"})
        real_slots = [s for s in data["action_map"]["item_slots"] if s is not None]
        assert all(s["available_for_pickup"] for s in real_slots)


class TestNoDockedRobots:
    def test_no_dock_has_robot(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"no_docked_robots": "true"})
        for dock in data["dock_states"]:
            assert dock["robot_id"] == ""
            assert dock["status"] == "available"


class TestAllWzAvailable:
    def test_all_wz_available(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"all_wz_available": "true"})
        for dock in data["dock_states"]:
            if dock["dock_type"] == "waiting_zone":
                assert dock["status"] == "available"
                assert dock["robot_id"] == ""

    def test_waiting_zones_list_reflects_flag(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"all_wz_available": "true"})
        for wz in data["waiting_zones"]:
            assert wz["status"] == "available"
            assert wz["robot_id"] == ""


class TestAllReceiversAvailable:
    def test_all_receivers_available(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"all_receivers_available": "true"})
        for dock in data["dock_states"]:
            if dock["dock_type"] == "receiver":
                assert dock["status"] == "available"
                assert dock["robot_id"] == ""


class TestPickerItemFlags:
    def test_pickers_have_item(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"pickers_have_item": "true"})
        assert all(v is True for v in data["picker_has_item"].values())

    def test_pickers_no_item(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"pickers_no_item": "true"})
        assert all(v is False for v in data["picker_has_item"].values())


class TestTransporterFlags:
    def test_transporters_empty(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"transporters_empty": "true"})
        for ns, load in data["transporter_loads"].items():
            assert load[0] == 0.0
        for ns, carried in data["transporter_carried"].items():
            assert carried == []

    def test_transporters_in_wz(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"transporters_in_wz": "true"})
        assert all(v is True for v in data["transporter_in_wz"].values())

    def test_transporters_not_in_wz(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {"transporters_not_in_wz": "true"})
        assert all(v is False for v in data["transporter_in_wz"].values())


# ---------------------------------------------------------------------------
# Stacked flags
# ---------------------------------------------------------------------------

class TestStackedFlags:
    def test_clean_start_scenario(self, test_client):
        client, config_id = test_client
        data = get_dummy(client, config_id, {
            "all_items_available": "true",
            "no_docked_robots": "true",
            "pickers_no_item": "true",
            "transporters_empty": "true",
        })
        for dock in data["dock_states"]:
            assert dock["robot_id"] == ""
            assert dock["status"] == "available"
        real_slots = [s for s in data["action_map"]["item_slots"] if s is not None]
        assert all(s["available_for_pickup"] for s in real_slots)
        assert all(v is False for v in data["picker_has_item"].values())
        for ns, carried in data["transporter_carried"].items():
            assert carried == []


# ---------------------------------------------------------------------------
# RL constants correctness
# ---------------------------------------------------------------------------

class TestRlConstants:
    def test_action_dims(self, test_client):
        client, config_id = test_client
        num_pickers = 3
        num_transporters = 2
        data = get_dummy(client, config_id, {"num_pickers": num_pickers, "num_transporters": num_transporters})
        rc = data["rl_constants"]
        assert rc["NUM_PICKERS"] == num_pickers
        assert rc["NUM_TRANSPORTERS"] == num_transporters
        assert rc["PICKER_ACTION_DIM"] == 1 + RL_CONSTANTS["MAX_ITEMS"] + num_transporters
        assert rc["TRANSPORTER_ACTION_DIM"] == 1 + RL_CONSTANTS["MAX_WZ"] + RL_CONSTANTS["MAX_ITEMS"]
