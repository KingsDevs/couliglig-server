"""
Tests for GET /rl/builder-inputs/dummy

No Redis, no DB, no external HTTP calls needed — the endpoint is fully self-contained.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from services.redis_dock_runtime import RL_CONSTANTS

client = TestClient(app)

ROBOT_ID = "couliglig_bot_99"
BASE_URL = "/rl/builder-inputs/dummy"


def get_dummy(params: dict | None = None) -> dict:
    params = {"robot_id": ROBOT_ID, **(params or {})}
    resp = client.get(BASE_URL, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class TestStructure:
    def test_top_level_keys_present(self):
        data = get_dummy()
        expected_keys = {
            "dock_states", "dock_positions", "item_weights", "item_receiver_docks",
            "robot_positions", "picker_has_item", "transporter_loads",
            "transporter_carried", "transporter_in_wz", "waiting_zones",
            "action_map", "rl_constants",
        }
        assert expected_keys.issubset(data.keys())

    def test_action_map_keys_present(self):
        data = get_dummy()
        am = data["action_map"]
        assert set(am.keys()) == {
            "item_slots", "wz_slots", "receiver_slots", "picker_slots", "transporter_slots"
        }

    def test_rl_constants_keys_present(self):
        data = get_dummy()
        rc = data["rl_constants"]
        for key in ("MAX_ITEMS", "MAX_WZ", "MAX_PICKERS", "MAX_TRANSPORTERS",
                    "PICKER_ACTION_DIM", "TRANSPORTER_ACTION_DIM",
                    "NUM_PICKERS", "NUM_TRANSPORTERS"):
            assert key in rc


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

class TestCounts:
    def test_default_dock_counts(self):
        data = get_dummy()
        types = [d["dock_type"] for d in data["dock_states"]]
        assert types.count("pickup") == 5
        assert types.count("waiting_zone") == 3
        assert types.count("receiver") == 3

    def test_custom_counts(self):
        data = get_dummy({"num_items": 3, "num_wz": 2, "num_receivers": 2,
                          "num_pickers": 2, "num_transporters": 1})
        types = [d["dock_type"] for d in data["dock_states"]]
        assert types.count("pickup") == 3
        assert types.count("waiting_zone") == 2
        assert types.count("receiver") == 2
        assert len(data["picker_has_item"]) == 2
        assert len(data["transporter_loads"]) == 1

    def test_action_map_slots_padded_to_max(self):
        data = get_dummy({"num_items": 3, "num_wz": 2,
                          "num_pickers": 2, "num_transporters": 1})
        am = data["action_map"]
        assert len(am["item_slots"]) == RL_CONSTANTS["MAX_ITEMS"]
        assert len(am["wz_slots"]) == RL_CONSTANTS["MAX_WZ"]
        assert len(am["picker_slots"]) == RL_CONSTANTS["MAX_PICKERS"]
        assert len(am["transporter_slots"]) == RL_CONSTANTS["MAX_TRANSPORTERS"]

    def test_counts_capped_at_max(self):
        data = get_dummy({
            "num_items": 999, "num_wz": 999,
            "num_pickers": 999, "num_transporters": 999,
        })
        types = [d["dock_type"] for d in data["dock_states"]]
        assert types.count("pickup") <= RL_CONSTANTS["MAX_ITEMS"]
        assert types.count("waiting_zone") <= RL_CONSTANTS["MAX_WZ"]
        assert len(data["picker_has_item"]) <= RL_CONSTANTS["MAX_PICKERS"]
        assert len(data["transporter_loads"]) <= RL_CONSTANTS["MAX_TRANSPORTERS"]


# ---------------------------------------------------------------------------
# robot_id always present
# ---------------------------------------------------------------------------

class TestRobotIdPresent:
    def test_robot_id_in_robot_positions(self):
        data = get_dummy()
        assert ROBOT_ID in data["robot_positions"]

    def test_robot_id_in_picker_has_item(self):
        data = get_dummy()
        assert ROBOT_ID in data["picker_has_item"]

    def test_robot_id_in_picker_slots(self):
        data = get_dummy()
        namespaces = [s["namespace"] for s in data["action_map"]["picker_slots"] if s]
        assert ROBOT_ID in namespaces

    def test_robot_id_is_first_picker_slot(self):
        data = get_dummy()
        first = data["action_map"]["picker_slots"][0]
        assert first is not None
        assert first["namespace"] == ROBOT_ID

    def test_robot_id_not_in_transporter_slots(self):
        """robot_id goes into pickers, never transporters."""
        data = get_dummy()
        namespaces = [s["namespace"] for s in data["action_map"]["transporter_slots"] if s]
        assert ROBOT_ID not in namespaces

    def test_dock_robot_ids_are_known_robots(self):
        """Any robot_id on a dock must be from the generated robot pool."""
        data = get_dummy()
        known = set(data["robot_positions"].keys())
        for dock in data["dock_states"]:
            rid = dock.get("robot_id", "")
            if rid:
                assert rid in known, f"Unknown robot_id '{rid}' on dock {dock['dock_id']}"


# ---------------------------------------------------------------------------
# Scenario flags
# ---------------------------------------------------------------------------

class TestAllItemsAvailable:
    def test_all_pickup_docks_available(self):
        data = get_dummy({"all_items_available": "true"})
        for dock in data["dock_states"]:
            if dock["dock_type"] == "pickup":
                assert dock["status"] == "available"
                assert dock["robot_id"] == ""
                assert dock["item_id"] != ""

    def test_action_map_all_available_for_pickup(self):
        data = get_dummy({"all_items_available": "true"})
        real_slots = [s for s in data["action_map"]["item_slots"] if s is not None]
        assert all(s["available_for_pickup"] for s in real_slots)


class TestNoDockedRobots:
    def test_no_dock_has_robot(self):
        data = get_dummy({"no_docked_robots": "true"})
        for dock in data["dock_states"]:
            assert dock["robot_id"] == ""
            assert dock["status"] == "available"


class TestAllWzAvailable:
    def test_all_wz_available(self):
        data = get_dummy({"all_wz_available": "true"})
        for dock in data["dock_states"]:
            if dock["dock_type"] == "waiting_zone":
                assert dock["status"] == "available"
                assert dock["robot_id"] == ""

    def test_waiting_zones_list_reflects_flag(self):
        data = get_dummy({"all_wz_available": "true"})
        for wz in data["waiting_zones"]:
            assert wz["status"] == "available"
            assert wz["robot_id"] == ""


class TestAllReceiversAvailable:
    def test_all_receivers_available(self):
        data = get_dummy({"all_receivers_available": "true"})
        for dock in data["dock_states"]:
            if dock["dock_type"] == "receiver":
                assert dock["status"] == "available"
                assert dock["robot_id"] == ""


class TestPickerItemFlags:
    def test_pickers_have_item(self):
        data = get_dummy({"pickers_have_item": "true"})
        assert all(v is True for v in data["picker_has_item"].values())

    def test_pickers_no_item(self):
        data = get_dummy({"pickers_no_item": "true"})
        assert all(v is False for v in data["picker_has_item"].values())


class TestTransporterFlags:
    def test_transporters_empty(self):
        data = get_dummy({"transporters_empty": "true"})
        for ns, (cap, _max) in data["transporter_loads"].items():
            assert cap == 0.0
        for ns, carried in data["transporter_carried"].items():
            assert carried == []

    def test_transporters_in_wz(self):
        data = get_dummy({"transporters_in_wz": "true"})
        assert all(v is True for v in data["transporter_in_wz"].values())

    def test_transporters_not_in_wz(self):
        data = get_dummy({"transporters_not_in_wz": "true"})
        assert all(v is False for v in data["transporter_in_wz"].values())


# ---------------------------------------------------------------------------
# Stacked flags
# ---------------------------------------------------------------------------

class TestStackedFlags:
    def test_clean_start_scenario(self):
        """all_items_available + no_docked_robots + pickers_no_item + transporters_empty"""
        data = get_dummy({
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
    def test_action_dims(self):
        num_pickers = 3
        num_transporters = 2
        data = get_dummy({"num_pickers": num_pickers, "num_transporters": num_transporters})
        rc = data["rl_constants"]
        assert rc["NUM_PICKERS"] == num_pickers
        assert rc["NUM_TRANSPORTERS"] == num_transporters
        assert rc["PICKER_ACTION_DIM"] == 1 + RL_CONSTANTS["MAX_ITEMS"] + num_transporters
        assert rc["TRANSPORTER_ACTION_DIM"] == 1 + RL_CONSTANTS["MAX_WZ"] + RL_CONSTANTS["MAX_ITEMS"]
