"""
Tests for bulk update/delete endpoints added to:
  - PUT  /robots/infos         (bulk update RobotInfo)
  - DELETE /robots/infos/bulk  (bulk delete RobotInfo)
  - PUT  /dock/update_dock     (bulk update Dock)
  - DELETE /dock/delete_dock   (bulk delete Dock)

Isolation strategy:
  - In-memory SQLite for all DB state.
  - fakeredis patched onto the docks router (not needed for robot-info routes).
"""

import pytest
import fakeredis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from schema.base import Base
from schema.docks import DockConfig, Dock
from schema.map import MapConfig
from schema.robot_info import RobotInfo
from definitions import DockType
from main import app
from services.database import get_db_session
import routes.docks as docks_route


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed_map_config(db_session) -> int:
    """Creates a DockConfig + MapConfig and returns the MapConfig id."""
    dock_cfg = DockConfig(name="map_dock_cfg", description="for map")
    db_session.add(dock_cfg)
    db_session.flush()

    map_cfg = MapConfig(
        dock_config_id=dock_cfg.id,
        map_yaml_filename="test.yaml",
        map_image_filename="test.png",
    )
    db_session.add(map_cfg)
    db_session.commit()
    db_session.refresh(map_cfg)
    return map_cfg.id


def _seed_robot_infos(db_session, map_id: int, count: int = 2) -> list[int]:
    """Creates `count` RobotInfo rows and returns their primary key ids."""
    ids = []
    for i in range(count):
        info = RobotInfo(
            robot_name=f"bot_{i}",
            initial_x=float(i),
            initial_y=float(i),
            initial_theta=0.0,
            map_id=map_id,
        )
        db_session.add(info)
        db_session.flush()
        ids.append(info.id)
    db_session.commit()
    return ids


def _seed_dock_config_and_docks(db_session, count: int = 2) -> tuple[int, list[int]]:
    """Creates a DockConfig with `count` docks; returns (config_id, [dock.id, ...])."""
    cfg = DockConfig(name="bulk_test_cfg", description="bulk tests")
    db_session.add(cfg)
    db_session.flush()

    dock_ids = []
    for i in range(count):
        dock = Dock(
            config_id=cfg.id,
            dock_id=f"dock_{i}",
            dock_type=DockType.PICKUP,
            aruco_id=i + 1,
            x=float(i),
            y=float(i),
            theta=0.0,
        )
        db_session.add(dock)
        db_session.flush()
        dock_ids.append(dock.id)
    db_session.commit()
    return cfg.id, dock_ids


# ===========================================================================
# PUT /robots/infos  –  bulk update
# ===========================================================================

class TestBulkUpdateRobotInfos:

    def test_updates_single_record(self, client, db_session):
        map_id = _seed_map_config(db_session)
        (rid,) = _seed_robot_infos(db_session, map_id, count=1)

        resp = client.put("/robots/infos", json=[{
            "id": rid, "robot_name": "updated_bot",
            "initial_x": 9.0, "initial_y": 8.0, "initial_theta": 1.5, "map_id": map_id,
        }])

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["robot_name"] == "updated_bot"
        assert data[0]["initial_x"] == 9.0

    def test_updates_multiple_records(self, client, db_session):
        map_id = _seed_map_config(db_session)
        rid_a, rid_b = _seed_robot_infos(db_session, map_id, count=2)

        resp = client.put("/robots/infos", json=[
            {"id": rid_a, "robot_name": "alpha", "initial_x": 1.0, "initial_y": 2.0, "initial_theta": 0.0, "map_id": map_id},
            {"id": rid_b, "robot_name": "beta",  "initial_x": 3.0, "initial_y": 4.0, "initial_theta": 0.1, "map_id": map_id},
        ])

        assert resp.status_code == 200
        names = {r["robot_name"] for r in resp.json()}
        assert names == {"alpha", "beta"}

    def test_returns_404_for_unknown_id(self, client, db_session):
        map_id = _seed_map_config(db_session)
        resp = client.put("/robots/infos", json=[{
            "id": 99999, "robot_name": "ghost",
            "initial_x": 0.0, "initial_y": 0.0, "initial_theta": 0.0, "map_id": map_id,
        }])
        assert resp.status_code == 404

    def test_returns_404_for_unknown_map_id(self, client, db_session):
        map_id = _seed_map_config(db_session)
        (rid,) = _seed_robot_infos(db_session, map_id, count=1)

        resp = client.put("/robots/infos", json=[{
            "id": rid, "robot_name": "bot",
            "initial_x": 0.0, "initial_y": 0.0, "initial_theta": 0.0, "map_id": 99999,
        }])
        assert resp.status_code == 404

    def test_returns_409_on_name_map_conflict_with_other_record(self, client, db_session):
        map_id = _seed_map_config(db_session)
        rid_a, rid_b = _seed_robot_infos(db_session, map_id, count=2)
        # bot_0 and bot_1 already exist; try renaming bot_1 to bot_0 (conflict)
        resp = client.put("/robots/infos", json=[{
            "id": rid_b, "robot_name": "bot_0",
            "initial_x": 0.0, "initial_y": 0.0, "initial_theta": 0.0, "map_id": map_id,
        }])
        assert resp.status_code == 409

    def test_updating_own_name_is_allowed(self, client, db_session):
        """A record can keep its own name (no self-conflict)."""
        map_id = _seed_map_config(db_session)
        (rid,) = _seed_robot_infos(db_session, map_id, count=1)

        resp = client.put("/robots/infos", json=[{
            "id": rid, "robot_name": "bot_0",
            "initial_x": 5.0, "initial_y": 5.0, "initial_theta": 0.5, "map_id": map_id,
        }])
        assert resp.status_code == 200
        assert resp.json()[0]["initial_x"] == 5.0


# ===========================================================================
# DELETE /robots/infos/bulk  –  bulk delete
# ===========================================================================

class TestBulkDeleteRobotInfos:

    def test_deletes_single_record(self, client, db_session):
        map_id = _seed_map_config(db_session)
        (rid,) = _seed_robot_infos(db_session, map_id, count=1)

        resp = client.request("DELETE", "/robots/infos/bulk", json={"ids": [rid]})

        assert resp.status_code == 200
        assert resp.json() == {"deleted": [rid]}
        assert db_session.query(RobotInfo).filter(RobotInfo.id == rid).first() is None

    def test_deletes_multiple_records(self, client, db_session):
        map_id = _seed_map_config(db_session)
        rid_a, rid_b = _seed_robot_infos(db_session, map_id, count=2)

        resp = client.request("DELETE", "/robots/infos/bulk", json={"ids": [rid_a, rid_b]})

        assert resp.status_code == 200
        assert set(resp.json()["deleted"]) == {rid_a, rid_b}
        assert db_session.query(RobotInfo).count() == 0

    def test_returns_404_for_unknown_id(self, client, db_session):
        map_id = _seed_map_config(db_session)
        (rid,) = _seed_robot_infos(db_session, map_id, count=1)

        resp = client.request("DELETE", "/robots/infos/bulk", json={"ids": [rid, 99999]})
        assert resp.status_code == 404

    def test_all_or_nothing_on_unknown_id(self, client, db_session):
        """When one id is missing, no records should be deleted."""
        map_id = _seed_map_config(db_session)
        (rid,) = _seed_robot_infos(db_session, map_id, count=1)

        client.request("DELETE", "/robots/infos/bulk", json={"ids": [rid, 99999]})

        # The valid record must still exist
        assert db_session.query(RobotInfo).filter(RobotInfo.id == rid).first() is not None


# ===========================================================================
# PUT /dock/update_dock  –  bulk update
# ===========================================================================

class TestBulkUpdateDock:

    def test_updates_single_dock(self, client, db_session):
        _, (dock_id,) = _seed_dock_config_and_docks(db_session, count=1)

        resp = client.put("/dock/update_dock", json=[{
            "id": dock_id, "x": 99.0, "y": 88.0,
        }])

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["x"] == 99.0
        assert data[0]["y"] == 88.0

    def test_updates_multiple_docks(self, client, db_session):
        _, (did_a, did_b) = _seed_dock_config_and_docks(db_session, count=2)

        resp = client.put("/dock/update_dock", json=[
            {"id": did_a, "x": 1.1},
            {"id": did_b, "x": 2.2},
        ])

        assert resp.status_code == 200
        xs = {r["x"] for r in resp.json()}
        assert xs == {1.1, 2.2}

    def test_returns_404_for_unknown_id(self, client, db_session):
        resp = client.put("/dock/update_dock", json=[{"id": 99999, "x": 0.0}])
        assert resp.status_code == 404

    def test_returns_409_on_dock_id_conflict(self, client, db_session):
        _, (did_a, did_b) = _seed_dock_config_and_docks(db_session, count=2)
        # Try renaming dock_b's dock_id to dock_0 (already used by dock_a)
        resp = client.put("/dock/update_dock", json=[{"id": did_b, "dock_id": "dock_0"}])
        assert resp.status_code == 409

    def test_returns_409_on_aruco_id_conflict(self, client, db_session):
        _, (did_a, did_b) = _seed_dock_config_and_docks(db_session, count=2)
        # aruco_id=1 belongs to dock_a; try assigning it to dock_b
        resp = client.put("/dock/update_dock", json=[{"id": did_b, "aruco_id": 1}])
        assert resp.status_code == 409

    def test_updating_own_dock_id_is_allowed(self, client, db_session):
        """A dock can keep its own dock_id without triggering a 409."""
        _, (dock_id,) = _seed_dock_config_and_docks(db_session, count=1)

        resp = client.put("/dock/update_dock", json=[{
            "id": dock_id, "dock_id": "dock_0", "x": 5.0,
        }])
        assert resp.status_code == 200
        assert resp.json()[0]["x"] == 5.0

    def test_partial_fields_preserve_existing_values(self, client, db_session):
        """Omitted optional fields should not overwrite existing values."""
        _, (dock_id,) = _seed_dock_config_and_docks(db_session, count=1)

        resp = client.put("/dock/update_dock", json=[{"id": dock_id, "theta": 3.14}])

        assert resp.status_code == 200
        d = resp.json()[0]
        assert d["theta"] == 3.14
        assert d["dock_id"] == "dock_0"   # unchanged


# ===========================================================================
# DELETE /dock/delete_dock  –  bulk delete
# ===========================================================================

class TestBulkDeleteDock:

    def test_deletes_single_dock(self, client, db_session):
        _, (dock_id,) = _seed_dock_config_and_docks(db_session, count=1)

        resp = client.request("DELETE", "/dock/delete_dock", json={"ids": [dock_id]})

        assert resp.status_code == 200
        assert resp.json() == {"deleted": [dock_id]}
        assert db_session.query(Dock).filter(Dock.id == dock_id).first() is None

    def test_deletes_multiple_docks(self, client, db_session):
        _, (did_a, did_b) = _seed_dock_config_and_docks(db_session, count=2)

        resp = client.request("DELETE", "/dock/delete_dock", json={"ids": [did_a, did_b]})

        assert resp.status_code == 200
        assert set(resp.json()["deleted"]) == {did_a, did_b}
        assert db_session.query(Dock).count() == 0

    def test_returns_404_for_unknown_id(self, client, db_session):
        resp = client.request("DELETE", "/dock/delete_dock", json={"ids": [99999]})
        assert resp.status_code == 404

    def test_all_or_nothing_on_unknown_id(self, client, db_session):
        """When one id is missing, no docks should be deleted."""
        _, (dock_id,) = _seed_dock_config_and_docks(db_session, count=1)

        client.request("DELETE", "/dock/delete_dock", json={"ids": [dock_id, 99999]})

        assert db_session.query(Dock).filter(Dock.id == dock_id).first() is not None
