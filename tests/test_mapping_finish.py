"""
Tests for POST /mapping/finish (pull map + docks snapshot from the robot and
persist DockConfig + Docks + MapConfig in one transaction), plus regression
tests for the routes refactored onto the shared services
(POST /map/upload, POST /dock/create_dock).

Isolation strategy:
  - In-memory SQLite for all DB state.
  - fakeredis patched onto the mapping router with `robot_ips` seeded.
  - MAP_STORAGE redirected to tmp_path.
  - httpx.AsyncClient.post patched to return a zip built in the test.
"""

import io
import json
import math
import zipfile

import fakeredis
import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from schema.base import Base
from schema.docks import DockConfig, Dock
from schema.map import MapConfig
from definitions import DockType
from main import app
from services.database import get_db_session
from services.generate_dock_yaml import generate_dock_yaml
import routes.maps as maps_route
import routes.mapping as mapping_route
import services.map_storage as map_storage_service


HOSTNAME = "couliglig_bot_1"
ROBOT_IP = "10.0.0.42"
MAP_YAML_NAME = "map_1712345678.yaml"
MAP_IMAGE_NAME = "map_1712345678.pgm"

SNAPSHOT_DOCKS = {
    "docks": {
        "dock_0": {
            "pose": [1.5, -2.25, 1.2],
            "aruco_markers": {"pos": [1.5, -2.25, -1.94], "center": 31},
            "observations": 42,
            "position_std": 0.012,
        },
        "dock_1": {
            "pose": [-3.0, 4.0, -0.5],
            "aruco_markers": {"pos": [-3.0, 4.0, 2.64], "center": 7},
            "observations": 18,
            "position_std": 0.02,
        },
        "dock_2": {
            "pose": [0.0, 0.5, 3.0],
            "aruco_markers": {"pos": [0.0, 0.5, -0.14], "center": 12},
            "observations": 5,
            "position_std": 0.05,
        },
    }
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_redis():
    r = fakeredis.FakeRedis(decode_responses=True)
    r.set("robot_ips", json.dumps({
        HOSTNAME: {
            "hostname": HOSTNAME,
            "ip": ROBOT_IP,
            "namespace": HOSTNAME,
            "ros_domain_id": 5,
            "battery": None,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    }))
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
def map_storage(tmp_path, monkeypatch):
    storage = tmp_path / "maps"
    storage.mkdir()
    monkeypatch.setattr(map_storage_service, "MAP_STORAGE", storage)
    monkeypatch.setattr(maps_route, "MAP_STORAGE", storage)
    return storage


@pytest.fixture()
def client(fake_redis, db_session, map_storage, monkeypatch):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    monkeypatch.setattr(mapping_route, "redis_client", fake_redis)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture()
def robot(monkeypatch):
    """Patch httpx.AsyncClient.post; set `robot.response` / `robot.error`."""

    class FakeRobot:
        def __init__(self):
            self.calls = []
            self.response = _zip_response()
            self.error = None

    fake = FakeRobot()

    async def fake_post(self, url, *args, **kwargs):
        fake.calls.append(url)
        if fake.error is not None:
            raise fake.error
        return fake.response

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return fake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_zip(docks: dict | None = SNAPSHOT_DOCKS, extra: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(MAP_YAML_NAME, f"image: {MAP_IMAGE_NAME}\nresolution: 0.05\norigin: [0, 0, 0]\n")
        zf.writestr(MAP_IMAGE_NAME, b"P5\n2 2\n255\n\x00\x00\x00\x00")
        if docks is not None:
            zf.writestr("docks.yaml", yaml.safe_dump(docks))
        for name, content in (extra or {}).items():
            zf.writestr(name, content)
    return buf.getvalue()


def _zip_response(content: bytes | None = None, status_code: int = 200, docks_count: int = 3) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=_build_zip() if content is None else content,
        headers={
            "content-type": "application/zip",
            "X-Docks-Status": "ok",
            "X-Docks-Count": str(docks_count),
        },
    )


def _request(name="warehouse_a", docks=None, hostname=HOSTNAME):
    if docks is None:
        docks = [
            {"aruco_id": 31, "dock_id": "pickup_1", "dock_type": "pickup"},
            {"aruco_id": 7, "dock_id": "recv_1", "dock_type": "receiver"},
        ]
    return {"hostname": hostname, "name": name, "description": "from slam", "docks": docks}


def _assert_nothing_persisted(db_session, map_storage):
    assert db_session.query(DockConfig).count() == 0
    assert db_session.query(Dock).count() == 0
    assert db_session.query(MapConfig).count() == 0
    assert list(map_storage.iterdir()) == []


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# POST /mapping/finish
# ---------------------------------------------------------------------------

class TestFinishMapping:

    def test_success_creates_config_docks_and_map(self, client, db_session, map_storage, robot):
        resp = client.post("/mapping/finish", json=_request())
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert robot.calls == [f"http://{ROBOT_IP}:8000/slam/save-map"]
        assert body["docks_created"] == 2
        assert body["unreviewed"] == [12]

        cfg = db_session.query(DockConfig).filter(DockConfig.id == body["dock_config_id"]).one()
        assert cfg.name == "warehouse_a"
        assert cfg.description == "from slam"

        docks = {d.aruco_id: d for d in db_session.query(Dock).filter(Dock.config_id == cfg.id).all()}
        assert set(docks) == {31, 7}

        assert docks[31].dock_id == "pickup_1"
        assert docks[31].dock_type == DockType.PICKUP
        assert (docks[31].x, docks[31].y, docks[31].theta) == (1.5, -2.25, 1.2)

        assert docks[7].dock_id == "recv_1"
        assert docks[7].dock_type == DockType.RECEIVER
        assert (docks[7].x, docks[7].y, docks[7].theta) == (-3.0, 4.0, -0.5)

        map_cfg = db_session.query(MapConfig).filter(MapConfig.id == body["map_id"]).one()
        assert map_cfg.dock_config_id == cfg.id
        assert map_cfg.map_yaml_filename == MAP_YAML_NAME
        assert map_cfg.map_image_filename == MAP_IMAGE_NAME

        assert sorted(p.name for p in map_storage.iterdir()) == sorted([MAP_YAML_NAME, MAP_IMAGE_NAME])
        assert f"image: {MAP_IMAGE_NAME}" in (map_storage / MAP_YAML_NAME).read_text()
        assert not (map_storage / "docks.yaml").exists()

        # Round-trip: generated dock yaml keeps the pose and flips the marker yaw by pi
        generated = yaml.safe_load(generate_dock_yaml(list(docks.values()), "").read())
        for dock in docks.values():
            entry = generated["docks"][dock.dock_id]
            assert entry["pose"] == [dock.x, dock.y, dock.theta]
            assert entry["aruco_markers"]["center"] == dock.aruco_id
            assert entry["aruco_markers"]["pos"][:2] == [dock.x, dock.y]
            assert entry["aruco_markers"]["pos"][2] == pytest.approx(_wrap(dock.theta + math.pi), abs=0.01)

    def test_success_with_no_docks_and_missing_docks_yaml(self, client, db_session, map_storage, robot):
        robot.response = _zip_response(_build_zip(docks=None), docks_count=0)

        resp = client.post("/mapping/finish", json=_request(docks=[]))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["docks_created"] == 0
        assert body["unreviewed"] == []
        assert db_session.query(MapConfig).count() == 1
        assert db_session.query(Dock).count() == 0

    def test_duplicate_config_name_returns_409_without_calling_robot(self, client, db_session, map_storage, robot):
        db_session.add(DockConfig(name="warehouse_a"))
        db_session.commit()

        resp = client.post("/mapping/finish", json=_request())
        assert resp.status_code == 409
        assert robot.calls == []
        assert db_session.query(DockConfig).count() == 1
        assert list(map_storage.iterdir()) == []

    def test_aruco_missing_from_snapshot_returns_409(self, client, db_session, map_storage, robot):
        req = _request(docks=[
            {"aruco_id": 31, "dock_id": "pickup_1", "dock_type": "pickup"},
            {"aruco_id": 99, "dock_id": "ghost", "dock_type": "receiver"},
        ])
        resp = client.post("/mapping/finish", json=req)
        assert resp.status_code == 409
        assert resp.json()["detail"]["missing_aruco_ids"] == [99]
        _assert_nothing_persisted(db_session, map_storage)

    def test_robot_500_returns_502(self, client, db_session, map_storage, robot):
        robot.response = httpx.Response(500, json={"detail": "map_saver failed"})

        resp = client.post("/mapping/finish", json=_request())
        assert resp.status_code == 502
        assert "500" in resp.json()["detail"]
        _assert_nothing_persisted(db_session, map_storage)

    def test_robot_unreachable_returns_502(self, client, db_session, map_storage, robot):
        robot.error = httpx.ConnectError("connection refused")

        resp = client.post("/mapping/finish", json=_request())
        assert resp.status_code == 502
        _assert_nothing_persisted(db_session, map_storage)

    def test_robot_timeout_returns_502(self, client, db_session, map_storage, robot):
        robot.error = httpx.ReadTimeout("timed out")

        resp = client.post("/mapping/finish", json=_request())
        assert resp.status_code == 502
        _assert_nothing_persisted(db_session, map_storage)

    def test_malformed_zip_returns_502(self, client, db_session, map_storage, robot):
        robot.response = _zip_response(b"this is not a zip")

        resp = client.post("/mapping/finish", json=_request())
        assert resp.status_code == 502
        _assert_nothing_persisted(db_session, map_storage)

    def test_robot_offline_returns_404(self, client, db_session, map_storage, robot):
        resp = client.post("/mapping/finish", json=_request(hostname="couliglig_bot_9"))
        assert resp.status_code == 404
        assert robot.calls == []
        _assert_nothing_persisted(db_session, map_storage)

    def test_duplicate_dock_id_returns_422(self, client, db_session, map_storage, robot):
        req = _request(docks=[
            {"aruco_id": 31, "dock_id": "same", "dock_type": "pickup"},
            {"aruco_id": 7, "dock_id": "same", "dock_type": "receiver"},
        ])
        resp = client.post("/mapping/finish", json=req)
        assert resp.status_code == 422
        assert robot.calls == []
        _assert_nothing_persisted(db_session, map_storage)

    def test_duplicate_aruco_id_returns_422(self, client, db_session, map_storage, robot):
        req = _request(docks=[
            {"aruco_id": 31, "dock_id": "a", "dock_type": "pickup"},
            {"aruco_id": 31, "dock_id": "b", "dock_type": "receiver"},
        ])
        resp = client.post("/mapping/finish", json=req)
        assert resp.status_code == 422
        assert robot.calls == []

    def test_map_file_conflict_rolls_back_and_keeps_existing_files(self, client, db_session, map_storage, robot):
        # The image already exists on disk: the map yaml must not be left behind,
        # the existing image must not be deleted and nothing is committed.
        (map_storage / MAP_IMAGE_NAME).write_bytes(b"someone else's map")

        resp = client.post("/mapping/finish", json=_request())
        assert resp.status_code == 409
        assert db_session.query(DockConfig).count() == 0
        assert db_session.query(Dock).count() == 0
        assert db_session.query(MapConfig).count() == 0
        assert [p.name for p in map_storage.iterdir()] == [MAP_IMAGE_NAME]
        assert (map_storage / MAP_IMAGE_NAME).read_bytes() == b"someone else's map"

    def test_commit_failure_removes_written_files(self, client, db_session, map_storage, robot):
        def failing_commit():
            raise RuntimeError("disk full")

        db_session.commit = failing_commit
        try:
            with pytest.raises(RuntimeError):
                client.post("/mapping/finish", json=_request())
        finally:
            del db_session.commit

        assert db_session.query(DockConfig).count() == 0
        assert db_session.query(MapConfig).count() == 0
        assert list(map_storage.iterdir()) == []


# ---------------------------------------------------------------------------
# Refactored routes keep their behavior
# ---------------------------------------------------------------------------

class TestRefactoredRoutes:

    def test_upload_map_stores_files_and_row(self, client, db_session, map_storage):
        cfg = DockConfig(name="cfg")
        db_session.add(cfg)
        db_session.commit()

        resp = client.post(
            "/map/upload",
            data={"dock_config_id": cfg.id},
            files={
                "map_yaml": ("m.yaml", b"yaml", "text/yaml"),
                "map_image": ("m.pgm", b"img", "image/x-portable-graymap"),
            },
        )
        assert resp.status_code == 200, resp.text
        assert (map_storage / "m.yaml").read_bytes() == b"yaml"
        assert (map_storage / "m.pgm").read_bytes() == b"img"
        assert db_session.query(MapConfig).count() == 1

        # Second upload for the same dock config is rejected and writes nothing
        resp = client.post(
            "/map/upload",
            data={"dock_config_id": cfg.id},
            files={
                "map_yaml": ("n.yaml", b"yaml", "text/yaml"),
                "map_image": ("n.pgm", b"img", "image/x-portable-graymap"),
            },
        )
        assert resp.status_code == 409
        assert not (map_storage / "n.yaml").exists()

    def test_upload_map_unknown_config_returns_404(self, client, map_storage):
        resp = client.post(
            "/map/upload",
            data={"dock_config_id": 999},
            files={
                "map_yaml": ("m.yaml", b"yaml", "text/yaml"),
                "map_image": ("m.pgm", b"img", "image/x-portable-graymap"),
            },
        )
        assert resp.status_code == 404
        assert list(map_storage.iterdir()) == []

    def test_create_dock_and_conflicts(self, client, db_session):
        cfg = DockConfig(name="cfg")
        db_session.add(cfg)
        db_session.commit()

        dock = {"config_id": cfg.id, "dock_id": "d1", "dock_type": "pickup", "aruco_id": 3, "x": 1.0, "y": 2.0, "theta": 0.5}
        resp = client.post("/dock/create_dock", json=[dock])
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["dock_id"] == "d1"

        resp = client.post("/dock/create_dock", json=[dock])
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Dock ID already exists in the specified config"

        resp = client.post("/dock/create_dock", json=[{**dock, "dock_id": "d2"}])
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Aruco ID already exists in the specified config"

        resp = client.post("/dock/create_dock", json=[{**dock, "config_id": 999}])
        assert resp.status_code == 404
