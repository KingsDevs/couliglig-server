"""
Tests for PUT /map/update (partial update of a single MapConfig).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from schema.base import Base
from schema.docks import DockConfig
from schema.map import MapConfig
from main import app
from services.database import get_db_session
import routes.maps as maps_route


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
    monkeypatch.setattr(maps_route, "MAP_STORAGE", storage)
    return storage


@pytest.fixture()
def client(db_session, map_storage):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _seed_map(db_session, storage: Path, yaml_name="orig.yaml", image_name="orig.png") -> tuple[int, int]:
    dock_cfg = DockConfig(name="cfg", description="d")
    db_session.add(dock_cfg)
    db_session.flush()

    (storage / yaml_name).write_text("orig-yaml")
    (storage / image_name).write_bytes(b"orig-image")

    map_cfg = MapConfig(
        dock_config_id=dock_cfg.id,
        map_yaml_filename=yaml_name,
        map_image_filename=image_name,
    )
    db_session.add(map_cfg)
    db_session.commit()
    db_session.refresh(map_cfg)
    return map_cfg.id, dock_cfg.id


class TestUpdateMap:

    def test_returns_404_for_unknown_map(self, client):
        resp = client.put("/map/update", data={"map_db_id": 9999})
        assert resp.status_code == 404

    def test_noop_update_keeps_files(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)

        resp = client.put("/map/update", data={"map_db_id": map_id})
        assert resp.status_code == 200
        assert (map_storage / "orig.yaml").exists()
        assert (map_storage / "orig.png").exists()

    def test_updates_yaml_file_and_deletes_old(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)

        resp = client.put(
            "/map/update",
            data={"map_db_id": map_id},
            files={"map_yaml": ("new.yaml", b"new-yaml-content", "text/yaml")},
        )
        assert resp.status_code == 200
        assert "new.yaml" in resp.json()["map_yaml_path"]

        assert not (map_storage / "orig.yaml").exists()
        assert (map_storage / "new.yaml").read_text() == "new-yaml-content"
        assert (map_storage / "orig.png").exists()  # image untouched

        row = db_session.query(MapConfig).filter(MapConfig.id == map_id).first()
        assert row.map_yaml_filename == "new.yaml"
        assert row.map_image_filename == "orig.png"

    def test_updates_image_file_and_deletes_old(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)

        resp = client.put(
            "/map/update",
            data={"map_db_id": map_id},
            files={"map_image": ("new.png", b"new-image-bytes", "image/png")},
        )
        assert resp.status_code == 200
        assert not (map_storage / "orig.png").exists()
        assert (map_storage / "new.png").read_bytes() == b"new-image-bytes"
        assert (map_storage / "orig.yaml").exists()

    def test_updates_both_files(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)

        resp = client.put(
            "/map/update",
            data={"map_db_id": map_id},
            files={
                "map_yaml": ("b.yaml", b"y", "text/yaml"),
                "map_image": ("b.png", b"i", "image/png"),
            },
        )
        assert resp.status_code == 200
        assert not (map_storage / "orig.yaml").exists()
        assert not (map_storage / "orig.png").exists()
        assert (map_storage / "b.yaml").exists()
        assert (map_storage / "b.png").exists()

    def test_updates_dock_config_id(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)
        other_cfg = DockConfig(name="other", description="x")
        db_session.add(other_cfg)
        db_session.commit()

        resp = client.put(
            "/map/update",
            data={"map_db_id": map_id, "dock_config_id": other_cfg.id},
        )
        assert resp.status_code == 200
        assert resp.json()["dock_config_id"] == other_cfg.id

    def test_dock_config_conflict(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)

        # seed a second MapConfig that already owns another dock config
        other_cfg = DockConfig(name="other", description="x")
        db_session.add(other_cfg)
        db_session.flush()
        (map_storage / "o.yaml").write_text("o")
        (map_storage / "o.png").write_bytes(b"o")
        db_session.add(MapConfig(
            dock_config_id=other_cfg.id,
            map_yaml_filename="o.yaml",
            map_image_filename="o.png",
        ))
        db_session.commit()

        resp = client.put(
            "/map/update",
            data={"map_db_id": map_id, "dock_config_id": other_cfg.id},
        )
        assert resp.status_code == 409

    def test_unknown_dock_config_returns_404(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)
        resp = client.put(
            "/map/update",
            data={"map_db_id": map_id, "dock_config_id": 99999},
        )
        assert resp.status_code == 404

    def test_yaml_filename_conflict(self, client, db_session, map_storage):
        map_id, _ = _seed_map(db_session, map_storage)

        # second map owns "taken.yaml"
        other_cfg = DockConfig(name="o", description="x")
        db_session.add(other_cfg)
        db_session.flush()
        (map_storage / "taken.yaml").write_text("t")
        (map_storage / "taken.png").write_bytes(b"t")
        db_session.add(MapConfig(
            dock_config_id=other_cfg.id,
            map_yaml_filename="taken.yaml",
            map_image_filename="taken.png",
        ))
        db_session.commit()

        resp = client.put(
            "/map/update",
            data={"map_db_id": map_id},
            files={"map_yaml": ("taken.yaml", b"x", "text/yaml")},
        )
        assert resp.status_code == 409
        # original file must still be intact
        assert (map_storage / "orig.yaml").exists()
