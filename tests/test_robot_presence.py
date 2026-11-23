"""robot_ips presence: entries no live /robots/ws socket refreshes are pruned.

The field case: the server restarted while a robot was connected (its socket's cleanup
never ran), Redis kept the entry, the robot was then switched off, and the dashboard
listed it as online for days.
"""
import asyncio
import json
from datetime import datetime, timedelta

import fakeredis
import pytest
from fastapi.testclient import TestClient

import routes.robots as robots_route
from main import app


@pytest.fixture()
def fake_redis(monkeypatch):
    r = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(robots_route, "redis_client", r)
    yield r
    r.flushall()


@pytest.fixture()
def shortened(monkeypatch):
    calls = []
    monkeypatch.setattr(robots_route, "shorten_robot_locks", lambda r, robot_id: calls.append(robot_id) or [])
    return calls


def entry(hostname, age_s, domain=5, now=None):
    now = now or datetime.now()
    return {"hostname": hostname, "ip": "192.168.20.105", "namespace": "couliglig",
            "ros_domain_id": domain, "battery": None, "session": "s1",
            "timestamp": (now - timedelta(seconds=age_s)).isoformat() + "Z"}


def robot_ips(r):
    return json.loads(r.get("robot_ips") or "{}")


def test_stale_entry_is_removed_and_fresh_kept(fake_redis, shortened):
    now = datetime.now()
    fake_redis.set("robot_ips", json.dumps({
        "couliglige": entry("couliglige", 3 * 24 * 3600, domain=5, now=now),   # 3 days, as in the field
        "couligligd": entry("couligligd", 4, domain=4, now=now),
    }))
    removed = robots_route.prune_stale_robots(now=now)
    assert [e["hostname"] for e in removed] == ["couliglige"]
    assert list(robot_ips(fake_redis)) == ["couligligd"]
    assert shortened == ["couliglig_bot_5"]                 # its dock locks expire like a disconnect


def test_threshold(fake_redis, shortened):
    now = datetime.now()
    limit = robots_route.ROBOT_STALE_S
    fake_redis.set("robot_ips", json.dumps({
        "a": entry("a", limit - 1, now=now),
        "b": entry("b", limit + 1, now=now),
    }))
    robots_route.prune_stale_robots(now=now)
    assert list(robot_ips(fake_redis)) == ["a"]


def test_entries_without_a_timestamp_are_removed(fake_redis, shortened):
    fake_redis.set("robot_ips", json.dumps({
        "legacy": "192.168.20.9",                                   # the old string form
        "no_ts": {"hostname": "no_ts", "ip": "1.2.3.4"},
        "bad_ts": {"hostname": "bad_ts", "ip": "1.2.3.4", "timestamp": "yesterday"},
        "ok": entry("ok", 1),
    }))
    removed = robots_route.prune_stale_robots()
    assert sorted(e["hostname"] for e in removed) == ["bad_ts", "legacy", "no_ts"]
    assert list(robot_ips(fake_redis)) == ["ok"]
    assert shortened == []                                          # no domain id: no locks to name


def test_nothing_stale_writes_nothing(fake_redis, shortened):
    fake_redis.set("robot_ips", json.dumps({"ok": entry("ok", 1)}))
    before = fake_redis.get("robot_ips")
    assert robots_route.prune_stale_robots() == []
    assert fake_redis.get("robot_ips") == before


def test_empty_registry(fake_redis, shortened):
    assert robots_route.prune_stale_robots() == []


def _hello():
    return {"hostname": "couliglig-1", "ip": "10.0.0.5", "namespace": "couliglig",
            "ros_domain_id": 1, "battery": 80.0}


def test_live_socket_survives_and_comes_back(fake_redis, shortened):
    """A connected robot is never pruned while it heartbeats; if its entry is pruned
    anyway (clock jump), its next heartbeat re-adds it."""
    with TestClient(app) as client, client.websocket_connect("/robots/ws") as ws:
        ws.send_json(_hello())
        ws.send_json({"type": "heartbeat"})
        ws.send_json({"type": "heartbeat"})          # round-trip: both processed
        assert robots_route.prune_stale_robots() == []
        assert "couliglig-1" in robot_ips(fake_redis)

        far_future = datetime.now() + timedelta(seconds=robots_route.ROBOT_STALE_S + 5)
        assert [e["hostname"] for e in robots_route.prune_stale_robots(now=far_future)] == ["couliglig-1"]
        ws.send_json({"type": "heartbeat"})
        ws.send_json({"type": "heartbeat"})
        assert "couliglig-1" in robot_ips(fake_redis)


def test_startup_prunes_left_over_entries(fake_redis, shortened):
    """The server starts with a stale entry from before its restart: it is gone at once."""
    fake_redis.set("robot_ips", json.dumps({"couliglige": entry("couliglige", 3600)}))
    with TestClient(app) as client:
        for _ in range(50):                          # the background task runs first thing
            if not robot_ips(fake_redis):
                break
            client.get("/robots/online")
        assert robot_ips(fake_redis) == {}
        assert client.get("/robots/online").json()["data"] == []


def test_pruner_survives_errors(monkeypatch):
    calls = []

    def boom():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("redis down")
        raise asyncio.CancelledError

    monkeypatch.setattr(robots_route, "prune_stale_robots", boom)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(robots_route.prune_stale_robots_forever(interval=0))
    assert len(calls) == 2                           # kept going after the first failure
