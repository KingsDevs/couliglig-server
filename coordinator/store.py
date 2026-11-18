"""The coordinator's persistent state (its own SQLite file, separate from the
FastAPI config database).

Write-through: every state change commits before the triggering event is
acked, so a restarted coordinator resumes from the last acked state.

    meta     key/value: epoch (incremented on every start), active config
    items    one row per item (JSON)
    tasks    one row per task (JSON)
    robots   persistent per-robot fields (role, zone, carried) (JSON)
    events   every robot event applied, keyed by event_id (deduplication + audit)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS meta   (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items  (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks  (id TEXT PRIMARY KEY, robot_id TEXT NOT NULL, status TEXT NOT NULL,
                                   data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS robots (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, robot_id TEXT NOT NULL,
                                   received_at REAL NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_DDL)
        self._lock = threading.Lock()
        version = self.get_meta("schema_version")
        if version is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        elif int(version) != SCHEMA_VERSION:
            raise RuntimeError(f"{self.path}: schema version {version}, expected {SCHEMA_VERSION}")

    def close(self):
        self._db.close()

    # ---------------- meta ----------------

    def get_meta(self, key: str, default=None):
        row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value) -> None:
        self._db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (key, json.dumps(value)))

    def next_epoch(self) -> int:
        with self._lock:
            epoch = int(self.get_meta("epoch", 0)) + 1
            self.set_meta("epoch", epoch)
            return epoch

    # ---------------- transactions ----------------

    def commit(self, *, items=(), tasks=(), robots=(), deleted_items=(), event=None, meta=None) -> None:
        """Write the given rows (dicts with an "id") in one transaction."""
        with self._lock:
            cur = self._db.cursor()
            cur.execute("BEGIN")
            try:
                for it in items:
                    cur.execute("INSERT INTO items(id, data) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                                (it["id"], json.dumps(it)))
                for item_id in deleted_items:
                    cur.execute("DELETE FROM items WHERE id=?", (item_id,))
                for t in tasks:
                    cur.execute("INSERT INTO tasks(id, robot_id, status, data) VALUES(?, ?, ?, ?) "
                                "ON CONFLICT(id) DO UPDATE SET status=excluded.status, data=excluded.data",
                                (t["id"], t["robot_id"], t["status"], json.dumps(t)))
                for r in robots:
                    cur.execute("INSERT INTO robots(id, data) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                                (r["id"], json.dumps(r)))
                if event is not None:
                    cur.execute("INSERT INTO events(event_id, robot_id, received_at, data) VALUES(?, ?, ?, ?)",
                                (event["event_id"], event["robot_id"], time.time(), json.dumps(event)))
                for key, value in (meta or {}).items():
                    cur.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                (key, json.dumps(value)))
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def seen_event(self, event_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM events WHERE event_id=?", (event_id,)).fetchone() is not None

    # ---------------- loading ----------------

    def load_items(self) -> list[dict]:
        return [json.loads(r[0]) for r in self._db.execute("SELECT data FROM items")]

    def load_tasks(self, active_only: bool = False) -> list[dict]:
        if active_only:
            rows = self._db.execute("SELECT data FROM tasks WHERE status IN ('SENT', 'ACCEPTED', 'STARTED')")
        else:
            rows = self._db.execute("SELECT data FROM tasks")
        return [json.loads(r[0]) for r in rows]

    def load_robots(self) -> list[dict]:
        return [json.loads(r[0]) for r in self._db.execute("SELECT data FROM robots")]

    def events_for(self, robot_id: str) -> list[dict]:
        return [json.loads(r[0]) for r in
                self._db.execute("SELECT data FROM events WHERE robot_id=? ORDER BY received_at", (robot_id,))]
