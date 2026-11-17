# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
pip install fakeredis  # required for tests, not in requirements.txt
```

**Run the server:**
```bash
uvicorn main:app --reload
```

**Run tests:**
```bash
pytest tests -v
```

**Run a single test:**
```bash
pytest tests/test_dock_runtime.py::test_function_name -v
```

**Run with Docker:**
```bash
docker-compose up
```

## Architecture

This is a FastAPI server that coordinates a fleet of warehouse robots. It manages dock lifecycle state and generates RL observations for agent training.

### Two-layer state model

- **SQLite** (via SQLAlchemy): Persistent configuration — `DockConfig`, `Dock`, `MapConfig`, `RobotInfo` schemas in `schema/`.
- **Redis** (via `services/redis_dock_runtime.py`): Runtime state — active dock config, per-dock status/locks, robot positions/states, item assignments.

Activating a dock config copies it from SQLite into Redis. All runtime operations (reserve, occupy, release) operate on Redis only.

### Key modules

- `main.py` — FastAPI app, mounts all routers
- `routes/` — Thin HTTP handlers; business logic lives in services
- `services/redis_dock_runtime.py` — Core dock lifecycle logic (~750 lines); all Redis key patterns are defined here
- `services/robot_watcher.py` — Background daemon; health-checks each robot every 5s via `GET /status/health`, removes offline robots from Redis
- `definitions/dock_types.py` — `DockType` and `DockStatus` enums used everywhere

### Dock lifecycle

```
add-item → reserve (atomic Redis SET NX lock) → occupy → release
```

Docks have types: `pickup`, `receiver`, `waiting_zone`, `hand_off`. Items are added to pickup docks and carry a `receiver_dock_id` for delivery routing.

Lock lifetimes (the `lock:dock:*` key is the source of truth; the `dock:*` hash mirrors it):
- `reserve` sets a TTL (`DOCK_LOCK_TTL_S`, default 300 s). The owner extends it with `POST /dock/renew` (409 = reservation lost); `occupy` also refreshes it.
- Readers call `reconcile_expired_locks()` first, which marks docks whose lock has expired as available again.
- An occupied `hand_off` dock's lock is persistent (the item sits on the transporter's slot). `occupy` records `item_id` on it, and the delivering transporter releases it with `release` + that `item_id`, even though the picker owns the lock.
- When a robot's `/robots/ws` socket drops, its expiring locks are cut to `DOCK_LOCK_DISCONNECT_GRACE_S` (default 30 s); reconnecting restores the full TTL. Robot ids for locks are `couliglig_bot_{ros_domain_id}`.
- `robot_ips` is updated with WATCH/MULTI, and each socket only removes the entry carrying its own `session`.

### RL observation endpoint

`GET /rl/builder-inputs?robot_id={id}` (in `routes/rl.py`) returns the full environment state for RL training — dock states, item weights, robot poses (fetched live from each robot's `/roslib/transform`), picker/transporter agent states, and a deterministic `action_map` with padded indices (MAX_ITEMS=20, MAX_WZ=8, MAX_PICKERS=10, MAX_TRANSPORTERS=5).

### Fleet coordinator (in progress, replaces the endpoint above)

`coordinator/` is the new central decision layer (design: the "Couliglig MARL Deployment Redesign" doc). It is a Python port of couliglig-gamesim's policy layer:

- `coordinator/spec/` — `World` snapshot in grid cells (port of gamesim `PolicyWorld`), `observe()` (port of `RLObserver`, mirrors marl-couliglig `PickerTransporterEnv1` observations), `policy.json` loader. Masks are the env's rules narrowed by a `Feasibility` object (`EnvRules` = env rules only).
- `coordinator/policies/` — `MarlOnnxPolicy` (onnxruntime, in-process), `GreedyPolicy`, `StayPolicy`, `FallbackPolicy` (per-robot substitution on timeout, error or illegal index), `decode()` from action index to verb + target.
- `policy_models/<name>/` — an exported model directory from gamesim's `tools/export_policy.py` (`--out` can write here directly): `picker.onnx`, `transporter.onnx`, `policy.json`, `fixtures.json`. Loading a directory re-runs every fixture through the observer and the ONNX models and refuses it on any mismatch. (`models/` is the pydantic package, hence the different name.)

Tests: `pytest tests/test_coordinator_policy.py -v` (observer, heuristics and ONNX against the fixtures).

### Test isolation

Tests use `fakeredis` (in-memory Redis) and SQLite in-memory. The `conftest.py` adds the project root to `sys.path`. External HTTP calls to robots are mocked via `unittest.mock.patch`.
