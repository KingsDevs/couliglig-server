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
pytest tests/test_dock_runtime.py -v
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

Docks have types: `pickup`, `receiver`, `waiting_zone`. Items are added to pickup docks and carry a `receiver_dock_id` for delivery routing.

### RL observation endpoint

`GET /rl/builder-inputs?robot_id={id}` (in `routes/rl.py`) returns the full environment state for RL training — dock states, item weights, robot poses (fetched live from each robot's `/roslib/transform`), picker/transporter agent states, and a deterministic `action_map` with padded indices (MAX_ITEMS=20, MAX_WZ=8, MAX_PICKERS=10, MAX_TRANSPORTERS=5).

### Test isolation

Tests use `fakeredis` (in-memory Redis) and SQLite in-memory. The `conftest.py` adds the project root to `sys.path`. External HTTP calls to robots are mocked via `unittest.mock.patch`.
