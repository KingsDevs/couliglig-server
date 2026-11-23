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
- `routes/robots.py` — robot registry in Redis `robot_ips`: each robot's `/robots/ws` socket registers it and heartbeats every 5 s; a background task (`prune_stale_robots_forever`, started in `main.py`) removes entries with no heartbeat for `ROBOT_STALE_S` (30 s), e.g. ones left over from a server restart
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

### Fleet coordinator (Phase 2; not yet driving robots)

`coordinator/` is the single writer of fleet state that replaces the per-robot RL loop (design: the "Couliglig MARL Deployment Redesign" doc). It runs beside the legacy Redis path; FastAPI routes are not wired to it yet (that is the Phase 3 cut-over).

- `coordinator/protocol/` — envelope (`couliglig.fleet/1`, session, seq, epoch), pydantic message payloads, `PROTOCOL_HASH`, RabbitMQ topology. couliglig-main's fleet_agent will keep a copy; `robot.hello` carries the hash. A welcome names the hello it answers (`hello_id`); robots ignore any other.
- `coordinator/core.py` — the synchronous state machine: registry and liveness, world model, decision rounds with the env's conflict rules, task state machine, reconciliation. Commitments are applied at decision time; failures roll back using the robot's physical report. Only a new session (robot restart) can lose a task; a same-session hello never fails one.
- `coordinator/store.py` — the coordinator's own SQLite file (`db/coordinator.db`): epoch, items, tasks, robots, event log (dedup by `event_id`). Write-through before acking an event.
- `coordinator/service.py` — aio-pika wiring (`CoordinatorService`) and `OpsClient` (operator RPC over `fleet.ops`). Publishing failures during a broker outage are logged, never fatal.
- `coordinator/grid.py` — metres <-> 0.75 m training cells; `GridDistance` (BFS) until the Nav2 provider exists. Unknown map pixels (205) are blocked.
- `coordinator/spec/`, `coordinator/policies/` — the decision layer ported from couliglig-gamesim (`World`, `observe()`, `MarlOnnxPolicy`, `GreedyPolicy`, `StayPolicy`, `FallbackPolicy`, `decode()`). Prefer `stay` over `greedy`: greedy transporters hop between free zones, as in the env.
- `coordinator/sim/` — `FakeRobot` (the fleet_agent contract), `SimFleet` (in-process, simulated clock, message loss/duplication/delay, restarts, partitions), `AmqpRobot` (the same robot over RabbitMQ).
- `policy_models/<name>/` — an exported model directory from gamesim's `tools/export_policy.py`: `picker.onnx`, `transporter.onnx`, `policy.json`, `fixtures.json`. Loading re-runs every fixture and refuses the directory on any mismatch. (`models/` is the pydantic package, hence the name.)

Run: `python -m coordinator --config coordinator.yaml` (or the `coordinator` service in docker-compose, with `rabbitmq`).

Tests: `tests/test_coordinator_policy.py` (fixtures), `tests/test_coordinator_protocol.py`, `tests/test_coordinator_core.py` (in-process scenarios incl. chaos and restarts), `tests/test_coordinator_amqp.py` (real RabbitMQ in a throwaway Docker container; skipped without Docker).

### SAM3 server (`sam3_server/`)

Text-prompt segmentation + grasp server, moved from GraspGen-couliglig (which keeps its own copy). ZMQ REQ/REP on 5557, msgpack + msgpack-numpy; actions `infer` (image + organized cloud + text -> ranked grasps, zed_camera frame, metres), `segment` (mask only), `metadata`, `health`. Clients: couliglig-main's `couliglig_grasp_bridge` / `robotic_arm_utils` (`sam3_host`, `sam3_port`). `infer` forwards the cleaned cloud to the GraspGen ZMQ server (5556) on the GPU PC (`GRASPGEN_HOST`); `--pose-mode manual` skips GraspGen.

- Own container on the Jetson (Orin Nano 8 GB, JetPack 6.2): `Dockerfile.sam3` on `dustynv/pytorch:2.7-r36.4.0` (stable torch 2.7, CUDA 12.6, Python 3.10), compose service `sam3` behind the `sam3` profile. The Jetson's `.env` sets `COMPOSE_PROFILES=sam3` (so plain `docker compose up` includes it) and `HF_TOKEN` (gated `facebook/sam3`); `GRASPGEN_HOST` defaults to 192.168.20.113.
- Keep torch/transformers out of `requirements.txt`. `transformers` is `<5.21` (5.21+ needs Python 3.11). Don't use NVIDIA's `pytorch:24.12-py3-igpu`: its torch `2.6.0a0` lacks `torch.accelerator`, which Sam3Model needs.
- Only `segmenter.py` imports torch; `server.py` takes the segmenter as an argument, so `tests/test_sam3_server.py` runs the protocol with a fake model and a fake GraspGen. Those tests need `pip install -r requirements-sam3.txt` and skip without it.

### Test isolation

Tests use `fakeredis` (in-memory Redis) and SQLite in-memory. The `conftest.py` adds the project root to `sys.path`. External HTTP calls to robots are mocked via `unittest.mock.patch`.
