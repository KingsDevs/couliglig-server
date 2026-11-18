#!/usr/bin/env bash
# Server side of the Webots fleet sim, in one terminal (Ctrl-C stops everything):
#   RabbitMQ   Docker container couliglig-sim-rabbitmq on localhost:5673 (user sim/sim), fresh each run
#   coordinator  coordinator.sim.yaml (db/coordinator_sim.db)
#   server     FLEET_MODE=bus on 127.0.0.1:8000 (not 0.0.0.0: sim robot APIs take 127.0.0.x:8000)
# then rebuilds the "webots-fleet-2" dock config from couliglig-main (scripts/seed_webots_sim.py
# --replace; edits made to it in the UI are lost) and activates it.
#
#   scripts/sim_up.sh            fresh coordinator state and fleet_agent state files
#   scripts/sim_up.sh --keep     keep db/coordinator_sim.db and ~/.couliglig/sim
#
# Then, in a shell with ROS and couliglig-main's install sourced:
#   ros2 launch couliglig_task_manager sim_fleet_launch.py amqp_url:=amqp://sim:sim@localhost:5673/
# and add an item:
#   curl -X POST localhost:8000/dock/add-item -H 'content-type: application/json' \
#        -d '{"dock_id": "dock_0", "item_id": "box1", "receiver_dock_id": "dock_12"}'
#   curl -X POST localhost:8000/fleet/resume     # the coordinator starts in manual
#   curl localhost:8000/fleet/snapshot
# Logs: db/sim/coordinator.log, db/sim/server.log
set -euo pipefail
cd "$(dirname "$0")/.."

KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1
PY=${PYTHON:-python3}
BROKER=couliglig-sim-rabbitmq
AMQP=amqp://sim:sim@localhost:5673/
LOGS=db/sim
mkdir -p "$LOGS"

# A ROS-sourced shell puts ROS's Python packages first; the server does not want them.
unset PYTHONPATH AMENT_PREFIX_PATH

pids=()
cleanup() {
    trap - EXIT INT TERM
    echo; echo "stopping..."
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
    docker stop "$BROKER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ $KEEP == 0 ]]; then
    rm -f db/coordinator_sim.db db/coordinator_sim.db-wal db/coordinator_sim.db-shm
    rm -rf "$HOME/.couliglig/sim"
fi

echo "RabbitMQ: $BROKER on :5673"
docker rm -f "$BROKER" >/dev/null 2>&1 || true
docker run -d --rm --name "$BROKER" -p 127.0.0.1:5673:5672 -p 127.0.0.1:15673:15672 \
    -e RABBITMQ_DEFAULT_USER=sim -e RABBITMQ_DEFAULT_PASS=sim rabbitmq:3.13.7-management >/dev/null
for i in $(seq 60); do
    $PY - "$AMQP" <<'EOF' >/dev/null 2>&1 && break
import asyncio, sys, aio_pika
async def main():
    c = await aio_pika.connect(sys.argv[1], timeout=2)
    await c.close()
asyncio.run(main())
EOF
    [[ $i == 60 ]] && { echo "RabbitMQ did not come up"; exit 1; }
    sleep 1
done

SEED=$($PY scripts/seed_webots_sim.py --replace)   # mirrors couliglig-main's files: always rebuilt
echo "$SEED" | head -n -1
CONFIG_ID=$(echo "$SEED" | tail -1)

echo "coordinator -> $LOGS/coordinator.log"
AMQP_URL=$AMQP $PY -m coordinator --config coordinator.sim.yaml >"$LOGS/coordinator.log" 2>&1 &
pids+=($!)

echo "server (FLEET_MODE=bus) on 127.0.0.1:8000 -> $LOGS/server.log"
FLEET_MODE=bus AMQP_URL=$AMQP $PY -m uvicorn main:app --host 127.0.0.1 --port 8000 >"$LOGS/server.log" 2>&1 &
pids+=($!)

for i in $(seq 60); do
    out=$(curl -s -X POST "http://127.0.0.1:8000/dock/activate?config_id=$CONFIG_ID" || true)
    if [[ "$out" == *'"success":true'* || "$out" == *'"ok":true'* ]]; then
        echo "activated dock config $CONFIG_ID: $out"; break
    fi
    [[ $i == 60 ]] && { echo "activate failed: $out"; exit 1; }
    sleep 1
done

echo
echo "ready. Start the robots:"
echo "  ros2 launch couliglig_task_manager sim_fleet_launch.py amqp_url:=$AMQP"
echo "The coordinator starts in MANUAL: robots get no tasks until you switch to auto"
echo "(frontend, or: curl -X POST localhost:8000/fleet/resume)."
echo "RabbitMQ UI: http://127.0.0.1:15673 (sim/sim). Ctrl-C stops."
tail -n 0 -F "$LOGS/coordinator.log" &
pids+=($!)
wait -n "${pids[0]}" "${pids[1]}"
echo "a process exited; see $LOGS/"
