"""Run the fleet coordinator:  python -m coordinator --config coordinator.yaml"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

import yaml

from coordinator.core import Config, Coordinator
from coordinator.grid import CELL_SIZE_M, Grid
from coordinator.policies import make_policy
from coordinator.service import CoordinatorService
from coordinator.store import Store


def build(config: dict) -> tuple[Coordinator, str]:
    policy, spec = make_policy(config["policy"])
    grid_cfg = config.get("grid", {})
    cell = float(grid_cfg.get("cell_size_m", CELL_SIZE_M))
    if grid_cfg.get("map_yaml"):
        grid = Grid.from_map_yaml(grid_cfg["map_yaml"], cell_size=cell)
    else:
        grid = Grid.open(int(grid_cfg.get("width", 40)), int(grid_cfg.get("height", 40)), cell_size=cell)
    timing = Config(**config.get("timing", {}))
    store = Store(config.get("db", "db/coordinator.db"))
    url = os.getenv("AMQP_URL", config.get("amqp_url", "amqp://guest:guest@localhost:5672/"))
    # Manual unless the config says otherwise: robots get no tasks until an operator
    # switches to auto (POST /fleet/resume). Only a fresh db/coordinator.db uses this;
    # afterwards the last manual / auto setting is kept.
    control = str(config.get("control", "manual")).strip().lower()
    if control not in ("manual", "auto"):
        raise SystemExit(f"control must be 'manual' or 'auto', not {control!r}")
    return Coordinator(store, policy, spec, grid, config=timing, manual=control == "manual"), url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="coordinator.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = yaml.safe_load(Path(args.config).read_text())
    coord, url = build(config)
    coord.start()
    service = CoordinatorService(coord, url)

    async def run():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, service.stop)
        await service.run()

    asyncio.run(run())


if __name__ == "__main__":
    main()
