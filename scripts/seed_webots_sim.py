"""Create the dock config, map and robot poses of couliglig-main's Webots fleet sim.

    python scripts/seed_webots_sim.py [--replace]

Reads, from the couliglig-main checkout (COULIGLIG_MAIN, or ../couliglig-main):
  src/navigation/config/robot_fleet_configs_sim.yaml    robots, start poses, dock database file
  src/couliglig/config/<dock_database_file>             dock ids, poses, ArUco ids
  src/mapping/maps/<map_file>                           the map Nav2 uses
and writes one DockConfig (default name "webots-fleet-2") with its docks, a
MapConfig (files copied into db/maps) and RobotInfo rows. Dock roles follow
ROLES below; the hand-off docks (couliglig_bot_N_<slot>) are hand_off.
An existing config of that name is kept unless --replace. The last line printed
is the config id, for POST /dock/activate?config_id=<id>.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # services.database and map storage use paths under db/

from definitions import DockType                                   # noqa: E402
from schema import Dock, DockConfig, MapConfig, RobotInfo          # noqa: E402
from services.database import SessionLocal                         # noqa: E402
from services.map_storage import MAP_STORAGE, remove_map_files, store_map_files  # noqa: E402

# Sim dock roles (warehouse_fleet_2.wbt): both station aisles are pickups, the row along the
# south wall receivers, and dock_15-17 against the east wall the transporter's waiting zones.
ROLES = {DockType.PICKUP: range(0, 10), DockType.RECEIVER: range(10, 15), DockType.WAITING_ZONE: range(15, 18)}


def dock_type(dock_id: str) -> DockType:
    m = re.fullmatch(r"dock_(\d+)", dock_id)
    if not m:
        return DockType.HANDOFF          # couliglig_bot_3_left etc.: on a transporter
    for t, ids in ROLES.items():
        if int(m.group(1)) in ids:
            return t
    raise ValueError(f"{dock_id} has no role in ROLES")


def delete_config(db, cfg: DockConfig) -> None:
    for map_cfg in db.query(MapConfig).filter_by(dock_config_id=cfg.id).all():
        db.query(RobotInfo).filter_by(map_id=map_cfg.id).delete()
        remove_map_files([MAP_STORAGE / map_cfg.map_yaml_filename, MAP_STORAGE / map_cfg.map_image_filename])
        db.delete(map_cfg)
    db.query(Dock).filter_by(config_id=cfg.id).delete()
    db.delete(cfg)
    db.commit()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default=os.environ.get("COULIGLIG_MAIN", str(ROOT.parent / "couliglig-main")))
    ap.add_argument("--robots", default="src/navigation/config/robot_fleet_configs_sim.yaml",
                    help="robots file, relative to --main")
    ap.add_argument("--name", default="webots-fleet-2")
    ap.add_argument("--replace", action="store_true", help="delete an existing config of that name first")
    args = ap.parse_args()

    main_dir = Path(args.main)
    robots_cfg = yaml.safe_load((main_dir / args.robots).read_text())
    db_files = {r.get("dock_database_file", "dock_database.yaml") for r in robots_cfg["robots"]}
    if len(db_files) != 1:
        raise SystemExit(f"robots use different dock databases: {sorted(db_files)}")
    dock_db = main_dir / "src/couliglig/config" / db_files.pop()
    docks = yaml.safe_load(dock_db.read_text())["docks"]
    map_yaml = main_dir / "src/mapping/maps" / robots_cfg["map_file"]
    map_meta = yaml.safe_load(map_yaml.read_text())
    map_image = map_yaml.parent / map_meta["image"]

    db = SessionLocal()
    try:
        old = db.query(DockConfig).filter_by(name=args.name).first()
        if old is not None:
            if not args.replace:
                print(f"dock config {args.name!r} exists; keeping it (--replace recreates it)")
                print(old.id)
                return 0
            delete_config(db, old)

        cfg = DockConfig(name=args.name, description=f"Webots fleet sim ({args.robots})")
        db.add(cfg)
        db.flush()
        for dock_id, d in docks.items():
            x, y, yaw = (float(v) for v in d["pose"])
            t = dock_type(dock_id)
            db.add(Dock(config_id=cfg.id, dock_id=dock_id, dock_type=t,
                        aruco_id=(d.get("aruco_markers") or {}).get("center"),
                        # a hand-off dock moves with its transporter: no fixed pose
                        x=None if t == DockType.HANDOFF else x,
                        y=None if t == DockType.HANDOFF else y,
                        theta=None if t == DockType.HANDOFF else yaw))

        # The map, renamed after the config so it cannot clash with an uploaded one.
        stem = f"{args.name}_{map_yaml.stem}"
        map_meta["image"] = f"{stem}{map_image.suffix}"
        map_cfg = store_map_files(db, cfg.id, f"{stem}.yaml", yaml.safe_dump(map_meta, sort_keys=False).encode(),
                                  map_meta["image"], map_image.read_bytes(), commit=False)
        for r in robots_cfg["robots"]:
            yaw = 2.0 * math.atan2(float(r.get("orientation_z", 0.0)), float(r.get("orientation_w", 1.0)))
            db.add(RobotInfo(robot_name=r["name"], initial_x=float(r["x"]), initial_y=float(r["y"]),
                             initial_theta=yaw, map_id=map_cfg.id))
        db.commit()
        counts = {t.value: sum(dock_type(k) == t for k in docks) for t in DockType}
        print(f"dock config {args.name!r}: id {cfg.id}; docks {counts}; "
              f"map db/maps/{stem}.yaml; robots {[r['name'] for r in robots_cfg['robots']]}")
        print(cfg.id)
        return 0
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
