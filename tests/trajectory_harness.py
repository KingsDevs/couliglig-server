"""Builds a Coordinator from one recorded training-env step (trajectories.json)
and compares its conflict resolution and commitments with the env's."""
from __future__ import annotations

import math
from collections import defaultdict

from coordinator.core import (
    AVAILABLE, CLAIMED, DELIVERING, HANDING, HELD, ON_TRANSPORTER, ONLINE, STARTED, Config, Coordinator, ItemRec,
    RobotRec, TaskRec, numeric_id,
)
from coordinator.grid import Grid
from coordinator.policies import GreedyPolicy
from coordinator.policies.base import Choice
from coordinator.protocol import messages as m
from coordinator.store import Store


def _cell(c):
    return (int(c[0]), int(c[1]))


def build(step, spec):
    w = step["world"]
    height, width = w["shape"]
    grid = Grid(width=width, height=height, cell_size=1.0)
    table = {(_cell(a), _cell(b)): (math.inf if d is None else d) for a, b, d in step["distances"]}
    coord = Coordinator(Store(), GreedyPolicy(spec), spec, grid,
                        distance=lambda a, b: table.get((_cell(a), _cell(b)), math.inf),
                        config=Config(handoff_slots_default=64), clock=lambda: 0.0)
    coord.start()

    def dock(dock_id, kind, cell):
        x, y = grid.position_of(cell)
        coord._add_dock({"id": dock_id, "type": kind, "x": x, "y": y})
        return dock_id

    zones = [dock(f"Z{z['id']}", "waiting_zone", z["cell"]) for z in w["zones"]]
    zone_at = {_cell(z["cell"]): f"Z{z['id']}" for z in w["zones"]}

    def place(rid, role, cell, busy, capacity=4.0):
        x, y = grid.position_of(cell)
        r = RobotRec(id=rid, role=role, capacity=capacity, status=ONLINE,
                     exec_state=(m.ExecState.EXECUTING if busy else m.ExecState.IDLE).value,
                     pose=m.Pose(x=x, y=y), pose_at=0.0, last_seen=0.0)
        coord.robots[rid] = r
        return r

    pickers = w["agent_ids"]["picker"]
    transporters = w["agent_ids"]["transporter"]
    for p in w["pickers"]:
        place(pickers[p["id"]], "picker", p["cell"], p["busy"])
    carried_by = {}
    for t in w["transporters"]:
        moving = _cell(t["cell"]) != _cell(t["dest"])
        r = place(transporters[t["id"]], "transporter", t["cell"], moving, capacity=t["cap"])
        if t["zone"]:
            r.zone = zone_at[_cell(t["dest"])]
            r.parked = not moving
        if moving:
            # on its way: an active task ending at its destination, as the coordinator would have
            task = TaskRec(id=f"moving-{r.id}", robot_id=r.id, kind=m.TaskKind.PARK_ZONE.value if t["zone"]
                           else m.TaskKind.GO_HOME.value, target=r.zone, dest_cell=_cell(t["dest"]), status=STARTED)
            coord.tasks[task.id] = task
            r.task_id = task.id
        for i in t["carried"]:
            carried_by[i] = r.id
            r.carried.append(str(i))
    for it in w["items"]:
        iid = str(it["id"])
        dock(f"pk{iid}", "pickup", it["cell"])
        dock(f"rc{it['recv_cell'][0]}_{it['recv_cell'][1]}", "receiver", it["recv_cell"])
        item = ItemRec(id=iid, pickup_dock=f"pk{iid}", receiver_dock=f"rc{it['recv_cell'][0]}_{it['recv_cell'][1]}",
                       weight=it["w"], created_at=float(it["id"]))
        if not it["picked"]:
            item.status = AVAILABLE
        elif not it["handed"]:
            item.status, item.holder = HELD, pickers[it["holder"]]
            coord.robots[item.holder].held_item = iid
        elif it["id"] in carried_by:
            item.status, item.transporter = ON_TRANSPORTER, carried_by[it["id"]]
        else:
            item.status = DELIVERING
        coord.items[iid] = item

    choices = []
    for d in step["decisions"]:
        rid = (pickers if d["role"] == "picker" else transporters)[d["agent"]]
        target = d["target"]
        if d["verb"] in ("pick", "deliver"):
            target = str(target)
        choices.append((rid, Choice(d["verb"], target)))
    return coord, choices, transporters, zones


def env_winners(step) -> set[str]:
    return {aid for aid, _ in step["trips"]}


def new_tasks(coord) -> set[str]:
    """Robots the coordinator gave a task in this round (not the pre-built moving ones)."""
    return {t.robot_id for t in coord.tasks.values() if not t.id.startswith("moving-")}


def tie_defects(step) -> dict[str, str]:
    """Claims the env settled by comparing ids as text where numeric order differs.

    Returns {env winner: numeric winner} for each such claim group.
    """
    w = step["world"]
    ids = w["agent_ids"]
    dist = {(_cell(a), _cell(b)): (math.inf if d is None else d) for a, b, d in step["distances"]}
    cells = {"picker": {p["id"]: _cell(p["cell"]) for p in w["pickers"]},
             "transporter": {t["id"]: _cell(t["cell"]) for t in w["transporters"]}}
    item_cell = {it["id"]: _cell(it["cell"]) for it in w["items"]}
    t_dest = {t["id"]: _cell(t["dest"]) for t in w["transporters"]}
    zone_cell = {z["id"]: _cell(z["cell"]) for z in w["zones"]}
    groups = defaultdict(list)
    for d in step["decisions"]:
        if d["verb"] == "deliver":
            continue
        rid = ids[d["role"]][d["agent"]]
        start = cells[d["role"]][d["agent"]]
        goal = {"pick": item_cell, "handoff": t_dest, "zone": zone_cell}[d["verb"]][d["target"]]
        groups[(d["verb"], d["target"])].append((dist.get((start, goal), math.inf), rid))
    out = {}
    for claimants in groups.values():
        if len(claimants) < 2:
            continue
        text = min(claimants, key=lambda c: (c[0], c[1]))[1]
        numeric = min(claimants, key=lambda c: (c[0], numeric_id(c[1])))[1]
        if text != numeric:
            out[text] = numeric
    return out


def coordinator_commitments(coord, transporters, zones):
    items = {}
    for iid, it in coord.items.items():
        items[iid] = [it.status != AVAILABLE, it.status in (HANDING, ON_TRANSPORTER, DELIVERING)]
    hold = {}
    for rid, r in coord.robots.items():
        if r.role == "picker":
            item = coord.items.get(r.held_item) if r.held_item else None
            hold[rid] = item is not None and item.status in (CLAIMED, HELD) and item.holder == rid
    load = [coord.robots[t].capacity - coord._room(coord.robots[t]) for t in transporters]
    in_zone = [coord.robots[t].zone is not None for t in transporters]
    occ = [coord._zone_holder(z) is not None for z in zones]
    return items, hold, load, in_zone, occ
