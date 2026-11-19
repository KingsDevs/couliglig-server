"""The central assignment heuristic ("assign"): one plan for the whole fleet each round.

Port of marl-couliglig's assign_policy.py (assign_intents, encode) onto the
coordinator's World instead of the env's StaySim; couliglig-gamesim runs the
same rules in scripts/policy/heuristics.gd. Four rules:

1. Matching: free pickers are matched to items (and holding pickers to
   transporters) with one min-cost assignment instead of each taking its
   nearest target, so two robots never race for the same target.
2. Critical-path priority: an item's tail is the shortest time from its shelf
   to its receiver through a waiting zone, tail(i) = min_z T(i->z) + T(z->rcv).
   Long tails start early:
       cost(picker p, item i)        = T(p->i) + T(i->z*(i)) - lam_eff * tail(i)
       cost(picker p, transporter t) = T(p->t) + lam_eff * T(zone of t->rcv)
   lam_eff = lam * min(1, pickers / items left) with phase on: travel dominates
   while there is plenty of work, tails near the end.
3. Zone balancing: a transporter outside a zone goes to the free zone z that
   minimises T(t->z) + beta * the weighted mean tail of the pending items once
   z is staffed.
4. Departure: a loaded transporter in its zone waits while a picker is expected
   to bring it an item within wait * T(zone -> nearest receiver of its load)
   steps, and leaves otherwise.
Every robot only chooses among the targets its observation lists (the K
nearest), so the plan is encoded as an ordinary action index.

Coordinator additions, which change nothing under the env's rules (EnvRules):
targets the robots cannot take right now (World.feasibility: a blacklisted dock,
a transporter without a free hand-off slot, a busy receiver) are left out of the
plan, as gamesim's port does, so a robot is matched to work it can actually do;
and only robots deciding this round are planned for (the others are on a task).

tests/test_coordinator_assign.py checks this port against assign_policy.py's own
choices on recorded env states (tests/data/assign_cases.json.gz).
"""
from __future__ import annotations

import math

import numpy as np

from coordinator.policies.base import IDLE, DecisionRequest
from coordinator.spec.obs import Observation, nearest
from coordinator.spec.policy_spec import PolicySpec
from coordinator.spec.world import World

BIG = 1e6
GROUP = {"pick": "items", "handoff": "transporters", "zone": "zones", "deliver": "carried"}
DEFAULTS = {"lam": 1.0, "beta": 1.0, "phase": True, "wait": 1.0}


class AssignPolicy:
    name = "assign"

    def __init__(self, spec: PolicySpec, lam: float = 1.0, beta: float = 1.0, phase: bool = True,
                 wait: float = 1.0):
        self.spec = spec
        self.k = spec.k
        self.params = {"lam": float(lam), "beta": float(beta), "phase": bool(phase), "wait": float(wait)}

    def decide(self, req: DecisionRequest) -> dict[str, int]:
        w = req.world
        index = {("picker", str(p.id)): ("p", k) for k, p in enumerate(w.pickers)}
        index.update({("transporter", str(t.id)): ("t", k) for k, t in enumerate(w.transporters)})
        keys = {d.robot_id: index.get((d.observation.role, str(d.robot_id))) for d in req.deciders}
        intents = assign_intents(w, self.k, deciding={key for key in keys.values() if key}, **self.params)
        return {d.robot_id: encode(self.spec, d.observation, intents.get(keys[d.robot_id]), w)
                for d in req.deciders}


def encode(spec: PolicySpec, observation: Observation, intent, world: World) -> int:
    """An intent -> the action index pointing at its target in the robot's observation;
    idle if the target is not listed or its action is masked."""
    if intent is None:
        return IDLE
    verb, target = intent
    if verb in ("pick", "deliver"):
        target = world.items[target].id             # observation rows hold item ids
    group = GROUP[verb]
    offset = 1
    for name, rows in spec.action_layout[observation.role]:
        if name == group:
            listed = observation.targets.get(group, [])
            if target not in listed:
                return IDLE
            action = offset + listed.index(target)
            return action if observation.obs["action_mask"][action] > 0.5 else IDLE
        offset += rows
    return IDLE


def assign_intents(w: World, k: dict, lam: float = 1.0, beta: float = 1.0, phase: bool = True,
                   wait: float = 1.0, deciding: set | None = None) -> dict:
    """Intents of the robots free to act, keyed ("p", picker index) / ("t", transporter index):
    ("pick", item index), ("handoff", transporter index), ("zone", zone index),
    ("deliver", item index). Robots left out stay idle. `deciding` (keys) limits the plan to
    the robots deciding this round; None plans for every robot that is not busy."""
    feas = w.feasibility
    m = picker_matchings(w, k, lam, phase, deciding)
    st, held, lam_eff = m["statics"], m["held"], m["lam_eff"]
    intents: dict = {}

    def free(key, busy):
        return not busy and (deciding is None or key in deciding)

    # 1+2) empty pickers -> items; holding pickers -> transporters waiting in a zone with room
    for r, c in min_cost_matching(m["pick_cost"]):
        intents[("p", m["empty"][r])] = ("pick", m["available"][c])
    for r, c in min_cost_matching(m["handoff_cost"]):
        intents[("p", m["holding"][r])] = ("handoff", c)

    # transporters: 4) deliver or keep waiting for more; 3) zone balancing outside a zone
    occ = np.array([z.occ for z in w.zones], dtype=bool)
    z_occ = occ.copy()
    best_staffed = arrivals = None
    pend_tail_z = pend_w = None
    for t, tr in enumerate(w.transporters):
        if not free(("t", t), tr.busy):
            continue
        if tr.carried:
            carried = sorted(tr.carried)
            d = np.array([w.dist(tr.cell, w.items[i].recv_cell) for i in carried])
            rows = nearest(d, k["k_carried"])
            if not rows:
                continue
            if wait > 0 and tr.zone:
                if arrivals is None:
                    arrivals = _expected_arrivals(w, held, intents, lam_eff)
                if arrivals[t] <= wait * w.steps(d[rows[0]]):
                    continue  # an item is on its way: wait for it
            for row in rows:   # the nearest receiver that can take it now
                if feas.can_deliver(tr, w.items[carried[row]]):
                    intents[("t", t)] = ("deliver", carried[row])
                    break
            continue
        if tr.zone or z_occ.all():
            continue
        if best_staffed is None:
            pend_tail_z, pend_w = _pending_tails(w, st, held)
            best_staffed = (pend_tail_z[:, occ].min(axis=1) if occ.any() else np.full(len(pend_w), np.inf))
        d = np.array([w.dist(tr.cell, z.cell) for z in w.zones])
        # the zones its observation offers; zones taken by transporters planned before it
        # this round stay listed (the observation doesn't know them) but are skipped
        listed = [z for z in nearest(np.where(~occ, d, np.inf), k["k_zones"])
                  if not z_occ[z] and feas.can_enter_zone(tr, w.zones[z])]
        if not listed:
            continue
        cost = np.array([w.steps(d[z]) for z in listed], dtype=np.float64)
        if len(pend_w):
            with_z = np.minimum(best_staffed[:, None], pend_tail_z[:, listed])   # (items, zones)
            cost = cost + beta * (pend_w @ with_z) / pend_w.sum()
        z = listed[int(np.argmin(cost))]
        intents[("t", t)] = ("zone", z)
        z_occ[z] = True  # the next transporters plan around this one
        if len(pend_w):
            best_staffed = np.minimum(best_staffed, pend_tail_z[:, z])
    return intents


def picker_matchings(w: World, k: dict, lam: float = 1.0, phase: bool = True, deciding: set | None = None) -> dict:
    """Rules 1+2 for the pickers: the cost matrices of the two matchings.

    pick_cost[r, c]     empty picker empty[r] -> item available[c]
    handoff_cost[r, c]  holding picker holding[r] -> transporter c
    BIG = not allowed (not in its observation's list, no room, or not feasible now)."""
    feas = w.feasibility
    st = _statics(w)
    held = _held(w)
    available = [i for i, it in enumerate(w.items) if not it.picked]
    n_p = len(w.pickers)
    lam_eff = lam * min(1.0, n_p / max(1, len(available))) if phase else lam

    def free(p):
        return not w.pickers[p].busy and (deciding is None or ("p", p) in deciding)

    empty = [p for p in range(n_p) if free(p) and held[p] < 0]
    pick_cost = np.full((len(empty), len(available)), BIG)
    if empty and available:
        prio = np.array([st["head_to_best"][i] - lam_eff * st["tail"][i] for i in available])
        for r, p in enumerate(empty):
            pk = w.pickers[p]
            d = np.array([w.dist(pk.cell, w.items[i].cell) for i in available])
            for c in nearest(d, k["k_items"]):       # the items its observation offers
                if feas.can_pick(pk, w.items[available[c]]):
                    pick_cost[r, c] = w.steps(d[c]) + prio[c]

    holding = [p for p in range(n_p) if free(p) and held[p] >= 0]
    handoff_cost = np.full((len(holding), len(w.transporters)), BIG)
    for r, p in enumerate(holding):
        pk, item = w.pickers[p], w.items[held[p]]
        can_take = np.array([t.zone and t.cap - t.load >= item.w for t in w.transporters], dtype=bool)
        if not can_take.any():
            continue
        d = np.array([w.dist(pk.cell, t.dest) for t in w.transporters])
        for c in nearest(np.where(can_take, d, np.inf), k["k_transporters"]):
            if feas.can_hand_to(pk, w.transporters[c]):
                handoff_cost[r, c] = w.steps(d[c]) + lam_eff * _zone_to(w, w.transporters[c].dest, item.recv_cell)
    return {"statics": st, "held": held, "lam_eff": lam_eff, "available": available, "empty": empty,
            "pick_cost": pick_cost, "holding": holding, "handoff_cost": handoff_cost}


def _held(w: World) -> list[int]:
    """Per picker: the item it holds or is on its way to pick (index), -1 if none. An item
    already handed over (the picker is taking it to a transporter) no longer counts."""
    held = [-1] * len(w.pickers)
    for i, it in enumerate(w.items):
        if it.holder >= 0 and not it.handed and it.holder < len(held):
            held[it.holder] = i
    return held


def _statics(w: World) -> dict:
    """Per item: T(item -> zone), T(zone -> its receiver), the best tail and the head to it."""
    n_i, n_z = len(w.items), len(w.zones)
    to_zone = np.zeros((n_i, n_z))
    zone_to_recv = np.zeros((n_i, n_z))
    for i, it in enumerate(w.items):
        for z, zone in enumerate(w.zones):
            to_zone[i, z] = w.steps(w.dist(zone.cell, it.cell))
            zone_to_recv[i, z] = w.steps(w.dist(zone.cell, it.recv_cell))
    tail_z = to_zone + zone_to_recv
    rows = np.arange(n_i)
    best = tail_z.argmin(axis=1) if n_z else np.zeros(n_i, dtype=np.int64)
    return {"zone_to_recv": zone_to_recv, "tail_z": tail_z,
            "tail": tail_z[rows, best] if n_z else np.zeros(n_i),
            "head_to_best": to_zone[rows, best] if n_z else np.zeros(n_i)}


def _zone_to(w: World, cell, recv) -> float:
    """T(waiting zone at cell -> receiver)."""
    return w.steps(w.dist(cell, recv))


def _expected_arrivals(w: World, held: list[int], intents: dict, lam_eff: float) -> np.ndarray:
    """Per transporter: steps until the first picker is expected to hand it an item that
    fits (inf if none). Each item a picker holds or heads to pick goes to the transporter
    the hand-off rule would choose."""
    arrivals = np.full(len(w.transporters), np.inf)
    room = np.array([t.cap - t.load for t in w.transporters])
    zone = np.array([t.zone for t in w.transporters], dtype=bool)
    for p, pk in enumerate(w.pickers):
        intent = intents.get(("p", p))
        if pk.busy:
            item = held[p]
            if item < 0:
                continue  # walking to a hand-off it already made
            loc, eta = pk.dest, float(pk.free_in)
        elif intent is not None and intent[0] == "handoff":
            t = intent[1]
            arrivals[t] = min(arrivals[t], w.steps(w.dist(pk.cell, w.transporters[t].dest)))
            continue
        elif intent is not None and intent[0] == "pick":
            item = intent[1]
            loc = w.items[item].cell
            eta = w.steps(w.dist(pk.cell, loc))
        elif held[p] >= 0:
            item, loc, eta = held[p], pk.cell, 0.0
        else:
            continue
        cand = np.flatnonzero(zone & (room >= w.items[item].w))
        if not len(cand):
            continue
        to_t = np.array([w.steps(w.dist(loc, w.transporters[t].dest)) for t in cand])
        to_recv = np.array([_zone_to(w, w.transporters[t].dest, w.items[item].recv_cell) for t in cand])
        j = int(np.argmin(to_t + lam_eff * to_recv))
        arrivals[cand[j]] = min(arrivals[cand[j]], eta + float(to_t[j]))
    return arrivals


def _pending_tails(w: World, st: dict, held: list[int]):
    """(tail through each zone, weight) of the items not handed over yet."""
    tails, weights = [], []
    for i, it in enumerate(w.items):
        if not it.picked:
            tails.append(st["tail_z"][i])
            weights.append(it.w)
    for p, item in enumerate(held):
        if item >= 0:  # with a picker: from where it is heading
            dest = w.pickers[p].dest
            to_zone = np.array([w.steps(w.dist(z.cell, dest)) for z in w.zones])
            tails.append(to_zone + st["zone_to_recv"][item])
            weights.append(w.items[item].w)
    if not tails:
        return np.zeros((0, len(w.zones))), np.zeros(0)
    return np.array(tails, dtype=np.float64).reshape(len(tails), len(w.zones)), np.array(weights, dtype=np.float64)


# ---------------------------------------------------------------- min-cost matching

def min_cost_matching(cost: np.ndarray) -> list[tuple[int, int]]:
    """Rows matched to distinct columns at minimum total cost (as scipy's
    linear_sum_assignment on a rectangular matrix); pairs at BIG (not allowed) are
    dropped. Kuhn-Munkres with potentials, O(n^2 m)."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    a = cost.T if transposed else cost
    col_of = _hungarian(a)
    out = []
    for r, c in enumerate(col_of):
        if c >= 0 and a[r, c] < BIG / 2:
            out.append((c, r) if transposed else (r, c))
    return sorted(out)


def _hungarian(a: np.ndarray) -> list[int]:
    """The column of each row of a (rows <= cols) in a minimum-cost assignment."""
    n, m = a.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)     # p[j]: row (1-based) matched to column j
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, math.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = a[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], math.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_of = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            col_of[p[j] - 1] = j - 1
    return col_of
