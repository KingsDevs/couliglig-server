"""Observations for the trained policies, from a World snapshot.

Port of couliglig-gamesim's RLObserver (scripts/rl_observer.gd), which mirrors
PickerTransporterEnv1._picker_obs / _transporter_obs column for column:
K-nearest entity tables, nearest first, on the env's fixed scales
(path d -> d / (d + 50), offsets dx -> dx / (|dx| + 50), weights / 10,
steps t -> t / (t + 10)). Table sizes, optional columns and scales come from
the model's policy.json.

Actions are [idle] + one per row of each pointer group, as in the env:
    picker       items, transporters
    transporter  zones, carried
Each row's target (Observation.targets) is the entity it points at: an item
id, a transporter / zone index into the World's lists, a carried item id.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from coordinator.spec.policy_spec import PolicySpec
from coordinator.spec.world import World


@dataclass
class Observation:
    obs: dict[str, np.ndarray]       # model input name -> float32 array in the model's shape
    targets: dict[str, list]          # pointer group -> what each row points at
    choice: bool                      # any legal action besides idle
    role: str                         # "picker" | "transporter"


def observe(world: World, spec: PolicySpec, is_picker: bool, index: int) -> Observation:
    out = _Builder(world, spec).picker(index) if is_picker else _Builder(world, spec).transporter(index)
    _check_shapes(out, spec)
    return out


def nearest(d, k: int, first=None) -> list[int]:
    """Indices of the k smallest finite values, nearest first, ties by index.

    With `first` (bool per entry) those entries come first, then the nearest
    of the rest (env._nearest).
    """
    d = np.asarray(d, dtype=np.float64)
    if first is not None:
        first = np.asarray(first, dtype=bool)
        head = nearest(np.where(first, d, np.inf), k)
        if len(head) < k:
            head += nearest(np.where(first, np.inf, d), k - len(head))
        return head
    idx = np.flatnonzero(np.isfinite(d))
    return [int(i) for i in idx[np.lexsort((idx, d[idx]))][:k]]


class _Table:
    """A K-row entity table, zero-padded, with its row mask."""

    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.data = np.zeros((rows, cols), dtype=np.float32)
        self.mask = np.zeros(rows, dtype=np.float32)
        self.n = 0

    def add(self, row: list[float]) -> None:
        assert len(row) == self.cols, (len(row), self.cols)
        self.data[self.n] = row
        self.mask[self.n] = 1.0
        self.n += 1


def _legal(flags: list[bool], rows: int) -> list[float]:
    out = [0.0] * rows
    for i, f in enumerate(flags):
        out[i] = 1.0 if f else 0.0
    return out


class _Builder:
    def __init__(self, w: World, spec: PolicySpec):
        self.w, self.spec = w, spec
        self.timing = "timing" in spec.obs_features
        self.competition = "competition" in spec.obs_features
        self.ws = spec.weight_scale

    # env._dist_f / _dir_f / _time_f
    def dist_f(self, d: float) -> float:
        return 1.0 if math.isinf(d) else d / (d + self.spec.dist_half_cells)

    def dir_f(self, delta: float) -> float:
        return delta / (abs(delta) + self.spec.dir_half_cells)

    def time_f(self, t: float) -> float:
        return 1.0 if math.isinf(t) else t / (t + self.spec.time_half_steps)

    # ---------------- picker ----------------

    def picker(self, me: int) -> Observation:
        w, k, feas = self.w, self.spec.k, self.w.feasibility
        p = w.pickers[me]
        pos, holding = p.cell, p.hold
        t_items = _Table(k["k_items"], 5 + int(self.competition))
        t_trans = _Table(k["k_transporters"], 7 + int(self.timing))
        t_pick = _Table(k["k_pickers"], 5 + int(self.timing))
        targets = {"items": [], "transporters": []}
        item_legal, trans_legal = [], []

        avail = [i for i, it in enumerate(w.items) if not it.picked]
        d = [w.dist(pos, w.items[i].cell) for i in avail]
        comp = self._competition(avail) if self.competition else None
        handoff = self._handoff_dist(avail)
        for j in nearest(d, t_items.rows):
            it = w.items[avail[j]]
            row = [self.dist_f(d[j]), it.w / self.ws, self.dir_f(w.dx(pos, it.cell)),
                   self.dir_f(w.dy(pos, it.cell)), self.dist_f(handoff[j])]
            if comp is not None:
                best, second, who = comp
                row.append(self.time_f(second[j] if who[j] == me else best[j]))
            t_items.add(row)
            targets["items"].append(it.id)
            item_legal.append(not holding and feas.can_pick(p, it))

        d = [w.dist(pos, t.dest) for t in w.transporters]
        can_take = [holding and t.cap - t.load >= p.w and t.zone for t in w.transporters]
        for j in nearest(d, t_trans.rows, can_take):
            t = w.transporters[j]
            fits = holding and t.cap - t.load >= p.w
            row = [self.dist_f(d[j]), self.dir_f(w.dx(pos, t.dest)), self.dir_f(w.dy(pos, t.dest)),
                   min(t.load / t.cap, 1.0), (t.cap - t.load) / self.ws, float(t.zone), float(fits)]
            if self.timing:
                row.append(self.time_f(t.free_in))
            t_trans.add(row)
            targets["transporters"].append(j)
            trans_legal.append(fits and t.zone and feas.can_hand_to(p, t))

        others = [i for i in range(len(w.pickers)) if i != me]
        d = [w.dist(pos, w.pickers[i].dest) for i in others]
        for j in nearest(d, t_pick.rows):
            o = w.pickers[others[j]]
            row = [self.dist_f(d[j]), self.dir_f(w.dx(pos, o.dest)), self.dir_f(w.dy(pos, o.dest)),
                   float(o.hold), float(o.busy)]
            if self.timing:
                row.append(self.time_f(o.free_in))
            t_pick.add(row)

        obs_vec = [float(holding), p.w / self.ws, w.remaining_frac, w.time_frac]
        mask = [1.0] + _legal(item_legal, t_items.rows) + _legal(trans_legal, t_trans.rows)
        return self._result("picker", obs_vec, mask,
                            {"items": t_items, "transporters": t_trans, "pickers": t_pick}, targets)

    def _competition(self, avail: list[int]):
        """Per available item: earliest arrival (steps) of a picker holding
        nothing, counting the rest of its trip, and the runner-up
        (env: comp_best / comp_second / comp_who)."""
        w = self.w
        best, second, who = [], [], []
        for i in avail:
            b, s, bw = math.inf, math.inf, -1
            for k, p in enumerate(w.pickers):
                if p.hold:
                    continue
                dd = w.dist(p.dest, w.items[i].cell)
                arrive = p.free_in + (math.ceil(dd / w.agent_speed) if not math.isinf(dd) else math.inf)
                if arrive < b:
                    s, b, bw = b, arrive, k
                elif arrive < s:
                    s = arrive
            best.append(b)
            second.append(s)
            who.append(bw)
        return best, second, who

    def _handoff_dist(self, avail: list[int]) -> list[float]:
        """Per available item: path cells to the nearest transporter that could
        take it now (in a zone, with room), from that transporter's destination."""
        w = self.w
        out = []
        for i in avail:
            best = math.inf
            for t in w.transporters:
                if t.zone and t.cap - t.load >= w.items[i].w:
                    best = min(best, w.dist(t.dest, w.items[i].cell))
            out.append(best)
        return out

    # ---------------- transporter ----------------

    def transporter(self, me: int) -> Observation:
        w, k, feas = self.w, self.spec.k, self.w.feasibility
        t = w.transporters[me]
        pos = t.cell
        t_zones = _Table(k["k_zones"], 4)
        t_carried = _Table(k["k_carried"], 4)
        t_pick = _Table(k["k_pickers"], 6 + int(self.timing))
        t_trans = _Table(k["k_transporters"], 6 + int(self.timing))
        t_pending = _Table(k["k_pending"], 5)
        targets = {"zones": [], "carried": []}
        zone_legal, carried_legal = [], []

        d = [w.dist(pos, z.cell) for z in w.zones]
        for j in nearest(d, t_zones.rows, [not z.occ for z in w.zones]):
            z = w.zones[j]
            t_zones.add([self.dist_f(d[j]), self.dir_f(w.dx(pos, z.cell)), self.dir_f(w.dy(pos, z.cell)),
                         float(z.occ)])
            targets["zones"].append(j)
            zone_legal.append(not z.occ and feas.can_enter_zone(t, z))

        # my load, by distance to each box's receiver
        d = [w.dist(pos, w.items[i].recv_cell) for i in t.carried]
        for j in nearest(d, t_carried.rows):
            it = w.items[t.carried[j]]
            t_carried.add([self.dist_f(d[j]), it.w / self.ws, self.dir_f(w.dx(pos, it.recv_cell)),
                           self.dir_f(w.dy(pos, it.recv_cell))])
            targets["carried"].append(it.id)
            carried_legal.append(feas.can_deliver(t, it))

        # pickers; heading to me = on their way to hand me a box
        heading = sum(1 for p in w.pickers if p.handoff_to == me)
        d = [w.dist(pos, p.dest) for p in w.pickers]
        for j in nearest(d, t_pick.rows):
            p = w.pickers[j]
            row = [self.dist_f(d[j]), self.dir_f(w.dx(pos, p.dest)), self.dir_f(w.dy(pos, p.dest)),
                   float(p.hold), p.w / self.ws, float(p.handoff_to == me)]
            if self.timing:
                row.append(self.time_f(p.free_in))
            t_pick.add(row)

        others = [i for i in range(len(w.transporters)) if i != me]
        d = [w.dist(pos, w.transporters[i].dest) for i in others]
        for j in nearest(d, t_trans.rows):
            o = w.transporters[others[j]]
            row = [self.dist_f(d[j]), self.dir_f(w.dx(pos, o.dest)), self.dir_f(w.dy(pos, o.dest)),
                   min(o.load / o.cap, 1.0), (o.cap - o.load) / self.ws, float(o.zone)]
            if self.timing:
                row.append(self.time_f(o.free_in))
            t_trans.add(row)

        # boxes not handed over yet: on a rack, or held by a picker (then where it is heading)
        pending, at = [], []
        for i, it in enumerate(w.items):
            if it.handed:
                continue
            pending.append(i)
            at.append(w.pickers[it.holder].dest if it.holder >= 0 else it.cell)
        d = [w.dist(pos, a) for a in at]
        for j in nearest(d, t_pending.rows):
            it = w.items[pending[j]]
            t_pending.add([self.dist_f(d[j]), self.dir_f(w.dx(pos, at[j])), self.dir_f(w.dy(pos, at[j])),
                           it.w / self.ws, float(it.holder >= 0)])

        obs_vec = [min(t.load / t.cap, 1.0), (t.cap - t.load) / self.ws, float(t.zone),
                   min(len(t.carried) / t_carried.rows, 1.0), w.remaining_frac, w.time_frac,
                   min(heading / 5.0, 1.0)]
        mask = [1.0] + _legal(zone_legal, t_zones.rows) + _legal(carried_legal, t_carried.rows)
        return self._result("transporter", obs_vec, mask,
                            {"zones": t_zones, "carried": t_carried, "pickers": t_pick,
                             "transporters": t_trans, "pending": t_pending}, targets)

    # ---------------- helpers ----------------

    @staticmethod
    def _result(role, obs_vec, mask, tables, targets) -> Observation:
        obs = {"obs_vec": np.asarray(obs_vec, dtype=np.float32),
               "action_mask": np.asarray(mask, dtype=np.float32)}
        for name, table in tables.items():
            obs[name] = table.data
            obs[f"{name}_mask"] = table.mask
        return Observation(obs=obs, targets=targets, choice=bool(obs["action_mask"][1:].any()), role=role)


def _check_shapes(out: Observation, spec: PolicySpec) -> None:
    """Every model input present, in the shape policy.json gives it."""
    for name, shape in spec.inputs[out.role]:
        got = out.obs.get(name)
        if got is None or tuple(got.shape) != shape:
            raise ValueError(f"{out.role} {name}: built {None if got is None else got.shape}, model expects {shape}")
