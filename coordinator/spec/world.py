"""The fleet's state in PickerTransporterEnv1's terms, one snapshot per decision round.

Port of couliglig-gamesim's PolicyWorld (scripts/policy/policy_world.gd):

    positions   grid cells (row, col), floats allowed. The coordinator converts
                map metres to cells (cell_size_m); fixtures are already in cells
    distances   path length in cells from any cell to any cell, supplied by the
                caller (Nav2 planner on the real map; a table for fixtures).
                inf = no path
    steps       an agent moves agent_speed cells per step:
                steps(d) = 0 for d ~ 0, else max(1, ceil(d / agent_speed))
    items       live boxes (delivered ones are dropped). A box counts as picked
                as soon as a picker sets off for it, and as handed over as soon
                as a picker sets off to a transporter with it (the env commits
                at decision time)
    pickers     dest = where the current leg ends; hold = has a box or is on
                its way to pick one; handoff_to = index of the transporter it
                is taking a box to, else -1
    transporters zone = holds a waiting zone (driving there or parked);
                load counts boxes on board plus pickers on their way with one;
                carried = indices into items
    time        remaining_frac = undelivered / batch size; time_frac =
                elapsed / limit. Continuous operation reads as (1.0, 0.0)

What the robots can do right now (a dock free, a side slot free, a station
free) is asked through a Feasibility object, so the observer's masks are the
env's rules plus those checks. EnvRules (the default) applies the env's rules
alone, as in training and in the fixtures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

Cell = tuple[float, float]  # (row, col)


@dataclass
class Item:
    id: int | str
    cell: Cell
    w: float
    recv_cell: Cell
    picked: bool = False
    handed: bool = False
    holder: int = -1            # index of the picker holding it, -1 if none


@dataclass
class Picker:
    id: int | str
    cell: Cell
    dest: Cell
    busy: bool = False
    hold: bool = False
    w: float = 0.0              # weight of the box it holds
    free_in: float = 0.0        # steps until it can act again
    handoff_to: int = -1        # transporter index it is bringing a box to


@dataclass
class Transporter:
    id: int | str
    cell: Cell
    dest: Cell
    cap: float
    load: float = 0.0
    zone: bool = False
    free_in: float = 0.0
    carried: list[int] = field(default_factory=list)


@dataclass
class Zone:
    id: int | str
    cell: Cell
    occ: bool = False


class Feasibility(Protocol):
    """What the fleet can physically do now, beyond the env's rules."""

    def can_pick(self, picker: Picker, item: Item) -> bool: ...
    def can_hand_to(self, picker: Picker, transporter: Transporter) -> bool: ...
    def can_enter_zone(self, transporter: Transporter, zone: Zone) -> bool: ...
    def can_deliver(self, transporter: Transporter, item: Item) -> bool: ...


class EnvRules:
    """The training env's rules only: nothing beyond them is ruled out."""

    def can_pick(self, picker, item):
        return True

    def can_hand_to(self, picker, transporter):
        return True

    def can_enter_zone(self, transporter, zone):
        return True

    def can_deliver(self, transporter, item):
        return True


@dataclass
class World:
    items: list[Item]
    pickers: list[Picker]
    transporters: list[Transporter]
    zones: list[Zone]
    distance: Callable[[Cell, Cell], float]
    agent_speed: float = 5.0
    remaining_frac: float = 1.0
    time_frac: float = 0.0
    feasibility: Feasibility = field(default_factory=EnvRules)

    def dist(self, a: Cell, b: Cell) -> float:
        return self.distance(a, b)

    def steps(self, d: float) -> float:
        if math.isinf(d):
            return 1.0  # env: unreachable trips take one step
        if d < 0.5:
            return 0.0
        return max(1.0, math.ceil(d / self.agent_speed))

    @staticmethod
    def dx(a: Cell, b: Cell) -> float:
        return b[1] - a[1]

    @staticmethod
    def dy(a: Cell, b: Cell) -> float:
        return b[0] - a[0]

    @classmethod
    def from_fixture(cls, f: dict, agent_speed: float = 5.0) -> "World":
        """A world from one fixture's "world" (gamesim tools/export_policy.py)."""
        index_of = {}
        items = []
        for it in f["items"]:
            index_of[int(it["id"])] = len(items)
            items.append(Item(id=int(it["id"]), cell=_cell(it["cell"]), w=float(it["w"]),
                              recv_cell=_cell(it["recv_cell"]), picked=bool(it["picked"]),
                              handed=bool(it["handed"]), holder=int(it["holder"])))
        pickers = [Picker(id=int(p["id"]), cell=_cell(p["cell"]), dest=_cell(p["dest"]), busy=bool(p["busy"]),
                          hold=bool(p["hold"]), w=float(p["w"]), free_in=float(p["free_in"]),
                          handoff_to=int(p["handoff_to"]))
                   for p in f["pickers"]]
        transporters = [Transporter(id=int(t["id"]), cell=_cell(t["cell"]), dest=_cell(t["dest"]),
                                    cap=float(t["cap"]), load=float(t["load"]), zone=bool(t["zone"]),
                                    free_in=float(t["free_in"]),
                                    carried=[index_of[int(i)] for i in t["carried"]])
                        for t in f["transporters"]]
        zones = [Zone(id=int(z["id"]), cell=_cell(z["cell"]), occ=bool(z["occ"])) for z in f["zones"]]
        table = DistanceTable.from_blocks(f["dist"])
        return cls(items=items, pickers=pickers, transporters=transporters, zones=zones,
                   distance=table, agent_speed=agent_speed,
                   remaining_frac=float(f["remaining_frac"]), time_frac=float(f["time_frac"]))


class DistanceTable:
    """Path cells looked up from a fixed table; counts lookups it cannot answer."""

    def __init__(self, table: dict[tuple[Cell, Cell], float]):
        self._table = table
        self.missing = 0

    @classmethod
    def from_blocks(cls, blocks: list[dict]) -> "DistanceTable":
        table = {}
        for block in blocks:
            for i, src in enumerate(block["sources"]):
                for j, tgt in enumerate(block["targets"]):
                    d = block["d"][i][j]
                    table[(_cell(src), _cell(tgt))] = math.inf if d is None else float(d)
        return cls(table)

    def __call__(self, a: Cell, b: Cell) -> float:
        d = self._table.get((_cell(a), _cell(b)))
        if d is None:
            self.missing += 1
            return math.inf
        return d


def _cell(c) -> Cell:
    return (float(c[0]), float(c[1]))
