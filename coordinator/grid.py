"""Map metres <-> training-grid cells, and path lengths on the grid.

One cell is cell_size metres (0.75 m, as couliglig-gamesim; see the plan's
grid-scale answer). Rows grow downwards like the training maps' image rows,
so row = height - 1 - floor((y - origin_y) / cell_size).

GridDistance (BFS on free cells, cached per source) is the distance source
until the Nav2 planner provider exists (Phase 3). It also moves the fake
robots, so they and the coordinator agree on paths.
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CELL_SIZE_M = 0.75
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class Grid:
    width: int                      # columns
    height: int                     # rows
    cell_size: float = CELL_SIZE_M
    origin_x: float = 0.0
    origin_y: float = 0.0
    blocked: set[tuple[int, int]] = field(default_factory=set)

    @classmethod
    def open(cls, width: int, height: int, cell_size: float = CELL_SIZE_M) -> "Grid":
        return cls(width=width, height=height, cell_size=cell_size)

    @classmethod
    def from_map_yaml(cls, yaml_path: str | Path, cell_size: float = CELL_SIZE_M,
                      free_fraction: float = 0.5) -> "Grid":
        """A ROS map_server map (yaml + pgm), downsampled: a cell is free when at
        least free_fraction of its pixels are free."""
        import yaml

        yaml_path = Path(yaml_path)
        meta = yaml.safe_load(yaml_path.read_text())
        img = _load_pgm(yaml_path.parent / meta["image"])
        res = float(meta["resolution"])
        free_thresh = float(meta.get("free_thresh", 0.196))
        occ = (255.0 - img) / 255.0 if not meta.get("negate") else img / 255.0
        # map_saver writes unknown space as 205, which sits under the usual
        # free_thresh (0.25); never plan through unexplored space
        free = (occ < free_thresh) & (img != 205)
        f = max(1, int(round(cell_size / res)))
        h, w = free.shape[0] // f, free.shape[1] // f
        frac = free[:h * f, :w * f].reshape(h, f, w, f).mean(axis=(1, 3))
        # image row 0 is the top of the map, as grid row 0
        blocked = {(int(r), int(c)) for r, c in np.argwhere(frac < free_fraction)}
        ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
        # the image may be taller than h*f pixels; anchor the grid at the image's bottom edge
        oy += (img.shape[0] - h * f) * res
        return cls(width=w, height=h, cell_size=f * res, origin_x=ox, origin_y=oy, blocked=blocked)

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x - self.origin_x) / self.cell_size))
        row = self.height - 1 - int(math.floor((y - self.origin_y) / self.cell_size))
        return (row, col)

    def position_of(self, cell) -> tuple[float, float]:
        """Centre of a cell in map metres."""
        row, col = int(round(cell[0])), int(round(cell[1]))
        x = self.origin_x + (col + 0.5) * self.cell_size
        y = self.origin_y + (self.height - 1 - row + 0.5) * self.cell_size
        return (x, y)

    def inside(self, cell) -> bool:
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def free(self, cell) -> bool:
        return self.inside(cell) and (cell[0], cell[1]) not in self.blocked


class GridDistance:
    """Path cells between any two cells (4-connected BFS over free cells).

    A blocked target (a dock against a wall) is reached through its nearest
    free neighbour, as the env's stations are not walkable either. Fields are
    cached per source cell; docks don't move, so most lookups are cache hits.
    """

    def __init__(self, grid: Grid, cache_size: int = 512):
        self.grid = grid
        self._fields: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_size = cache_size

    def __call__(self, a, b) -> float:
        src, dst = _int_cell(a), _int_cell(b)
        if src == dst:
            return 0.0
        dist, _ = self._field(src)
        if self.grid.inside(dst) and dist[dst] >= 0:
            return float(dist[dst])
        best = math.inf
        for dr, dc in _NEIGHBOURS:
            n = (dst[0] + dr, dst[1] + dc)
            if self.grid.inside(n) and dist[n] >= 0:
                best = min(best, float(dist[n]) + 1.0)
        return best

    def path(self, a, b) -> list[tuple[int, int]] | None:
        """Cells from a to b inclusive (shortest), or None if unreachable."""
        src, dst = _int_cell(a), _int_cell(b)
        if src == dst:
            return [src]
        dist, parent = self._field(src)
        end = dst
        if not (self.grid.inside(dst) and dist[dst] >= 0):
            options = [(dist[n], n) for n in ((dst[0] + dr, dst[1] + dc) for dr, dc in _NEIGHBOURS)
                       if self.grid.inside(n) and dist[n] >= 0]
            if not options:
                return None
            end = min(options)[1]
        cells = [end]
        while cells[-1] != src:
            r, c = cells[-1]
            p = int(parent[r, c])
            cells.append((p // self.grid.width, p % self.grid.width))
        cells.reverse()
        if end != dst:
            cells.append(dst)
        return cells

    def _field(self, src):
        if src in self._fields:
            self._fields.move_to_end(src)
            return self._fields[src]
        g = self.grid
        dist = np.full((g.height, g.width), -1, dtype=np.int32)
        parent = np.full((g.height, g.width), -1, dtype=np.int64)
        if g.inside(src):
            dist[src] = 0
            q = deque([src])
            while q:
                r, c = q.popleft()
                for dr, dc in _NEIGHBOURS:
                    n = (r + dr, c + dc)
                    if g.free(n) and dist[n] < 0:
                        dist[n] = dist[r, c] + 1
                        parent[n] = r * g.width + c
                        q.append(n)
        self._fields[src] = (dist, parent)
        if len(self._fields) > self._cache_size:
            self._fields.popitem(last=False)
        return dist, parent


def _int_cell(c) -> tuple[int, int]:
    return (int(round(c[0])), int(round(c[1])))


def _load_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    parts, i = [], 0
    while len(parts) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        parts.append(data[i:j])
        i = j
    i += 1
    w, h = int(parts[1]), int(parts[2])
    if parts[0] == b"P5":
        return np.frombuffer(data[i:i + w * h], dtype=np.uint8).reshape(h, w)
    return np.array(data[i:].split(), dtype=np.uint8).reshape(h, w)
