"""policy.json: the contract between an exported model and the observer.

Written by couliglig-gamesim's tools/export_policy.py next to the ONNX files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Optional observation columns this observer can build. Others exist in the
# training env (zone_demand, receiver) but no exported model uses them yet.
SUPPORTED_FEATURES = {"competition", "timing"}

# Action verb for each pointer group of the action layout.
VERBS = {"items": "pick", "transporters": "handoff", "zones": "zone", "carried": "deliver"}


@dataclass(frozen=True)
class PolicySpec:
    k: dict[str, int]
    obs_features: frozenset[str]
    action_layout: dict[str, list[tuple[str, int]]]   # role -> [(group, rows)]
    inputs: dict[str, list[tuple[str, tuple[int, ...]]]]  # role -> [(name, shape)]
    dist_half_cells: float
    dir_half_cells: float
    weight_scale: float
    time_half_steps: float
    agent_speed: float
    checkpoint: str = ""

    def action_dim(self, role: str) -> int:
        return 1 + sum(rows for _, rows in self.action_layout[role])


def load_spec(path: str | Path) -> PolicySpec:
    """Read policy.json (a file, or the model directory that holds it)."""
    path = Path(path)
    if path.is_dir():
        path = path / "policy.json"
    meta = json.loads(path.read_text())
    features = frozenset(meta.get("obs_features") or [])
    unsupported = features - SUPPORTED_FEATURES
    if unsupported:
        raise ValueError(f"{path}: observer cannot build obs_features {sorted(unsupported)}")
    if meta.get("model") not in (None, "entity_pointer_network"):
        raise ValueError(f"{path}: unsupported model {meta.get('model')!r}")
    scales = meta.get("scales", {})
    spec = PolicySpec(
        k={name: int(v) for name, v in meta["k"].items()},
        obs_features=features,
        action_layout={role: [(g, int(n)) for g, n in layout] for role, layout in meta["action_layout"].items()},
        inputs={role: [(n, tuple(int(d) for d in shape)) for n, shape in entries]
                for role, entries in meta["inputs"].items()},
        dist_half_cells=float(scales.get("dist_half_cells", 50)),
        dir_half_cells=float(scales.get("dir_half_cells", 50)),
        weight_scale=float(scales.get("weight_scale", 10)),
        time_half_steps=float(scales.get("time_half_steps", 10)),
        agent_speed=float(scales.get("agent_speed", 5)),
        checkpoint=str(meta.get("checkpoint", "")),
    )
    for role in ("picker", "transporter"):
        if role not in spec.action_layout or role not in spec.inputs:
            raise ValueError(f"{path}: no {role} entry")
        if set(g for g, _ in spec.action_layout[role]) - set(VERBS):
            raise ValueError(f"{path}: unknown action group in {spec.action_layout[role]}")
    return spec
