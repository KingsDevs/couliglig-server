"""Rule-based baselines that read only the agent's own observation.

Port of couliglig-gamesim's scripts/policy/heuristics.gd (greedy / stay),
which matches marl-couliglig's environments/heuristics.py on the fixtures.
Observation lists are nearest first, so "first legal row" is "nearest legal".

The central assignment heuristic (assign_policy.py) reads the whole world
rather than one observation; it is ported with the coordinator's world model.
"""
from __future__ import annotations

from coordinator.policies.base import IDLE, DecisionRequest
from coordinator.spec.obs import Observation
from coordinator.spec.policy_spec import PolicySpec


def _first_legal(mask, start: int, count: int) -> int:
    for i in range(start, min(start + count, len(mask))):
        if mask[i] > 0.5:
            return i
    return -1


def picker_action(obs: dict, k: dict) -> int:
    """Nearest available box; once holding one, the nearest transporter that can take it."""
    mask = obs["action_mask"]
    a = _first_legal(mask, 1, k["k_items"])
    if a < 0:
        a = _first_legal(mask, 1 + k["k_items"], k["k_transporters"])
    return IDLE if a < 0 else a


def greedy_transporter_action(obs: dict, k: dict) -> int:
    """Deliver the carried box with the nearest receiver, else the nearest free zone."""
    mask = obs["action_mask"]
    a = _first_legal(mask, 1 + k["k_zones"], k["k_carried"])
    if a < 0:
        a = _first_legal(mask, 1, k["k_zones"])
    return IDLE if a < 0 else a


def stay_transporter_action(obs: dict, k: dict) -> int:
    """Like greedy, but wait in the current zone instead of hopping to another one."""
    mask = obs["action_mask"]
    a = _first_legal(mask, 1 + k["k_zones"], k["k_carried"])
    if a >= 0:
        return a
    if obs["obs_vec"][2] > 0.5:  # in a zone
        return IDLE
    a = _first_legal(mask, 1, k["k_zones"])
    return IDLE if a < 0 else a


def greedy_action(observation: Observation, k: dict) -> int:
    if observation.role == "picker":
        return picker_action(observation.obs, k)
    return greedy_transporter_action(observation.obs, k)


def stay_action(observation: Observation, k: dict) -> int:
    if observation.role == "picker":
        return picker_action(observation.obs, k)
    return stay_transporter_action(observation.obs, k)


class GreedyPolicy:
    name = "greedy"

    def __init__(self, spec: PolicySpec):
        self.k = spec.k

    def decide(self, req: DecisionRequest) -> dict[str, int]:
        return {d.robot_id: greedy_action(d.observation, self.k) for d in req.deciders}


class StayPolicy:
    name = "stay"

    def __init__(self, spec: PolicySpec):
        self.k = spec.k

    def decide(self, req: DecisionRequest) -> dict[str, int]:
        return {d.robot_id: stay_action(d.observation, self.k) for d in req.deciders}
