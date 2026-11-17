"""One small interface for every decision maker (RL or heuristic).

Mirrors couliglig-gamesim's PolicyController: each round, every idle robot
with a real choice gets an observation; a policy returns one action index per
robot. The coordinator owns everything around decide(): building
observations, validating indices against the mask, conflict resolution,
decoding, timeouts. A policy cannot name a dock or bypass a mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coordinator.spec.obs import Observation
from coordinator.spec.policy_spec import VERBS, PolicySpec
from coordinator.spec.world import World

IDLE = 0


@dataclass(frozen=True)
class Decider:
    robot_id: str
    observation: Observation

    @property
    def is_picker(self) -> bool:
        return self.observation.role == "picker"


@dataclass(frozen=True)
class DecisionRequest:
    world: World               # whole snapshot, for centralized policies
    deciders: list[Decider]    # robots that must act this round


class DecisionPolicy(Protocol):
    name: str

    def decide(self, req: DecisionRequest) -> dict[str, int]:
        """Action index per decider's robot_id."""
        ...


def is_legal(observation: Observation, action: int) -> bool:
    mask = observation.obs["action_mask"]
    return 0 <= action < len(mask) and mask[action] > 0.5


@dataclass(frozen=True)
class Choice:
    verb: str                  # "idle" | "pick" | "handoff" | "zone" | "deliver"
    target: object = None      # item id, transporter index, zone index, carried item id


def decode(spec: PolicySpec, observation: Observation, action: int) -> Choice:
    """Action index -> what to do, through the row targets captured with the observation."""
    if action == IDLE:
        return Choice("idle")
    offset = 1
    for group, rows in spec.action_layout[observation.role]:
        if action < offset + rows:
            targets = observation.targets.get(group, [])
            row = action - offset
            if row >= len(targets):
                raise ValueError(f"action {action} points at empty row {row} of {group}")
            return Choice(VERBS[group], targets[row])
        offset += rows
    raise ValueError(f"action {action} outside the {observation.role} layout")


def masked_argmax(logits: np.ndarray, mask: np.ndarray) -> int:
    """Argmax over legal actions (first on ties)."""
    masked = np.where(mask > 0.5, logits, -np.inf)
    return int(np.argmax(masked))
