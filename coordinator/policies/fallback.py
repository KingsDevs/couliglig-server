"""Primary policy with a fallback, per robot.

The fallback decides for a robot when the primary times out, raises, leaves
the robot out, or returns an index outside its mask. Every substitution is
recorded with its reason, so a round's log shows who actually decided.
"""
from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field

from coordinator.policies.base import DecisionPolicy, DecisionRequest, is_legal

log = logging.getLogger(__name__)


@dataclass
class Substitution:
    robot_id: str
    reason: str


@dataclass
class FallbackPolicy:
    primary: DecisionPolicy
    fallback: DecisionPolicy
    timeout_s: float = 0.2
    last_substitutions: list[Substitution] = field(default_factory=list)
    _pool: concurrent.futures.ThreadPoolExecutor = field(
        default_factory=lambda: concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy"))

    @property
    def name(self) -> str:
        return f"{self.primary.name}+{self.fallback.name}"

    def decide(self, req: DecisionRequest) -> dict[str, int]:
        self.last_substitutions = []
        reason_all = None
        future = self._pool.submit(self.primary.decide, req)
        try:
            chosen = future.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            # The primary keeps running in its thread; its answer is discarded.
            chosen, reason_all = {}, f"timeout after {self.timeout_s * 1000:.0f} ms"
        except Exception as e:  # noqa: BLE001 - any primary failure falls back
            chosen, reason_all = {}, f"{type(e).__name__}: {e}"

        out = {}
        needs_fallback = []
        for d in req.deciders:
            action = chosen.get(d.robot_id)
            if action is None:
                needs_fallback.append(d)
                self.last_substitutions.append(Substitution(d.robot_id, reason_all or "no decision"))
            elif not is_legal(d.observation, action):
                needs_fallback.append(d)
                self.last_substitutions.append(Substitution(d.robot_id, f"illegal action {action}"))
            else:
                out[d.robot_id] = action
        if needs_fallback:
            fallback = self.fallback.decide(DecisionRequest(world=req.world, deciders=needs_fallback))
            for d in needs_fallback:
                out[d.robot_id] = fallback[d.robot_id]
            for s in self.last_substitutions:
                log.warning("policy %s -> %s for %s: %s", self.primary.name, self.fallback.name, s.robot_id, s.reason)
        return out
