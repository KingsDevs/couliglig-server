"""Decision policies behind one interface (coordinator.policies.base).

    marl_onnx   the trained policy (MarlOnnxPolicy), verified on load
    greedy      nearest legal target
    stay        greedy, but transporters wait in their zone
    fallback    FallbackPolicy(primary, fallback): per-robot substitution on
                timeout, error or an illegal index

make_policy() builds one from coordinator.yaml's "policy" section:

    policy:
      primary: marl_onnx
      fallback: greedy
      decide_timeout_ms: 200
      marl_onnx:
        model_dir: policy_models/imit_pointer_obs_tc-round_2
        deterministic: true
"""
from __future__ import annotations

from pathlib import Path

from coordinator.policies.base import (
    IDLE, Choice, Decider, DecisionPolicy, DecisionRequest, decode, is_legal, masked_argmax,
)
from coordinator.policies.fallback import FallbackPolicy
from coordinator.policies.heuristics import GreedyPolicy, StayPolicy
from coordinator.policies.marl_onnx import MarlOnnxPolicy, ModelVerificationError, verify_model_dir
from coordinator.spec.policy_spec import PolicySpec, load_spec

__all__ = [
    "IDLE", "Choice", "Decider", "DecisionPolicy", "DecisionRequest", "decode", "is_legal", "masked_argmax",
    "FallbackPolicy", "GreedyPolicy", "StayPolicy", "MarlOnnxPolicy", "ModelVerificationError",
    "verify_model_dir", "make_policy",
]


def _single(name: str, config: dict, spec: PolicySpec) -> DecisionPolicy:
    if name == "greedy":
        return GreedyPolicy(spec)
    if name == "stay":
        return StayPolicy(spec)
    if name == "marl_onnx":
        opts = config.get("marl_onnx", {})
        return MarlOnnxPolicy(opts["model_dir"], deterministic=opts.get("deterministic", True),
                              seed=opts.get("seed"))
    raise ValueError(f"unknown policy {name!r}")


def make_policy(config: dict, spec: PolicySpec | None = None) -> tuple[DecisionPolicy, PolicySpec]:
    """(policy, spec) from the "policy" config section. The spec comes from the
    RL model's directory when one is configured, else from `spec`."""
    model_dir = config.get("marl_onnx", {}).get("model_dir")
    if model_dir:
        spec = load_spec(Path(model_dir))
    if spec is None:
        raise ValueError("no policy.json: configure marl_onnx.model_dir or pass a spec")
    primary = _single(config.get("primary", "greedy"), config, spec)
    fallback_name = config.get("fallback")
    if not fallback_name:
        return primary, spec
    return FallbackPolicy(primary, _single(fallback_name, config, spec),
                          timeout_s=config.get("decide_timeout_ms", 200) / 1000.0), spec
