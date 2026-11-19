"""Decision policies behind one interface (coordinator.policies.base).

    marl_onnx   the trained policy (MarlOnnxPolicy), verified on load
    greedy      nearest legal target
    assign      one central min-cost plan for the whole fleet (assign_policy.py)
    stay        greedy, but transporters wait in their zone
    fallback    FallbackPolicy(primary, fallback): per-robot substitution on
                timeout, error or an illegal index

make_policy() builds one from coordinator.yaml's "policy" section; make_policy_set()
builds every choice (stay, greedy, marl_onnx) for operators to switch between at
run time, with `primary` as the default:

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
from coordinator.policies.assign import AssignPolicy
from coordinator.policies.fallback import FallbackPolicy
from coordinator.policies.heuristics import GreedyPolicy, StayPolicy
from coordinator.policies.marl_onnx import MarlOnnxPolicy, ModelVerificationError, verify_model_dir
from coordinator.spec.policy_spec import PolicySpec, load_spec

__all__ = [
    "IDLE", "Choice", "Decider", "DecisionPolicy", "DecisionRequest", "decode", "is_legal", "masked_argmax",
    "AssignPolicy", "FallbackPolicy", "GreedyPolicy", "StayPolicy", "MarlOnnxPolicy", "ModelVerificationError",
    "verify_model_dir", "make_policy", "make_policy_set", "POLICY_CHOICES",
]


def _single(name: str, config: dict, spec: PolicySpec) -> DecisionPolicy:
    if name == "greedy":
        return GreedyPolicy(spec)
    if name == "stay":
        return StayPolicy(spec)
    if name == "assign":
        return AssignPolicy(spec, **config.get("assign", {}))
    if name == "marl_onnx":
        opts = config.get("marl_onnx", {})
        return MarlOnnxPolicy(opts["model_dir"], deterministic=opts.get("deterministic", True),
                              seed=opts.get("seed"))
    raise ValueError(f"unknown policy {name!r}")


# What an operator may pick at run time (POST /fleet/policy): label, kind
POLICY_CHOICES = {
    "stay": ("Stay", "heuristic"),            # nearest work; transporters wait in their zone
    "greedy": ("Greedy", "heuristic"),        # nearest work; transporters may hop between zones
    "assign": ("Assign", "heuristic"),        # one central plan for the fleet (assign_policy.py)
    "marl_onnx": ("RL model", "rl"),          # the trained policy, with a heuristic fallback
}


def make_policy_set(config: dict, spec: PolicySpec | None = None
                    ) -> tuple[dict[str, DecisionPolicy], dict[str, str], str, PolicySpec]:
    """Every choice in POLICY_CHOICES, built from the "policy" config section, for the
    coordinator to switch between: (choices, unavailable {name: reason}, default, spec).

    The RL model is wrapped in FallbackPolicy with the configured fallback heuristic
    (stay by default). A model that cannot be loaded or fails its fixtures is listed
    as unavailable instead of stopping the coordinator; the default (`primary`) then
    falls back to that heuristic.
    """
    model_dir = config.get("marl_onnx", {}).get("model_dir")
    if model_dir:
        spec = load_spec(Path(model_dir))
    if spec is None:
        raise ValueError("no policy.json: configure marl_onnx.model_dir or pass a spec")
    fallback_name = config.get("fallback") or "stay"
    if fallback_name not in ("stay", "greedy", "assign"):
        raise ValueError(f"policy.fallback must be stay, greedy or assign, not {fallback_name!r}")
    choices: dict[str, DecisionPolicy] = {name: _single(name, config, spec)
                                          for name in ("stay", "greedy", "assign")}
    unavailable: dict[str, str] = {}
    if model_dir:
        try:
            choices["marl_onnx"] = FallbackPolicy(_single("marl_onnx", config, spec), choices[fallback_name],
                                                  timeout_s=config.get("decide_timeout_ms", 200) / 1000.0)
        except Exception as e:  # noqa: BLE001 - a broken model must not stop the fleet
            unavailable["marl_onnx"] = f"{type(e).__name__}: {e}"
    else:
        unavailable["marl_onnx"] = "no marl_onnx.model_dir configured"
    default = config.get("primary", "stay")
    if default not in POLICY_CHOICES:
        raise ValueError(f"policy.primary must be one of {sorted(POLICY_CHOICES)}, not {default!r}")
    if default not in choices:
        default = fallback_name
    return choices, unavailable, default, spec


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
