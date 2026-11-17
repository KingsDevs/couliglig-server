"""The trained picker / transporter policies, run in-process with onnxruntime.

A model directory is what couliglig-gamesim's tools/export_policy.py writes:
    picker.onnx, transporter.onnx   inputs named after the observation keys,
                                    output "logits" (illegal actions at -1e9)
    policy.json                     the observation spec (coordinator.spec)
    fixtures.json                   env observations with their world state,
                                    the checkpoint's logits and heuristic actions

Loading a directory verifies it end to end: the observer must rebuild every
fixture's observation from its world, and the ONNX logits must match the
checkpoint's. A directory that fails is refused.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from coordinator.policies.base import DecisionRequest, masked_argmax
from coordinator.spec.obs import observe
from coordinator.spec.policy_spec import PolicySpec, load_spec
from coordinator.spec.world import World

ROLES = ("picker", "transporter")
OBS_TOLERANCE = 1e-5
LOGIT_TOLERANCE = 1e-4


class ModelVerificationError(ValueError):
    pass


@dataclass
class VerificationReport:
    fixtures: int = 0
    observer_failures: list[str] = field(default_factory=list)
    logit_failures: list[str] = field(default_factory=list)
    max_logit_error: float = 0.0

    @property
    def ok(self) -> bool:
        return self.fixtures > 0 and not self.observer_failures and not self.logit_failures


class OnnxModels:
    """One onnxruntime session per role."""

    def __init__(self, model_dir: Path, spec: PolicySpec):
        import onnxruntime as ort

        self.spec = spec
        self.sessions = {}
        for role in ROLES:
            session = ort.InferenceSession(str(model_dir / f"{role}.onnx"), providers=["CPUExecutionProvider"])
            have = {i.name for i in session.get_inputs()}
            missing = [n for n, _ in spec.inputs[role] if n not in have]
            if missing:
                raise ModelVerificationError(f"{role}.onnx lacks inputs {missing} listed in policy.json")
            self.sessions[role] = session

    def logits(self, role: str, observations: list[dict[str, np.ndarray]]) -> np.ndarray:
        if not observations:
            return np.zeros((0, self.spec.action_dim(role)), dtype=np.float32)
        feed = {name: np.stack([np.asarray(o[name], dtype=np.float32).reshape(shape) for o in observations])
                for name, shape in self.spec.inputs[role]}
        return self.sessions[role].run(["logits"], feed)[0]


def verify_model_dir(model_dir: str | Path, models: OnnxModels | None = None,
                     spec: PolicySpec | None = None) -> VerificationReport:
    """Observer and ONNX checked against the directory's fixtures."""
    model_dir = Path(model_dir)
    spec = spec or load_spec(model_dir)
    models = models or OnnxModels(model_dir, spec)
    fixtures = json.loads((model_dir / "fixtures.json").read_text())
    report = VerificationReport()
    for role in ROLES:
        names = [n for n, _ in spec.inputs[role]]
        for n, f in enumerate(fixtures.get(role, [])):
            report.fixtures += 1
            label = f"{role} #{n}"
            if "world" not in f:
                report.observer_failures.append(f"{label}: no world (re-export with gamesim's export_policy.py)")
            else:
                world = World.from_fixture(f["world"], spec.agent_speed)
                built = observe(world, spec, role == "picker", int(f["agent"]))
                if world.distance.missing:
                    report.observer_failures.append(f"{label}: {world.distance.missing} distances missing")
                for name in names:
                    want = np.asarray(f["obs"][name], dtype=np.float32)
                    got = built.obs[name].reshape(-1)
                    if got.shape != want.shape or np.abs(got - want).max(initial=0.0) > OBS_TOLERANCE:
                        report.observer_failures.append(f"{label}: {name} differs")
                        break
                for group, want in f.get("targets", {}).items():
                    if [int(t) for t in built.targets.get(group, [])] != [int(t) for t in want]:
                        report.observer_failures.append(f"{label}: targets {group} differ")
            obs = {name: np.asarray(f["obs"][name], dtype=np.float32) for name in names}
            got = models.logits(role, [obs])[0]
            want = np.asarray(f["logits"], dtype=np.float32)
            legal = obs["action_mask"] > 0.5
            err = float(np.abs(got - want)[legal].max(initial=0.0))
            report.max_logit_error = max(report.max_logit_error, err)
            if err > LOGIT_TOLERANCE or masked_argmax(got, obs["action_mask"]) != masked_argmax(want, obs["action_mask"]):
                report.logit_failures.append(f"{label}: logits differ by {err:.2e}")
    return report


class MarlOnnxPolicy:
    name = "marl_onnx"

    def __init__(self, model_dir: str | Path, deterministic: bool = True, seed: int | None = None,
                 verify: bool = True):
        self.model_dir = Path(model_dir)
        self.spec = load_spec(self.model_dir)
        self.models = OnnxModels(self.model_dir, self.spec)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.report = None
        if verify:
            self.report = verify_model_dir(self.model_dir, self.models, self.spec)
            if not self.report.ok:
                problems = (self.report.observer_failures + self.report.logit_failures)[:5]
                raise ModelVerificationError(
                    f"{self.model_dir}: refused ({self.report.fixtures} fixtures; "
                    f"{len(self.report.observer_failures)} observer and {len(self.report.logit_failures)} "
                    f"logit failures): {problems}")

    def decide(self, req: DecisionRequest) -> dict[str, int]:
        out = {}
        for role in ROLES:
            deciders = [d for d in req.deciders if d.observation.role == role]
            if not deciders:
                continue
            logits = self.models.logits(role, [d.observation.obs for d in deciders])
            for d, row in zip(deciders, logits):
                out[d.robot_id] = self._choose(row, d.observation.obs["action_mask"])
        return out

    def _choose(self, logits: np.ndarray, mask: np.ndarray) -> int:
        """Argmax over legal actions, or a sample from their softmax."""
        if self.deterministic:
            return masked_argmax(logits, mask)
        legal = mask > 0.5
        z = np.where(legal, logits.astype(np.float64), -np.inf)
        z = np.exp(z - z[legal].max())
        return int(self.rng.choice(len(z), p=z / z.sum()))
