"""What the trained policy sees, in the training environment's terms.

Ported from couliglig-gamesim (scripts/policy/policy_world.gd,
scripts/rl_observer.gd), which mirrors marl-couliglig's
PickerTransporterEnv1 observations column for column. The model's
policy.json is the spec: K sizes, optional feature columns, scales and the
action layout all come from it.
"""
from coordinator.spec.policy_spec import PolicySpec, load_spec
from coordinator.spec.world import EnvRules, Feasibility, Item, Picker, Transporter, World, Zone
from coordinator.spec.obs import Observation, observe

__all__ = [
    "PolicySpec", "load_spec",
    "World", "Item", "Picker", "Transporter", "Zone", "Feasibility", "EnvRules",
    "Observation", "observe",
]
