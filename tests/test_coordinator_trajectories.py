"""
The coordinator's conflict resolution and commitments against the training env.

trajectories.json (written by couliglig-gamesim's tools/export_policy.py from
marl-couliglig's PickerTransporterEnv1) records env steps: the world before,
every agent's decision, the path lengths the env compared, the trips it
started (the winning claims) and the committed state after the step. For each
step the coordinator is built in the same state, given the same decisions, and
must pick the same winners and reach the same committed state:

    items picked / handed over, pickers holding, transporter loads,
    transporters in a zone, zones occupied

One known difference is expected: the env breaks distance ties by comparing
agent ids as text (picker_10 before picker_2); the coordinator compares them
as numbers. Steps with such a tie are checked for that substitution only.

Run with:
    pytest tests/test_coordinator_trajectories.py -v
"""
import json
from pathlib import Path

import pytest

from coordinator.spec import load_spec

from tests.trajectory_harness import build, coordinator_commitments, env_winners, new_tasks, tie_defects

MODEL_DIR = Path(__file__).resolve().parent.parent / "policy_models" / "imit_pointer_obs_tc-round_2"
SPEC = load_spec(MODEL_DIR)
STEPS = json.loads((MODEL_DIR / "trajectories.json").read_text())["steps"]


def test_recording_covers_conflicts():
    contested = {"pick": 0, "handoff": 0, "zone": 0}
    for step in STEPS:
        seen = {}
        for d in step["decisions"]:
            key = (d["verb"], d["target"])
            seen[key] = seen.get(key, 0) + 1
        for (verb, _), n in seen.items():
            if n > 1 and verb in contested:
                contested[verb] += 1
    assert len(STEPS) >= 100
    assert all(n >= 10 for n in contested.values()), contested


@pytest.mark.parametrize("n", range(len(STEPS)))
def test_step_matches_env(n):
    step = STEPS[n]
    coord, choices, transporters, zones = build(step, SPEC)
    coord._resolve_and_commit(choices, transporters, zones, 0.0)

    got = new_tasks(coord)
    defects = tie_defects(step)
    want = {defects.get(aid, aid) for aid in env_winners(step)}
    assert got == want, f"winners: coordinator {sorted(got - want)} vs env {sorted(want - got)}"
    if defects:
        return  # the env took a different branch on a text tie-break; states diverge from here

    items, hold, load, in_zone, occ = coordinator_commitments(coord, transporters, zones)
    after = step["after"]
    for item_id, status in after["items"].items():
        assert items[item_id] == status, f"item {item_id}: coordinator {items[item_id]} vs env {status}"
    assert [hold[p] for p in step["world"]["agent_ids"]["picker"]] == after["picker_hold"]
    assert load == pytest.approx(after["transporter_load"])
    assert in_zone == after["transporter_zone"]
    assert occ == after["zone_occ"]


def test_text_tie_breaks_are_rare():
    affected = sum(1 for step in STEPS if tie_defects(step))
    assert affected <= len(STEPS) // 10, affected
