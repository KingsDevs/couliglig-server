"""
The coordinator's assign heuristic (coordinator/policies/assign.py) against the
training repo's own: marl-couliglig assign_policy.AssignActor chose every free
agent's action on 116 recorded PickerTransporterEnv1 states (24 episodes, fleets
of 3 to 36 pickers); tests/data/assign_cases.json.gz holds those states in the
coordinator's World fixture format with every path length between the cells
involved, and tests/data/gen_assign_cases.py rebuilds it:

    ~/marl-couliglig/.venv/bin/python tests/data/gen_assign_cases.py tests/data/assign_cases.json.gz

Run with:
    pytest tests/test_coordinator_assign.py -v
"""
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from coordinator.policies import AssignPolicy, Decider, DecisionRequest, decode
from coordinator.policies.assign import BIG, min_cost_matching, picker_matchings
from coordinator.spec import World, load_spec, observe

ROOT = Path(__file__).resolve().parent.parent
SPEC = load_spec(ROOT / "policy_models" / "imit_pointer_obs_tc-round_2")
DATA = json.loads(gzip.open(Path(__file__).parent / "data" / "assign_cases.json.gz", "rt").read())


def run_case(case, policy):
    world = World.from_fixture(case["world"], agent_speed=DATA["agent_speed"])
    for k, p in enumerate(world.pickers):          # unique robot ids, as in the coordinator
        p.id = f"picker_{k}"
    for k, t in enumerate(world.transporters):
        t.id = f"transporter_{k}"
    deciders, expected = [], {}
    for d in case["deciders"]:
        is_picker = d["role"] == "picker"
        rid = (world.pickers if is_picker else world.transporters)[d["agent"]].id
        deciders.append(Decider(rid, observe(world, SPEC, is_picker, d["agent"])))
        expected[rid] = d["action"]
    got = policy.decide(DecisionRequest(world=world, deciders=deciders))
    return world, got, expected


def test_same_actions_as_the_training_heuristic():
    """Every action as assign_policy.py chose it, except where a matching has several
    minimum-cost solutions: scipy and this port may then pick different ones, so a
    picker may differ only if the reference's pairs cost exactly as much as ours."""
    policy = AssignPolicy(SPEC)
    decisions = moving = tied = 0
    wrong = []
    for n, case in enumerate(DATA["cases"]):
        world, got, expected = run_case(case, policy)
        assert world.distance.missing == 0, f"case {n}: the recording lacks a path length"
        decisions += len(expected)
        moving += sum(1 for a in expected.values() if a)
        differ = [rid for rid in expected if got[rid] != expected[rid]]
        if not differ:
            continue
        if any(rid.startswith("transporter") for rid in differ) or not _tie(world, expected, got):
            wrong += [(n, rid, expected[rid], got[rid]) for rid in differ]
        else:
            tied += len(differ)
    assert moving > 300, "the recording should exercise the heuristic"
    assert not wrong, f"{len(wrong)} of {decisions} actions differ: {wrong[:10]}"
    assert tied < 0.1 * decisions, f"{tied} of {decisions} actions differ by ties"


def _tie(world, expected, got) -> bool:
    """Both matchings: the reference's pairs and ours are allowed and cost the same."""
    deciding = {("p", k) for k, p in enumerate(world.pickers) if p.id in expected}
    m = picker_matchings(world, SPEC.k, deciding=deciding)
    for rows, cost, verb, column_of in (
            (m["empty"], m["pick_cost"], "pick",
             {world.items[i].id: c for c, i in enumerate(m["available"])}),
            (m["holding"], m["handoff_cost"], "handoff", {t: t for t in range(len(world.transporters))})):
        totals = []
        for actions in (expected, got):
            pairs = []
            for r, p in enumerate(rows):
                rid = world.pickers[p].id
                choice = decode(SPEC, observe(world, SPEC, True, p), actions[rid])
                if choice.verb == verb:
                    pairs.append((r, column_of[choice.target]))
            if any(cost[r, c] >= BIG / 2 for r, c in pairs) or len({c for _, c in pairs}) < len(pairs):
                return False
            totals.append((len(pairs), sum(cost[r, c] for r, c in pairs)))
        if totals[0][0] != totals[1][0] or abs(totals[0][1] - totals[1][1]) > 1e-6:
            return False
    return True


def test_matching_is_minimal():
    rng = np.random.default_rng(7)
    for _ in range(300):
        n, m = rng.integers(1, 6, size=2)
        cost = rng.integers(0, 6, size=(n, m)).astype(float)
        cost[rng.random((n, m)) < 0.25] = BIG
        pairs = min_cost_matching(cost)
        assert len({r for r, _ in pairs}) == len(pairs) == len({c for _, c in pairs})
        assert all(cost[r, c] < BIG / 2 for r, c in pairs)
        best = _brute(cost)
        assert (len(pairs), sum(cost[r, c] for r, c in pairs)) == pytest.approx(best)


def _brute(cost):
    """(pairs, cost) of scipy's choice: the least total over the assignments of
    min(n, m) pairs, then the allowed (non-BIG) pairs kept."""
    n, m = cost.shape
    best_total, best_pairs = None, None
    if n <= m:
        options = (list(zip(range(n), cols)) for cols in itertools.permutations(range(m), n))
    else:
        options = (list(zip(rows, range(m))) for rows in itertools.permutations(range(n), m))
    for pairs in options:
        total = sum(cost[r, c] for r, c in pairs)
        if best_total is None or total < best_total:
            best_total, best_pairs = total, pairs
    kept = [(r, c) for r, c in best_pairs if cost[r, c] < BIG / 2]
    return len(kept), sum(cost[r, c] for r, c in kept)
