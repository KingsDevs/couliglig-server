"""
Phase 2a: the coordinator's decision layer, ported from couliglig-gamesim.

The fixtures (policy_models/<model>/fixtures.json, written by gamesim's
tools/export_policy.py from marl-couliglig's PickerTransporterEnv1) hold real
env observations with the world state behind them, the checkpoint's logits
and the greedy / stay actions. Everything here is checked against them.

Run with:
    pytest tests/test_coordinator_policy.py -v
"""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from coordinator.policies import (
    Choice, Decider, DecisionRequest, FallbackPolicy, GreedyPolicy, MarlOnnxPolicy, ModelVerificationError,
    StayPolicy, decode, make_policy, masked_argmax, verify_model_dir,
)
from coordinator.policies.heuristics import greedy_action, stay_action
from coordinator.spec import Item, Picker, Transporter, World, Zone, load_spec, observe
from coordinator.spec.obs import nearest

MODEL_DIR = Path(__file__).resolve().parent.parent / "policy_models" / "imit_pointer_obs_tc-round_2"
SPEC = load_spec(MODEL_DIR)
FIXTURES = json.loads((MODEL_DIR / "fixtures.json").read_text())
CASES = [(role, n) for role in ("picker", "transporter") for n in range(len(FIXTURES[role]))]


def _fixture(role, n):
    return FIXTURES[role][n]


def _observe_fixture(role, n):
    f = _fixture(role, n)
    world = World.from_fixture(f["world"], SPEC.agent_speed)
    return f, world, observe(world, SPEC, role == "picker", f["agent"])


@pytest.fixture(scope="module")
def marl():
    return MarlOnnxPolicy(MODEL_DIR)


# ===========================================================================
# Observer
# ===========================================================================

class TestObserver:

    def test_fixtures_cover_both_roles(self):
        assert len(FIXTURES["picker"]) >= 24 and len(FIXTURES["transporter"]) >= 24

    @pytest.mark.parametrize("role,n", CASES)
    def test_rebuilds_fixture(self, role, n):
        f, world, built = _observe_fixture(role, n)
        assert world.distance.missing == 0
        for name, want in f["obs"].items():
            got = built.obs[name].reshape(-1)
            np.testing.assert_allclose(got, np.asarray(want, dtype=np.float32), atol=1e-5, rtol=0,
                                       err_msg=f"{role} #{n} {name}")
        assert {g: [int(t) for t in v] for g, v in built.targets.items()} == f["targets"]

    def test_shapes_match_policy_json(self):
        _, _, built = _observe_fixture("transporter", 0)
        for name, shape in SPEC.inputs["transporter"]:
            assert built.obs[name].shape == shape

    def test_nearest_orders_like_env(self):
        d = [3.0, math.inf, 1.0, 1.0, 0.5]
        assert nearest(d, 3) == [4, 2, 3]
        assert nearest(d, 9) == [4, 2, 3, 0]          # inf dropped
        assert nearest(d, 3, [False, True, False, True, False]) == [3, 4, 2]

    def test_nearest_breaks_ties_at_the_cutoff_by_index(self):
        assert nearest([1.0, 2.0, 2.0, 2.0, 0.0], 3) == [4, 0, 1]

    def test_unreachable_entities_are_skipped_and_scaled(self):
        far = (9.0, 9.0)
        world = World(
            items=[Item(id="a", cell=(0, 3), w=1.0, recv_cell=(0, 9)), Item(id="b", cell=far, w=1.0, recv_cell=(0, 9))],
            pickers=[Picker(id="p", cell=(0, 0), dest=(0, 0))],
            transporters=[Transporter(id="t", cell=(0, 5), dest=(0, 5), cap=4, zone=True)],
            zones=[Zone(id="z", cell=(0, 5), occ=True)],
            distance=lambda a, b: math.inf if far in (a, b) else abs(a[0] - b[0]) + abs(a[1] - b[1]),
        )
        built = observe(world, SPEC, True, 0)
        assert built.targets["items"] == ["a"]
        assert built.obs["items_mask"].tolist()[:2] == [1.0, 0.0]
        assert built.obs["items"][0, 0] == pytest.approx(3 / 53)

    def test_feasibility_narrows_masks(self):
        f = _fixture("picker", 0)
        world = World.from_fixture(f["world"], SPEC.agent_speed)

        class NothingPickable:
            def can_pick(self, picker, item): return False
            def can_hand_to(self, picker, transporter): return True
            def can_enter_zone(self, transporter, zone): return True
            def can_deliver(self, transporter, item): return True

        world.feasibility = NothingPickable()
        built = observe(world, SPEC, True, f["agent"])
        assert built.obs["action_mask"][1:1 + SPEC.k["k_items"]].sum() == 0
        # the table itself is unchanged: only legality is narrowed
        np.testing.assert_allclose(built.obs["items"].reshape(-1), f["obs"]["items"], atol=1e-5)


# ===========================================================================
# Heuristics
# ===========================================================================

class TestHeuristics:

    @pytest.mark.parametrize("role,n", CASES)
    def test_match_env_heuristics(self, role, n):
        f, _, built = _observe_fixture(role, n)
        assert greedy_action(built, SPEC.k) == f["greedy"]
        assert stay_action(built, SPEC.k) == f["stay"]

    def test_policies_return_legal_actions(self):
        deciders = [Decider(f"{role}{n}", _observe_fixture(role, n)[2]) for role, n in CASES]
        req = DecisionRequest(world=None, deciders=deciders)
        for policy in (GreedyPolicy(SPEC), StayPolicy(SPEC)):
            actions = policy.decide(req)
            for d in deciders:
                assert d.observation.obs["action_mask"][actions[d.robot_id]] > 0.5


# ===========================================================================
# Trained policy (ONNX)
# ===========================================================================

class TestMarlOnnx:

    def test_model_dir_verifies(self, marl):
        assert marl.report.ok
        assert marl.report.fixtures == len(CASES)
        assert marl.report.max_logit_error < 1e-4

    def test_decides_checkpoint_argmax_from_rebuilt_observations(self, marl):
        deciders, expected = [], {}
        for role, n in CASES:
            f, _, built = _observe_fixture(role, n)
            rid = f"{role}{n}"
            deciders.append(Decider(rid, built))
            expected[rid] = masked_argmax(np.asarray(f["logits"]), built.obs["action_mask"])
        assert marl.decide(DecisionRequest(world=None, deciders=deciders)) == expected

    def test_sampling_stays_legal(self):
        policy = MarlOnnxPolicy(MODEL_DIR, deterministic=False, seed=3, verify=False)
        deciders = [Decider(f"{role}{n}", _observe_fixture(role, n)[2]) for role, n in CASES]
        actions = policy.decide(DecisionRequest(world=None, deciders=deciders))
        for d in deciders:
            assert d.observation.obs["action_mask"][actions[d.robot_id]] > 0.5

    def test_refuses_dir_whose_observer_check_fails(self, tmp_path):
        bad = tmp_path / "model"
        shutil.copytree(MODEL_DIR, bad)
        fixtures = json.loads((bad / "fixtures.json").read_text())
        fixtures["picker"][0]["world"]["pickers"][fixtures["picker"][0]["agent"]]["hold"] ^= True
        (bad / "fixtures.json").write_text(json.dumps(fixtures))
        with pytest.raises(ModelVerificationError, match="observer"):
            MarlOnnxPolicy(bad)

    def test_refuses_dir_whose_logits_differ(self, tmp_path):
        bad = tmp_path / "model"
        shutil.copytree(MODEL_DIR, bad)
        fixtures = json.loads((bad / "fixtures.json").read_text())
        f = fixtures["transporter"][0]
        legal = [i for i, m in enumerate(f["obs"]["action_mask"]) if m > 0.5]
        f["logits"][legal[0]] += 1.0
        (bad / "fixtures.json").write_text(json.dumps(fixtures))
        report = verify_model_dir(bad)
        assert not report.ok and report.logit_failures

    def test_refuses_fixtures_without_world(self, tmp_path):
        bad = tmp_path / "model"
        shutil.copytree(MODEL_DIR, bad)
        fixtures = json.loads((bad / "fixtures.json").read_text())
        del fixtures["picker"][0]["world"]
        (bad / "fixtures.json").write_text(json.dumps(fixtures))
        with pytest.raises(ModelVerificationError, match="no world"):
            MarlOnnxPolicy(bad)


# ===========================================================================
# Decoding
# ===========================================================================

class TestDecode:

    def test_picker_layout(self):
        f, _, built = _observe_fixture("picker", 0)
        k_items = SPEC.k["k_items"]
        assert decode(SPEC, built, 0) == Choice("idle")
        assert decode(SPEC, built, 1) == Choice("pick", built.targets["items"][0])
        if built.targets["transporters"]:
            assert decode(SPEC, built, 1 + k_items) == Choice("handoff", built.targets["transporters"][0])

    def test_transporter_layout(self):
        for n in range(len(FIXTURES["transporter"])):
            _, _, built = _observe_fixture("transporter", n)
            if built.targets["carried"]:
                k_zones = SPEC.k["k_zones"]
                assert decode(SPEC, built, 1) == Choice("zone", built.targets["zones"][0])
                assert decode(SPEC, built, 1 + k_zones) == Choice("deliver", built.targets["carried"][0])
                return
        pytest.skip("no fixture with a carried item")

    def test_empty_row_and_out_of_range_rejected(self):
        _, _, built = _observe_fixture("picker", 0)
        with pytest.raises(ValueError):
            decode(SPEC, built, SPEC.action_dim("picker"))
        empty_row = len(built.targets["items"]) + 1
        if empty_row <= SPEC.k["k_items"]:
            with pytest.raises(ValueError, match="empty row"):
                decode(SPEC, built, empty_row)


# ===========================================================================
# Fallback
# ===========================================================================

class _Fixed:
    def __init__(self, name, action=None, error=None, delay=0.0):
        self.name, self.action, self.error, self.delay = name, action, error, delay

    def decide(self, req):
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return {d.robot_id: self.action for d in req.deciders}


def _request():
    deciders = [Decider(f"picker{n}", _observe_fixture("picker", n)[2]) for n in range(3)]
    return DecisionRequest(world=None, deciders=deciders)


class TestFallback:

    def test_primary_answer_used(self):
        policy = FallbackPolicy(_Fixed("p", action=0), GreedyPolicy(SPEC))
        assert set(policy.decide(_request()).values()) == {0}
        assert policy.last_substitutions == []

    def test_error_falls_back(self):
        req = _request()
        policy = FallbackPolicy(_Fixed("p", error=RuntimeError("boom")), GreedyPolicy(SPEC))
        assert policy.decide(req) == GreedyPolicy(SPEC).decide(req)
        assert all("boom" in s.reason for s in policy.last_substitutions)

    def test_illegal_index_falls_back(self):
        req = _request()
        policy = FallbackPolicy(_Fixed("p", action=999), StayPolicy(SPEC))
        assert policy.decide(req) == StayPolicy(SPEC).decide(req)
        assert all("illegal" in s.reason for s in policy.last_substitutions)

    def test_timeout_falls_back(self):
        req = _request()
        policy = FallbackPolicy(_Fixed("p", action=0, delay=0.5), GreedyPolicy(SPEC), timeout_s=0.05)
        start = time.monotonic()
        assert policy.decide(req) == GreedyPolicy(SPEC).decide(req)
        assert time.monotonic() - start < 0.4
        assert all("timeout" in s.reason for s in policy.last_substitutions)

    def test_make_policy_from_config(self):
        policy, spec = make_policy({"primary": "marl_onnx", "fallback": "greedy", "decide_timeout_ms": 150,
                                    "marl_onnx": {"model_dir": str(MODEL_DIR)}})
        assert isinstance(policy, FallbackPolicy)
        assert policy.name == "marl_onnx+greedy" and policy.timeout_s == pytest.approx(0.15)
        assert spec.k == SPEC.k


# ===========================================================================
# policy.json
# ===========================================================================

class TestSpec:

    def test_reads_export(self):
        assert SPEC.obs_features == {"competition", "timing"}
        assert SPEC.action_dim("picker") == 1 + SPEC.k["k_items"] + SPEC.k["k_transporters"]
        assert SPEC.agent_speed == 5

    def test_rejects_features_the_observer_cannot_build(self, tmp_path):
        meta = json.loads((MODEL_DIR / "policy.json").read_text())
        meta["obs_features"] = ["timing", "zone_demand"]
        (tmp_path / "policy.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="zone_demand"):
            load_spec(tmp_path)

    def test_shape_mismatch_is_caught(self, tmp_path):
        meta = json.loads((MODEL_DIR / "policy.json").read_text())
        for entry in meta["inputs"]["picker"]:
            if entry[0] == "items":
                entry[1] = [12, 6]
        (tmp_path / "policy.json").write_text(json.dumps(meta))
        bad_spec = load_spec(tmp_path)
        f = _fixture("picker", 0)
        with pytest.raises(ValueError, match="items"):
            observe(World.from_fixture(f["world"]), bad_spec, True, f["agent"])
