"""Reference cases for the coordinator's assign port, from marl-couliglig's own assign_policy.

Drives PickerTransporterEnv1 episodes with AssignActor (lam=1, beta=1, phase on, wait=1) and,
at decision steps, records the env state in coordinator.spec.World's fixture format
(couliglig-gamesim tools/export_policy.world_of) with ALL path lengths between the cells
the heuristic and the observer read, plus every free agent's reference action.

    ~/marl-couliglig/.venv/bin/python gen_assign_cases.py OUT.json.gz [--states 120]
"""
import argparse
import contextlib
import gzip
import io
import json
import os
import sys

GAMESIM = os.path.expanduser("~/couliglig-gamesim")
CHECKPOINT = os.path.expanduser("~/marl-couliglig/results/imit_pointer_obs_tc/round_2")
sys.path.insert(0, os.path.join(GAMESIM, "tools"))
import export_policy as ep  # noqa: E402  (chdirs into marl-couliglig, puts it on sys.path)
import numpy as np  # noqa: E402
from ray.rllib.policy.policy import Policy  # noqa: E402
from assign_policy import AssignActor  # noqa: E402
from environments.picker_transporter_env_1 import PickerTransporterEnv1  # noqa: E402


def all_pairs(env, world):
    cells = set()
    for it in world["items"]:
        cells.add(tuple(it["cell"]))
        cells.add(tuple(it["recv_cell"]))
    for z in world["zones"]:
        cells.add(tuple(z["cell"]))
    for a in world["pickers"] + world["transporters"]:
        cells.add(tuple(a["cell"]))
        cells.add(tuple(a["dest"]))
    cells = sorted(cells)
    return ep._dist_block(env, cells, cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=31_000)
    args = ap.parse_args()

    policy = Policy.from_checkpoint(os.path.join(CHECKPOINT, "policies", "picker_policy"))
    env_config = dict(policy.config["env_config"])
    env = PickerTransporterEnv1(dict(env_config, render_mode="none", specific_map=None))
    actor = AssignActor(env)
    per_episode = -(-args.states // args.episodes)
    cases = []
    for e in range(args.episodes):
        np.random.seed(args.seed + e)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset()
        taken, done, step = 0, False, 0
        while not done:
            actions = actor.joint(obs)
            # spread the samples over the episode: every 3rd step with a decision
            if taken < per_episode and any(actions.values()) and step % 3 == 0:
                try:
                    world = ep.world_of(env)
                except ValueError:
                    world = None
                if world is not None:
                    for k, a in enumerate(env.transporter_ids):
                        world["transporters"][k]["busy"] = bool(env._is_busy(a))
                    world["dist"] = [all_pairs(env, world)]
                    deciders = []
                    for aid, a in actions.items():
                        role = "picker" if aid in env.pickers else "transporter"
                        ids = env.picker_ids if role == "picker" else env.transporter_ids
                        deciders.append({"role": role, "agent": ids.index(aid), "action": int(a)})
                    cases.append({"episode": e, "step": step, "world": world, "deciders": deciders})
                    taken += 1
            with contextlib.redirect_stdout(io.StringIO()):
                obs, _, term, trunc, _ = env.step(actions)
            step += 1
            done = term["__all__"] or trunc["__all__"]
        print(f"episode {e}: {taken} states ({len(env.picker_ids)} pickers, "
              f"{len(env.transporter_ids)} transporters, {len(env.waiting_zones)} zones)", flush=True)
        if len(cases) >= args.states:
            break
    meta = {"source": "marl-couliglig assign_policy.AssignActor (lam=1, beta=1, phase=1, wait=1)",
            "checkpoint_env": CHECKPOINT, "agent_speed": float(env.agent_speed), "cases": cases}
    with gzip.open(args.out, "wt") as f:
        json.dump(meta, f)
    print(f"{len(cases)} states -> {args.out}")


if __name__ == "__main__":
    main()
