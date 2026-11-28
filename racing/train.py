"""Train a racing policy with PPO. RSL-RL does the PPO maths and the networks, this loop does the
logging, evaluation and saving.

    python -m racing.train --name first --minutes 10
    python -m racing.train --name tuned --minutes 30 --from runs/<run>/checkpoints/model_3100.pt

Everything lands in runs/<date>_<name>/:
    config.json         every setting, the git commit, library versions, GPU
    updates.csv         one row per PPO update (trainer stats + episode averages)
    episodes.csv.gz     one row per finished training episode
    eval.csv            deterministic test on the same fixed courses every --eval-every updates
    checkpoints/        model_<iteration>.pt, and model_final.pt

Watch the result with: python -m racing.live runs/<run>
"""

import argparse
import csv
import gzip
import json
import math
import platform
import subprocess
import time
from dataclasses import replace
from datetime import datetime
from importlib import metadata
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, EmpiricalNormalization

from racing.env import END_NAMES, EPISODE_COLUMNS, REWARD_TERMS, RaceConfig, RaceEnv
from sim.params import DroneParams

REPO = Path(__file__).resolve().parents[1]

POLICY = dict(
    actor_hidden_dims=[128, 128], critic_hidden_dims=[128, 128], activation="elu", init_noise_std=0.5
)
PPO_SETTINGS = dict(
    num_learning_epochs=5,
    num_mini_batches=4,  # few big minibatches, the update is the slow part on a GPU
    clip_param=0.2,
    gamma=0.995,  # looks ~4 s ahead at 50 Hz (0.99 was ~2 s): wear only costs later on
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.005,
    learning_rate=1e-3,
    max_grad_norm=1.0,
    schedule="adaptive",  # learning rate follows the KL between old and new policy
    desired_kl=0.01,
)
STEPS_PER_ENV = 24  # per update, so 4096 envs -> ~98k samples per update
BOUNDS_COEF = 0.01  # keeps the policy's mean action inside [-1, 1], see BoundedActorCritic


class BoundedActorCritic(ActorCritic):
    """RSL-RL's actor-critic plus a loss on the mean action leaving [-1, 1].

    The env clamps actions to [-1, 1], so past the edge a mean of 1.5 and a mean of 50 fly the
    same. Nothing pulls the means back, they drift out, and the entropy bonus widens the Gaussian
    for free since most samples get clamped anyway. Noise went 0.2 -> 2+ and crashes 6% -> 18%
    over a 40 min run. Fix: add BOUNDS_COEF * sum(relu(|mu| - 1)^2) to the training loss. A loss
    on the policy's output, not a reward term, so the reward is untouched.

    RSL-RL's update does `loss -= entropy_coef * entropy`, so the penalty rides in on the entropy
    term (divided by entropy_coef) instead of copying its whole update function. Side effect:
    the "entropy" PPO reports includes it, so train() logs entropy and the penalty separately.
    """

    def __init__(self, *args, entropy_coef, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coef = entropy_coef

    @property
    def entropy(self):
        over = torch.relu(self.distribution.mean.abs() - 1.0).pow(2).sum(dim=-1)
        return super().entropy - BOUNDS_COEF / self.entropy_coef * over


def git_state():
    def run(*cmd):
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def episode_stats(eps, prefix="", hz=50):
    """Averages over a batch of finished episodes (rows as in EPISODE_COLUMNS). An empty batch
    still gives every key (as NaN) so the CSV columns never change."""
    if len(eps) == 0:
        return {k: float("nan") for k in episode_stats(torch.zeros(1, len(EPISODE_COLUMNS)), prefix, hz)}
    col = {name: eps[:, i] for i, name in enumerate(EPISODE_COLUMNS)}
    out = {
        "episodes": len(eps),
        "mean_return": col["return"].mean().item(),
        "mean_steps": col["steps"].mean().item(),
        "mean_gates": col["gates"].mean().item(),
        "max_gates": col["gates"].max().item(),
        "gates_per_second": (col["gates"] / (col["steps"] / hz)).mean().item(),
        "mean_speed": col["mean_speed"].mean().item(),
        "mean_command": col["mean_command"].mean().item(),
        "mean_spin": col["mean_spin"].mean().item(),  # rad/s, yaw
        "max_spin": col["max_spin"].mean().item(),
        "mean_life_used": col["life_used"].mean().item(),
    }
    for k, name in enumerate(END_NAMES):
        out[f"end_{name}"] = (col["end"] == k).float().mean().item()
    out["crash_rate"] = 1.0 - out["end_timeout"]  # timeout = flew the whole race
    for t in REWARD_TERMS:
        out[f"r_{t}"] = col[f"r_{t}"].mean().item()
    return {prefix + k: v for k, v in out.items()}


class CsvLog:
    """Appends dict rows; the header comes from the first row."""

    def __init__(self, path, compress=False):
        self.file = gzip.open(path, "wt", newline="") if compress else open(path, "w", newline="")
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row), extrasaction="ignore", restval="")
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


@torch.no_grad()
def evaluate(policy, normalizer, env, seed):
    """Mean action (no sampling) on the same courses every time. Each env's first episode only,
    otherwise quick crashers would count several times."""
    env.gen.manual_seed(seed)
    env._finished = []
    obs, _ = env.reset()
    for _ in range(env.max_episode_length):
        obs, _, _, _ = env.step(policy.act_inference(normalizer(obs)))
    first = {}
    for row in env.pop_episodes():  # time order, so setdefault keeps each env's first
        first.setdefault(int(row[0]), row)
    rows = torch.stack(list(first.values())) if first else torch.zeros(0, len(EPISODE_COLUMNS))
    return episode_stats(rows, prefix="eval_", hz=env.cfg.drone.control_hz)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="run")
    ap.add_argument("--minutes", type=float, default=10.0, help="stop after this much wall time")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wear-rate", type=float, default=0.0, help="life lost per second at full speed")
    ap.add_argument("--wear-sensor", action="store_true", help="agent sees each motor's life")
    ap.add_argument("--eval-every", type=int, default=50, help="updates between deterministic evals")
    ap.add_argument("--eval-envs", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--from", dest="start_from", help="checkpoint to carry on training from")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    cfg = RaceConfig(drone=replace(DroneParams(), wear_rate=args.wear_rate), wear_sensor=args.wear_sensor)
    env = RaceEnv(args.envs, cfg, device=device, seed=args.seed)
    EVAL_SEED = 10_000  # same eval courses for every run, so runs compare
    eval_env = RaceEnv(args.eval_envs, cfg, device=device, seed=EVAL_SEED)

    run = REPO / "runs" / f"{datetime.now():%Y-%m-%d_%H%M}_{args.name}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=False)
    config = {
        "args": vars(args),
        "race": cfg.to_dict(),
        "policy": POLICY,
        "ppo": PPO_SETTINGS,
        "steps_per_env": STEPS_PER_ENV,
        "bounds_coef": BOUNDS_COEF,
        "eval_seed": EVAL_SEED,
        "git": git_state(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "rsl_rl": metadata.version("rsl-rl-lib"),
        },
        "gpu": torch.cuda.get_device_name(0),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))

    obs, _ = env.get_observations()
    policy = BoundedActorCritic(
        obs.shape[1], obs.shape[1], env.num_actions, entropy_coef=PPO_SETTINGS["entropy_coef"], **POLICY
    ).to(device)
    normalizer = EmpiricalNormalization(shape=[obs.shape[1]], until=1.0e8).to(device)
    ppo = PPO(policy, device=device, **PPO_SETTINGS)
    ppo.init_storage("rl", env.num_envs, STEPS_PER_ENV, [obs.shape[1]], [obs.shape[1]], [env.num_actions])
    it = 0
    if args.start_from:
        # carry on from a saved model: weights, observation normaliser, optimiser, update count
        ckpt = torch.load(args.start_from, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        normalizer.load_state_dict(ckpt["normalizer"])
        ppo.optimizer.load_state_dict(ckpt["optimizer"])
        it = ckpt["iteration"]
        config["started_from"] = {"checkpoint": args.start_from, "iteration": it}
        (run / "config.json").write_text(json.dumps(config, indent=2))

    updates_log, eval_log = CsvLog(run / "updates.csv"), CsvLog(run / "eval.csv")
    episodes_log = CsvLog(run / "episodes.csv.gz", compress=True)
    print(f"run: {run}\nobs {obs.shape[1]}, envs {args.envs}, budget {args.minutes} min", flush=True)

    def save(name, it):
        torch.save(
            {
                "iteration": it,
                "policy": policy.state_dict(),
                "normalizer": normalizer.state_dict(),
                "optimizer": ppo.optimizer.state_dict(),
                "config": config,
            },
            run / "checkpoints" / name,
        )

    obs = normalizer(obs)
    start = time.time()
    while time.time() - start < args.minutes * 60:
        t0 = time.time()
        # rollout
        policy.train()
        normalizer.train()
        with torch.inference_mode():
            for _ in range(STEPS_PER_ENV):
                actions = ppo.act(obs, obs)
                obs, rewards, dones, extras = env.step(actions)
                obs = normalizer(obs)
                ppo.process_env_step(rewards, dones, extras)
            ppo.compute_returns(obs)
            mu = ppo.storage.mu  # the policy's mean actions this rollout, before update() clears it
            over = torch.relu(mu.abs() - 1.0)
            drift = {
                "mean_abs_mu": mu.abs().mean().item(),
                "share_mu_out": (over > 0).float().mean().item(),  # means past the [-1, 1] clamp
                "loss_bounds": BOUNDS_COEF * over.pow(2).sum(-1).mean().item(),
            }
        t1 = time.time()
        losses = ppo.update()
        t2 = time.time()

        eps = env.pop_episodes()
        for row in eps.tolist():
            episodes_log.write({"iteration": it, **dict(zip(EPISODE_COLUMNS, row))})
        row = {
            "iteration": it,
            "wall_seconds": round(t2 - start, 2),
            "env_steps": (it + 1) * STEPS_PER_ENV * args.envs,
            "steps_per_second": round(STEPS_PER_ENV * args.envs / (t2 - t0)),
            "collect_seconds": round(t1 - t0, 3),
            "learn_seconds": round(t2 - t1, 3),
            "learning_rate": ppo.learning_rate,
            "action_std": policy.std.mean().item(),  # exploration noise, shrinks as it learns
            "loss_value": losses["value_function"],
            "loss_surrogate": losses["surrogate"],
            # Gaussian entropy from the std (PPO's own number has the bounds loss mixed in)
            "entropy": (0.5 * torch.log(2 * math.pi * math.e * policy.std.detach() ** 2)).sum().item(),
            **drift,
            **episode_stats(eps, hz=cfg.drone.control_hz),
        }
        updates_log.write(row)

        if it % args.eval_every == 0:
            policy.eval()
            normalizer.eval()
            ev = evaluate(policy, normalizer, eval_env, EVAL_SEED)
            eval_log.write(
                {"iteration": it, "env_steps": row["env_steps"], "wall_seconds": row["wall_seconds"], **ev}
            )
            print(
                f"it {it:5d}  {row['wall_seconds']:7.0f}s  {row['steps_per_second'] / 1e3:5.0f}k steps/s  "
                f"train: return {row.get('mean_return', float('nan')):6.1f} gates {row.get('mean_gates', 0):4.2f} "
                f"crash {row.get('crash_rate', float('nan')):.2f}   eval: gates {ev.get('eval_mean_gates', 0):5.2f} "
                f"({ev.get('eval_gates_per_second', 0):.2f}/s) crash {ev.get('eval_crash_rate', float('nan')):.2f}",
                flush=True,
            )
        if it % args.save_every == 0:
            save(f"model_{it}.pt", it)
        it += 1

    save("model_final.pt", it)
    for log in (updates_log, eval_log, episodes_log):
        log.close()
    print(f"done: {it} updates in {(time.time() - start) / 60:.1f} min -> {run}", flush=True)


if __name__ == "__main__":
    main()
