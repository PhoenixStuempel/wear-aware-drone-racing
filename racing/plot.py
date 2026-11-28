"""Training curves for a run, as PNGs in runs/<run>/plots/.

    python -m racing.plot runs/2026-11-26_2200_endless

One y axis per chart, so measures with different scales get their own panel.
"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRAIN, EVAL = SLOTS[0], SLOTS[1]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def read(path):
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: [] for k in rows[0]} if rows else {}
    for r in rows:
        for k, v in r.items():
            try:
                cols[k].append(float(v))
            except (TypeError, ValueError):
                cols[k].append(math.nan)
    return cols


def millions(xs):
    return [x / 1e6 for x in xs]


def finish(ax, title, ylabel=None):
    ax.set_title(title)
    ax.set_xlabel("environment steps (millions)")
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_run(run):
    up, ev = read(run / "updates.csv"), read(run / "eval.csv")
    seconds = json.loads((run / "config.json").read_text())["race"]["episode_seconds"]
    out = run / "plots"
    out.mkdir(exist_ok=True)
    x, xe = millions(up["env_steps"]), millions(ev.get("env_steps", []))

    # learning: return, gates, outcomes
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, up["mean_return"], color=TRAIN, label="training (sampled actions)")
    ax.plot(xe, ev["eval_mean_return"], color=EVAL, label="test (mean action, fixed courses)")
    finish(ax, "Return per episode")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(x, up["mean_gates"], color=TRAIN, label="training")
    ax.plot(xe, ev["eval_mean_gates"], color=EVAL, label="test")
    finish(ax, f"Gates passed per {seconds:g} s race")
    ax.legend(loc="upper left")
    ax = axes[2]
    ends = ["timeout", "floor", "wall", "gate_frame", "wrong_way"]
    names = [f"flew the full {seconds:g} s", "hit floor", "hit wall", "hit gate frame", "went in backwards"]
    # going in backwards was a crash only in a few runs on Nov 27, the others have no such column
    names = [n for n, e in zip(names, ends) if f"eval_end_{e}" in ev]
    ends = [e for e in ends if f"eval_end_{e}" in ev]
    ax.stackplot(
        xe,
        [ev[f"eval_end_{e}"] for e in ends],
        colors=SLOTS[: len(ends)],
        labels=names,
        edgecolor=SURFACE,
        lw=1,
    )
    finish(ax, "How test episodes end", "share of episodes")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(out / "learning.png", dpi=160)
    plt.close(fig)

    # reward broken into its terms (training episodes)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    terms = ["progress", "gate", "action_rate", "crash"]
    for k, t in enumerate(terms):
        ax.plot(x, up[f"r_{t}"], color=SLOTS[k], label=t.replace("_", " "))
    ax.axhline(0, color=INK_SOFT, lw=0.8)
    finish(ax, "Reward per episode, by term (training)")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(out / "reward_terms.png", dpi=160)
    plt.close(fig)

    # flying: speed, motor use, spin
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    axes[0].plot(x, up["mean_speed"], color=TRAIN, label="training")
    axes[0].plot(xe, ev["eval_mean_speed"], color=EVAL, label="test")
    finish(axes[0], "Mean speed", "m/s")
    axes[0].legend(loc="upper left")
    axes[1].plot(x, up["mean_command"], color=TRAIN, label="training")
    axes[1].plot(xe, ev["eval_mean_command"], color=EVAL, label="test")
    finish(axes[1], "Mean motor command", "fraction of max rotor speed")
    axes[1].legend(loc="upper left")
    rev = 1 / (2 * math.pi)
    axes[2].plot(x, [v * rev for v in up["mean_spin"]], color=TRAIN, label="training")
    axes[2].plot(xe, [v * rev for v in ev["eval_mean_spin"]], color=EVAL, label="test")
    finish(axes[2], "Mean spin (yaw rate)", "rev/s")
    axes[2].legend(loc="upper left")
    fig.savefig(out / "flying.png", dpi=160)
    plt.close(fig)

    # the trainer itself
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
    axes[0].plot(x, up["learning_rate"], color=TRAIN, label="learning rate")
    finish(axes[0], "Learning rate (adaptive)")
    axes[0].set_yscale("log")
    axes[1].plot(x, up["action_std"], color=TRAIN, label="action noise")
    finish(axes[1], "Exploration noise (action std)")
    axes[2].plot(x, up["loss_value"], color=TRAIN, label="value loss")
    finish(axes[2], "Value loss")
    axes[2].set_yscale("log")
    fig.savefig(out / "trainer.png", dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("plots in", plot_run(Path(arg)))
