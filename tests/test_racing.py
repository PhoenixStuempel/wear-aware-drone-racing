import math

import pytest
import torch

from racing.course import AHEAD, MIN_GATE_GAP, PADDING, check_gate, first_gates, in_box, random_gate
from racing.env import END_NAMES, EPISODE_COLUMNS, RaceEnv
from racing.train import POLICY, BoundedActorCritic

CENTRE = torch.tensor([[0.0, 0.0, 2.0]])
FORWARD = torch.tensor([[1.0, 0.0, 0.0]])
SIDE = torch.tensor([[0.0, 1.0, 0.0]])


# a step from x = -0.1 to x = +0.1 through a gate at the origin facing +x, at different offsets
@pytest.mark.parametrize(
    "offset, passed, hit",
    [
        ((0.0, 0.0), True, False),  # straight through the middle
        ((0.3, 0.0), True, False),  # off centre but the drone still fits
        ((0.0, 0.5), False, True),  # props would clip the frame
        ((0.7, 0.0), False, True),  # into the frame itself
        ((2.0, 0.0), False, False),  # well clear, nothing there
    ],
)
def test_gate_pass_and_frame(offset, passed, hit):
    y, z = offset
    before = CENTRE + torch.tensor([[-0.1, y, z]])
    after = CENTRE + torch.tensor([[0.1, y, z]])
    p, h = check_gate(before, after, CENTRE, FORWARD, SIDE)
    assert bool(p) == passed and bool(h) == hit


@pytest.mark.parametrize("y", [0.0, 0.7, 2.0])
def test_gate_backwards_counts_for_nothing(y):
    # back to front, through the opening, the frame, or well clear: not a pass, not a crash
    p, h = check_gate(
        CENTRE + torch.tensor([[0.1, y, 0]]), CENTRE + torch.tensor([[-0.1, y, 0]]), CENTRE, FORWARD, SIDE
    )
    assert not p and not h


def test_gate_checked_where_it_crossed():
    # a big sideways step: starts 0.9 m off to the side, crosses the plane 0.1 m off centre.
    # Checking the start point instead of the crossing point would call it a frame hit
    before = CENTRE + torch.tensor([[-0.1, 0.9, 0]])
    after = CENTRE + torch.tensor([[0.1, -0.7, 0]])
    p, h = check_gate(before, after, CENTRE, FORWARD, SIDE)
    assert p and not h


def test_rolled_gate():
    # 0.5 m to the side hits an upright gate's frame, but rolled 45 deg the opening's corner is there
    before, after = CENTRE + torch.tensor([[-0.1, 0.5, 0]]), CENTRE + torch.tensor([[0.1, 0.5, 0]])
    rolled = torch.tensor([[0.0, 1.0, 1.0]]) / math.sqrt(2)
    assert not check_gate(before, after, CENTRE, FORWARD, SIDE)[0]
    assert check_gate(before, after, CENTRE, FORWARD, rolled)[0]


def test_gate_facing_up():
    # flying straight up through a flat gate counts, falling down through it doesn't
    up, side = torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]])
    below, above = CENTRE - 0.1 * up, CENTRE + 0.1 * up
    assert check_gate(below, above, CENTRE, up, side)[0]
    assert not check_gate(above, below, CENTRE, up, side)[0]


def test_courses_follow_the_rules():
    # 2000 courses, 30 gates each, made the way the env makes them (first few, then one at a time
    # against the gates still in the window)
    g = torch.Generator().manual_seed(0)
    n = 2000
    back = 2.0 + 2.0 * torch.rand(n, generator=g)
    pos, dirs, across = first_gates(g, back, n, "cpu")
    pos, dirs, across = list(pos.unbind(1)), list(dirs.unbind(1)), list(across.unbind(1))
    for _ in range(27):
        c, d, a = random_gate(torch.stack(pos[-AHEAD:], dim=1), g, prev=pos[-1])
        pos.append(c), dirs.append(d), across.append(a)
    pos, dirs, across = (torch.stack(x, dim=1) for x in (pos, dirs, across))

    assert in_box(pos, PADDING - 1e-6).all()
    assert in_box(pos[:, 0] - back.unsqueeze(1) * dirs[:, 0], 0.8 - 1e-6).float().mean() > 0.999  # start
    ones = torch.ones(n, 30)
    assert torch.allclose(dirs.norm(dim=-1), ones, atol=1e-5)
    assert torch.allclose(across.norm(dim=-1), ones, atol=1e-5)
    assert (dirs * across).sum(-1).abs().max() < 1e-4  # the side lies in the gate's plane

    # each gate is clear of the ones that were in the window when it was made
    close = 0
    for i in range(1, 30):
        gap = (pos[:, i : i + 1] - pos[:, max(i - AHEAD, 0) : i]).norm(dim=-1).min(dim=1).values
        close += (gap < MIN_GATE_GAP).sum()
    assert close <= 2  # TRIES draws can all miss, very rarely

    # every gate after the first faces the one before: you arrive at its front
    towards = (pos[:, 1:] - pos[:, :-1]) / (pos[:, 1:] - pos[:, :-1]).norm(dim=-1, keepdim=True)
    assert torch.allclose(dirs[:, 1:], towards, atol=1e-5)

    # really random: some gates face nearly straight up or down, some are near the walls
    assert (dirs[..., 2].abs() > 0.95).any()
    assert (pos[..., :2].abs().max(-1).values > 3.3).any()


def test_env_basics():
    env = RaceEnv(64, device="cpu", seed=0)
    obs, _ = env.get_observations()
    assert obs.shape == (64, 29) and torch.isfinite(obs).all()

    # hovering (all zero actions) earns about nothing and doesn't crash for a while
    total = torch.zeros(64)
    for _ in range(50):
        obs, reward, done, _ = env.step(torch.zeros(64, 4))
        total += reward
    assert not done.any()
    assert total.abs().max() < 1.0

    # random flailing ends episodes, and each one is logged with a known ending
    for _ in range(300):
        env.step(torch.randn(64, 4))
    eps = env.pop_episodes()
    assert len(eps) > 0 and eps.shape[1] == len(EPISODE_COLUMNS)
    assert set(eps[:, 1].long().tolist()) <= set(range(len(END_NAMES)))


def test_passing_a_gate_moves_the_course_on():
    env = RaceEnv(1, device="cpu", seed=3)
    old = env.gate_pos[0].clone()
    through = env.gate_dir[0, 0]
    env.drones.state[0, 0:3] = old[0] - 0.05 * through  # just in front of the next gate
    env.drones.state[0, 3:6] = 5.0 * through  # flying straight through it
    _, reward, done, _ = env.step(torch.zeros(1, 4))
    assert env.gates_passed[0] == 1 and not done[0]
    assert torch.allclose(env.gate_pos[0, :2], old[1:])  # the rest moved up, a new one at the end
    assert (env.gate_pos[0, 2] - old).norm(dim=-1).min() >= MIN_GATE_GAP
    assert reward[0] > 4.0  # the gate bonus


def test_bounds_loss_only_past_the_edge():
    # the penalty rides in on the entropy term: no change while the means are inside [-1, 1],
    # lower "entropy" (so a higher loss) once they're outside
    torch.manual_seed(0)
    policy = BoundedActorCritic(29, 29, 4, entropy_coef=0.005, **POLICY)
    obs = torch.randn(64, 29)
    policy.update_distribution(obs)
    plain = policy.distribution.entropy().sum(-1)
    inside = policy.distribution.mean.abs().max() <= 1
    assert inside and torch.allclose(policy.entropy, plain)

    with torch.no_grad():
        policy.actor[-1].bias += 3.0  # push every mean to ~3
    policy.update_distribution(obs)
    plain = policy.distribution.entropy().sum(-1)
    assert (policy.entropy < plain - 10).all()  # 0.01 / 0.005 * 4 motors * ~2^2 = ~32 each
