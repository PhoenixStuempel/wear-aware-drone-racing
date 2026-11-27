"""Endless random gate courses, made in a batch (one per env). Sim axes: z up, metres.

Every gate is at a random spot in the box (kept PADDING in from the walls, floor and ceiling) and
rolled to a random angle. It faces the gate before it: you fly through it heading away from the
last one, so you always arrive at the front, never have to turn round a gate, and the course is
random but smooth. That can be steeply up or down. The first gate faces a random way and the drone
starts straight in front of it. A gate sits at least MIN_GATE_GAP from the other gates, which also
means two gates can never overlap. Only a few gates exist at a time; passing one makes a new one.
"""

import torch

BOX_HALF = 5.0  # m, the box is 10 x 10 x 10 with the floor at z = 0
PADDING = 1.5  # m, gate centres stay this far from the walls, floor and ceiling
MIN_GATE_GAP = 2.5  # m between gate centres. Over 2.12 (two frame corners touching) so no overlaps
AHEAD = 3  # gates kept ahead of the drone
TRIES = 32  # random gates drawn at once per env, the first one that follows the rules is used

# gate size, matches viewer/gates.js. The drone counts as a point plus DRONE_RADIUS
GATE_OPEN_HALF = 0.6  # m, half the opening
GATE_FRAME_HALF = 0.75  # m, half the outer frame
DRONE_RADIUS = 0.2  # m, a bit generous: centre to prop tip is ~0.27


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def in_box(p, margin):
    """Is p (..., 3) at least margin inside the box (floor and ceiling too)?"""
    return (
        (p[..., :2].abs() <= BOX_HALF - margin).all(-1)
        & (p[..., 2] >= margin)
        & (p[..., 2] <= 2 * BOX_HALF - margin)
    )


def random_gate(others, generator=None, start_back=None, prev=None):
    """One new gate per env, at least MIN_GATE_GAP from every gate in others (n, k, 3).

    Draws TRIES gates at once and keeps the first that follows the rules: same as redrawing
    until one fits, but with no loop, so the GPU never waits. If none fit (rare), the one
    furthest from the others is used (for a first gate: the first one drawn).

    prev (n, 3): the gate before it. The new gate faces it (way through = from prev to the new
    centre). Without prev, a random facing: a course's first gate.

    start_back (n,) is for a course's first gate: the drone starts that far in front of it, so
    that point has to be in the box too.

    Returns centre, direction (the way through) and across (along one side of the square),
    each (n, 3).
    """
    n, k, device = others.shape[0], others.shape[1], others.device
    shape = (n, TRIES, 3)

    # where: uniform in the padded box
    lo = torch.tensor([-BOX_HALF + PADDING, -BOX_HALF + PADDING, PADDING], device=device)
    hi = torch.tensor([BOX_HALF - PADDING, BOX_HALF - PADDING, 2 * BOX_HALF - PADDING], device=device)
    centre = lo + (hi - lo) * torch.rand(shape, device=device, generator=generator)
    # facing: away from the gate before, or for a first gate a random direction (a normal
    # distribution is the same every way, so this is uniform)
    if prev is None:
        direction = unit(torch.randn(shape, device=device, generator=generator))
    else:
        direction = unit(centre - prev.unsqueeze(1))
    # roll: another random direction with its part along `direction` taken out
    across = torch.randn(shape, device=device, generator=generator)
    across = unit(across - (across * direction).sum(-1, keepdim=True) * direction)

    # the rules
    if k:
        gap = (centre.unsqueeze(2) - others.unsqueeze(1)).norm(dim=-1)  # (n, TRIES, k)
        nearest = gap.min(-1).values
        ok = nearest >= MIN_GATE_GAP
    else:  # no other gates yet
        nearest = torch.zeros(n, TRIES, device=device)
        ok = torch.ones(n, TRIES, dtype=torch.bool, device=device)
    if start_back is not None:
        ok &= in_box(centre - start_back.view(n, 1, 1) * direction, 0.8)

    # first try that's ok (argmax picks the first of equal values), else the furthest one
    pick = torch.where(ok, torch.inf, nearest).argmax(dim=1)
    take = pick.view(n, 1, 1).expand(n, 1, 3)
    return tuple(t.gather(1, take).squeeze(1) for t in (centre, direction, across))


def first_gates(generator, start_back, n, device):
    """A course's first AHEAD gates. Returns centres, directions, acrosses, each (n, AHEAD, 3)."""
    gates = [random_gate(torch.zeros(n, 0, 3, device=device), generator, start_back)]
    for _ in range(AHEAD - 1):
        so_far = torch.stack([g[0] for g in gates], dim=1)
        gates.append(random_gate(so_far, generator, prev=gates[-1][0]))
    return tuple(torch.stack(parts, dim=1) for parts in zip(*gates))


def check_gate(prev_pos, pos, centre, direction, across):
    """Did the drone cross the gate's plane (going the right way) this step, and where?
    Returns (passed, hit_frame), both bool (n,). Going through backwards counts for nothing."""
    before = ((prev_pos - centre) * direction).sum(-1)
    after = ((pos - centre) * direction).sum(-1)
    crossed = (before < 0) & (after >= 0)  # front to back, the right way

    # where on the plane it crossed: interpolate between the two positions. before == after only
    # when it didn't cross, the nan/inf from that gets clamped away
    t = (before / (before - after)).nan_to_num(0.0).clamp(0, 1).unsqueeze(-1)
    hit = prev_pos + t * (pos - prev_pos) - centre
    up = torch.linalg.cross(direction, across)  # the gate's other side
    off = torch.maximum((hit * across).sum(-1).abs(), (hit * up).sum(-1).abs())  # square opening

    passed = crossed & (off < GATE_OPEN_HALF - DRONE_RADIUS)
    hit_frame = crossed & ~passed & (off < GATE_FRAME_HALF + DRONE_RADIUS)
    return passed, hit_frame
