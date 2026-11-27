"""Gate race environment, many races at once on the GPU. Follows RSL-RL's VecEnv interface.

A race is 20 s on an endless random course: as many gates as it can pass. The drone starts
hovering a few metres in front of the first gate, and the episode ends at the 20 s mark or on a
crash.
Reward has no wear term on purpose.
"""

from dataclasses import asdict, dataclass, field

import torch
from rsl_rl.env import VecEnv

from racing.course import AHEAD, BOX_HALF, check_gate, first_gates, random_gate
from sim import quat
from sim.drones import Drones
from sim.params import DroneParams


@dataclass(frozen=True)
class Rewards:
    progress: float = 2.0  # per metre closer to the next gate
    gate: float = 5.0  # per gate passed
    action_rate: float = -0.005  # times |a - a_prev|^2, at most 16 so -0.08 per step
    crash: float = -5.0  # mild on purpose, big crash penalties taught "give up" in my old projects
    clamp: float = 100.0  # per step total


@dataclass(frozen=True)
class RaceConfig:
    episode_seconds: float = 20.0  # was 10 until Nov 27: longer races leave room to pace wear
    wear_sensor: bool = False  # agent sees each motor's life
    action_scale: float = 0.55  # action 1 -> hover + 0.55 of max rotor speed
    start_distance: tuple[float, float] = (2.0, 4.0)  # m before the first gate
    drone: DroneParams = field(default_factory=DroneParams)
    rewards: Rewards = field(default_factory=Rewards)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Back from to_dict (eg out of a checkpoint)."""
        d = dict(d)
        drone = dict(d.pop("drone"))
        drone["inertia"] = tuple(drone["inertia"])
        d["start_distance"] = tuple(d["start_distance"])
        return cls(drone=DroneParams(**drone), rewards=Rewards(**d.pop("rewards")), **d)


# why an episode ended, stored per episode. "timeout" = flew the whole race without crashing
END_TIMEOUT, END_FLOOR, END_WALL, END_GATE_FRAME, END_NAN = range(5)
END_NAMES = ["timeout", "floor", "wall", "gate_frame", "nan"]
REWARD_TERMS = ["progress", "gate", "action_rate", "crash"]


class RaceEnv(VecEnv):
    def __init__(self, num_envs: int, cfg: RaceConfig = RaceConfig(), device="cuda", seed: int = 0):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_actions = 4
        self.max_episode_length = round(cfg.episode_seconds * cfg.drone.control_hz)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        n, t = num_envs, dict(device=self.device)
        self.drones = Drones(n, cfg.drone, device=self.device)
        self.hover = cfg.drone.hover_rotor_speed / cfg.drone.max_rotor_speed
        # course: the next AHEAD gates, each a centre, the way through, and one side (its roll)
        self.gate_pos = torch.zeros(n, AHEAD, 3, **t)
        self.gate_dir = torch.zeros(n, AHEAD, 3, **t)
        self.gate_across = torch.zeros(n, AHEAD, 3, **t)
        self.gates_passed = torch.zeros(n, dtype=torch.long, **t)
        self.prev_action = torch.zeros(n, 4, **t)
        self.prev_dist = torch.zeros(n, **t)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, **t)

        # per-episode running totals, for the logs
        self.ep_terms = torch.zeros(n, len(REWARD_TERMS), **t)
        self.ep_speed_sum = torch.zeros(n, **t)
        self.ep_command_sum = torch.zeros(n, **t)
        self.ep_spin_sum = torch.zeros(n, **t)  # |yaw rate|, rad/s
        self.ep_spin_max = torch.zeros(n, **t)
        self._finished = []  # per-step snapshots of episodes that ended, pulled once per rollout
        self.record = False  # keep the state from just before resets (live view)
        self.last_state = None

        self._reset(torch.ones(n, dtype=torch.bool, **t))
        self.obs = self._observe()

    # --- VecEnv interface ---------------------------------------------------------------

    def get_observations(self):
        return self.obs, {"observations": {}}

    def reset(self):
        self._reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self.obs = self._observe()
        return self.obs, {"observations": {}}

    def step(self, actions):
        cfg, w, d = self.cfg, self.cfg.rewards, self.drones
        actions = actions.clamp(-1.0, 1.0)  # the policy's Gaussian isn't bounded
        prev_pos = d.position.clone()
        d.step(self.hover + cfg.action_scale * actions)
        self.episode_length_buf += 1
        pos = d.position

        # next gate (the only solid one). Passing it moves the window on and adds a new random
        # gate at the end, away from the ones in the window (the one just passed included) and
        # facing the last one
        centre = self.gate_pos[:, 0]
        passed, hit_frame = check_gate(prev_pos, pos, centre, self.gate_dir[:, 0], self.gate_across[:, 0])
        new = random_gate(self.gate_pos, self.gen, prev=self.gate_pos[:, -1])
        p = passed.view(-1, 1, 1)
        for name, part in zip(("gate_pos", "gate_dir", "gate_across"), new):
            old = getattr(self, name)
            shifted = torch.cat((old[:, 1:], part.unsqueeze(1)), dim=1)
            setattr(self, name, torch.where(p, shifted, old))
        self.gates_passed += passed.long()

        # crashes, and the end of the race
        bad = ~torch.isfinite(d.state).all(dim=1)
        floor = pos[:, 2] < 0.1
        wall = (pos[:, :2].abs() > BOX_HALF).any(dim=1) | (pos[:, 2] > 2 * BOX_HALF)
        crashed = bad | floor | wall | hit_frame
        done = crashed | (self.episode_length_buf >= self.max_episode_length)

        # reward terms. Progress uses the distance to the gate that was next before this step,
        # so passing a gate doesn't make the distance jump
        terms = torch.stack(
            (
                w.progress * (self.prev_dist - (centre - pos).norm(dim=-1)),
                w.gate * passed.float(),
                w.action_rate * ((actions - self.prev_action) ** 2).sum(-1),
                w.crash * crashed.float(),
            ),
            dim=1,
        )
        terms = torch.nan_to_num(terms)  # a NaN drone still has to give PPO a number
        reward = terms.sum(1).clamp(-w.clamp, w.clamp)

        # bookkeeping for the logs
        spin = d.body_rate[:, 2].abs().nan_to_num()
        self.ep_terms += terms
        self.ep_speed_sum += d.velocity.norm(dim=-1).nan_to_num()
        self.ep_command_sum += (self.hover + cfg.action_scale * actions).clamp(0, 1).mean(-1)
        self.ep_spin_sum += spin
        self.ep_spin_max = torch.maximum(self.ep_spin_max, spin)
        why = torch.full_like(self.gates_passed, END_TIMEOUT)
        why = torch.where(hit_frame, END_GATE_FRAME, why)
        why = torch.where(wall, END_WALL, why)
        why = torch.where(floor, END_FLOOR, why)
        why = torch.where(bad, END_NAN, why)
        self._snapshot(done, why)

        if self.record:
            self.last_state = d.state.clone()
            self.last_gates = self.gate_pos[:, :2], self.gate_dir[:, :2], self.gate_across[:, :2]

        self.prev_action = actions
        self.prev_dist = (self.gate_pos[:, 0] - pos).norm(dim=-1).nan_to_num()

        # ended races start again straight away. No time_outs for PPO: the time limit is the real
        # end of a race (and the agent sees the time left), not a cut-off to bootstrap through
        self._reset(done)
        self.obs = self._observe()
        return self.obs, reward, done.long(), {"observations": {}}

    # --- internals ----------------------------------------------------------------------

    def _reset(self, mask):
        """Fresh course and drone for every env where mask is True. Everything is computed for
        all envs and blended with torch.where, so nothing waits on the GPU."""
        cfg, n, dev = self.cfg, self.num_envs, self.device

        def uniform(lo, hi):
            return lo + (hi - lo) * torch.rand(n, device=dev, generator=self.gen)

        # a fresh course. The drone hovers start_distance in front of the first gate, on the line
        # through it, nose turned towards it (level, so for a gate facing up it's just below)
        back = uniform(*cfg.start_distance)
        centres, through, across = first_gates(self.gen, back, n, dev)
        spawn = centres[:, 0] - back.unsqueeze(1) * through[:, 0]
        # first_gates already keeps this in the box, the clamp is for the rare gate that couldn't
        spawn[:, :2] = spawn[:, :2].clamp(-BOX_HALF + 0.5, BOX_HALF - 0.5)
        spawn[:, 2] = spawn[:, 2].clamp(0.8, 2 * BOX_HALF - 0.5)
        z_axis = torch.zeros(n, 3, device=dev)
        z_axis[:, 2] = 1.0
        fresh = torch.zeros_like(self.drones.state)
        fresh[:, 0:3] = spawn
        fresh[:, 6:10] = quat.from_axis_angle(z_axis, torch.atan2(through[:, 0, 1], through[:, 0, 0]))
        fresh[:, 13:17] = cfg.drone.hover_rotor_speed
        fresh[:, 17:21] = 1.0

        m1, m2 = mask.unsqueeze(1), mask.view(n, 1, 1)
        self.drones.state = torch.where(m1, fresh, self.drones.state)
        self.gate_pos = torch.where(m2, centres, self.gate_pos)
        self.gate_dir = torch.where(m2, through, self.gate_dir)
        self.gate_across = torch.where(m2, across, self.gate_across)
        self.gates_passed = torch.where(mask, 0, self.gates_passed)
        self.prev_action = torch.where(m1, 0.0, self.prev_action)
        self.prev_dist = torch.where(mask, (centres[:, 0] - spawn).norm(dim=-1), self.prev_dist)
        self.episode_length_buf = torch.where(mask, 0, self.episode_length_buf)
        self.ep_terms = torch.where(m1, 0.0, self.ep_terms)
        for name in ("ep_speed_sum", "ep_command_sum", "ep_spin_sum", "ep_spin_max"):
            setattr(self, name, torch.where(mask, 0.0, getattr(self, name)))

    def _observe(self):
        """29 numbers (33 with the wear sensor), all roughly in [-1, 1]. Gate and velocity
        vectors are in the drone's body frame, so the same skill works whichever way the course
        faces."""
        d = self.drones
        to_body = quat.to_matrix(d.quat).transpose(1, 2)  # world -> body

        def body(v):
            return (to_body @ v.unsqueeze(-1)).squeeze(-1)

        down = torch.zeros_like(d.position)
        down[:, 2] = -1.0
        parts = [
            body(self.gate_pos[:, 0] - d.position) / 5.0,  # next gate, where
            body(self.gate_dir[:, 0]),  # and which way through (its roll isn't seen)
            body(self.gate_pos[:, 1] - d.position) / 5.0,  # the one after
            body(self.gate_dir[:, 1]),
            body(d.velocity) / 5.0,
            body(down),  # which way is down, ie the tilt
            d.body_rate / 10.0,
            d.position / BOX_HALF,  # where in the box, to keep off the walls and floor
            self.prev_action,
            (1.0 - self.episode_length_buf.float() / self.max_episode_length).unsqueeze(1),  # time left
        ]
        if self.cfg.wear_sensor:
            parts.append(d.life)
        return torch.nan_to_num(torch.cat(parts, dim=1)).clamp(-10.0, 10.0)

    def _snapshot(self, done, why):
        """Keep this step's ending episodes (masked later, so no GPU sync here)."""
        length = self.episode_length_buf.float().clamp(min=1)
        columns = [
            done.float(),
            torch.arange(self.num_envs, device=self.device).float(),
            why.float(),
            length,
            self.gates_passed.float(),
            self.ep_terms.sum(1),
            *self.ep_terms.T,
            self.ep_speed_sum / length,
            self.ep_command_sum / length,
            self.ep_spin_sum / length,
            self.ep_spin_max,
            self.drones.life.min(dim=1).values,
            1.0 - self.drones.life.mean(dim=1),  # every race starts at full life
        ]
        self._finished.append(torch.stack(columns, dim=1))

    def pop_episodes(self):
        """Every episode that ended since the last call, as a (k, len(EPISODE_COLUMNS)) CPU
        tensor. One GPU sync, call it once per rollout."""
        if not self._finished:
            return torch.zeros(0, len(EPISODE_COLUMNS))
        rows = torch.cat(self._finished)
        self._finished = []
        return rows[rows[:, 0] > 0][:, 1:].cpu()


EPISODE_COLUMNS = (
    ["env", "end", "steps", "gates", "return"]
    + [f"r_{t}" for t in REWARD_TERMS]
    + ["mean_speed", "mean_command", "mean_spin", "max_spin", "min_life", "life_used"]
)
