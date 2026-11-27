"""Batched quadcopter physics with motor wear. One row of `state` per drone, all on one device.

Rotors, seen from above (x forward, y left):

    3 FL (cw)    0 FR (ccw)
          \\    /
           body
          /    \\
    2 BL (ccw)   1 BR (cw)

Same order as viewer/drone.js, so wear shows on the right motor there.
"""

import torch

from sim import quat
from sim.params import GRAVITY, DroneParams

# state layout
POS = slice(0, 3)
VEL = slice(3, 6)  # world frame
QUAT = slice(6, 10)  # body to world, w x y z
RATE = slice(10, 13)  # body rates, body frame
ROTOR = slice(13, 17)  # rad/s
LIFE = slice(17, 21)  # 1 = new, 0 = worn out
STATE_SIZE = 21


class Drones:
    def __init__(self, count: int, params: DroneParams = DroneParams(), device="cpu", dtype=torch.float32):
        self.p = params
        self.count = count
        self.device = torch.device(device)
        self.dtype = dtype
        t = dict(device=self.device, dtype=dtype)

        angles = -torch.deg2rad(torch.tensor([45.0, 135.0, 225.0, 315.0], **t))
        self.rotor_x = params.arm * torch.cos(angles)
        self.rotor_y = params.arm * torch.sin(angles)
        self.spin = torch.tensor([1.0, -1.0, 1.0, -1.0], **t)  # +1 = ccw from above
        self.inertia = torch.tensor(params.inertia, **t)
        self.gravity = torch.tensor([0.0, 0.0, -GRAVITY], **t)

        self.state = torch.zeros(count, STATE_SIZE, **t)
        self.reset()

    # views into the state
    @property
    def position(self):
        return self.state[:, POS]

    @property
    def velocity(self):
        return self.state[:, VEL]

    @property
    def quat(self):
        return self.state[:, QUAT]

    @property
    def body_rate(self):
        return self.state[:, RATE]

    @property
    def rotor_speed(self):
        return self.state[:, ROTOR]

    @property
    def life(self):
        return self.state[:, LIFE]

    def reset(self, ids=None, position=None):
        """Level and still at `position` (default origin), rotors at hover speed, new motors."""
        rows = self.state if ids is None else self.state[ids]
        rows.zero_()
        if position is not None:
            rows[:, POS] = position
        rows[:, 6] = 1.0  # identity quaternion
        rows[:, ROTOR] = self.p.hover_rotor_speed
        rows[:, LIFE] = 1.0
        if ids is not None:
            self.state[ids] = rows

    def step(self, command: torch.Tensor):
        """One control step. command is (count, 4), rotor speed as a fraction of max, held for
        the whole step."""
        target = command.clamp(0.0, 1.0) * self.p.max_rotor_speed
        for _ in range(self.p.substeps):
            self.state = self._rk4(self.state, target, self.p.dt)

    def thrust(self, rotor_speed, life):
        return self.p.thrust_coeff * life.clamp(min=0.0) * rotor_speed**2  # N, per rotor

    def derivative(self, x, target):
        p = self.p
        v, q, w = x[:, VEL], x[:, QUAT], x[:, RATE]
        rotor, life = x[:, ROTOR], x[:, LIFE]

        # motors: first order lag towards the target speed
        tau = torch.where(target > rotor, p.motor_tau_up, p.motor_tau_down)
        rotor_acc = (target - rotor) / tau

        # wear, stops once a motor is fully worn
        wear = -p.wear_rate * (rotor.clamp(min=0.0) / p.max_rotor_speed) ** p.wear_exponent
        wear = torch.where(life > 0, wear, torch.zeros_like(wear))

        # forces
        thrust = self.thrust(rotor, life)
        body_thrust = torch.zeros_like(v)
        body_thrust[:, 2] = thrust.sum(1)
        force = quat.rotate(q, body_thrust) - p.drag * v.norm(dim=1, keepdim=True) * v
        acc = force / p.mass + self.gravity

        # torques (body frame). Yaw: prop drag pushes the body opposite to the spin, and so does
        # spinning a rotor up (motor torque reacts on the body)
        torque = torch.stack(
            (
                (thrust * self.rotor_y).sum(1),
                -(thrust * self.rotor_x).sum(1),
                -(self.spin * (p.torque_per_thrust * thrust + p.rotor_inertia * rotor_acc)).sum(1),
            ),
            dim=1,
        )
        # gyroscopic torque from the rotors' net angular momentum (along body z)
        rotor_momentum = torch.zeros_like(w)
        rotor_momentum[:, 2] = p.rotor_inertia * (self.spin * rotor).sum(1)
        torque = torque - torch.linalg.cross(w, rotor_momentum) - p.rate_damping * w

        # Euler's equation, inertia is diagonal
        w_dot = (torque - torch.linalg.cross(w, self.inertia * w)) / self.inertia
        q_dot = 0.5 * quat.mul(q, torch.cat((torch.zeros_like(w[:, :1]), w), dim=1))

        return torch.cat((v, acc, q_dot, w_dot, rotor_acc, wear), dim=1)

    def _rk4(self, x, target, dt):
        k1 = self.derivative(x, target)
        k2 = self.derivative(x + 0.5 * dt * k1, target)
        k3 = self.derivative(x + 0.5 * dt * k2, target)
        k4 = self.derivative(x + dt * k3, target)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # RK4 drifts the quaternion off unit length, and can overshoot 0 on speed and life
        x[:, QUAT] = x[:, QUAT] / x[:, QUAT].norm(dim=1, keepdim=True)
        x[:, ROTOR] = x[:, ROTOR].clamp(min=0.0)
        x[:, LIFE] = x[:, LIFE].clamp(min=0.0)
        return x
