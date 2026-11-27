"""Hand-tuned position controller for the check flight (proves the physics can be flown).
The agents never use this.

Position PD gives a wanted thrust vector, then a geometric attitude controller (Lee et al.
2010) turns that into torques, and the mixer splits thrust + torques across the 4 rotors.
"""

import math

import torch

from sim import quat
from sim.drones import Drones
from sim.params import GRAVITY


def _vee(m):
    return torch.stack((m[:, 2, 1], m[:, 0, 2], m[:, 1, 0]), dim=1)  # skew matrix -> vector


class PositionController:
    def __init__(
        self,
        drones: Drones,
        pos_gain=4.0,
        vel_gain=3.0,
        att_gain=64.0,
        rate_gain=11.0,
        max_acc=15.0,
        max_tilt=45.0,
    ):
        self.d = drones
        self.pos_gain, self.vel_gain = pos_gain, vel_gain
        self.att_gain, self.rate_gain = att_gain, rate_gain
        self.max_acc = max_acc  # m/s^2
        self.tan_max_tilt = math.tan(math.radians(max_tilt))  # max_tilt in degrees

        # rows: collective thrust, roll, pitch, yaw torque, all per unit rotor thrust
        p = drones.p
        mix = torch.stack(
            (
                torch.ones_like(drones.rotor_x),
                drones.rotor_y,
                -drones.rotor_x,
                -drones.spin * p.torque_per_thrust,
            )
        )
        self.unmix = torch.linalg.inv(mix)

    def command(self, target: torch.Tensor, yaw=0.0) -> torch.Tensor:
        """Motor commands that fly to `target` (world, m) with the nose at `yaw` (rad from +x)."""
        d = self.d
        acc = self.pos_gain * (target - d.position) - self.vel_gain * d.velocity
        return self.accelerate(acc, yaw)

    def accelerate(self, acc: torch.Tensor, yaw=0.0) -> torch.Tensor:
        """Motor commands for a wanted acceleration (world, m/s^2, capped) and nose heading."""
        d, p = self.d, self.d.p
        size = acc.norm(dim=1, keepdim=True)
        acc = acc * (self.max_acc / size.clamp(min=self.max_acc))

        # more than 1 g down would mean pointing the thrust at the ground (flips it), so 0.6 g max.
        # Then cap sideways so the tilt atan(side / (up + g)) stays under max_tilt
        acc = acc.clone()
        acc[:, 2] = acc[:, 2].clamp(min=-0.6 * GRAVITY)
        side = acc[:, :2].norm(dim=1, keepdim=True)
        side_max = (acc[:, 2:] + GRAVITY) * self.tan_max_tilt
        acc[:, :2] = acc[:, :2] * (side_max / side.clamp(min=1e-9)).clamp(max=1.0)

        force = p.mass * (acc - d.gravity)

        # wanted attitude: body z along the force, nose turned towards yaw
        z = force / force.norm(dim=1, keepdim=True)
        yaw = torch.as_tensor(yaw, device=d.device, dtype=d.dtype).expand(d.count)
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=1)
        y = torch.linalg.cross(z, forward)
        y = y / y.norm(dim=1, keepdim=True)
        x = torch.linalg.cross(y, z)
        want = torch.stack((x, y, z), dim=2)  # columns are the body axes

        rot = quat.to_matrix(d.quat)
        att_error = 0.5 * _vee(want.transpose(1, 2) @ rot - rot.transpose(1, 2) @ want)
        w = d.body_rate
        torque = d.inertia * (-self.att_gain * att_error - self.rate_gain * w)
        torque = torque + torch.linalg.cross(w, d.inertia * w)

        collective = (force * rot[:, :, 2]).sum(1, keepdim=True)  # only the part along body z helps
        thrust = (torch.cat((collective, torque), dim=1) @ self.unmix.T).clamp(min=0.0)

        # thrust -> rotor speed, counting wear (a worn motor needs more speed for the same thrust)
        life = d.life.clamp(min=1e-3)
        speed = torch.sqrt(thrust / (p.thrust_coeff * life))
        return speed / p.max_rotor_speed
