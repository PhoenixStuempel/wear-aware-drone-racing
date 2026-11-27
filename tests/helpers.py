import torch

from sim.drones import Drones
from sim.params import DroneParams

NO_DRAG = DroneParams(drag=0.0, rate_damping=0.0)


def make(params=DroneParams(), count=1):
    return Drones(count, params, dtype=torch.float64)  # float64 so the tolerances can be tight


def hover_command(d):
    return torch.full((d.count, 4), d.p.hover_rotor_speed / d.p.max_rotor_speed, dtype=d.dtype)


def fly(d, command, seconds):
    for _ in range(round(seconds * d.p.control_hz)):
        d.step(command)
