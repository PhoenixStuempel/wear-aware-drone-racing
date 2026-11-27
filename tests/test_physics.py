import math
from dataclasses import replace

import pytest
import torch

from sim import quat
from sim.drones import QUAT, RATE, ROTOR, Drones
from sim.params import GRAVITY, DroneParams
from tests.helpers import NO_DRAG, fly, hover_command, make


def test_hover_stays_put():
    d = make()
    fly(d, hover_command(d), 5.0)
    assert d.position.abs().max() < 1e-9
    assert torch.allclose(d.quat, torch.tensor([[1.0, 0, 0, 0]], dtype=d.dtype), atol=1e-12)


def test_free_fall():
    d = make(NO_DRAG)
    d.state[:, ROTOR] = 0.0
    fly(d, torch.zeros(1, 4, dtype=d.dtype), 1.0)
    assert d.position[0, 2].item() == pytest.approx(-GRAVITY / 2, abs=1e-9)
    assert d.velocity[0, 2].item() == pytest.approx(-GRAVITY, abs=1e-9)


def test_full_throttle_climb():
    # T/W 5, so straight up at 4g once the rotors are at full speed
    d = make(NO_DRAG)
    d.state[:, ROTOR] = d.p.max_rotor_speed
    fly(d, torch.ones(1, 4, dtype=d.dtype), 0.2)
    assert d.velocity[0, 2].item() == pytest.approx((d.p.thrust_to_weight - 1) * GRAVITY * 0.2, rel=1e-9)


def test_motor_lag():
    # first order lag, so after one time constant the rotor has covered 1 - 1/e of the gap
    p = DroneParams()
    d = make(p)
    d.state[:, ROTOR] = 0.0
    fly(d, torch.ones(1, 4, dtype=d.dtype), p.motor_tau_up)
    expected = (1 - math.exp(-1)) * p.max_rotor_speed
    assert d.rotor_speed[0, 0].item() == pytest.approx(expected, rel=1e-4)

    d.state[:, ROTOR] = p.max_rotor_speed
    fly(d, torch.zeros(1, 4, dtype=d.dtype), p.motor_tau_down)
    assert d.rotor_speed[0, 0].item() == pytest.approx(math.exp(-1) * p.max_rotor_speed, rel=1e-4)


# rotors: 0 FR, 1 BR, 2 BL, 3 FL. Speeding up a pair should turn the body one way only
@pytest.mark.parametrize(
    "rotors, axis, sign",
    [
        ((2, 3), 0, 1),  # left side up -> +roll
        ((0, 3), 1, -1),  # nose up -> -pitch (+pitch takes the nose down)
        ((0, 2), 2, -1),  # ccw rotors faster -> body yaws cw
    ],
)
def test_turn_directions(rotors, axis, sign):
    d = make()
    command = hover_command(d)
    command[0, list(rotors)] *= 1.1
    fly(d, command, 0.1)
    rate = d.body_rate[0]
    assert rate[axis].item() * sign > 0.1
    others = [i for i in range(3) if i != axis]
    assert rate[others].abs().max() < 1e-9


def test_spin_keeps_angular_momentum():
    # no rotors, no drag: the world frame angular momentum R I w can't change
    d = make(NO_DRAG)
    d.state[:, ROTOR] = 0.0
    d.state[:, RATE] = torch.tensor([3.0, 1.0, 5.0], dtype=d.dtype)  # off the principal axes

    def momentum():
        return (quat.to_matrix(d.quat) @ (d.inertia * d.body_rate).unsqueeze(-1)).squeeze(-1)

    start = momentum()
    fly(d, torch.zeros(1, 4, dtype=d.dtype), 2.0)
    assert torch.allclose(momentum(), start, rtol=1e-6, atol=0)


def test_gyroscopic_torque():
    # only the ccw pair spinning (diagonal, so no roll/pitch from thrust), net rotor momentum
    # H = J_r (w0 + w2) up body z. Rolling at 1 rad/s, torque = -w x H = (0, H, 0)
    d = make(NO_DRAG)
    speed = 2000.0
    d.state[:, ROTOR] = torch.tensor([speed, 0.0, speed, 0.0], dtype=d.dtype)
    d.state[:, RATE] = torch.tensor([1.0, 0.0, 0.0], dtype=d.dtype)
    rate_dot = d.derivative(d.state, d.rotor_speed.clone())[0, RATE]  # target = current, no spin-up
    momentum = d.p.rotor_inertia * 2 * speed
    assert rate_dot[1].item() == pytest.approx(momentum / d.p.inertia[1], rel=1e-12)
    assert rate_dot[0].item() == pytest.approx(0.0, abs=1e-12)


def test_rk4_is_fourth_order():
    # halving the step should cut the error ~16x. Reference is a much finer step
    start = make().state.clone()
    start[:, QUAT] = quat.from_axis_angle(
        torch.tensor([[1.0, 0.5, 0.2]], dtype=torch.float64), torch.tensor([0.4], dtype=torch.float64)
    )
    start[:, RATE] = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    command = torch.tensor(
        [[0.9, 0.6, 0.8, 0.7]], dtype=torch.float64
    )  # all above hover, motors only spin up

    def run(substeps):
        d = make(replace(DroneParams(), substeps=substeps))
        d.state = start.clone()
        fly(d, command, 1.0)
        return d.state

    reference = run(64)
    coarse = (run(2) - reference).abs().max()
    fine = (run(4) - reference).abs().max()
    assert 10 < coarse / fine < 22


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_cpu():
    torch.manual_seed(0)
    command = 0.35 + 0.2 * torch.rand(256, 4, dtype=torch.float64)
    cpu = Drones(256, dtype=torch.float64)
    gpu = Drones(256, device="cuda", dtype=torch.float32)
    for _ in range(50):
        cpu.step(command)
        gpu.step(command.to("cuda", torch.float32))
    assert (gpu.position.cpu().double() - cpu.position).abs().max() < 1e-3
