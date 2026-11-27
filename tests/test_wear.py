from dataclasses import replace

import pytest
import torch

from sim.drones import LIFE, ROTOR
from sim.params import GRAVITY, DroneParams
from tests.helpers import NO_DRAG, fly, make


def test_no_wear_when_off():
    d = make(replace(DroneParams(), wear_rate=0.0))
    fly(d, torch.ones(1, 4, dtype=d.dtype), 2.0)
    assert (d.life == 1.0).all()


def test_life_only_goes_down_and_stops_at_zero():
    d = make(replace(DroneParams(), wear_rate=1.0))
    command = torch.ones(1, 4, dtype=d.dtype)
    previous = d.life.clone()
    for _ in range(150):  # 3 s, at full speed it's gone in about 1
        d.step(command)
        assert (d.life <= previous).all()
        previous = d.life.clone()
    assert (d.life == 0.0).all()


@pytest.mark.parametrize("exponent", [1.0, 3.0])
def test_wear_scales_with_speed_to_the_power(exponent):
    # rotors held at half and full speed, life lost should differ by 0.5^p
    d = make(replace(DroneParams(), wear_rate=0.1, wear_exponent=exponent), count=2)
    speeds = torch.tensor([[0.5], [1.0]], dtype=d.dtype).expand(2, 4).contiguous()
    d.state[:, ROTOR] = speeds * d.p.max_rotor_speed
    fly(d, speeds, 0.5)
    lost = 1 - d.life[:, 0]
    assert (lost[0] / lost[1]).item() == pytest.approx(0.5**exponent, rel=1e-6)
    assert lost[1].item() == pytest.approx(0.1 * 0.5, rel=1e-6)


def test_half_worn_motors_give_half_thrust():
    d = make(NO_DRAG)
    d.state[:, ROTOR] = d.p.max_rotor_speed
    d.state[:, LIFE] = 0.5
    fly(d, torch.ones(1, 4, dtype=d.dtype), 0.1)
    expected = (0.5 * d.p.thrust_to_weight - 1) * GRAVITY * 0.1
    assert d.velocity[0, 2].item() == pytest.approx(expected, rel=1e-9)


def test_worn_out_motors_give_nothing():
    d = make(NO_DRAG)
    d.state[:, ROTOR] = d.p.max_rotor_speed
    d.state[:, LIFE] = 0.0
    fly(d, torch.ones(1, 4, dtype=d.dtype), 0.5)
    assert d.velocity[0, 2].item() == pytest.approx(-GRAVITY * 0.5, abs=1e-9)
