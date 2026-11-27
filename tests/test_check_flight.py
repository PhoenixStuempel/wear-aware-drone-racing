import torch

from sim import quat
from sim.drones import LIFE
from sim.pid import PositionController
from tests.helpers import make


def fly_to(d, target, seconds):
    ctrl = PositionController(d)
    for _ in range(round(seconds * d.p.control_hz)):
        d.step(ctrl.command(target))


def test_holds_hover():
    d = make()
    fly_to(d, torch.zeros(1, 3, dtype=d.dtype), 3.0)
    assert d.position.norm() < 0.02


def test_flies_to_a_point():
    d = make()
    target = torch.tensor([[3.0, -2.0, 1.0]], dtype=d.dtype)
    fly_to(d, target, 5.0)
    assert (d.position - target).norm() < 0.1
    assert d.velocity.norm() < 0.2


def test_flies_downhill_without_flipping():
    # a target well below used to ask for more than 1 g down, which points the thrust at the floor
    d = make()
    d.state[:, 2] = 5.0
    ctrl = PositionController(d)
    target = torch.tensor([[3.0, 0.0, 0.5]], dtype=d.dtype)
    for _ in range(round(5.0 * d.p.control_hz)):
        d.step(ctrl.command(target))
        # aims for 45 deg max, the attitude loop overshoots a few degrees. 60 still catches a flip
        assert quat.to_matrix(d.quat)[0, 2, 2] > 0.5
    assert (d.position - target).norm() < 0.1


def test_turns_to_a_heading():
    # yaw 90 degrees left, body x should end up along world +y
    d = make()
    ctrl = PositionController(d)
    target = torch.zeros(1, 3, dtype=d.dtype)
    for _ in range(round(4.0 * d.p.control_hz)):
        d.step(ctrl.command(target, yaw=torch.pi / 2))
    nose = quat.to_matrix(d.quat)[0, :, 0]
    assert torch.allclose(nose, torch.tensor([0.0, 1.0, 0.0], dtype=d.dtype), atol=0.02)
    assert d.position.norm() < 0.05


def test_flies_on_worn_motors():
    # one motor half worn, the rest 70%. The mixer just spins the worn ones faster
    d = make()
    d.state[:, LIFE] = torch.tensor([0.7, 0.7, 0.7, 0.5], dtype=d.dtype)
    target = torch.tensor([[-2.0, 1.0, 2.0]], dtype=d.dtype)
    fly_to(d, target, 5.0)
    assert (d.position - target).norm() < 0.1
