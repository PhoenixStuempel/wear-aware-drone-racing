"""Vehicle and sim settings. SI units. World is z up, body is x forward, y left, z up."""

import math
from dataclasses import dataclass

GRAVITY = 9.81


@dataclass(frozen=True)
class DroneParams:
    mass: float = 0.75  # kg
    arm: float = 0.17  # m, centre to motor axis
    inertia: tuple[float, float, float] = (0.0023, 0.0023, 0.004)  # kg m^2, diagonal
    thrust_to_weight: float = 5.0  # all 4 motors at full speed
    max_rotor_speed: float = 2500.0  # rad/s, about 24k rpm
    torque_per_thrust: float = 0.016  # m, prop drag torque / thrust
    rotor_inertia: float = 1e-5  # kg m^2, prop + motor bell

    # props have momentum so rotor speed lags the command. Spin-down is slower since only
    # drag slows the prop
    motor_tau_up: float = 0.04  # s
    motor_tau_down: float = 0.08  # s

    drag: float = 0.012  # N/(m/s)^2, body drag F = -drag |v| v
    rate_damping: float = 0.002  # N m s, angular drag

    # wear: dL/dt = -wear_rate (w / w_max)^wear_exponent, thrust and drag torque scale by L.
    # wear_rate is life lost per second at full speed, 0 turns wear off
    wear_rate: float = 0.0  # 1/s
    wear_exponent: float = 3.0

    control_hz: int = 50
    substeps: int = 2  # RK4 steps per control step

    @property
    def max_thrust(self) -> float:
        return self.thrust_to_weight * self.mass * GRAVITY / 4  # N, per motor

    @property
    def thrust_coeff(self) -> float:
        return self.max_thrust / self.max_rotor_speed**2  # N/(rad/s)^2

    @property
    def hover_rotor_speed(self) -> float:
        return self.max_rotor_speed / math.sqrt(self.thrust_to_weight)

    @property
    def dt(self) -> float:
        return 1.0 / (self.control_hz * self.substeps)  # s, one RK4 step
