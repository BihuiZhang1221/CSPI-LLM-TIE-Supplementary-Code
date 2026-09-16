"""Hardware interface contracts, thruster allocation and PWM calibration, and the deployable state estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable
from dataclasses import dataclass
from typing import Sequence
import numpy as np


"""Hardware interface contracts for the BSA-D2 tank platform.

The vehicle-side bridge exposes one synchronized sensor observation and accepts a
normalized 5D continuous action (surge, sway, heave, yaw, pitch) at the control
rate.  Implementations live on the onboard computer; the controller and state
estimator below only depend on these narrow method names.
"""
REQUIRED_SENSOR_KEYS: tuple[str, ...] = (
    "dvl_velocity",
    "imu_orientation",
    "imu_angular_velocity",
    "pressure_depth",
    "ray_depth",
)
@dataclass
class HardwareConfig:
    """Deployment parameters shared by the bridge and the controller."""

    control_rate_hz: float = 10.0
    semantic_rate_hz: float = 2.0
    action_dim: int = 5
    thrust_limit_n: float = 80.0
    pwm_min_us: int = 1100
    pwm_mid_us: int = 1500
    pwm_max_us: int = 1900
    temperature_limit_c: float = 75.0
    leak_alarm_enabled: bool = True
    required_sensor_keys: tuple[str, ...] = field(default=REQUIRED_SENSOR_KEYS)
@dataclass
class SensorFrame:
    """One synchronized sensor sample delivered by the hardware bridge."""

    dvl_velocity: tuple[float, float, float]
    imu_orientation: tuple[float, float, float]
    imu_angular_velocity: tuple[float, float, float]
    pressure_depth: float
    ray_depth: object | None = None
    timestamp_s: float = 0.0
@runtime_checkable
class AuvHardwareInterface(Protocol):
    """Bridge between the onboard computer and the vehicle actuators."""

    def read_observation(self) -> SensorFrame:
        """Read one synchronized sensor observation."""

    def write_action(self, action: Sequence[float]) -> None:
        """Send a normalized 5D control action to the vehicle bridge."""

    def stop(self) -> None:
        """Command a safe stop (zero thrust, motors disarmed)."""


"""Six-thruster allocation and firmware PWM calibration.

The allocation matrix and the PWM calibration polynomials match the STM32H743
firmware that executed the tank experiments.  A body wrench (forces and moments in
the body frame) is mapped to six thruster commands in newtons, each command is
clipped to the 80 N thruster limit, and the clipped command is converted to the
firmware PWM pulse width.  Saturating thrusters are reported so that the caller
can log actuator-limit events.
"""
FIRMWARE_A_INV = np.asarray(
    [
        [0.0, -0.5, 0.5, 0.075, 1.25, 0.0],
        [0.0, 0.5, 0.5, -0.075, 1.25, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 3.33],
        [-0.5, 0.0, 0.0, 0.0, 0.0, 3.33],
        [0.0, -0.5, -0.5, -0.075, 1.25, 0.0],
        [0.0, 0.5, -0.5, 0.075, 1.25, 0.0],
    ],
    dtype=np.float64,
)
POSITIVE_PWM_COEFF = np.asarray([2.55947856e-04, -4.88849907e-02, 4.91272694, 1563.52382])
NEGATIVE_PWM_COEFF = np.asarray([-1.90315851e-04, 3.89391698e-02, -4.53990612, 1428.40154])
THRUST_LIMIT_N = 80.0
PWM_MIN_US = 1100
PWM_MID_US = 1500
PWM_MAX_US = 1900
FORWARD_ACTION_SCALE_N = 120.0
LATERAL_ACTION_SCALE_N = 40.0
VERTICAL_ACTION_SCALE_N = 40.0
YAW_ACTION_SCALE_NM = 20.0
PITCH_ACTION_SCALE_NM = 12.0
@dataclass
class AllocationResult:
    """Allocation output for one control step."""

    commanded_wrench: np.ndarray
    raw_thrust_n: np.ndarray
    limited_thrust_n: np.ndarray
    pwm_us: np.ndarray
    saturation_mask: np.ndarray
class SixThrusterAllocator:
    """Firmware-consistent thruster allocation and PWM mapping."""

    def __init__(self, action_scales: Sequence[float] | None = None):
        self.action_scales = np.asarray(
            action_scales
            or (
                FORWARD_ACTION_SCALE_N,
                LATERAL_ACTION_SCALE_N,
                VERTICAL_ACTION_SCALE_N,
                YAW_ACTION_SCALE_NM,
                PITCH_ACTION_SCALE_NM,
            ),
            dtype=np.float64,
        )
        if self.action_scales.shape != (5,):
            raise ValueError("action_scales must contain five gains")

    def action_to_wrench(self, action: Sequence[float]) -> np.ndarray:
        """Map a normalized 5D action to a 6D body wrench."""

        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.size != 5:
            raise ValueError(f"expected a 5D action, got {values.size}D")
        scaled = np.clip(values, -1.0, 1.0) * self.action_scales
        return np.asarray(
            [scaled[0], scaled[1], scaled[2], 0.0, 0.0, scaled[3] + scaled[4]],
            dtype=np.float64,
        )

    def allocate(self, action: Sequence[float]) -> AllocationResult:
        """Return thrust and PWM commands for one normalized action."""

        return self.allocate_wrench(self.action_to_wrench(action))

    def allocate_wrench(self, wrench: Sequence[float]) -> AllocationResult:
        """Return thrust and PWM commands for a 6D body wrench."""

        commanded = np.asarray(wrench, dtype=np.float64).reshape(-1)
        if commanded.size != 6:
            raise ValueError(f"expected a 6D body wrench, got {commanded.size}D")
        raw = FIRMWARE_A_INV @ commanded
        limited = np.clip(raw, -THRUST_LIMIT_N, THRUST_LIMIT_N)
        pwm = thrust_to_pwm(limited)
        return AllocationResult(
            commanded_wrench=commanded,
            raw_thrust_n=raw,
            limited_thrust_n=limited,
            pwm_us=pwm,
            saturation_mask=np.abs(raw) > THRUST_LIMIT_N + 1.0e-12,
        )

    def stop(self) -> AllocationResult:
        """Return the zero-thrust, mid-PWM safe stop command."""

        return self.allocate(np.zeros(5, dtype=np.float64))
def thrust_to_pwm(thrust_n: np.ndarray) -> np.ndarray:
    """Convert clipped thruster commands in newtons to firmware PWM microseconds."""

    values = np.asarray(thrust_n, dtype=np.float64)
    pwm = np.full(values.shape, float(PWM_MID_US), dtype=np.float64)
    positive = values > 0.0
    negative = values < 0.0
    if np.any(positive):
        pwm[positive] = np.polyval(POSITIVE_PWM_COEFF, values[positive])
    if np.any(negative):
        pwm[negative] = np.polyval(NEGATIVE_PWM_COEFF, -values[negative])
    return np.clip(np.floor(pwm + 0.5), PWM_MIN_US, PWM_MAX_US).astype(np.int64)


"""Deployable state estimation from tank sensors.

The estimator turns a synchronized :class:`~src.hardware.interface.SensorFrame`
into the deployable 28D observation contract used by the controller: body-frame
DVL velocity, IMU attitude and angular rate, depth, goal direction, goal distance,
and the previous executed action.  It contains no privileged simulator state
(route identity, current truth, waypoint or cross-track information).
"""
DVL_VELOCITY_SCALES_M_S = np.asarray((2.0, 1.5, 1.0), dtype=np.float32)
MAX_DEPTH_M = 100.0
GOAL_BODY_FRAME = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
@dataclass
class EstimatorConfig:
    """Configuration of the deployable state estimator."""

    dvl_velocity_scales_m_s: np.ndarray = field(
        default_factory=lambda: DVL_VELOCITY_SCALES_M_S.copy()
    )
    max_depth_m: float = MAX_DEPTH_M
    goal_distance_scale_m: float = 10.0
class AuvStateEstimator:
    """Convert sensor frames into the 28D deployable state vector."""

    def __init__(self, config: EstimatorConfig | None = None):
        self.config = config or EstimatorConfig()
        self._previous_action = np.zeros(5, dtype=np.float32)

    def reset(self) -> None:
        """Clear the previous-action history between episodes."""

        self._previous_action = np.zeros(5, dtype=np.float32)

    def update_action(self, action: Sequence[float]) -> None:
        """Record the last executed action for the next observation."""

        self._previous_action = np.clip(np.asarray(action, dtype=np.float32).reshape(5), -1.0, 1.0)

    def estimate(
        self,
        depth_m: float,
        goal_direction_body: Sequence[float] | None = None,
        goal_distance_m: float = 0.0,
    ) -> np.ndarray:
        """Assemble the 28D state from the latest sensor frame.

        The first three entries follow the DVL convention used during training
        (``[-vx, -vy, vz]`` divided by the per-axis DVL range), entries 3-5 hold
        the IMU angular rate, entries 6-11 the Euler attitude, entry 12 the
        normalized depth, entries 19-21 the goal-direction proxy, entry 22 the
        normalized goal distance, and the final five the previous action.
        """

        state = np.zeros(28, dtype=np.float32)
        state[12] = float(np.clip(depth_m / self.config.max_depth_m, 0.0, 1.0))
        if goal_direction_body is not None:
            direction = np.asarray(goal_direction_body, dtype=np.float32).reshape(3)
            state[19:22] = direction
        goal = goal_distance_m if goal_distance_m > 0.0 else float(self.config.goal_distance_scale_m)
        state[22] = float(np.clip(goal / self.config.goal_distance_scale_m, 0.0, 1.0))
        state[23:28] = self._previous_action
        return state

    def fill_velocity(self, state: np.ndarray, dvl_velocity_m_s: Sequence[float]) -> np.ndarray:
        """Write the three DVL velocity channels into the state vector."""

        values = np.asarray(dvl_velocity_m_s, dtype=np.float32).reshape(3)
        scales = np.asarray(self.config.dvl_velocity_scales_m_s, dtype=np.float32).reshape(3)
        state[:3] = np.asarray([-values[0], -values[1], values[2]], dtype=np.float32) / scales
        return state

    def fill_angular_rate(self, state: np.ndarray, angular_rate_rad_s: Sequence[float]) -> np.ndarray:
        """Write the three IMU angular-rate channels into the state vector."""

        state[3:6] = np.asarray(angular_rate_rad_s, dtype=np.float32).reshape(3)
        return state

    def fill_attitude(self, state: np.ndarray, euler_xyz_rad: Sequence[float]) -> np.ndarray:
        """Write the three Euler attitude channels into the state vector."""

        state[6:9] = np.asarray(euler_xyz_rad, dtype=np.float32).reshape(3)
        return state
