"""Series elastic actuator (tendon model) for Isaac Lab: motor -> spring (tendon) -> joint.

    motor:   J_m * dw_m/dt = tau_m - tau_s,   tau_m = clip(kp (q_des - th_m) + kd (qd_des - w_m), +-tau_max)
             |w_m| <= motor velocity limit (muscle shortening speed)
    tendon:  tau_s = k_s (th_m - q) + d_s (w_m - qd)            -> applied to the joint

Energy is stored in the spring when the joint lags the motor (e.g. shoulder layback) and returned
when it recoils, so the joint can briefly move faster than the motor itself.
The actuator is explicit and integrated at the physics rate (compute() is called every physics step).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class SeriesElasticActuator(ActuatorBase):
    cfg: "SeriesElasticActuatorCfg"

    def __init__(self, cfg: "SeriesElasticActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        shape = (self._num_envs, self.num_joints)
        self.motor_pos = torch.zeros(shape, device=self._device)
        self.motor_vel = torch.zeros(shape, device=self._device)
        self.spring_torque = torch.zeros(shape, device=self._device)
        self._last_deflection = torch.zeros(shape, device=self._device)
        self.needs_sync = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        # joint states are written after the actuator reset: align the motor with the joint on the next step
        self.needs_sync[env_ids] = True

    @property
    def deflection(self) -> torch.Tensor:
        return self._last_deflection

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
                ) -> ArticulationActions:
        c = self.cfg
        if self.needs_sync.any():
            m = self.needs_sync
            self.motor_pos[m] = joint_pos[m]
            self.motor_vel[m] = joint_vel[m]
            self.needs_sync[:] = False
        q_des = control_action.joint_positions
        qd_des = control_action.joint_velocities if control_action.joint_velocities is not None else torch.zeros_like(q_des)
        tau_m = c.motor_kp * (q_des - self.motor_pos) + c.motor_kd * (qd_des - self.motor_vel)
        tau_m = tau_m.clamp(-c.motor_effort, c.motor_effort)
        tau_s = c.spring_stiffness * (self.motor_pos - joint_pos) + c.spring_damping * (self.motor_vel - joint_vel)
        # semi-implicit Euler at the physics rate
        self.motor_vel = (self.motor_vel + c.dt * (tau_m - tau_s) / c.motor_inertia).clamp(-c.motor_velocity, c.motor_velocity)
        self.motor_pos = self.motor_pos + c.dt * self.motor_vel
        self._last_deflection = self.motor_pos - joint_pos
        self.spring_torque = tau_s
        self.computed_effort = tau_s
        self.applied_effort = tau_s.clamp(-c.spring_torque_limit, c.spring_torque_limit)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class SeriesElasticActuatorCfg(ActuatorBaseCfg):
    class_type: type = SeriesElasticActuator
    dt: float = MISSING  # physics time step
    motor_effort: float = MISSING  # Nm (muscle force limit, human level)
    motor_velocity: float = MISSING  # rad/s (muscle shortening speed limit)
    motor_inertia: float = 0.01  # kg m^2 (reflected motor / muscle mass)
    motor_kp: float = 300.0
    motor_kd: float = 6.0
    spring_stiffness: float = MISSING  # Nm/rad (tendon)
    spring_damping: float = 0.5
    spring_torque_limit: float = 1000.0  # safety clamp only
