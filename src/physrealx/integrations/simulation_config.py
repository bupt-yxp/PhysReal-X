"""Shared control, camera and perturbation contract for physical simulation engines."""

from __future__ import annotations

import math
import re

from pydantic import Field, model_validator

from physrealx.domain import DomainModel
from physrealx.integrations.sensors import CameraSpec


class BodyWrench(DomainModel):
    """Constant world-axis load applied at the named body's center of mass."""

    force_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> BodyWrench:
        if not all(math.isfinite(v) for v in (*self.force_n, *self.torque_nm)):
            raise ValueError("external forces and torques must be finite")
        return self


class DomainRandomization(DomainModel):
    mass_scale: tuple[float, float] = (1.0, 1.0)
    damping_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    initial_position_jitter: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    color_jitter: float = Field(default=0.0, ge=0, le=1)
    light_jitter: float = Field(default=0.0, ge=0, le=1)
    material_shininess_jitter: float = Field(default=0.0, ge=0, le=1)
    material_specular_jitter: float = Field(default=0.0, ge=0, le=1)
    camera_position_jitter_m: float = Field(default=0.0, ge=0, le=1)
    occlusion_probability: float = Field(default=0.0, ge=0, le=1)
    occlusion_fraction: tuple[float, float] = (0.08, 0.2)

    @model_validator(mode="after")
    def validate_intervals(self) -> DomainRandomization:
        for interval in (self.mass_scale, self.damping_scale, self.friction_scale):
            if (
                not all(math.isfinite(v) for v in interval)
                or interval[0] <= 0
                or interval[0] > interval[1]
            ):
                raise ValueError(
                    "physical randomization needs positive finite ordered scale bounds"
                )
        if not 0 < self.occlusion_fraction[0] <= self.occlusion_fraction[1] <= 0.5:
            raise ValueError("occluder size must occupy a bounded fraction of the camera view")
        return self


class PhysicsConfig(DomainModel):
    cameras: dict[str, CameraSpec] = Field(default_factory=dict)
    controlled_joints: tuple[str, ...] = Field(min_length=1)
    timestep_seconds: float = Field(default=0.002, gt=0, le=0.01)
    substeps: int = Field(default=25, ge=1, le=1000)
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    initial_positions: tuple[float, ...]
    passive_positions: dict[str, float] = Field(default_factory=dict)
    effort_limits: tuple[float, ...]
    position_gain: float = Field(default=200.0, gt=0, allow_inf_nan=False)
    velocity_gain: float = Field(default=20.0, ge=0, allow_inf_nan=False)
    external_wrenches: dict[str, BodyWrench] = Field(default_factory=dict)
    randomization: DomainRandomization = Field(default_factory=DomainRandomization)

    @model_validator(mode="after")
    def validate_vectors(self) -> PhysicsConfig:
        if len(self.cameras) > 8 or any(
            re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None for name in self.cameras
        ):
            raise ValueError("at most eight uniquely named cameras are supported")
        count = len(self.controlled_joints)
        if (
            len(set(self.controlled_joints)) != count
            or len(self.initial_positions) != count
            or len(self.effort_limits) != count
        ):
            raise ValueError("joint names, initial positions and effort limits must align uniquely")
        if any(
            not math.isfinite(v)
            for v in (*self.gravity_m_s2, *self.initial_positions, *self.effort_limits)
        ) or any(v <= 0 for v in self.effort_limits):
            raise ValueError("physical configuration must be finite with positive effort limits")
        if set(self.passive_positions) & set(self.controlled_joints) or any(
            not math.isfinite(value) for value in self.passive_positions.values()
        ):
            raise ValueError("passive positions must be finite and exclude controlled joints")
        return self
