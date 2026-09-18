"""Small canonical environment and robot assets used by scene-generation examples."""

from __future__ import annotations

import math

from physrealx.domain import (
    AssetEvidenceState,
    AssetSpec,
    BodyPhysicsSpec,
    GeometryRepresentation,
    GeometryRole,
    GeometrySpec,
    ParameterSource,
    PoseSpec,
    ScalarParameter,
)
from physrealx.errors import DomainValidationError


def _prior(value: float) -> ScalarParameter:
    return ScalarParameter(
        value=value,
        uncertainty=abs(value),
        source=ParameterSource.DEFAULT,
        confidence=0.0,
    )


def _body(link: str, mass: float, dimensions: tuple[float, float, float]) -> BodyPhysicsSpec:
    x, y, z = dimensions
    return BodyPhysicsSpec(
        link=link,
        mass_kg=_prior(mass),
        center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_kg_m2=(
            (mass * (y * y + z * z) / 12, 0.0, 0.0),
            (0.0, mass * (x * x + z * z) / 12, 0.0),
            (0.0, 0.0, mass * (x * x + y * y) / 12),
        ),
        static_friction=_prior(0.6),
        dynamic_friction=_prior(0.5),
        restitution=_prior(0.0),
    )


def _box(
    name: str,
    link: str,
    role: GeometryRole,
    dimensions: tuple[float, float, float],
    translation: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> GeometrySpec:
    return GeometrySpec(
        name=name,
        link=link,
        role=role,
        representation=GeometryRepresentation.PRIMITIVE,
        uri="primitive://box",
        local_pose=PoseSpec(translation_m=translation, rotation_xyzw=rotation_xyzw),
        primitive_dimensions_m=dimensions,
    )


def room_shell_asset(
    *, resource_id: str, version: str, width_m: float, depth_m: float, height_m: float
) -> AssetSpec:
    """Build a floor plus two visual walls; the floor is the only collision surface."""
    values = (width_m, depth_m, height_m)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise DomainValidationError("room dimensions must be finite and positive")
    thickness = 0.04
    floor = (width_m, depth_m, thickness)
    geometries = [
        _box("floor_visual", "room", GeometryRole.VISUAL, floor, (0.0, 0.0, -0.02)),
        _box("floor_collision", "room", GeometryRole.COLLISION, floor, (0.0, 0.0, -0.02)),
        _box(
            "back_wall_visual",
            "room",
            GeometryRole.VISUAL,
            (width_m, thickness, height_m),
            (0.0, depth_m / 2, height_m / 2),
        ),
        _box(
            "left_wall_visual",
            "room",
            GeometryRole.VISUAL,
            (thickness, depth_m, height_m),
            (-width_m / 2, 0.0, height_m / 2),
        ),
    ]
    return AssetSpec(
        resource_id=resource_id,
        version=version,
        display_name="Room shell",
        root_link="room",
        links=("room",),
        geometries=tuple(geometries),
        physics=(_body("room", 100.0, floor),),
        evidence_state=AssetEvidenceState.PRIOR,
        metadata={
            "asset_kind": "room_shell",
            "dimensions_m": list(values),
            "collision_scope": "floor",
        },
    )
