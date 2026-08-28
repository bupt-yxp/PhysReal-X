"""Backend-independent quality gates for trainable digital-twin assets."""

from __future__ import annotations

from collections import defaultdict

from physrealx.contracts import ArtifactStore
from physrealx.digital_twin import (
    AssetValidationIssue,
    AssetValidationProfile,
    AssetValidationReport,
    IssueSeverity,
)
from physrealx.domain import AssetSpec, GeometryRepresentation, GeometryRole, ParameterSource


def validate_asset(
    asset: AssetSpec,
    profile: AssetValidationProfile,
    artifacts: ArtifactStore,
) -> AssetValidationReport:
    """Validate invariants that require an assembled asset or artifact store."""

    issues: list[AssetValidationIssue] = []
    by_link: dict[str, set[GeometryRole]] = defaultdict(set)
    for geometry in asset.geometries:
        by_link[geometry.link].add(geometry.role)
        if profile is not AssetValidationProfile.SCHEMA and (
            geometry.representation is not GeometryRepresentation.PRIMITIVE
        ):
            if geometry.checksum is None:
                issues.append(
                    AssetValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code="missing_geometry_checksum",
                        message="non-primitive geometry requires a content checksum",
                        path=f"geometries.{geometry.name}",
                    )
                )
            elif not artifacts.verify(geometry.checksum):
                issues.append(
                    AssetValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code="unavailable_geometry_artifact",
                        message="geometry checksum is not available in the artifact store",
                        path=f"geometries.{geometry.name}",
                    )
                )

    if profile is not AssetValidationProfile.SCHEMA:
        for link in asset.links:
            missing = {GeometryRole.VISUAL, GeometryRole.COLLISION} - by_link[link]
            for role in sorted(missing, key=lambda item: item.value):
                issues.append(
                    AssetValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code=f"missing_{role.value}_geometry",
                        message=f"link {link!r} has no {role.value} geometry",
                        path=f"links.{link}",
                    )
                )
        for body in asset.physics:
            if body.static_friction.value < body.dynamic_friction.value:
                issues.append(
                    AssetValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code="invalid_friction_order",
                        message="static friction must not be lower than dynamic friction",
                        path=f"physics.{body.link}",
                    )
                )

    if profile is AssetValidationProfile.DIGITAL_TWIN:
        for body in asset.physics:
            parameters = {
                "mass_kg": body.mass_kg,
                "static_friction": body.static_friction,
                "dynamic_friction": body.dynamic_friction,
                "restitution": body.restitution,
            }
            for name, parameter in parameters.items():
                if parameter.source is ParameterSource.DEFAULT:
                    issues.append(
                        AssetValidationIssue(
                            severity=IssueSeverity.WARNING,
                            code="default_physics_parameter",
                            message="default physical parameters require later calibration",
                            path=f"physics.{body.link}.{name}",
                        )
                    )
                if parameter.confidence == 0:
                    issues.append(
                        AssetValidationIssue(
                            severity=IssueSeverity.WARNING,
                            code="zero_parameter_confidence",
                            message="physical parameter confidence is zero",
                            path=f"physics.{body.link}.{name}",
                        )
                    )
        for joint in asset.joints:
            if joint.effort_limit is None or joint.velocity_limit is None:
                issues.append(
                    AssetValidationIssue(
                        severity=IssueSeverity.WARNING,
                        code="incomplete_joint_limits",
                        message=(
                            "joint effort and velocity limits should be measured before deployment"
                        ),
                        path=f"joints.{joint.name}",
                    )
                )

    return AssetValidationReport(
        profile=profile,
        issues=tuple(issues),
        metrics={
            "link_count": float(len(asset.links)),
            "joint_count": float(len(asset.joints)),
            "geometry_count": float(len(asset.geometries)),
            "error_count": float(sum(item.severity is IssueSeverity.ERROR for item in issues)),
            "warning_count": float(sum(item.severity is IssueSeverity.WARNING for item in issues)),
        },
    )


def merge_validation_reports(
    worker: AssetValidationReport,
    platform: AssetValidationReport,
) -> AssetValidationReport:
    """Combine worker evidence with platform-owned structural checks."""

    issues = worker.issues + tuple(item for item in platform.issues if item not in worker.issues)
    metrics = {f"worker.{key}": value for key, value in worker.metrics.items()}
    metrics.update({f"platform.{key}": value for key, value in platform.metrics.items()})
    return AssetValidationReport(
        profile=AssetValidationProfile.DIGITAL_TWIN,
        issues=issues,
        metrics=metrics,
    )


__all__ = ["merge_validation_reports", "validate_asset"]
