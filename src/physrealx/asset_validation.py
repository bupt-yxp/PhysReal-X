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
from physrealx.domain import (
    AssetEvidenceState,
    AssetSpec,
    GeometryRole,
    ParameterSource,
)


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

    if profile in {
        AssetValidationProfile.DIGITAL_TWIN,
        AssetValidationProfile.TRAINABLE,
    }:
        strict = profile is AssetValidationProfile.TRAINABLE
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
                            severity=IssueSeverity.ERROR if strict else IssueSeverity.WARNING,
                            code="default_physics_parameter",
                            message="default physical parameters require later calibration",
                            path=f"physics.{body.link}.{name}",
                        )
                    )
                if parameter.confidence == 0:
                    issues.append(
                        AssetValidationIssue(
                            severity=IssueSeverity.ERROR if strict else IssueSeverity.WARNING,
                            code="zero_parameter_confidence",
                            message="physical parameter confidence is zero",
                            path=f"physics.{body.link}.{name}",
                        )
                    )
        for joint in asset.joints:
            if joint.effort_limit is None or joint.velocity_limit is None:
                issues.append(
                    AssetValidationIssue(
                        severity=IssueSeverity.ERROR if strict else IssueSeverity.WARNING,
                        code="incomplete_joint_limits",
                        message=(
                            "joint effort and velocity limits should be measured before deployment"
                        ),
                        path=f"joints.{joint.name}",
                    )
                )

        if strict:
            if asset.evidence_state is not AssetEvidenceState.VALIDATED:
                issues.append(
                    AssetValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code="asset_not_validated",
                        message="a trainable asset requires validated physical evidence",
                        path="evidence_state",
                    )
                )
            for index, evidence in enumerate(asset.evidence):
                if not artifacts.verify(evidence):
                    issues.append(
                        AssetValidationIssue(
                            severity=IssueSeverity.ERROR,
                            code="unavailable_validation_evidence",
                            message="asset validation evidence is unavailable or corrupt",
                            path=f"evidence.{index}",
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


__all__ = ["validate_asset"]
