from __future__ import annotations

from pathlib import Path

from physrealx import (
    ArtifactRef,
    AssetEvidenceState,
    AssetValidationProfile,
    AssetValidationReport,
    PhysRealX,
)
from physrealx.domain import GeometryRole, ParameterSource, ScalarParameter
from physrealx.reference import reference_asset


def _codes(report: AssetValidationReport) -> set[str]:
    return {item.code for item in report.issues}


def test_validation_profiles_add_only_their_own_quality_gates(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()

    schema = platform.assets.validate(asset, profile=AssetValidationProfile.SCHEMA)
    digital_twin = platform.assets.validate(asset, profile=AssetValidationProfile.DIGITAL_TWIN)
    trainable = platform.assets.validate(asset, profile=AssetValidationProfile.TRAINABLE)

    assert schema.metrics["error_count"] == 0
    assert digital_twin.metrics["error_count"] == 0
    assert _codes(digital_twin) == {"incomplete_joint_limits"}
    assert trainable.metrics["error_count"] > 0
    assert _codes(trainable) == {"asset_not_validated", "incomplete_joint_limits"}


def test_trainable_validation_rejects_missing_evidence_and_invalid_friction(
    tmp_path: Path,
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()
    missing = ArtifactRef(
        artifact_id=f"art_{'a' * 32}",
        size_bytes=1,
        media_type="application/json",
    )
    low_static_friction = ScalarParameter(
        value=0.1,
        source=ParameterSource.DEFAULT,
        confidence=0.0,
    )
    body = asset.physics[0].model_copy(update={"static_friction": low_static_friction})
    candidate = asset.model_copy(
        update={
            "evidence_state": AssetEvidenceState.VALIDATED,
            "evidence": (missing,),
            "geometries": tuple(
                geometry
                for geometry in asset.geometries
                if not (geometry.link == "drawer" and geometry.role is GeometryRole.COLLISION)
            ),
            "physics": (body, *asset.physics[1:]),
        }
    )

    report = platform.assets.validate(candidate, profile=AssetValidationProfile.TRAINABLE)

    assert report.metrics["error_count"] > 0
    assert {
        "default_physics_parameter",
        "invalid_friction_order",
        "missing_collision_geometry",
        "unavailable_validation_evidence",
        "zero_parameter_confidence",
    } <= _codes(report)
