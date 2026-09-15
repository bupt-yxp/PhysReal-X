from __future__ import annotations

import pytest
from pydantic import ValidationError

import physrealx.digital_twin as digital_twin
from physrealx.digital_twin import (
    FiducialSpec,
    ObservationModality,
    ObservationStateRole,
    ObservationStream,
    ViewAlignmentMode,
    ViewAlignmentSpec,
)
from physrealx.domain import ArtifactRef


def test_removed_composite_pipeline_models_are_not_available() -> None:
    removed = {
        "DigitalTwinBuildManifest",
        "DigitalTwinBuildResult",
        "DigitalTwinBuildSpec",
        "DigitalTwinRecipe",
        "DigitalTwinStage",
        "DigitalTwinStep",
        "StageExecutionRecord",
    }
    assert removed.isdisjoint(digital_twin.__all__)
    assert not any(hasattr(digital_twin, name) for name in removed)


def test_depth_observation_requires_explicit_unit() -> None:
    artifact = ArtifactRef(
        artifact_id=f"art_{'b' * 32}",
        size_bytes=1,
        media_type="application/x-npy",
    )
    with pytest.raises(ValidationError, match="explicit unit"):
        ObservationStream(
            name="can.start.depth",
            modality=ObservationModality.DEPTH,
            state="can.start",
            role=ObservationStateRole.RECONSTRUCTION,
            encoding="npy.depth.uint16",
            frame_count=1,
            artifacts=(artifact,),
            coordinate_frame="camera",
        )


def test_view_alignment_policy_is_explicit_and_modality_scoped() -> None:
    with pytest.raises(ValidationError, match="requires a fiducial"):
        ViewAlignmentSpec(mode=ViewAlignmentMode.FIDUCIAL_ASSISTED)
    with pytest.raises(ValidationError, match="cannot declare"):
        ViewAlignmentSpec(
            mode=ViewAlignmentMode.NATURAL_FEATURES,
            fiducial=FiducialSpec(family="aruco_6x6_50"),
        )

    depth = ArtifactRef(
        artifact_id=f"art_{'c' * 32}",
        size_bytes=1,
        media_type="application/x-npy",
    )
    with pytest.raises(ValidationError, match="only valid for RGB"):
        ObservationStream(
            name="can.start.depth",
            modality=ObservationModality.DEPTH,
            state="can.start",
            role=ObservationStateRole.RECONSTRUCTION,
            encoding="npy.depth.uint16",
            frame_count=1,
            artifacts=(depth,),
            unit="mm",
            view_alignment=ViewAlignmentSpec(),
        )
