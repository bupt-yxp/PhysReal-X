from __future__ import annotations

import pytest
from pydantic import ValidationError

import physrealx.digital_twin as digital_twin
from physrealx.digital_twin import ObservationModality, ObservationStateRole, ObservationStream
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
        digest=f"sha256:{'b' * 64}",
        size_bytes=1,
        media_type="application/x-npy",
    )
    with pytest.raises(ValidationError, match="explicit unit"):
        ObservationStream(
            name="can.start.001.depth",
            modality=ObservationModality.DEPTH,
            state="can.start",
            role=ObservationStateRole.RECONSTRUCTION,
            encoding="npy.packed_png16",
            frame_count=1,
            artifacts=(artifact,),
            coordinate_frame="camera",
        )
