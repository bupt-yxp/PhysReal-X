from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from physrealx.digital_twin import (
    AlgorithmSpec,
    AssetValidationProfile,
    AssetValidationReport,
    DigitalTwinBuildManifest,
    DigitalTwinRecipe,
    DigitalTwinStage,
    DigitalTwinStep,
    ObservationModality,
    ObservationStateRole,
    ObservationStream,
    StageExecutionRecord,
)
from physrealx.domain import ArtifactRef, ResourceType, VersionedRef


def _algorithm(name: str) -> AlgorithmSpec:
    return AlgorithmSpec(name=name, version="1.0.0")


def _steps() -> tuple[DigitalTwinStep, ...]:
    return (
        DigitalTwinStep(
            stage=DigitalTwinStage.NORMALIZATION,
            algorithm=_algorithm("normalize"),
        ),
        DigitalTwinStep(
            stage=DigitalTwinStage.CAMERA_RECOVERY,
            algorithm=_algorithm("camera_recovery"),
        ),
        DigitalTwinStep(
            stage=DigitalTwinStage.ARTICULATED_RECONSTRUCTION,
            algorithm=_algorithm("artgs"),
        ),
        DigitalTwinStep(
            stage=DigitalTwinStage.PHYSICS_ENRICHMENT,
            algorithm=_algorithm("physics_measurement"),
        ),
        DigitalTwinStep(
            stage=DigitalTwinStage.SIMULATION_PACKAGING,
            algorithm=_algorithm("openusd"),
        ),
        DigitalTwinStep(
            stage=DigitalTwinStage.QUALITY_VALIDATION,
            algorithm=_algorithm("physrealx_validator"),
        ),
    )


def _recipe(steps: tuple[DigitalTwinStep, ...] | None = None) -> DigitalTwinRecipe:
    return DigitalTwinRecipe(
        name="cabinet_twin",
        version="1.0.0",
        pipeline="test_pipeline",
        pipeline_version="1.0.0",
        steps=steps or _steps(),
    )


def _reference(resource_type: ResourceType, resource_id: str) -> VersionedRef:
    return VersionedRef(
        resource_type=resource_type,
        resource_id=resource_id,
        version="1.0.0",
        digest=f"sha256:{'a' * 64}",
    )


def test_recipe_requires_unique_ordered_pipeline_stages() -> None:
    steps = _steps()
    with pytest.raises(ValidationError, match="execution order"):
        _recipe((steps[1], steps[0], *steps[2:]))
    with pytest.raises(ValidationError, match="missing required stages"):
        _recipe(steps[:-1])
    with pytest.raises(ValidationError, match="must be unique"):
        _recipe((*steps, steps[-1]))


def test_build_manifest_requires_exact_recipe_evidence() -> None:
    recipe = _recipe()
    now = datetime.now(timezone.utc)
    records = tuple(
        StageExecutionRecord(
            stage=step.stage,
            algorithm=step.algorithm,
            started_at=now,
            completed_at=now,
        )
        for step in recipe.steps
    )

    manifest = DigitalTwinBuildManifest(
        resource_id="cabinet",
        version="1.0.0",
        source=_reference(ResourceType.OBSERVATION_SET, "cabinet_source"),
        recipe=recipe,
        asset=_reference(ResourceType.ASSET, "cabinet"),
        stages=records,
        validation=AssetValidationReport(profile=AssetValidationProfile.DIGITAL_TWIN),
    )
    assert manifest.reference().resource_type is ResourceType.ASSET_BUILD

    with pytest.raises(ValidationError, match="exactly match"):
        DigitalTwinBuildManifest(
            **manifest.model_dump(exclude={"stages"}, round_trip=True),
            stages=records[:-1],
        )


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
