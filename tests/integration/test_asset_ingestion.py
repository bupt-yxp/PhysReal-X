from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

from physrealx.contracts import ArtifactStore, DigitalTwinPipelineOutput
from physrealx.digital_twin import (
    AlgorithmSpec,
    AssetValidationProfile,
    AssetValidationReport,
    DigitalTwinBuildSpec,
    DigitalTwinRecipe,
    DigitalTwinStage,
    DigitalTwinStep,
    ObservationSetManifest,
    StageExecutionRecord,
)
from physrealx.errors import ResourceConflictError, SourcePackageError
from physrealx.platform import PhysRealX
from physrealx.reference import reference_asset


def _write_npy(path: Path, count: int) -> None:
    header = f"{{'descr': '|S1', 'fortran_order': False, 'shape': ({count},), }}"
    padding = (-((10 + len(header) + 1) % 16)) % 16
    encoded_header = f"{header}{' ' * padding}\n".encode("latin1")
    path.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(encoded_header))
        + encoded_header
        + bytes(range(count))
    )


def _source(root: Path, *, start_count: int = 2, end_count: int = 3) -> Path:
    for state, count in (("start", start_count), ("end", end_count)):
        directory = root / "real" / "can" / state / "img" / "001"
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.png").write_bytes(f"rgb-{state}-{index}".encode())
        depth = root / "real" / "can" / state / "depth"
        depth.mkdir()
        _write_npy(depth / "001_depth.npy", count)
    recon = root / "real" / "recon"
    recon.mkdir(parents=True)
    (recon / "candidate.ply").write_bytes(b"ply\n")
    simulation = root / "sim"
    simulation.mkdir()
    (simulation / "candidate.usd").write_text("#usda 1.0\n", encoding="utf-8")
    return root


def _recipe() -> DigitalTwinRecipe:
    algorithms = {
        DigitalTwinStage.NORMALIZATION: "normalize",
        DigitalTwinStage.CAMERA_RECOVERY: "camera_recovery",
        DigitalTwinStage.ARTICULATED_RECONSTRUCTION: "artgs",
        DigitalTwinStage.PHYSICS_ENRICHMENT: "physics_measurement",
        DigitalTwinStage.SIMULATION_PACKAGING: "openusd",
        DigitalTwinStage.QUALITY_VALIDATION: "physrealx_validator",
    }
    return DigitalTwinRecipe(
        name="cabinet_twin",
        version="1.0.0",
        pipeline="test_pipeline",
        pipeline_version="1.0.0",
        steps=tuple(
            DigitalTwinStep(
                stage=stage,
                algorithm=AlgorithmSpec(name=name, version="1.0.0"),
            )
            for stage, name in algorithms.items()
        ),
    )


class _Pipeline:
    name = "test_pipeline"
    plugin_version = "1.0.0"
    api_version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def build(
        self,
        spec: DigitalTwinBuildSpec,
        source: ObservationSetManifest,
        artifacts: ArtifactStore,
    ) -> DigitalTwinPipelineOutput:
        del source, artifacts
        self.calls += 1
        now = datetime.now(timezone.utc)
        stages = tuple(
            StageExecutionRecord(
                stage=step.stage,
                algorithm=step.algorithm,
                started_at=now,
                completed_at=now,
            )
            for step in spec.recipe.steps
        )
        asset = reference_asset().model_copy(
            update={
                "resource_id": spec.resource_id,
                "version": spec.version,
                "display_name": "Imported cabinet digital twin",
            }
        )
        return DigitalTwinPipelineOutput(
            asset=asset,
            stages=stages,
            validation=AssetValidationReport(profile=AssetValidationProfile.DIGITAL_TWIN),
        )


@pytest.mark.integration
def test_artmodeling_source_is_inspected_and_ingested_immutably(tmp_path: Path) -> None:
    source = _source(tmp_path / "cabinet2")
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    inspection = platform.assets.inspect_source(source)
    assert inspection.ingestible
    assert {stream.name for stream in inspection.streams} == {
        "can.start.001.rgb",
        "can.start.001.depth",
        "can.end.001.rgb",
        "can.end.001.depth",
    }
    assert {issue.code for issue in inspection.issues} == {
        "missing_camera_calibration",
        "unspecified_depth_unit",
    }

    reference = platform.assets.ingest(
        source,
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2 observations",
        config={"depth_unit": "mm", "coordinate_frame": "camera"},
    )
    assert reference == platform.assets.ingest(
        source,
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2 observations",
        config={"depth_unit": "mm", "coordinate_frame": "camera"},
    )
    manifest = platform.assets.get_observations(reference)
    assert manifest.metadata["depth_unit"] == "mm"
    assert len(manifest.attachments) == 2
    assert all(
        platform.artifacts.verify(artifact)
        for stream in manifest.streams
        for artifact in stream.artifacts
    )


@pytest.mark.integration
def test_artmodeling_ingestion_rejects_incomplete_or_ambiguous_sources(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "bad", start_count=2)
    _write_npy(source / "real" / "can" / "start" / "depth" / "001_depth.npy", 1)
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    inspection = platform.assets.inspect_source(source)
    assert not inspection.ingestible
    assert "rgb_depth_count_mismatch" in {issue.code for issue in inspection.issues}
    with pytest.raises(SourcePackageError, match="failed inspection"):
        platform.assets.ingest(
            source,
            resource_id="bad.observations",
            version="1.0.0",
            display_name="Bad observations",
            config={"depth_unit": "mm", "coordinate_frame": "camera"},
        )


@pytest.mark.integration
def test_digital_twin_build_is_validated_persisted_and_idempotent(tmp_path: Path) -> None:
    source_path = _source(tmp_path / "cabinet2")
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    source = platform.assets.ingest(
        source_path,
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2 observations",
        config={"depth_unit": "mm", "coordinate_frame": "camera"},
    )
    pipeline = _Pipeline()
    platform.plugins.register_asset_pipeline(pipeline)
    spec = DigitalTwinBuildSpec(
        resource_id="cabinet2",
        version="1.0.0",
        source=source,
        recipe=_recipe(),
    )

    result = platform.assets.build(spec)
    assert result.validation.passed
    assert platform.assets.get(result.asset).display_name == "Imported cabinet digital twin"
    assert platform.assets.get_build(result.build).source == source
    assert platform.assets.build(spec) == result
    assert pipeline.calls == 1

    changed = spec.model_copy(
        update={
            "recipe": spec.recipe.model_copy(update={"version": "1.0.1"}),
        }
    )
    with pytest.raises(ResourceConflictError, match="different source or recipe"):
        platform.assets.build(changed)
