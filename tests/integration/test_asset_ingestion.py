from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physrealx import ViewAlignmentMode
from physrealx.errors import ResourceConflictError, SourcePackageError
from physrealx.platform import PhysRealX


def _write_depth(directory: Path, count: int) -> None:
    directory.mkdir()
    for index in range(count):
        np.save(directory / f"{index}.npy", np.full((2, 3), index, np.uint16))


def _source(root: Path, *, start_count: int = 2, end_count: int = 3) -> Path:
    for state, count in (("start", start_count), ("end", end_count)):
        directory = root / "real" / "can" / state / "img"
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.png").write_bytes(f"rgb-{state}-{index}".encode())
        masks = root / "real" / "can" / state / "mask"
        masks.mkdir(parents=True)
        for index in range(count):
            (masks / f"{index}.png").write_bytes(f"mask-{state}-{index}".encode())
        _write_depth(root / "real" / "can" / state / "depth", count)
    recon = root / "real" / "recon"
    recon.mkdir(parents=True)
    (recon / "candidate.ply").write_bytes(b"ply\n")
    simulation = root / "sim"
    simulation.mkdir()
    (simulation / "candidate.usd").write_text("#usda 1.0\n", encoding="utf-8")
    return root


@pytest.mark.integration
def test_artmodeling_source_is_inspected_and_ingested_immutably(tmp_path: Path) -> None:
    source = _source(tmp_path / "cabinet2")
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    inspection = platform.assets.inspect_source(source)
    assert inspection.ingestible
    assert {stream.name for stream in inspection.streams} == {
        "can.start.rgb",
        "can.start.depth",
        "can.start.mask",
        "can.end.rgb",
        "can.end.depth",
        "can.end.mask",
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
        config={
            "depth_unit": "mm",
            "coordinate_frame": "camera",
            "view_alignment": {
                "can.start.rgb": {
                    "mode": "fiducial_assisted",
                    "fiducial": {"family": "aruco_6x6_50"},
                },
                "can.end.rgb": {"mode": "natural_features"},
            },
        },
    )
    assert reference == platform.assets.ingest(
        source,
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2 observations",
        config={
            "depth_unit": "mm",
            "coordinate_frame": "camera",
            "view_alignment": {
                "can.start.rgb": {
                    "mode": "fiducial_assisted",
                    "fiducial": {"family": "aruco_6x6_50"},
                },
                "can.end.rgb": {"mode": "natural_features"},
            },
        },
    )
    manifest = platform.assets.get_observations(reference)
    assert platform.assets.resolve_observation_source(reference) == source
    assert manifest.metadata["depth_unit"] == "mm"
    rgb_streams = {
        stream.name: stream for stream in manifest.streams if stream.name.endswith(".rgb")
    }
    assert rgb_streams["can.start.rgb"].view_alignment.mode is ViewAlignmentMode.FIDUCIAL_ASSISTED
    assert rgb_streams["can.end.rgb"].view_alignment.mode is ViewAlignmentMode.NATURAL_FEATURES
    assert len(manifest.attachments) == 2
    assert all(
        platform.artifacts.verify(artifact)
        for stream in manifest.streams
        for artifact in stream.artifacts
    )


@pytest.mark.integration
def test_project_source_identity_survives_symlinked_checkout_access(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = _source(project / "input" / "cabinet2")
    alias = tmp_path / "project-alias"
    alias.symlink_to(project, target_is_directory=True)
    platform = PhysRealX.open(alias / ".physrealx", discover_plugins=False)
    config = {"depth_unit": "mm", "coordinate_frame": "camera"}

    inspection = platform.assets.inspect_source(alias / "input" / "cabinet2", config=config)
    assert inspection.source_uri == "project:///input/cabinet2"
    reference = platform.assets.ingest(
        alias / "input" / "cabinet2",
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2 observations",
        config=config,
    )
    assert reference == platform.assets.ingest(
        source,
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2 observations",
        config=config,
    )
    assert platform.assets.get_observations(reference).source_uri == "project:///input/cabinet2"
    assert platform.assets.resolve_observation_source(reference) == source

    different_source = _source(project / "input" / "other")
    with pytest.raises(ResourceConflictError, match="different or unaudited"):
        platform.assets.ingest(
            different_source,
            resource_id="cabinet2.observations",
            version="1.0.0",
            display_name="Cabinet 2 observations",
            config=config,
        )


@pytest.mark.integration
@pytest.mark.parametrize("project_uri", [False, True])
def test_observation_source_resolution_rejects_missing_captures(
    tmp_path: Path, project_uri: bool
) -> None:
    source = _source(tmp_path / "input" / "captured laptop")
    platform = PhysRealX.open(
        tmp_path / (".physrealx" if project_uri else "workspace"), discover_plugins=False
    )
    reference = platform.assets.ingest(
        source,
        resource_id="capture",
        version="1.0.0",
        display_name="Capture",
        config={"depth_unit": "mm", "coordinate_frame": "camera"},
    )
    assert platform.assets.resolve_observation_source(reference) == source
    source.rename(source.with_name("moved capture"))
    with pytest.raises(SourcePackageError, match="missing or cannot be resolved locally"):
        platform.assets.resolve_observation_source(reference)


@pytest.mark.integration
def test_project_observation_source_rejects_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = _source(project / "input" / "capture")
    platform = PhysRealX.open(project / ".physrealx", discover_plugins=False)
    reference = platform.assets.ingest(
        source,
        resource_id="capture",
        version="1.0.0",
        display_name="Capture",
        config={"depth_unit": "mm", "coordinate_frame": "camera"},
    )
    outside = tmp_path / "outside"
    source.rename(outside)
    source.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourcePackageError, match="cannot be resolved locally"):
        platform.assets.resolve_observation_source(reference)


@pytest.mark.integration
def test_artmodeling_ingestion_rejects_incomplete_or_ambiguous_sources(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "bad", start_count=2)
    (source / "real" / "can" / "start" / "depth" / "1.npy").unlink()
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    inspection = platform.assets.inspect_source(source)
    assert not inspection.ingestible
    assert "rgb_depth_mask_frame_mismatch" in {issue.code for issue in inspection.issues}
    with pytest.raises(SourcePackageError, match="failed inspection"):
        platform.assets.ingest(
            source,
            resource_id="bad.observations",
            version="1.0.0",
            display_name="Bad observations",
            config={"depth_unit": "mm", "coordinate_frame": "camera"},
        )


@pytest.mark.integration
def test_artmodeling_ingestion_rejects_unknown_alignment_stream(tmp_path: Path) -> None:
    source = _source(tmp_path / "cabinet2")
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    with pytest.raises(SourcePackageError, match="unknown or non-RGB"):
        platform.assets.ingest(
            source,
            resource_id="cabinet2.observations",
            version="1.0.0",
            display_name="Cabinet 2 observations",
            config={
                "depth_unit": "mm",
                "coordinate_frame": "camera",
                "view_alignment": {
                    "missing.rgb": {"mode": "natural_features"},
                },
            },
        )


@pytest.mark.integration
def test_selected_states_exclude_incomplete_nested_video(tmp_path):
    source = _source(tmp_path / "oven4")
    images = source / "real/video/end/img"
    images.mkdir(parents=True)
    (images / "0.png").write_bytes(b"image")
    depth = source / "real/video/end/depth"
    _write_depth(depth, 2)
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    assert not px.assets.inspect_source(source).ingestible
    selected = {
        "states": ["can.start", "can.end"],
        "depth_unit": "mm",
        "coordinate_frame": "camera",
    }
    inspection = px.assets.inspect_source(source, config=selected)
    assert inspection.ingestible
    assert len(inspection.streams) == 6
    obs = px.assets.ingest(
        source, resource_id="oven4", version="1.0.0", display_name="Oven", config=selected
    )
    assert len(px.assets.get_observations(obs).streams) == 6
    for states in ([], ["can.start", "can.start"], ["missing"]):
        with pytest.raises(SourcePackageError, match="states"):
            px.assets.inspect_source(source, config={"states": states})


@pytest.mark.integration
def test_artmodeling_preserves_matching_non_contiguous_frame_ids(tmp_path: Path) -> None:
    source = _source(tmp_path / "capture")
    for modality, suffix in (("img", ".png"), ("depth", ".npy"), ("mask", ".png")):
        directory = source / "real/can/start" / modality
        (directory / f"1{suffix}").rename(directory / f"7{suffix}")
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    inspection = platform.assets.inspect_source(source)
    assert inspection.ingestible
    reference = platform.assets.ingest(
        source,
        resource_id="capture",
        version="1.0.0",
        display_name="Capture",
        config={"depth_unit": "mm", "coordinate_frame": "camera"},
    )
    manifest = platform.assets.get_observations(reference)
    streams = {stream.name: stream for stream in manifest.streams}
    assert streams["can.start.rgb"].metadata["frame_ids"] == ["0", "7"]
    assert streams["can.start.depth"].metadata["frame_ids"] == ["0", "7"]
    assert streams["can.start.mask"].metadata["frame_ids"] == ["0", "7"]
