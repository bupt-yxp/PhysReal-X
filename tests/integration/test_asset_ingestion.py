from __future__ import annotations

import struct
from pathlib import Path

import pytest

from physrealx.errors import SourcePackageError
from physrealx.platform import PhysRealX


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
