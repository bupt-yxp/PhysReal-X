from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "src"
WORKERS_SOURCE = PROJECT_ROOT / "workers" / "src"
sys.path.insert(0, str(PROJECT_SOURCE))
sys.path.insert(0, str(WORKERS_SOURCE))

from physrealx_workers.colmap_worker import (  # noqa: E402
    ColmapExecutionError,
    _handle,
)

from physrealx.digital_twin import (  # noqa: E402
    ObservationModality,
    ObservationSetManifest,
    ObservationStateRole,
    ObservationStream,
)
from physrealx.domain import (  # noqa: E402
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
    VersionedRef,
    canonical_json_bytes,
    sha256_digest,
)
from physrealx.external_workers import (  # noqa: E402
    WorkerInputFile,
    WorkerRequest,
    WorkerResourceArtifact,
)


def _fake_colmap(path: Path) -> None:
    path.write_text(
        f"""\
#!{sys.executable}
import json
import sys
from pathlib import Path

arguments = sys.argv[1:]
if arguments == ["-h"]:
    print("COLMAP 4.1.0.dev0 -- Structure-from-Motion")
    raise SystemExit(0)
command = arguments[0]
if arguments[1:] == ["-h"]:
    if command == "feature_extractor":
        print("--FeatureExtraction.use_gpu --FeatureExtraction.num_threads")
    elif command == "sequential_matcher":
        print("--FeatureMatching.use_gpu --FeatureMatching.num_threads")
    raise SystemExit(0)
values = {{arguments[index]: arguments[index + 1] for index in range(1, len(arguments), 2)}}
if command == "feature_extractor":
    if values["--FeatureExtraction.use_gpu"] == "1":
        print("CUDA device is unavailable", file=sys.stderr)
        raise SystemExit(2)
    Path(values["--database_path"]).write_bytes(b"database")
elif command == "sequential_matcher":
    pass
elif command == "mapper":
    model = Path(values["--output_path"]) / "0"
    model.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(name.encode())
    (model / "project.ini").write_text(
        f"output_path={{values['--output_path']}}\\n",
        encoding="utf-8",
    )
    image_root = Path(values["--image_path"])
    names = sorted(item.relative_to(image_root).as_posix() for item in image_root.rglob("*.*"))
    (model / "names.json").write_text(json.dumps(names), encoding="utf-8")
elif command == "model_converter":
    source = Path(values["--input_path"])
    output = Path(values["--output_path"])
    names = json.loads((source / "names.json").read_text(encoding="utf-8"))
    (output / "cameras.txt").write_text(
        "# cameras\\n1 PINHOLE 640 480 500 500 320 240\\n",
        encoding="utf-8",
    )
    (output / "images.txt").write_text(
        "# images\\n"
        f"1 1 0 0 0 0 0 0 1 {{names[0]}}\\n10 10 1 20 20 2\\n"
        f"2 1 0 0 0 1 0 0 1 {{names[1]}}\\n12 12 1\\n",
        encoding="utf-8",
    )
    (output / "points3D.txt").write_text(
        "# points\\n"
        "1 0 0 0 255 0 0 0.5 1 0 2 0\\n"
        "2 1 0 0 0 255 0 0.25 1 1\\n",
        encoding="utf-8",
    )
else:
    raise SystemExit(64)
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _request(tmp_path: Path, *, allow_fallback: bool) -> WorkerRequest:
    inputs = tmp_path / "inputs"
    dependencies = inputs / "000-observations.artifacts"
    dependencies.mkdir(parents=True)
    artifacts = []
    resource_artifacts = []
    for index in range(3):
        payload = f"png-{index}".encode()
        reference = ArtifactRef(
            digest=sha256_digest(payload),
            size_bytes=len(payload),
            media_type="image/png",
        )
        destination = dependencies / f"{index:06d}.png"
        destination.write_bytes(payload)
        artifacts.append(reference)
        resource_artifacts.append(
            WorkerResourceArtifact(
                owner="observations",
                json_pointer=f"/streams/0/artifacts/{index}",
                digest=reference.digest,
                size_bytes=reference.size_bytes,
                media_type=reference.media_type,
                path=destination.relative_to(tmp_path).as_posix(),
            )
        )
    manifest = ObservationSetManifest(
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="Cabinet 2",
        importer="test_importer",
        importer_version="1.0.0",
        source_uri="file:///capture/cabinet2",
        streams=(
            ObservationStream(
                name="can.start.001.rgb",
                modality=ObservationModality.RGB,
                state="can.start",
                role=ObservationStateRole.RECONSTRUCTION,
                encoding="png",
                frame_count=3,
                artifacts=tuple(artifacts),
                coordinate_frame="camera",
            ),
        ),
    )
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path = inputs / "000-observations"
    manifest_path.write_bytes(manifest_payload)
    manifest_digest = sha256_digest(manifest_payload)
    descriptor = MethodDescriptor(
        capability=MethodCapability.CAMERA_RECOVER,
        name="colmap",
        version="1.0.0",
        api_version="1",
        source_uri="https://github.com/colmap/colmap",
        source_revision="fixture",
        license_spdx="BSD-3-Clause",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator="any",
            python="3.11",
            environment_digest=f"sha256:{'e' * 64}",
        ),
        input_contract="ObservationSetManifest/1",
        output_contract=(
            "CameraRigCalibration/1+SparseReconstructionBundle/1+CameraRecoveryReport/1"
        ),
    )
    operation = OperationSpec(
        resource_id="operation.camera.recover.fixture",
        version="1.0.0",
        method=descriptor.bind(
            config={
                "execution_device": "auto" if allow_fallback else "cuda",
                "allow_cpu_fallback": allow_fallback,
                "mapper_min_model_size": 2,
            }
        ),
        inputs=(
            OperationInput(
                name="observations",
                resource=VersionedRef(
                    resource_type=ResourceType.OBSERVATION_SET,
                    resource_id=manifest.resource_id,
                    version=manifest.version,
                    digest=manifest_digest,
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(
                name="camera_recovery_report",
                contract="CameraRecoveryReport/1",
            ),
            OperationOutputRequirement(
                name="camera_rig_calibration",
                contract="CameraRigCalibration/1",
            ),
            OperationOutputRequirement(
                name="sparse_reconstruction_bundle",
                contract="SparseReconstructionBundle/1",
            ),
        ),
    )
    return WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="observations",
                kind="resource",
                digest=manifest_digest,
                size_bytes=len(manifest_payload),
                media_type="application/vnd.physrealx.observation_set+json",
                path=manifest_path.relative_to(tmp_path).as_posix(),
                resource_type=ResourceType.OBSERVATION_SET,
            ),
        ),
        resource_artifacts=tuple(resource_artifacts),
    )


def test_colmap_worker_recovers_camera_bundle_and_records_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "fake-colmap"
    _fake_colmap(binary)
    request = _request(tmp_path, allow_fallback=True)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    response = _handle(request, output, colmap_binary=binary)

    assert [item.name for item in response.outputs] == [
        "camera_recovery_report",
        "camera_rig_calibration",
        "sparse_reconstruction_bundle",
    ]
    assert response.metrics["registered_images"] == 2.0
    assert response.metrics["registration_rate"] == pytest.approx(2 / 3)
    assert response.metrics["sparse_points"] == 2.0
    assert response.metrics["mean_track_length"] == 1.5
    assert response.metrics["mean_reprojection_error_px"] == pytest.approx(5 / 12)
    assert response.metrics["used_gpu"] == 0.0
    assert response.metrics["cpu_fallback_used"] == 1.0
    report = json.loads((output / "camera-recovery-report.json").read_text(encoding="utf-8"))
    assert report["execution"]["requested_device"] == "auto"
    assert report["execution"]["effective_device"] == "cpu"
    assert "CUDA device is unavailable" in report["execution"]["fallback_reason"]
    assert report["scale_status"] == "unresolved"
    rig = json.loads((output / "camera-rig-calibration.json").read_text(encoding="utf-8"))
    assert rig["schema"] == "physrealx.camera-rig-calibration/1"
    assert rig["scale_status"] == "unresolved"
    assert len(rig["components"]) == 1
    assert len(rig["components"][0]["registered_views"]) == 2
    with zipfile.ZipFile(output / "sparse-reconstruction.zip") as archive:
        assert set(archive.namelist()) >= {
            "components/0/binary/cameras.bin",
            "components/0/binary/images.bin",
            "components/0/binary/points3D.bin",
            "components/0/text/cameras.txt",
            "components/0/text/images.txt",
            "components/0/text/points3D.txt",
            "bundle-manifest.json",
        }
        assert "components/0/binary/project.ini" not in archive.namelist()
        assert "components/0/binary/names.json" not in archive.namelist()
        bundle = json.loads(archive.read("bundle-manifest.json"))
        assert bundle["scale_status"] == "unresolved"
        assert all("project.ini" not in item["path"] for item in bundle["files"])
    assert "log_tail" not in report["execution"]
    assert not (tmp_path / "colmap-work").exists()


def test_colmap_worker_does_not_hide_explicit_cuda_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "fake-colmap"
    _fake_colmap(binary)
    request = _request(tmp_path, allow_fallback=False)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ColmapExecutionError, match="CUDA device is unavailable"):
        _handle(request, output, colmap_binary=binary)

    assert not (tmp_path / "colmap-work").exists()
    assert list(output.iterdir()) == []
