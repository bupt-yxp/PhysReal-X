"""COLMAP camera recovery worker for immutable observation-set resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physrealx.digital_twin import (
    ObservationModality,
    ObservationSetManifest,
    ObservationStateRole,
)
from physrealx.external_workers import (
    WorkerProducedFile,
    WorkerRequest,
    WorkerResponse,
    run_worker,
)
from physrealx_workers.common import config, dependency_path, report_probe, resource_json

_VERSION = re.compile(r"COLMAP\s+([^\s]+)")
_ALLOWED_CONFIG = {
    "allow_cpu_fallback",
    "camera_model",
    "camera_params",
    "camera_policy",
    "execution_device",
    "frame_stride",
    "gpu_index",
    "mapper_min_model_size",
    "max_frames_per_stream",
    "num_threads",
    "quadratic_overlap",
    "random_seed",
    "roles",
    "sequential_overlap",
    "states",
    "stream_names",
}
_OUTPUT_FILES = {
    "camera_recovery_report": (
        "camera-recovery-report.json",
        "application/json",
        "CameraRecoveryReport/1",
    ),
    "camera_rig_calibration": (
        "camera-rig-calibration.json",
        "application/json",
        "CameraRigCalibration/1",
    ),
    "sparse_reconstruction_bundle": (
        "sparse-reconstruction.zip",
        "application/zip",
        "SparseReconstructionBundle/1",
    ),
}
_SPARSE_MODEL_FILES = {
    "binary": frozenset(
        {
            "cameras.bin",
            "frames.bin",
            "images.bin",
            "points3D.bin",
            "rigs.bin",
        }
    ),
    "text": frozenset(
        {
            "cameras.txt",
            "frames.txt",
            "images.txt",
            "points3D.txt",
            "rigs.txt",
        }
    ),
}


class ColmapExecutionError(RuntimeError):
    """One native COLMAP stage failed with bounded diagnostic context."""

    def __init__(self, stage: str, returncode: int, detail: str) -> None:
        super().__init__(
            f"COLMAP {stage} failed with exit code {returncode}: {detail or 'no diagnostic output'}"
        )
        self.stage = stage
        self.returncode = returncode
        self.detail = detail


@dataclass(frozen=True)
class _SelectedImage:
    staged_name: str
    stream_name: str
    state: str
    frame_index: int
    digest: str
    source: Path
    suffix: str


@dataclass(frozen=True)
class _SparseStatistics:
    registered_images: int
    point_count: int
    observation_count: int
    mean_track_length: float
    mean_reprojection_error_px: float


@dataclass(frozen=True)
class _SparseComponent:
    component_id: str
    binary_model: Path
    text_model: Path
    cameras: list[dict[str, Any]]
    images: list[dict[str, Any]]
    statistics: _SparseStatistics


def _require_integer(options: dict[str, Any], name: str, default: int, minimum: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"COLMAP option {name!r} must be an integer >= {minimum}")
    return value


def _require_boolean(options: dict[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"COLMAP option {name!r} must be a boolean")
    return value


def _string_sequence(
    options: dict[str, Any],
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = options.get(name, default)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"COLMAP option {name!r} must be a non-empty string array")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"COLMAP option {name!r} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _validated_options(request: WorkerRequest) -> dict[str, Any]:
    options = config(request)
    unknown = sorted(set(options) - _ALLOWED_CONFIG)
    if unknown:
        raise ValueError(f"unknown COLMAP worker options: {unknown}")
    device = options.get("execution_device", "auto")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("execution_device must be one of: auto, cpu, cuda")
    policy = options.get("camera_policy", "single")
    if policy not in {"single", "per_stream", "per_image"}:
        raise ValueError("camera_policy must be one of: single, per_stream, per_image")
    camera_model = options.get("camera_model", "SIMPLE_RADIAL")
    if not isinstance(camera_model, str) or not camera_model.strip():
        raise ValueError("camera_model must be a non-empty COLMAP camera model")
    camera_params = options.get("camera_params")
    if camera_params is not None and (
        not isinstance(camera_params, str) or not camera_params.strip()
    ):
        raise ValueError("camera_params must be a non-empty comma-separated string")
    _require_boolean(options, "allow_cpu_fallback", True)
    _require_boolean(options, "quadratic_overlap", True)
    _require_integer(options, "frame_stride", 1, 1)
    _require_integer(options, "max_frames_per_stream", 100_000, 2)
    _require_integer(options, "sequential_overlap", 10, 1)
    _require_integer(options, "mapper_min_model_size", 3, 2)
    _require_integer(options, "num_threads", 1, 1)
    _require_integer(options, "random_seed", 0, 0)
    gpu_index = options.get("gpu_index", -1)
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < -1:
        raise ValueError("gpu_index must be an integer >= -1")
    for name in ("states", "stream_names"):
        if name in options:
            _string_sequence(options, name, ())
    if "roles" in options:
        roles = _string_sequence(options, "roles", ())
        known = {item.value for item in ObservationStateRole}
        invalid = sorted(set(roles) - known)
        if invalid:
            raise ValueError(f"unknown observation roles: {invalid}")
    return options


def _select_images(
    request: WorkerRequest,
    manifest: ObservationSetManifest,
    options: dict[str, Any],
) -> tuple[_SelectedImage, ...]:
    roles = set(
        _string_sequence(
            options,
            "roles",
            (ObservationStateRole.RECONSTRUCTION.value,),
        )
    )
    states = set(_string_sequence(options, "states", ())) if "states" in options else None
    stream_names = (
        set(_string_sequence(options, "stream_names", ())) if "stream_names" in options else None
    )
    stride = _require_integer(options, "frame_stride", 1, 1)
    maximum = _require_integer(options, "max_frames_per_stream", 100_000, 2)
    selected: list[_SelectedImage] = []
    for stream_index, stream in enumerate(manifest.streams):
        if stream.modality is not ObservationModality.RGB or stream.role.value not in roles:
            continue
        if states is not None and stream.state not in states:
            continue
        if stream_names is not None and stream.name not in stream_names:
            continue
        artifacts = stream.artifacts[::stride][:maximum]
        for sampled_index, artifact in enumerate(artifacts):
            suffixes = {"image/jpeg": ".jpg", "image/png": ".png", "image/tiff": ".tiff"}
            suffix = suffixes.get(artifact.media_type)
            if suffix is None:
                raise ValueError(
                    f"COLMAP RGB stream {stream.name!r} uses unsupported media type "
                    f"{artifact.media_type!r}"
                )
            frame_index = sampled_index * stride
            selected.append(
                _SelectedImage(
                    staged_name=f"stream-{stream_index:04d}/frame-{frame_index:08d}{suffix}",
                    stream_name=stream.name,
                    state=stream.state,
                    frame_index=frame_index,
                    digest=artifact.digest,
                    source=dependency_path(request, "observations", artifact.digest),
                    suffix=suffix,
                )
            )
    if len(selected) < 2:
        raise ValueError("COLMAP camera recovery requires at least two selected RGB frames")
    return tuple(selected)


def _stage_images(images: tuple[_SelectedImage, ...], image_root: Path) -> None:
    for image in images:
        destination = image_root / image.staged_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(image.source, destination)
        except OSError:
            shutil.copyfile(image.source, destination)
        destination.chmod(0o400)


def _tail(path: Path, limit: int = 16_384) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _run_colmap(binary: Path, stage: str, arguments: list[str], cwd: Path, log: Path) -> None:
    command = [str(binary), stage, *arguments]
    with log.open("ab") as stream:
        stream.write((f"$ {shlex.join(command)}\n").encode())
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise ColmapExecutionError(stage, completed.returncode, _tail(log))


def _help(binary: Path, command: str) -> str:
    completed = subprocess.run(
        [str(binary), command, "-h"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    text = completed.stdout[-256_000:].decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"cannot inspect COLMAP {command} options: {text[-4096:]}")
    return text


def _gpu_option(help_text: str, modern: str, legacy: str) -> str:
    if modern in help_text:
        return modern
    if legacy in help_text:
        return legacy
    raise RuntimeError(f"installed COLMAP does not expose {modern!r} or {legacy!r}")


def _gpu_failure(error: ColmapExecutionError) -> bool:
    if error.stage not in {"feature_extractor", "sequential_matcher"}:
        return False
    detail = error.detail.lower()
    return any(marker in detail for marker in ("cuda", "gpu", "device"))


def _camera_policy_arguments(policy: str) -> list[str]:
    if policy == "single":
        return ["--ImageReader.single_camera", "1"]
    if policy == "per_stream":
        return ["--ImageReader.single_camera_per_folder", "1"]
    return ["--ImageReader.single_camera_per_image", "1"]


def _attempt_reconstruction(
    binary: Path,
    images: tuple[_SelectedImage, ...],
    options: dict[str, Any],
    attempt_root: Path,
    *,
    use_gpu: bool,
) -> tuple[Path, tuple[Path, ...]]:
    image_root = attempt_root / "images"
    sparse_root = attempt_root / "sparse"
    text_root = attempt_root / "text"
    database = attempt_root / "database.db"
    log = attempt_root / "colmap.log"
    image_root.mkdir(parents=True)
    sparse_root.mkdir()
    text_root.mkdir()
    _stage_images(images, image_root)
    feature_help = _help(binary, "feature_extractor")
    matcher_help = _help(binary, "sequential_matcher")
    mapper_help = _help(binary, "mapper")
    feature_gpu = _gpu_option(
        feature_help,
        "--FeatureExtraction.use_gpu",
        "--SiftExtraction.use_gpu",
    )
    matcher_gpu = _gpu_option(
        matcher_help,
        "--FeatureMatching.use_gpu",
        "--SiftMatching.use_gpu",
    )
    threads = _require_integer(options, "num_threads", 1, 1)
    random_seed = _require_integer(options, "random_seed", 0, 0)
    gpu_index = int(options.get("gpu_index", -1))
    feature_prefix = feature_gpu.rsplit(".", 1)[0]
    matcher_prefix = matcher_gpu.rsplit(".", 1)[0]
    feature_arguments = [
        "--database_path",
        str(database),
        "--image_path",
        str(image_root),
        "--ImageReader.camera_model",
        str(options.get("camera_model", "SIMPLE_RADIAL")),
        *_camera_policy_arguments(str(options.get("camera_policy", "single"))),
        feature_gpu,
        str(int(use_gpu)),
        f"{feature_prefix}.num_threads",
        str(threads),
    ]
    if "--default_random_seed" in feature_help:
        feature_arguments.extend(("--default_random_seed", str(random_seed)))
    if gpu_index >= 0:
        gpu_name = feature_prefix + ".gpu_index"
        feature_arguments.extend((gpu_name, str(gpu_index)))
    camera_params = options.get("camera_params")
    if camera_params is not None:
        feature_arguments.extend(("--ImageReader.camera_params", str(camera_params)))
    _run_colmap(binary, "feature_extractor", feature_arguments, attempt_root, log)
    matcher_arguments = [
        "--database_path",
        str(database),
        matcher_gpu,
        str(int(use_gpu)),
        f"{matcher_prefix}.num_threads",
        str(threads),
        "--SequentialMatching.overlap",
        str(_require_integer(options, "sequential_overlap", 10, 1)),
        "--SequentialMatching.quadratic_overlap",
        str(int(_require_boolean(options, "quadratic_overlap", True))),
    ]
    if "--default_random_seed" in matcher_help:
        matcher_arguments.extend(("--default_random_seed", str(random_seed)))
    if "--TwoViewGeometry.random_seed" in matcher_help:
        matcher_arguments.extend(("--TwoViewGeometry.random_seed", str(random_seed)))
    if gpu_index >= 0:
        gpu_name = matcher_prefix + ".gpu_index"
        matcher_arguments.extend((gpu_name, str(gpu_index)))
    _run_colmap(binary, "sequential_matcher", matcher_arguments, attempt_root, log)
    mapper_arguments = [
        "--database_path",
        str(database),
        "--image_path",
        str(image_root),
        "--output_path",
        str(sparse_root),
        "--Mapper.min_model_size",
        str(_require_integer(options, "mapper_min_model_size", 3, 2)),
        "--Mapper.num_threads",
        str(threads),
    ]
    if "--default_random_seed" in mapper_help:
        mapper_arguments.extend(("--default_random_seed", str(random_seed)))
    if "--Mapper.random_seed" in mapper_help:
        mapper_arguments.extend(("--Mapper.random_seed", str(random_seed)))
    _run_colmap(
        binary,
        "mapper",
        mapper_arguments,
        attempt_root,
        log,
    )
    models = tuple(
        path
        for path in sorted(sparse_root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
        and not path.is_symlink()
        and any(path.glob("cameras.*"))
        and any(path.glob("images.*"))
    )
    if not models:
        raise RuntimeError("COLMAP mapper produced no registered sparse model")
    text_models: list[Path] = []
    for model in models:
        destination = text_root / model.name
        destination.mkdir()
        _run_colmap(
            binary,
            "model_converter",
            [
                "--input_path",
                str(model),
                "--output_path",
                str(destination),
                "--output_type",
                "TXT",
            ],
            attempt_root,
            log,
        )
        text_models.append(destination)
    return sparse_root, tuple(text_models)


def _data_lines(path: Path) -> list[str]:
    return [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]


def _parse_cameras(path: Path) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for line in _data_lines(path):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise RuntimeError(f"invalid COLMAP camera record: {line!r}")
        cameras.append(
            {
                "camera_id": int(fields[0]),
                "model": fields[1],
                "width": int(fields[2]),
                "height": int(fields[3]),
                "parameters": [float(value) for value in fields[4:]],
            }
        )
    return cameras


def _parse_images(path: Path, selected: dict[str, _SelectedImage]) -> list[dict[str, Any]]:
    lines = _data_lines(path)
    images: list[dict[str, Any]] = []
    expecting_image = True
    for line in lines:
        if expecting_image:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 10:
                raise RuntimeError(f"invalid COLMAP image record: {line!r}")
            name = " ".join(fields[9:])
            source = selected.get(name)
            if source is None:
                raise RuntimeError(f"COLMAP registered an undeclared staged image: {name!r}")
            images.append(
                {
                    "image_id": int(fields[0]),
                    "camera_id": int(fields[8]),
                    "image_name": name,
                    "source": {
                        "stream_name": source.stream_name,
                        "state": source.state,
                        "frame_index": source.frame_index,
                        "artifact_digest": source.digest,
                    },
                    "world_to_camera": {
                        "quaternion_wxyz": [float(value) for value in fields[1:5]],
                        "translation": [float(value) for value in fields[5:8]],
                    },
                }
            )
            expecting_image = False
        else:
            expecting_image = True
    if not expecting_image:
        raise RuntimeError("COLMAP images.txt is missing the final observations record")
    return images


def _parse_points(path: Path) -> tuple[int, int, float, float]:
    point_count = 0
    observation_count = 0
    weighted_error = 0.0
    for line in _data_lines(path):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            raise RuntimeError(f"invalid COLMAP point record: {line!r}")
        track_length = (len(fields) - 8) // 2
        error = float(fields[7])
        point_count += 1
        observation_count += track_length
        weighted_error += error * track_length
    mean_track_length = observation_count / point_count if point_count else 0.0
    mean_error = weighted_error / observation_count if observation_count else 0.0
    return point_count, observation_count, mean_track_length, mean_error


def _model_payload(
    model: Path,
    selected: dict[str, _SelectedImage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], _SparseStatistics]:
    cameras = _parse_cameras(model / "cameras.txt")
    images = _parse_images(model / "images.txt", selected)
    point_count, observations, mean_track, mean_error = _parse_points(model / "points3D.txt")
    return (
        cameras,
        images,
        _SparseStatistics(
            registered_images=len(images),
            point_count=point_count,
            observation_count=observations,
            mean_track_length=mean_track,
            mean_reprojection_error_px=mean_error,
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_sparse_bundle(
    destination: Path,
    components: tuple[_SparseComponent, ...],
    manifest: dict[str, Any],
) -> None:
    members: list[tuple[str, Path]] = []
    for component in components:
        for source_root, prefix in (
            (component.binary_model, "binary"),
            (component.text_model, "text"),
        ):
            for source in sorted(source_root.iterdir(), key=lambda item: item.name):
                if (
                    source.name in _SPARSE_MODEL_FILES[prefix]
                    and source.is_file()
                    and not source.is_symlink()
                ):
                    members.append(
                        (
                            f"components/{component.component_id}/{prefix}/{source.name}",
                            source,
                        )
                    )
    bundle_manifest = dict(manifest)
    bundle_manifest["files"] = [
        {"path": name, "sha256": _sha256(source), "size_bytes": source.stat().st_size}
        for name, source in members
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100400 << 16
            archive.writestr(info, source.read_bytes())
        info = zipfile.ZipInfo("bundle-manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100400 << 16
        archive.writestr(
            info,
            json.dumps(
                bundle_manifest,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )


def _colmap_version(binary: Path) -> str:
    completed = subprocess.run(
        [str(binary), "-h"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    text = completed.stdout.decode("utf-8", errors="replace")
    match = _VERSION.search(text)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(f"cannot determine COLMAP version: {text[-4096:]}")
    return match.group(1)


def _colmap_method_identity(binary: Path) -> tuple[str, str, str]:
    raw_version = _colmap_version(binary)
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(.*)", raw_version)
    if match is None:
        raise RuntimeError(f"COLMAP returned an unsupported version string: {raw_version!r}")
    major, minor, patch, suffix = match.groups()
    normalized_suffix = ""
    if suffix:
        normalized = re.sub(r"[^0-9A-Za-z-]+", ".", suffix).strip(".") or "build"
        normalized_suffix = f"-{normalized}"
    binary_digest = _sha256(binary).split(":", 1)[1]
    return (
        raw_version,
        f"{major}.{minor}.{patch}{normalized_suffix}",
        (f"binary-{binary_digest[:16]}"),
    )


def _handle(
    request: WorkerRequest,
    output_directory: Path,
    *,
    colmap_binary: Path,
) -> WorkerResponse:
    expected_outputs = {name: values[2] for name, values in _OUTPUT_FILES.items()}
    actual_outputs = {item.name: item.contract for item in request.operation.outputs}
    if actual_outputs != expected_outputs:
        raise ValueError(
            "COLMAP worker requires the exact versioned camera recovery output contracts"
        )
    options = _validated_options(request)
    manifest = ObservationSetManifest.model_validate(resource_json(request, "observations"))
    observation_reference = next(
        (
            item.resource
            for item in request.operation.inputs
            if item.name == "observations" and item.resource is not None
        ),
        None,
    )
    if observation_reference is None:
        raise ValueError("COLMAP worker requires a resolved observations resource input")
    selected = _select_images(request, manifest, options)
    requested_device = str(options.get("execution_device", "auto"))
    allow_fallback = _require_boolean(options, "allow_cpu_fallback", True)
    used_gpu = requested_device in {"auto", "cuda"}
    fallback_reason: str | None = None
    workspace_root = output_directory.parent / "colmap-work"
    if workspace_root.exists():
        raise RuntimeError("COLMAP temporary workspace already exists")
    workspace_root.mkdir(mode=0o700)
    try:
        attempt = workspace_root / ("cuda" if used_gpu else "cpu")
        attempt.mkdir()
        try:
            sparse_root, text_models = _attempt_reconstruction(
                colmap_binary,
                selected,
                options,
                attempt,
                use_gpu=used_gpu,
            )
        except ColmapExecutionError as error:
            if not used_gpu or not allow_fallback or not _gpu_failure(error):
                raise
            fallback_reason = str(error)
            shutil.rmtree(attempt)
            attempt = workspace_root / "cpu"
            attempt.mkdir()
            used_gpu = False
            sparse_root, text_models = _attempt_reconstruction(
                colmap_binary,
                selected,
                options,
                attempt,
                use_gpu=False,
            )
        selected_by_name = {item.staged_name: item for item in selected}
        components = []
        for text_model in text_models:
            cameras, images, statistics = _model_payload(text_model, selected_by_name)
            components.append(
                _SparseComponent(
                    component_id=text_model.name,
                    binary_model=sparse_root / text_model.name,
                    text_model=text_model,
                    cameras=cameras,
                    images=images,
                    statistics=statistics,
                )
            )
        primary = max(
            components,
            key=lambda item: (
                item.statistics.registered_images,
                item.statistics.point_count,
                item.component_id,
            ),
        )
        version = _colmap_version(colmap_binary)
        registered_names = {
            str(image["image_name"]) for component in components for image in component.images
        }
        registered_images = len(registered_names)
        registration_rate = registered_images / len(selected)
        point_count = sum(item.statistics.point_count for item in components)
        observation_count = sum(item.statistics.observation_count for item in components)
        mean_track_length = observation_count / point_count if point_count else 0.0
        mean_reprojection_error = (
            sum(
                item.statistics.mean_reprojection_error_px * item.statistics.observation_count
                for item in components
            )
            / observation_count
            if observation_count
            else 0.0
        )
        scale_reason = (
            "COLMAP monocular sparse reconstruction has similarity-gauge ambiguity; "
            "the observation set provides no verified metric-scale constraint."
        )
        rig_payload = {
            "schema": "physrealx.camera-rig-calibration/1",
            "method": {"name": "colmap", "version": version},
            "observation_set": observation_reference.model_dump(mode="json"),
            "calibration_type": "independent_camera_trajectory",
            "rig_constraints_applied": False,
            "primary_component": primary.component_id,
            "coordinate_convention": {
                "frame": "one independent colmap_world frame per component",
                "pose": "world_to_camera",
                "quaternion_order": "wxyz",
                "translation_unit": "arbitrary",
            },
            "scale_status": "unresolved",
            "scale_reason": scale_reason,
            "components": [
                {
                    "component_id": component.component_id,
                    "coordinate_frame": f"colmap_component_{component.component_id}",
                    "cameras": component.cameras,
                    "registered_views": component.images,
                }
                for component in components
            ],
        }
        report_payload = {
            "schema": "physrealx.camera-recovery-report/1",
            "method": {"name": "colmap", "version": version},
            "input": {
                "observation_set": observation_reference.model_dump(mode="json"),
                "selected_images": len(selected),
                "selected_streams": sorted({item.stream_name for item in selected}),
            },
            "execution": {
                "requested_device": requested_device,
                "effective_device": "cuda" if used_gpu else "cpu",
                "cpu_fallback_used": fallback_reason is not None,
                "fallback_reason": fallback_reason,
            },
            "model": {
                "component_count": len(components),
                "primary_component": primary.component_id,
                "registered_images": registered_images,
                "unregistered_images": sorted(
                    item.staged_name
                    for item in selected
                    if item.staged_name not in registered_names
                ),
                "registration_rate": registration_rate,
                "sparse_points": point_count,
                "observations": observation_count,
                "mean_track_length": mean_track_length,
                "mean_reprojection_error_px": mean_reprojection_error,
                "components": [
                    {
                        "component_id": component.component_id,
                        "registered_images": component.statistics.registered_images,
                        "sparse_points": component.statistics.point_count,
                        "observations": component.statistics.observation_count,
                        "mean_track_length": component.statistics.mean_track_length,
                        "mean_reprojection_error_px": (
                            component.statistics.mean_reprojection_error_px
                        ),
                    }
                    for component in components
                ],
            },
            "scale_status": "unresolved",
            "scale_reason": scale_reason,
        }
        bundle_manifest = {
            "schema": "physrealx.sparse-reconstruction-bundle/1",
            "method": {"name": "colmap", "version": version},
            "observation_set": observation_reference.model_dump(mode="json"),
            "primary_component": primary.component_id,
            "components": [component.component_id for component in components],
            "scale_status": "unresolved",
            "scale_reason": scale_reason,
        }
        rig_destination = output_directory / _OUTPUT_FILES["camera_rig_calibration"][0]
        report_destination = output_directory / _OUTPUT_FILES["camera_recovery_report"][0]
        sparse_destination = output_directory / _OUTPUT_FILES["sparse_reconstruction_bundle"][0]
        _write_json(rig_destination, rig_payload)
        _write_json(report_destination, report_payload)
        _write_sparse_bundle(
            sparse_destination,
            tuple(components),
            bundle_manifest,
        )
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)
    produced = []
    for requirement in request.operation.outputs:
        filename, media_type, contract = _OUTPUT_FILES[requirement.name]
        produced.append(
            WorkerProducedFile(
                name=requirement.name,
                contract=contract,
                path=filename,
                media_type=media_type,
            )
        )
    return WorkerResponse(
        outputs=tuple(produced),
        metrics={
            "selected_images": float(len(selected)),
            "registered_images": float(registered_images),
            "registration_rate": registration_rate,
            "sparse_points": float(point_count),
            "mean_track_length": mean_track_length,
            "mean_reprojection_error_px": mean_reprojection_error,
            "component_count": float(len(components)),
            "used_gpu": float(used_gpu),
            "cpu_fallback_used": float(fallback_reason is not None),
        },
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--colmap-binary", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    worker_arguments, remaining = parser.parse_known_args(arguments)
    binary = worker_arguments.colmap_binary.expanduser().resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"COLMAP binary is not executable: {binary}")
    if worker_arguments.probe:
        raw_version, method_version, source_revision = _colmap_method_identity(binary)
        return report_probe(
            capability="camera.recover",
            name="colmap",
            version=method_version,
            source_revision=source_revision,
            dependencies={"colmap": raw_version},
            self_test_checks={
                "binary_help": True,
                "binary_digest": _sha256(binary),
            },
        )
    return run_worker(
        lambda request, output: _handle(
            request,
            output,
            colmap_binary=binary,
        ),
        argv=remaining,
    )


if __name__ == "__main__":
    raise SystemExit(main())
