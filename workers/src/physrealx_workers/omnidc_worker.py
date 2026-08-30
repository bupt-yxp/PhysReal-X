"""Strict OMNI-DC depth-completion adapter.

The worker owns validation, evidence accounting and packaging only.  Neural
inference is delegated to an attested checkout of the official OMNI-DC source;
no model implementation is copied into PhysReal-X.  A missing source checkout,
checkpoint or CUDA runtime is therefore a hard availability failure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import re
import sys
import zipfile
from collections.abc import Callable, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from physrealx.external_workers import (
    WorkerProducedFile,
    WorkerRequest,
    WorkerResponse,
    run_worker,
)
from physrealx_workers.common import (
    dependency_file,
    dependency_path,
    report_probe,
    resource_json,
)

_BUNDLE_SCHEMA = "physrealx.calibrated-rgbd-bundle/1"
_OUTPUT_SCHEMA = "physrealx.enhanced-depth-bundle/1"
_REPORT_SCHEMA = "physrealx.depth-enhancement-report/1"
_INSTALLATION_SCHEMA = "physrealx.omnidc-installation/1"
_SOURCE_URI = "https://github.com/princeton-vl/OMNI-DC"
_METHOD = "omni_dc"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FRAME_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}$")
_RGB_MEDIA = {"application/x-npy", "image/jpeg", "image/png"}
_DEPTH_MEDIA = {"application/x-npy"}
_CHECKPOINT_MEDIA = {"application/octet-stream", "application/x-pytorch"}
_EVIDENCE_MEDIA = {"application/json", "application/octet-stream"}
_ALLOWED_CONFIG = {
    "absolute_tolerance_m",
    "maximum_holdout_abs_rel",
    "maximum_holdout_rmse_m",
    "minimum_coverage_gain",
    "minimum_holdout_points",
    "minimum_output_coverage",
    "relative_tolerance",
}


@dataclass(frozen=True)
class _Options:
    minimum_output_coverage: float
    minimum_coverage_gain: float
    minimum_holdout_points: int
    maximum_holdout_rmse_m: float
    maximum_holdout_abs_rel: float
    absolute_tolerance_m: float
    relative_tolerance: float


@dataclass(frozen=True)
class _Camera:
    camera_id: str
    width: int
    height: int
    intrinsic: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class _Frame:
    frame_id: str
    camera: _Camera
    rgb: Any
    sparse_depth_m: Any
    holdout_depth_m: Any
    rgb_digest: str
    sparse_digest: str
    holdout_digest: str


@dataclass(frozen=True)
class _ValidatedBundle:
    resource_id: str
    version: str
    source_digest: str
    source_revision: str
    release: str
    checkpoint_digest: str
    valid_min_m: float
    valid_max_m: float
    frames: tuple[_Frame, ...]


@dataclass(frozen=True)
class _Runtime:
    source_root: Path
    source_revision: str
    checkpoint_path: Path
    checkpoint_digest: str
    installation_manifest: Path
    self_test_rgb: Path
    self_test_depth: Path


Predictor = Callable[[Sequence[_Frame], Path, str, float], Sequence[Any]]


def _keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return result


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _artifact(value: Any, label: str, allowed_media: set[str]) -> dict[str, Any]:
    artifact = _object(value, label)
    _keys(artifact, required={"digest", "size_bytes", "media_type"}, label=label)
    _digest(artifact["digest"], f"{label}.digest")
    _integer(artifact["size_bytes"], f"{label}.size_bytes", minimum=1)
    if artifact["media_type"] not in allowed_media:
        raise ValueError(f"{label}.media_type must be one of: {', '.join(sorted(allowed_media))}")
    return artifact


def _options(raw: dict[str, Any]) -> _Options:
    unknown = sorted(set(raw) - _ALLOWED_CONFIG)
    if unknown:
        raise ValueError(f"unknown OMNI-DC options: {unknown}")
    minimum_output_coverage = _number(
        raw.get("minimum_output_coverage", 0.98),
        "minimum_output_coverage",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_coverage_gain = _number(
        raw.get("minimum_coverage_gain", 0.05),
        "minimum_coverage_gain",
        minimum=0.0,
        maximum=1.0,
    )
    return _Options(
        minimum_output_coverage=minimum_output_coverage,
        minimum_coverage_gain=minimum_coverage_gain,
        minimum_holdout_points=_integer(
            raw.get("minimum_holdout_points", 64),
            "minimum_holdout_points",
            minimum=1,
        ),
        maximum_holdout_rmse_m=_number(
            raw.get("maximum_holdout_rmse_m", 0.35),
            "maximum_holdout_rmse_m",
            minimum=1.0e-9,
        ),
        maximum_holdout_abs_rel=_number(
            raw.get("maximum_holdout_abs_rel", 0.20),
            "maximum_holdout_abs_rel",
            minimum=0.0,
        ),
        absolute_tolerance_m=_number(
            raw.get("absolute_tolerance_m", 1.0e-5),
            "absolute_tolerance_m",
            minimum=0.0,
        ),
        relative_tolerance=_number(
            raw.get("relative_tolerance", 1.0e-5),
            "relative_tolerance",
            minimum=0.0,
        ),
    )


def _load_npy(path: Path, label: str) -> Any:
    import numpy as np

    if path.suffix.lower() != ".npy":
        # Protocol materialization deliberately has no suffix; media type is authoritative.
        pass
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a safe NumPy array: {error}") from error
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} must contain exactly one NumPy array")
    return array


def _load_rgb(path: Path, media_type: str, width: int, height: int, label: str) -> Any:
    import numpy as np

    if media_type == "application/x-npy":
        rgb = _load_npy(path, label)
    else:
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Pillow is required to decode RGB image artifacts") from error
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{label} is not a decodable RGB image: {error}") from error
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"{label} must be uint8 HxWx3 with calibrated dimensions")
    return np.asarray(rgb)


def _load_depth(
    path: Path,
    width: int,
    height: int,
    label: str,
    *,
    valid_min_m: float,
    valid_max_m: float,
) -> Any:
    import numpy as np

    depth = _load_npy(path, label)
    if depth.shape != (height, width):
        raise ValueError(f"{label} dimensions do not match its calibrated camera")
    if depth.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError(f"{label} must use float32 or float64 metres")
    depth = np.asarray(depth, dtype=np.float32)
    if not bool(np.isfinite(depth).all()):
        raise ValueError(f"{label} contains NaN or infinity")
    if bool((depth < 0.0).any()):
        raise ValueError(f"{label} contains negative depths")
    valid = depth > 0.0
    if bool(valid.any()) and (
        float(depth[valid].min()) < valid_min_m or float(depth[valid].max()) > valid_max_m
    ):
        raise ValueError(f"{label} contains valid depths outside the declared metric range")
    return depth


def _validate_camera(value: Any, label: str) -> _Camera:
    camera = _object(value, label)
    _keys(
        camera,
        required={
            "camera_id",
            "image_width",
            "image_height",
            "intrinsic_matrix",
            "distortion",
            "sensor_to_world",
        },
        label=label,
    )
    camera_id = camera["camera_id"]
    if not isinstance(camera_id, str) or _FRAME_ID.fullmatch(camera_id) is None:
        raise ValueError(f"{label}.camera_id is invalid")
    width = _integer(camera["image_width"], f"{label}.image_width", minimum=2)
    height = _integer(camera["image_height"], f"{label}.image_height", minimum=2)
    intrinsic_rows = _array(camera["intrinsic_matrix"], f"{label}.intrinsic_matrix")
    if len(intrinsic_rows) != 3:
        raise ValueError(f"{label}.intrinsic_matrix must be 3x3")
    intrinsic = tuple(
        tuple(
            _number(item, f"{label}.intrinsic_matrix[{row_index}]") for item in _array(row, label)
        )
        for row_index, row in enumerate(intrinsic_rows)
    )
    if any(len(row) != 3 for row in intrinsic):
        raise ValueError(f"{label}.intrinsic_matrix must be 3x3")
    if intrinsic[0][0] <= 0.0 or intrinsic[1][1] <= 0.0 or intrinsic[2] != (0.0, 0.0, 1.0):
        raise ValueError(f"{label}.intrinsic_matrix is not a calibrated pinhole matrix")
    distortion = _object(camera["distortion"], f"{label}.distortion")
    _keys(distortion, required={"model", "coefficients"}, label=f"{label}.distortion")
    if distortion["model"] not in {"none", "opencv", "opencv_fisheye"}:
        raise ValueError(f"{label}.distortion.model is unsupported")
    coefficients = _array(distortion["coefficients"], f"{label}.distortion.coefficients")
    expected_count = {"none": 0, "opencv": 5, "opencv_fisheye": 4}[distortion["model"]]
    if len(coefficients) != expected_count:
        raise ValueError(f"{label}.distortion.coefficients has the wrong length")
    for index, item in enumerate(coefficients):
        _number(item, f"{label}.distortion.coefficients[{index}]")
    transform_rows = _array(camera["sensor_to_world"], f"{label}.sensor_to_world")
    if len(transform_rows) != 4:
        raise ValueError(f"{label}.sensor_to_world must be 4x4")
    transform = tuple(
        tuple(_number(item, f"{label}.sensor_to_world[{index}]") for item in _array(row, label))
        for index, row in enumerate(transform_rows)
    )
    if any(len(row) != 4 for row in transform) or transform[3] != (0.0, 0.0, 0.0, 1.0):
        raise ValueError(f"{label}.sensor_to_world must be a homogeneous 4x4 transform")
    return _Camera(camera_id, width, height, intrinsic)


def _indexed_artifact(
    request: WorkerRequest,
    value: Any,
    label: str,
    allowed_media: set[str],
) -> tuple[Path, dict[str, Any]]:
    artifact = _artifact(value, label, allowed_media)
    indexed = dependency_file(request, "rgbd", artifact["digest"])
    if indexed.size_bytes != artifact["size_bytes"] or indexed.media_type != artifact["media_type"]:
        raise ValueError(f"{label} does not match the immutable resource closure index")
    return dependency_path(request, "rgbd", artifact["digest"]), artifact


def _validate_bundle(request: WorkerRequest) -> _ValidatedBundle:
    import numpy as np

    manifest = resource_json(request, "rgbd")
    _keys(
        manifest,
        required={
            "schema",
            "resource_id",
            "version",
            "calibration",
            "depth_encoding",
            "frames",
            "checkpoint_manifest",
            "provenance",
        },
        label="CalibratedRGBDBundle",
    )
    if manifest["schema"] != _BUNDLE_SCHEMA:
        raise ValueError(f"rgbd schema must be {_BUNDLE_SCHEMA!r}")
    descriptor = request.operation.method.descriptor
    if (
        descriptor.capability.value != "depth.enhance"
        or descriptor.name != _METHOD
        or descriptor.source_uri != _SOURCE_URI
        or descriptor.input_contract != "CalibratedRGBDBundle/1"
        or descriptor.output_contract != "EnhancedDepthBundle/1+DepthEnhancementReport/1"
    ):
        raise ValueError("operation descriptor does not identify the OMNI-DC depth method")
    for name in ("resource_id", "version"):
        if not isinstance(manifest[name], str) or not manifest[name].strip():
            raise ValueError(f"rgbd.{name} must be a non-empty string")
    provenance = _object(manifest["provenance"], "rgbd.provenance")
    _keys(
        provenance,
        required={"source_capture", "calibration_evidence"},
        label="rgbd.provenance",
    )
    _indexed_artifact(
        request,
        provenance["source_capture"],
        "rgbd.provenance.source_capture",
        _EVIDENCE_MEDIA,
    )
    _indexed_artifact(
        request,
        provenance["calibration_evidence"],
        "rgbd.provenance.calibration_evidence",
        _EVIDENCE_MEDIA,
    )
    encoding = _object(manifest["depth_encoding"], "rgbd.depth_encoding")
    _keys(
        encoding,
        required={
            "unit",
            "meters_per_unit",
            "invalid_value",
            "valid_min_m",
            "valid_max_m",
        },
        label="rgbd.depth_encoding",
    )
    if encoding["unit"] != "m" or encoding["meters_per_unit"] != 1.0:
        raise ValueError("depth must be resolved to metres before OMNI-DC execution")
    if encoding["invalid_value"] != 0.0:
        raise ValueError("OMNI-DC sparse depth must use exactly 0.0 for invalid pixels")
    valid_min_m = _number(encoding["valid_min_m"], "valid_min_m", minimum=1.0e-6)
    valid_max_m = _number(encoding["valid_max_m"], "valid_max_m", minimum=valid_min_m)
    cameras_raw = _array(manifest["calibration"], "rgbd.calibration")
    if not cameras_raw:
        raise ValueError("rgbd.calibration cannot be empty")
    cameras = {
        camera.camera_id: camera
        for index, item in enumerate(cameras_raw)
        for camera in (_validate_camera(item, f"rgbd.calibration[{index}]"),)
    }
    if len(cameras) != len(cameras_raw):
        raise ValueError("rgbd.calibration camera IDs must be unique")
    checkpoint = _object(manifest["checkpoint_manifest"], "rgbd.checkpoint_manifest")
    _keys(
        checkpoint,
        required={"method", "release", "source_uri", "source_revision", "artifact"},
        label="rgbd.checkpoint_manifest",
    )
    if checkpoint["method"] != _METHOD or checkpoint["source_uri"] != _SOURCE_URI:
        raise ValueError("checkpoint manifest must identify the official OMNI-DC method")
    # The adapter currently supports the official v1.0 checkpoint.  V1.1 also
    # needs a separately attested Depth-Anything-V2 checkpoint and is therefore
    # intentionally rejected until that dependency is part of the closure.
    if checkpoint["release"] != "v1.0":
        raise ValueError("only OMNI-DC v1.0 is supported by this single-checkpoint contract")
    source_revision = checkpoint["source_revision"]
    if not isinstance(source_revision, str) or _REVISION.fullmatch(source_revision) is None:
        raise ValueError("checkpoint source_revision must be an immutable 40-hex commit")
    if descriptor.source_revision != source_revision:
        raise ValueError("method descriptor revision differs from checkpoint manifest")
    checkpoint_artifact = _artifact(
        checkpoint["artifact"],
        "rgbd.checkpoint_manifest.artifact",
        _CHECKPOINT_MEDIA,
    )
    if request.checkpoint is None:
        raise ValueError("OMNI-DC operation must bind an immutable checkpoint")
    if (
        request.checkpoint.digest != checkpoint_artifact["digest"]
        or request.checkpoint.size_bytes != checkpoint_artifact["size_bytes"]
        or request.checkpoint.media_type != checkpoint_artifact["media_type"]
    ):
        raise ValueError("bound checkpoint does not match rgbd.checkpoint_manifest")
    frames_raw = _array(manifest["frames"], "rgbd.frames")
    if not frames_raw:
        raise ValueError("rgbd.frames cannot be empty")
    frames: list[_Frame] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(frames_raw):
        label = f"rgbd.frames[{index}]"
        frame = _object(value, label)
        _keys(
            frame,
            required={"frame_id", "camera_id", "rgb", "sparse_depth", "holdout_depth"},
            label=label,
        )
        frame_id = frame["frame_id"]
        if not isinstance(frame_id, str) or _FRAME_ID.fullmatch(frame_id) is None:
            raise ValueError(f"{label}.frame_id is invalid")
        if frame_id in seen_ids:
            raise ValueError("rgbd frame IDs must be unique")
        seen_ids.add(frame_id)
        camera_id = frame["camera_id"]
        if camera_id not in cameras:
            raise ValueError(f"{label}.camera_id is not present in calibration")
        camera = cameras[camera_id]
        rgb_path, rgb_ref = _indexed_artifact(request, frame["rgb"], f"{label}.rgb", _RGB_MEDIA)
        sparse_path, sparse_ref = _indexed_artifact(
            request, frame["sparse_depth"], f"{label}.sparse_depth", _DEPTH_MEDIA
        )
        holdout_path, holdout_ref = _indexed_artifact(
            request, frame["holdout_depth"], f"{label}.holdout_depth", _DEPTH_MEDIA
        )
        rgb = _load_rgb(
            rgb_path,
            rgb_ref["media_type"],
            camera.width,
            camera.height,
            f"{label}.rgb",
        )
        sparse = _load_depth(
            sparse_path,
            camera.width,
            camera.height,
            f"{label}.sparse_depth",
            valid_min_m=valid_min_m,
            valid_max_m=valid_max_m,
        )
        holdout = _load_depth(
            holdout_path,
            camera.width,
            camera.height,
            f"{label}.holdout_depth",
            valid_min_m=valid_min_m,
            valid_max_m=valid_max_m,
        )
        if bool(np.logical_and(sparse > 0.0, holdout > 0.0).any()):
            raise ValueError(f"{label} sparse and holdout observations must be disjoint")
        if not bool((sparse > 0.0).any()):
            raise ValueError(f"{label}.sparse_depth has no valid observations")
        frames.append(
            _Frame(
                frame_id=frame_id,
                camera=camera,
                rgb=rgb,
                sparse_depth_m=sparse,
                holdout_depth_m=holdout,
                rgb_digest=rgb_ref["digest"],
                sparse_digest=sparse_ref["digest"],
                holdout_digest=holdout_ref["digest"],
            )
        )
    source_digest = next(item.digest for item in request.inputs if item.name == "rgbd")
    return _ValidatedBundle(
        resource_id=manifest["resource_id"],
        version=manifest["version"],
        source_digest=source_digest,
        source_revision=source_revision,
        release=checkpoint["release"],
        checkpoint_digest=checkpoint_artifact["digest"],
        valid_min_m=valid_min_m,
        valid_max_m=valid_max_m,
        frames=tuple(frames),
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _source_tree_digest(source_root: Path) -> str:
    """Digest all executable official Python source, independent of mtimes."""

    selected = [source_root / "LICENSE", *sorted((source_root / "src").rglob("*.py"))]
    if len(selected) < 3:
        raise RuntimeError("OMNI-DC source tree is incomplete")
    hasher = hashlib.sha256()
    for raw_path in selected:
        if raw_path.is_symlink():
            raise RuntimeError(f"OMNI-DC source tree contains a symlink: {raw_path}")
        path = raw_path.resolve(strict=True)
        if source_root not in path.parents or not path.is_file():
            raise RuntimeError(f"OMNI-DC source tree contains an unsafe file: {raw_path}")
        relative = path.relative_to(source_root).as_posix().encode()
        payload = path.read_bytes()
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return f"sha256:{hasher.hexdigest()}"


def _safe_absolute_file(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be an absolute path")
    raw = Path(value).expanduser()
    if not raw.is_absolute() or raw.is_symlink():
        raise RuntimeError(f"{label} must be an absolute non-symlink path")
    try:
        path = raw.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is unavailable: {raw}") from error
    if not path.is_file():
        raise RuntimeError(f"{label} must be a regular file")
    return path


def _load_runtime(path: Path | None = None) -> _Runtime:
    manifest_path = path or (Path(sys.prefix) / "share/physrealx/omnidc-installation.json")
    manifest_path = _safe_absolute_file(str(manifest_path), "OMNI-DC installation manifest")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("OMNI-DC installation manifest is not valid UTF-8 JSON") from error
    document = _object(document, "OMNI-DC installation manifest")
    _keys(
        document,
        required={
            "schema",
            "source_uri",
            "source_revision",
            "source_root",
            "source_tree_digest",
            "runtime_files",
            "checkpoint",
            "self_test",
        },
        label="OMNI-DC installation manifest",
    )
    if document["schema"] != _INSTALLATION_SCHEMA or document["source_uri"] != _SOURCE_URI:
        raise RuntimeError("OMNI-DC installation identity is unsupported")
    revision = document["source_revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise RuntimeError("OMNI-DC installation source_revision is not immutable")
    source_root_raw = document["source_root"]
    if not isinstance(source_root_raw, str):
        raise RuntimeError("OMNI-DC source_root must be absolute")
    source_root = Path(source_root_raw).expanduser()
    if not source_root.is_absolute() or source_root.is_symlink():
        raise RuntimeError("OMNI-DC source_root must be an absolute non-symlink directory")
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise RuntimeError("OMNI-DC source_root must be a directory")
    expected_tree_digest = _digest(document["source_tree_digest"], "OMNI-DC source_tree_digest")
    if _source_tree_digest(source_root) != expected_tree_digest:
        raise RuntimeError("OMNI-DC official source tree digest changed after installation")
    runtime_files = _array(document["runtime_files"], "OMNI-DC runtime_files")
    required_relative = {"src/config.py", "src/model/ognidc.py", "LICENSE"}
    observed_relative: set[str] = set()
    for index, item in enumerate(runtime_files):
        record = _object(item, f"runtime_files[{index}]")
        _keys(record, required={"path", "digest"}, label=f"runtime_files[{index}]")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("OMNI-DC runtime file paths must be safe and relative")
        raw_file_path = source_root / relative
        if raw_file_path.is_symlink():
            raise RuntimeError(f"OMNI-DC runtime file is unsafe: {relative}")
        file_path = raw_file_path.resolve(strict=True)
        if source_root not in file_path.parents or not file_path.is_file():
            raise RuntimeError(f"OMNI-DC runtime file is unsafe: {relative}")
        if _sha256(file_path) != _digest(record["digest"], f"runtime_files[{index}].digest"):
            raise RuntimeError(f"OMNI-DC runtime file digest changed: {relative}")
        observed_relative.add(relative)
    if not required_relative <= observed_relative:
        raise RuntimeError("OMNI-DC installation manifest omits required official source files")
    checkpoint_record = _object(document["checkpoint"], "OMNI-DC checkpoint")
    _keys(
        checkpoint_record,
        required={"path", "digest", "release"},
        label="OMNI-DC checkpoint",
    )
    if checkpoint_record["release"] != "v1.0":
        raise RuntimeError("OMNI-DC installation must use the supported v1.0 checkpoint")
    checkpoint_path = _safe_absolute_file(checkpoint_record["path"], "OMNI-DC checkpoint")
    checkpoint_digest = _digest(checkpoint_record["digest"], "OMNI-DC checkpoint.digest")
    if _sha256(checkpoint_path) != checkpoint_digest:
        raise RuntimeError("OMNI-DC checkpoint digest changed after installation")
    fixture = _object(document["self_test"], "OMNI-DC self_test")
    _keys(
        fixture,
        required={"rgb_path", "rgb_digest", "depth_path", "depth_digest"},
        label="OMNI-DC self_test",
    )
    self_test_rgb = _safe_absolute_file(fixture["rgb_path"], "OMNI-DC self-test RGB")
    self_test_depth = _safe_absolute_file(fixture["depth_path"], "OMNI-DC self-test depth")
    if _sha256(self_test_rgb) != _digest(fixture["rgb_digest"], "self_test.rgb_digest"):
        raise RuntimeError("OMNI-DC self-test RGB digest changed after installation")
    if _sha256(self_test_depth) != _digest(fixture["depth_digest"], "self_test.depth_digest"):
        raise RuntimeError("OMNI-DC self-test depth digest changed after installation")
    return _Runtime(
        source_root=source_root,
        source_revision=revision,
        checkpoint_path=checkpoint_path,
        checkpoint_digest=checkpoint_digest,
        installation_manifest=manifest_path,
        self_test_rgb=self_test_rgb,
        self_test_depth=self_test_depth,
    )


@contextmanager
def _official_import_path(source_root: Path) -> Iterator[None]:
    source = str(source_root / "src")
    original_argv = list(sys.argv)
    sys.path.insert(0, source)
    sys.argv = [str(source_root / "src/demo.py")]
    try:
        yield
    finally:
        sys.argv = original_argv
        with suppress(ValueError):
            sys.path.remove(source)


def _official_predictor(runtime: _Runtime) -> Predictor:
    """Build a thin callable around the official source checkout."""

    def predict(
        frames: Sequence[_Frame], checkpoint: Path, release: str, max_depth: float
    ) -> Sequence[Any]:
        import numpy as np
        import torch

        if release != "v1.0":
            raise RuntimeError("official predictor only accepts the attested v1.0 profile")
        if not torch.cuda.is_available():
            raise RuntimeError("OMNI-DC official inference requires a CUDA-capable worker")
        with _official_import_path(runtime.source_root):
            config_module = importlib.import_module("config")
            model_module = importlib.import_module("model.ognidc")
            args = copy.deepcopy(config_module.args)
            overrides = {
                "max_depth": max_depth,
                "data_normalize_median": 1,
                "num_resolution": 3,
                "multi_resolution_learnable_gradients_weights": "uniform",
                "load_dav2": 0,
                "gpus": "0",
                "GRU_iters": 1,
                "optim_layer_input_clamp": 1.0,
                "depth_activation_format": "exp",
                "whiten_sparse_depths": 1,
                "gru_internal_whiten_method": "median",
                "backbone_mode": "rgbd",
                "pred_confidence_input": 1,
            }
            for name, value in overrides.items():
                setattr(args, name, value)
            model = model_module.OGNIDC(args)
            try:
                checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:  # Official environment pins torch 1.10, which predates weights_only.
                checkpoint_payload = torch.load(checkpoint, map_location="cpu")
            if not isinstance(checkpoint_payload, dict) or not isinstance(
                checkpoint_payload.get("net"), dict
            ):
                raise RuntimeError("OMNI-DC checkpoint does not contain the official 'net' state")
            missing, unexpected = model.load_state_dict(checkpoint_payload["net"], strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "OMNI-DC checkpoint/model mismatch: "
                    f"missing={list(missing)}, unexpected={list(unexpected)}"
                )
            model = model.cuda().eval()
            outputs: list[Any] = []
            mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(3, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(3, 1, 1)
            divisor = int(4 * 2 ** (args.num_resolution - 1))
            with torch.no_grad():
                for frame in frames:
                    rgb = torch.from_numpy(np.ascontiguousarray(frame.rgb.transpose(2, 0, 1)))
                    rgb = rgb.cuda().float().div_(255.0)
                    rgb = ((rgb - mean) / std).unsqueeze(0)
                    depth = torch.from_numpy(np.ascontiguousarray(frame.sparse_depth_m))
                    depth = depth.cuda().float().unsqueeze(0).unsqueeze(0)
                    height, width = frame.sparse_depth_m.shape
                    pad_height = (-height) % divisor
                    pad_width = (-width) % divisor
                    rgb = torch.nn.functional.pad(rgb, (0, pad_width, 0, pad_height))
                    depth = torch.nn.functional.pad(depth, (0, pad_width, 0, pad_height))
                    intrinsic = torch.tensor(
                        frame.camera.intrinsic, dtype=torch.float32, device="cuda"
                    )
                    sample = {
                        "rgb": rgb,
                        "dep": depth,
                        "K": intrinsic.unsqueeze(0),
                        "pattern": 0,
                    }
                    result = model(sample)
                    if not isinstance(result, dict) or "pred" not in result:
                        raise RuntimeError("official OMNI-DC model returned no 'pred' tensor")
                    prediction = (
                        result["pred"][..., :height, :width].squeeze().detach().cpu().numpy()
                    )
                    outputs.append(np.asarray(prediction, dtype=np.float32))
            return outputs

    return predict


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _zip_write_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    record = zipfile.ZipInfo(name)
    record.date_time = (1980, 1, 1, 0, 0, 0)
    record.compress_type = zipfile.ZIP_DEFLATED
    record.external_attr = 0o100444 << 16
    archive.writestr(record, payload)


def _npy_bytes(array: Any) -> bytes:
    import io

    import numpy as np

    stream = io.BytesIO()
    np.save(stream, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return stream.getvalue()


def _evaluate_and_package(
    bundle: _ValidatedBundle,
    predictions: Sequence[Any],
    options: _Options,
    output_directory: Path,
) -> tuple[Path, Path, dict[str, float]]:
    import numpy as np

    if len(predictions) != len(bundle.frames):
        raise ValueError("OMNI-DC predictor returned the wrong number of frames")
    frame_records: list[dict[str, Any]] = []
    payloads: list[tuple[str, bytes]] = []
    total_pixels = 0
    total_sparse = 0
    total_holdout = 0
    total_enhanced = 0
    squared_errors: list[Any] = []
    absolute_relative_errors: list[Any] = []
    maximum_sparse_error = 0.0
    for frame, raw_prediction in zip(bundle.frames, predictions):
        prediction = np.asarray(raw_prediction)
        if prediction.shape != frame.sparse_depth_m.shape:
            raise ValueError(f"prediction shape mismatch for frame {frame.frame_id!r}")
        if prediction.dtype.kind not in {"f", "i", "u"}:
            raise ValueError(f"prediction for frame {frame.frame_id!r} is not numeric")
        prediction = np.asarray(prediction, dtype=np.float32)
        if not bool(np.isfinite(prediction).all()):
            raise ValueError(f"prediction for frame {frame.frame_id!r} contains NaN or infinity")
        if bool((prediction <= 0.0).any()):
            raise ValueError(f"prediction for frame {frame.frame_id!r} is not dense and positive")
        if (
            float(prediction.min()) < bundle.valid_min_m
            or float(prediction.max()) > bundle.valid_max_m
        ):
            raise ValueError(f"prediction for frame {frame.frame_id!r} exceeds the metric range")
        sparse_mask = frame.sparse_depth_m > 0.0
        holdout_mask = frame.holdout_depth_m > 0.0
        sparse_error_before_projection = np.abs(
            prediction[sparse_mask] - frame.sparse_depth_m[sparse_mask]
        )
        allowed_sparse_error = options.absolute_tolerance_m + options.relative_tolerance * np.abs(
            frame.sparse_depth_m[sparse_mask]
        )
        # The official network predicts every pixel.  Exact sensor observations remain
        # authoritative, so the adapter applies a deterministic projection after recording
        # the model's raw discrepancy.
        projection_applied = bool((sparse_error_before_projection > allowed_sparse_error).any())
        prediction[sparse_mask] = frame.sparse_depth_m[sparse_mask]
        sparse_error = np.abs(prediction[sparse_mask] - frame.sparse_depth_m[sparse_mask])
        if sparse_error.size:
            maximum_sparse_error = max(maximum_sparse_error, float(sparse_error.max()))
        if not bool(np.all(sparse_error <= allowed_sparse_error)):
            raise ValueError(f"valid sparse observations were not preserved for {frame.frame_id!r}")
        holdout_count = int(holdout_mask.sum())
        if holdout_count < options.minimum_holdout_points:
            raise ValueError(
                f"frame {frame.frame_id!r} has {holdout_count} holdout points; "
                f"at least {options.minimum_holdout_points} are required"
            )
        error = prediction[holdout_mask] - frame.holdout_depth_m[holdout_mask]
        frame_rmse = float(np.sqrt(np.mean(np.square(error))))
        frame_abs_rel = float(np.mean(np.abs(error) / frame.holdout_depth_m[holdout_mask]))
        if frame_rmse > options.maximum_holdout_rmse_m:
            raise ValueError(
                f"frame {frame.frame_id!r} holdout RMSE {frame_rmse:.6g} m exceeds "
                f"{options.maximum_holdout_rmse_m:.6g} m"
            )
        if frame_abs_rel > options.maximum_holdout_abs_rel:
            raise ValueError(
                f"frame {frame.frame_id!r} holdout AbsRel {frame_abs_rel:.6g} exceeds "
                f"{options.maximum_holdout_abs_rel:.6g}"
            )
        squared_errors.append(np.square(error.astype(np.float64)))
        absolute_relative_errors.append(
            np.abs(error.astype(np.float64)) / frame.holdout_depth_m[holdout_mask]
        )
        pixel_count = int(prediction.size)
        sparse_count = int(sparse_mask.sum())
        enhanced_count = int(np.logical_and(np.isfinite(prediction), prediction > 0.0).sum())
        coverage = enhanced_count / pixel_count
        sparse_coverage = sparse_count / pixel_count
        if coverage < options.minimum_output_coverage:
            raise ValueError(f"enhanced coverage gate failed for frame {frame.frame_id!r}")
        if coverage - sparse_coverage < options.minimum_coverage_gain:
            raise ValueError(f"coverage gain gate failed for frame {frame.frame_id!r}")
        output_name = f"depth/{frame.frame_id}.npy"
        payload = _npy_bytes(prediction)
        output_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        payloads.append((output_name, payload))
        frame_records.append(
            {
                "frame_id": frame.frame_id,
                "camera_id": frame.camera.camera_id,
                "depth": {
                    "path": output_name,
                    "digest": output_digest,
                    "size_bytes": len(payload),
                    "media_type": "application/x-npy",
                    "dtype": "float32",
                    "unit": "m",
                    "invalid_value": None,
                },
                "source": {
                    "rgb_digest": frame.rgb_digest,
                    "sparse_depth_digest": frame.sparse_digest,
                    "holdout_depth_digest": frame.holdout_digest,
                },
                "sparse_observation_projection": {
                    "applied": projection_applied,
                    "maximum_raw_discrepancy_m": float(
                        sparse_error_before_projection.max(initial=0.0)
                    ),
                },
                "metrics": {
                    "input_coverage": sparse_coverage,
                    "output_coverage": coverage,
                    "holdout_points": holdout_count,
                    "holdout_rmse_m": frame_rmse,
                    "holdout_abs_rel": frame_abs_rel,
                },
            }
        )
        total_pixels += pixel_count
        total_sparse += sparse_count
        total_holdout += holdout_count
        total_enhanced += enhanced_count
    all_squared = np.concatenate(squared_errors)
    all_abs_rel = np.concatenate(absolute_relative_errors)
    holdout_rmse = float(np.sqrt(np.mean(all_squared)))
    holdout_abs_rel = float(np.mean(all_abs_rel))
    if holdout_rmse > options.maximum_holdout_rmse_m:
        raise ValueError(
            f"aggregate holdout RMSE {holdout_rmse:.6g} m exceeds "
            f"{options.maximum_holdout_rmse_m:.6g} m"
        )
    if holdout_abs_rel > options.maximum_holdout_abs_rel:
        raise ValueError(
            f"aggregate holdout AbsRel {holdout_abs_rel:.6g} exceeds "
            f"{options.maximum_holdout_abs_rel:.6g}"
        )
    metrics = {
        "frames": float(len(bundle.frames)),
        "input_coverage": total_sparse / total_pixels,
        "output_coverage": total_enhanced / total_pixels,
        "coverage_gain": (total_enhanced - total_sparse) / total_pixels,
        "holdout_points": float(total_holdout),
        "holdout_rmse_m": holdout_rmse,
        "holdout_abs_rel": holdout_abs_rel,
        "maximum_preserved_observation_error_m": maximum_sparse_error,
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("depth enhancement produced a non-finite aggregate metric")
    output_manifest = {
        "schema": _OUTPUT_SCHEMA,
        "status": "PASSED",
        "resource_id": bundle.resource_id,
        "version": bundle.version,
        "method": _METHOD,
        "release": bundle.release,
        "checkpoint_digest": bundle.checkpoint_digest,
        "source_bundle_digest": bundle.source_digest,
        "depth_encoding": {
            "unit": "m",
            "meters_per_unit": 1.0,
            "invalid_value": None,
            "valid_min_m": bundle.valid_min_m,
            "valid_max_m": bundle.valid_max_m,
        },
        "frames": frame_records,
    }
    report = {
        "schema": _REPORT_SCHEMA,
        "status": "PASSED",
        "method": _METHOD,
        "release": bundle.release,
        "checkpoint_digest": bundle.checkpoint_digest,
        "source_revision": bundle.source_revision,
        "source_bundle_digest": bundle.source_digest,
        "metrics": metrics,
        "gates": {
            "finite": "PASSED",
            "positive": "PASSED",
            "metric_range": "PASSED",
            "valid_observation_preservation": "PASSED",
            "coverage": "PASSED",
            "holdout_rmse": "PASSED",
            "holdout_abs_rel": "PASSED",
        },
        "thresholds": {
            "minimum_output_coverage": options.minimum_output_coverage,
            "minimum_coverage_gain": options.minimum_coverage_gain,
            "minimum_holdout_points": options.minimum_holdout_points,
            "maximum_holdout_rmse_m": options.maximum_holdout_rmse_m,
            "maximum_holdout_abs_rel": options.maximum_holdout_abs_rel,
            "absolute_tolerance_m": options.absolute_tolerance_m,
            "relative_tolerance": options.relative_tolerance,
        },
        "frames": [{"frame_id": item["frame_id"], **item["metrics"]} for item in frame_records],
    }
    archive_path = output_directory / "enhanced_depth_bundle.zip"
    manifest_bytes = (
        json.dumps(output_manifest, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
        _zip_write_bytes(archive, "manifest.json", manifest_bytes)
        for name, payload in payloads:
            _zip_write_bytes(archive, name, payload)
    report_path = output_directory / "depth_enhancement_report.json"
    _write_json(report_path, report)
    return archive_path, report_path, metrics


def _handle(
    request: WorkerRequest,
    output_directory: Path,
    *,
    predictor: Predictor | None = None,
    runtime: _Runtime | None = None,
) -> WorkerResponse:
    options = _options(dict(request.operation.method.config))
    bundle = _validate_bundle(request)
    if request.checkpoint is None:  # Kept explicit for the type checker after validation.
        raise ValueError("OMNI-DC operation must bind an immutable checkpoint")
    checkpoint_path = Path.cwd() / request.checkpoint.path
    if predictor is None:
        selected_runtime = runtime or _load_runtime()
        if selected_runtime.source_revision != bundle.source_revision:
            raise RuntimeError("OMNI-DC source revision differs from checkpoint manifest")
        if selected_runtime.checkpoint_digest != bundle.checkpoint_digest:
            raise RuntimeError("OMNI-DC installed checkpoint differs from bound checkpoint")
        predictor = _official_predictor(selected_runtime)
    predictions = predictor(bundle.frames, checkpoint_path, bundle.release, bundle.valid_max_m)
    archive_path, report_path, metrics = _evaluate_and_package(
        bundle, predictions, options, output_directory
    )
    produced = {
        "enhanced_depth": WorkerProducedFile(
            name="enhanced_depth",
            contract="EnhancedDepthBundle/1",
            path=archive_path.name,
            media_type="application/zip",
        ),
        "enhancement_report": WorkerProducedFile(
            name="enhancement_report",
            contract="DepthEnhancementReport/1",
            path=report_path.name,
            media_type="application/json",
        ),
    }
    try:
        outputs = tuple(produced[item.name] for item in request.operation.outputs)
    except KeyError as error:
        raise ValueError(f"unsupported OMNI-DC output: {error.args[0]}") from error
    return WorkerResponse(outputs=outputs, metrics=metrics)


def _probe(runtime_path: Path | None) -> int:
    source_revision = "unavailable"
    checkpoint_digest: str | None = None
    dependencies: dict[str, str] = {}
    try:
        runtime = _load_runtime(runtime_path)
        source_revision = runtime.source_revision
        checkpoint_digest = runtime.checkpoint_digest
        import numpy as np
        import torch

        dependencies = {"numpy": np.__version__, "torch": torch.__version__}
        if int(np.__version__.split(".", 1)[0]) != 1:
            raise RuntimeError("OMNI-DC worker requires NumPy 1.x")
        if not torch.cuda.is_available():
            raise RuntimeError("OMNI-DC official inference requires CUDA")
        rgb = _load_rgb(runtime.self_test_rgb, "image/png", 64, 48, "self-test RGB")
        sparse = _load_npy(runtime.self_test_depth, "self-test sparse depth")
        if sparse.shape != (48, 64):
            raise RuntimeError("OMNI-DC self-test depth must be 48x64")
        camera = _Camera(
            "probe",
            64,
            48,
            ((50.0, 0.0, 31.5), (0.0, 50.0, 23.5), (0.0, 0.0, 1.0)),
        )
        frame = _Frame(
            "probe",
            camera,
            rgb,
            np.asarray(sparse, dtype=np.float32),
            np.zeros((48, 64), dtype=np.float32),
            f"sha256:{'0' * 64}",
            f"sha256:{'1' * 64}",
            f"sha256:{'2' * 64}",
        )
        output = _official_predictor(runtime)([frame], runtime.checkpoint_path, "v1.0", 300.0)
        prediction = np.asarray(output[0])
        if prediction.shape != (48, 64) or not bool(np.isfinite(prediction).all()):
            raise RuntimeError("OMNI-DC self-test produced an invalid prediction")
        if not bool((prediction > 0.0).all()) or float(np.std(prediction)) <= 1.0e-6:
            raise RuntimeError("OMNI-DC self-test produced a non-positive or constant prediction")
        return report_probe(
            capability="depth.enhance",
            name=_METHOD,
            version="1.0.0",
            source_revision=runtime.source_revision,
            dependencies=dependencies,
            checkpoint_digest=runtime.checkpoint_digest,
            checkpoint_source_revision=runtime.source_revision,
            self_test_checks={
                "self_test": "passed",
                "self_test_min_m": f"{float(prediction.min()):.8g}",
                "self_test_max_m": f"{float(prediction.max()):.8g}",
            },
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return report_probe(
            capability="depth.enhance",
            name=_METHOD,
            version="1.0.0",
            source_revision=source_revision,
            dependencies=dependencies,
            checkpoint_digest=checkpoint_digest,
            checkpoint_source_revision=(source_revision if checkpoint_digest is not None else None),
            self_test_checks={"runtime_attestation": False},
            ready=False,
            reason=str(error) or type(error).__name__,
            failure_exit_code=2,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="physrealx-worker-omnidc", add_help=False)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--installation-manifest", type=Path)
    arguments, remaining = parser.parse_known_args(argv)
    if arguments.probe:
        if remaining:
            raise SystemExit(f"unrecognized arguments: {' '.join(remaining)}")
        return _probe(arguments.installation_manifest)
    runtime = _load_runtime(arguments.installation_manifest)
    return run_worker(lambda request, output: _handle(request, output, runtime=runtime), remaining)


if __name__ == "__main__":
    raise SystemExit(main())
