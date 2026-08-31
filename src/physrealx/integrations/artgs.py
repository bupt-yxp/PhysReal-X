"""In-repository ArtGS integration for metric articulated visual geometry.

The adapter deliberately keeps Torch, CUDA and ArtGS imports outside the core
process.  Training still runs in a child process for fault containment, but the
child uses the same Python executable and package environment as PhysReal-X.
"""

from __future__ import annotations

import csv
import fcntl
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, NoReturn, Protocol

from pydantic import Field, field_validator, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    JSONValue,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodMaturity,
    MethodRuntime,
    OperationOutput,
    OperationSpec,
    canonical_json_bytes,
)
from physrealx.errors import DomainValidationError, MethodExecutionError

_INPUT_SCHEMA: Final = "physrealx.metric-multistate-rgbd-bundle/1"
_BUNDLE_SCHEMA: Final = "physrealx.visual-geometry-bundle/1"
_REPORT_SCHEMA: Final = "physrealx.geometry-reconstruction-report/1"
_UPSTREAM_REVISION: Final = "7c1f41be2cb8b96abca13c6a9668dcf7e06d8c2a"
_SOURCE_REVISION: Final = "7c1f41be2cb8b96a.physrealx1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SOURCE_REVISION_LINE = re.compile(r'^revision\s*=\s*"([0-9a-f]{40})"\s*$', re.MULTILINE)
_MAX_INPUT_FILES = 100_000
_MAX_INPUT_BYTES = 128 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_GENERATED_SOURCE_DIRECTORIES: Final = frozenset({"__pycache__", "build", "dist"})
_GENERATED_SOURCE_SUFFIXES: Final = frozenset({".o", ".pyc", ".pyd", ".so"})
_NATIVE_IMPORTS: Final = (
    (
        "diff_gaussian_rasterization._C",
        "submodules/diff-gaussian-rasterization",
    ),
    ("simple_knn._C", "submodules/simple-knn"),
    ("pointnet2_cuda", "utils/pointnet_lib"),
)


def _python_version() -> tuple[int, int]:
    """Return the active runtime version without making later probe code static."""

    return sys.version_info.major, sys.version_info.minor


def _safe_relative_path(value: str, label: str = "path") -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must not contain absolute or traversal components")
    return path.as_posix()


def _strict_json(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{label} contains forbidden non-finite value {value}")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{label} must be unambiguous finite UTF-8 JSON") from error


class ArtGSInputFile(DomainModel):
    """One immutable payload member in a prepared ArtGS input bundle."""

    path: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=127)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class ArtGSDepthBinding(DomainModel):
    """Auditable conversion from one enhanced metric frame to ArtGS millimetres."""

    image_path: str
    depth_path: str
    enhanced_frame_id: str = Field(min_length=1, max_length=255)

    @field_validator("image_path", "depth_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _safe_relative_path(value)


class ArtGSInputManifest(DomainModel):
    """Strict metric, two-state reconstruction input consumed by ArtGS."""

    schema_id: Literal["physrealx.metric-multistate-rgbd-bundle/1"] = Field(
        default=_INPUT_SCHEMA,
        alias="schema",
    )
    scene_id: str
    unit: Literal["m"] = "m"
    coordinate_frame: str = Field(min_length=1, max_length=127)
    source_camera: ArtifactRef
    source_depth: ArtifactRef
    num_slots: int = Field(ge=2, le=32)
    training_frames: int = Field(ge=4)
    validation_frames: int = Field(ge=0)
    depth_bindings: tuple[ArtGSDepthBinding, ...]
    files: tuple[ArtGSInputFile, ...]

    @field_validator("scene_id")
    @classmethod
    def validate_scene_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("scene_id must be a lowercase platform identifier")
        return value

    @field_validator("coordinate_frame")
    @classmethod
    def validate_coordinate_frame(cls, value: str) -> str:
        if ".." in value or any(character in value for character in "\x00\r\n"):
            raise ValueError("coordinate_frame must be a safe logical frame")
        return value

    @model_validator(mode="after")
    def validate_files(self) -> ArtGSInputManifest:
        if not self.files:
            raise ValueError("ArtGS input requires payload files")
        paths = [item.path for item in self.files]
        portable = [item.casefold() for item in paths]
        if len(paths) != len(set(paths)) or len(portable) != len(set(portable)):
            raise ValueError("ArtGS input file paths must be unique and portable")
        binding_images = [item.image_path for item in self.depth_bindings]
        binding_depths = [item.depth_path for item in self.depth_bindings]
        binding_frames = [item.enhanced_frame_id for item in self.depth_bindings]
        if (
            not binding_images
            or len(binding_images) != self.training_frames + self.validation_frames
            or len(binding_images) != len(set(binding_images))
            or len(binding_depths) != len(set(binding_depths))
            or len(binding_frames) != len(set(binding_frames))
            or not set(binding_images).issubset(paths)
            or not set(binding_depths).issubset(paths)
        ):
            raise ValueError("ArtGS depth bindings must be non-empty, unique and materialized")
        required = {
            "transforms_train.json",
            "transforms_train_start.json",
            "transforms_train_end.json",
        }
        missing = sorted(required.difference(paths))
        if missing:
            raise ValueError(f"ArtGS input is missing transform files: {', '.join(missing)}")
        validation_files = {
            "transforms_test_start.json",
            "transforms_test_end.json",
        }
        has_validation_files = validation_files.issubset(paths)
        if (self.validation_frames > 0) != has_validation_files:
            raise ValueError("validation frame count and test transform files disagree")
        return self


class ArtGSRunConfig(DomainModel):
    """Bound, finite ArtGS execution configuration."""

    coarse_iterations: int = Field(default=10_000, ge=100, le=200_000)
    joint_iterations: int = Field(default=5_000, ge=100, le=100_000)
    reconstruction_iterations: int = Field(default=20_000, ge=100, le=500_000)
    training_resolution: int = Field(default=1, ge=1, le=16)
    evaluation_resolution: int = Field(default=2, ge=1, le=16)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    max_stage_seconds: int = Field(default=86_400, ge=60, le=604_800)
    allow_training_view_evaluation: bool = False
    metric_depth_loss_weight: float = Field(default=1.0, ge=0.0, le=1_000.0)
    mono_depth_loss_weight: float = Field(default=0.0, ge=0.0, le=1_000.0)


ARTGS_DESCRIPTOR = MethodDescriptor(
    capability=MethodCapability.GEOMETRY_RECONSTRUCT,
    name="artgs",
    version="1.0.0",
    api_version="1",
    source_uri="https://github.com/bupt-yxp/PhysReal-X/tree/main/algorithms/artgs",
    source_revision=_SOURCE_REVISION,
    license_spdx="LicenseRef-ArtGS-Research-Only",
    maturity=MethodMaturity.EXPERIMENTAL,
    runtime=MethodRuntime(
        mode=MethodExecutionMode.IN_PROCESS,
        accelerator=MethodAccelerator.CUDA,
        python=">=3.10,<3.11",
    ),
    input_contract="MetricMultiStateRGBDBundle/1",
    output_contract="VisualGeometryBundle/1+GeometryReconstructionReport/1",
)


def artgs_source_root() -> Path:
    """Resolve the vendored source in editable and installed distributions."""

    module = Path(__file__).resolve()
    checkout = module.parents[3] / "algorithms" / "artgs"
    installed = module.parents[1] / "_embedded" / "artgs"
    for candidate in (checkout, installed):
        if (candidate / "SOURCE.toml").is_file() and (candidate / "train.py").is_file():
            return candidate
    raise FileNotFoundError("the embedded ArtGS source snapshot is unavailable")


def _validate_source_snapshot(root: Path) -> str:
    manifest = root / "SOURCE.toml"
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise MethodExecutionError("embedded ArtGS source manifest is unreadable") from error
    revision_match = _SOURCE_REVISION_LINE.search(text)
    if revision_match is None or revision_match.group(1) != _UPSTREAM_REVISION:
        raise MethodExecutionError("embedded ArtGS source revision does not match its descriptor")
    return revision_match.group(1)


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".png": "image/png",
        ".ply": "model/ply",
        ".npy": "application/x-npy",
    }.get(suffix, "application/octet-stream")


@dataclass(frozen=True)
class _DepthPayloadRecord:
    frame_id: str
    path: str
    size_bytes: int


def _validated_metric_evidence(
    camera_rig: ArtifactRef,
    enhanced_depth: ArtifactRef,
    artifacts: ArtifactStore,
) -> Mapping[str, _DepthPayloadRecord]:
    """Require resolved metric cameras and an accepted enhanced-depth bundle."""

    if camera_rig.media_type != "application/json" or not artifacts.verify(camera_rig):
        raise DomainValidationError("ArtGS camera evidence must be an available JSON artifact")
    if enhanced_depth.media_type != "application/zip" or not artifacts.verify(enhanced_depth):
        raise DomainValidationError("ArtGS depth evidence must be an available ZIP artifact")
    try:
        camera = _strict_json(artifacts.get_bytes(camera_rig), "ArtGS camera evidence")
    except ValueError as error:
        raise DomainValidationError("ArtGS camera evidence is invalid") from error
    convention = camera.get("coordinate_convention") if isinstance(camera, dict) else None
    components = camera.get("components") if isinstance(camera, dict) else None
    if (
        not isinstance(camera, dict)
        or camera.get("schema") != "physrealx.camera-rig-calibration/1"
        or camera.get("scale_status") != "resolved"
        or not isinstance(convention, dict)
        or convention.get("translation_unit") != "m"
        or not isinstance(components, list)
        or not components
    ):
        raise DomainValidationError(
            "ArtGS requires a resolved metric CameraRigCalibration/1 artifact"
        )

    records: dict[str, _DepthPayloadRecord] = {}
    with tempfile.TemporaryDirectory(prefix="physrealx-artgs-depth-check-") as temporary:
        archive_path = Path(temporary) / "enhanced-depth.zip"
        artifacts.materialize(enhanced_depth, archive_path)
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                members = archive.infolist()
                names = [_safe_relative_path(item.filename, "depth ZIP member") for item in members]
                if len(names) != len(set(names)) or any(item.is_dir() for item in members):
                    raise DomainValidationError("ArtGS enhanced-depth ZIP has unsafe members")
                info = archive.getinfo("manifest.json")
                if info.file_size > _MAX_MANIFEST_BYTES:
                    raise DomainValidationError("ArtGS enhanced-depth manifest is too large")
                depth = _strict_json(
                    archive.read(info),
                    "ArtGS enhanced-depth manifest",
                )
                frames = depth.get("frames") if isinstance(depth, dict) else None
                if not isinstance(frames, list) or not frames:
                    raise DomainValidationError("ArtGS enhanced-depth evidence has no frames")
                expected = {"manifest.json"}
                for frame in frames:
                    record = frame.get("depth") if isinstance(frame, dict) else None
                    frame_id = frame.get("frame_id") if isinstance(frame, dict) else None
                    if not isinstance(record, dict):
                        raise DomainValidationError(
                            "ArtGS enhanced-depth frame has no depth record"
                        )
                    path = record.get("path")
                    size = record.get("size_bytes")
                    if (
                        not isinstance(frame_id, str)
                        or not frame_id
                        or frame_id in records
                        or not isinstance(path, str)
                        or isinstance(size, bool)
                        or not isinstance(size, int)
                        or size <= 0
                        or record.get("media_type") != "application/x-npy"
                        or record.get("dtype") != "float32"
                        or record.get("unit") != "m"
                    ):
                        raise DomainValidationError("ArtGS enhanced-depth frame record is invalid")
                    normalized = _safe_relative_path(path, "enhanced-depth payload")
                    payload = archive.read(normalized)
                    if len(payload) != size:
                        raise DomainValidationError(
                            "ArtGS enhanced-depth payload size does not match its manifest"
                        )
                    records[frame_id] = _DepthPayloadRecord(
                        frame_id=frame_id,
                        path=normalized,
                        size_bytes=size,
                    )
                    expected.add(normalized)
                if set(names) != expected:
                    raise DomainValidationError(
                        "ArtGS enhanced-depth ZIP does not match its manifest"
                    )
        except (OSError, KeyError, zipfile.BadZipFile, ValueError) as error:
            raise DomainValidationError("ArtGS enhanced-depth evidence is invalid") from error
    if (
        not isinstance(depth, dict)
        or depth.get("schema") != "physrealx.enhanced-depth-bundle/1"
        or depth.get("status") != "PASSED"
        or not isinstance(depth.get("depth_encoding"), dict)
        or depth["depth_encoding"].get("unit") != "m"
    ):
        raise DomainValidationError("ArtGS requires a passed EnhancedDepthBundle/1 artifact")
    return records


def _transform_frames(path: Path, state: str) -> tuple[Path, ...]:
    document = _strict_json(path.read_bytes(), path.name)
    if not isinstance(document, dict) or not isinstance(document.get("frames"), list):
        raise DomainValidationError(f"{path.name} must contain a frames array")
    for field in ("camera_angle_x", "camera_angle_y"):
        angle = document.get(field)
        if (
            isinstance(angle, bool)
            or not isinstance(angle, (int, float))
            or not math.isfinite(float(angle))
            or not 0.0 < float(angle) < math.pi
        ):
            raise DomainValidationError(f"{path.name} requires a finite {field} in radians")
    frames = document["frames"]
    if len(frames) < 2:
        raise DomainValidationError(f"{path.name} requires at least two views")
    resolved: list[Path] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise DomainValidationError(f"{path.name} frame {index} must be an object")
        file_path = frame.get("file_path")
        matrix = frame.get("transform_matrix")
        frame_time = frame.get("time")
        if not isinstance(file_path, str):
            raise DomainValidationError(f"{path.name} frame {index} is missing file_path")
        relative = _safe_relative_path(file_path, f"{path.name} frame path")
        image = PurePosixPath(relative)
        if image.suffix == "":
            image = image.with_suffix(".png")
        if image.suffix.lower() != ".png" or not image.stem.isdigit():
            raise DomainValidationError(f"{path.name} frame paths must use numeric PNG file names")
        if not image.as_posix().startswith(f"{state}/"):
            raise DomainValidationError(f"{path.name} contains a frame from another state")
        if (
            isinstance(frame_time, bool)
            or not isinstance(frame_time, (int, float))
            or not math.isfinite(float(frame_time))
        ):
            raise DomainValidationError(f"{path.name} frame {index} has invalid time")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 4
            or any(
                not isinstance(row, list)
                or len(row) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in row
                )
                for row in matrix
            )
        ):
            raise DomainValidationError(f"{path.name} frame {index} has an invalid pose")
        numeric_matrix = [[float(value) for value in row] for row in matrix]
        if any(
            abs(value - expected) > 1e-6 for value, expected in zip(numeric_matrix[3], (0, 0, 0, 1))
        ):
            raise DomainValidationError(f"{path.name} frame {index} is not a rigid transform")
        rotation = [row[:3] for row in numeric_matrix[:3]]
        for row_index, row in enumerate(rotation):
            norm = sum(value * value for value in row)
            if abs(norm - 1.0) > 1e-3:
                raise DomainValidationError(f"{path.name} frame {index} rotation is not normalized")
            for other in rotation[row_index + 1 :]:
                if abs(sum(left * right for left, right in zip(row, other))) > 1e-3:
                    raise DomainValidationError(
                        f"{path.name} frame {index} rotation is not orthogonal"
                    )
        determinant = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if abs(determinant - 1.0) > 1e-3:
            raise DomainValidationError(f"{path.name} frame {index} rotation is improper")
        resolved.append(Path(image.as_posix()))
    if len(resolved) != len(set(resolved)):
        raise DomainValidationError(f"{path.name} contains duplicate frame paths")
    return tuple(resolved)


def _png_spec(path: Path, *, depth: bool) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(29)
    if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise DomainValidationError(f"ArtGS payload is not a valid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    bit_depth, color_type = header[24], header[25]
    if width <= 0 or height <= 0:
        raise DomainValidationError(f"ArtGS PNG has invalid dimensions: {path}")
    if depth and (bit_depth, color_type) != (16, 0):
        raise DomainValidationError("ArtGS metric depth must be 16-bit grayscale PNG")
    if not depth and (bit_depth != 8 or color_type not in {2, 6}):
        raise DomainValidationError("ArtGS RGB input must be 8-bit RGB or RGBA PNG")
    return width, height


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _metric_depth_png(payload: bytes) -> bytes:
    try:
        import numpy as np
    except ImportError as error:
        raise DomainValidationError(
            "ArtGS depth conversion requires the project artgs optional dependencies"
        ) from error
    try:
        depth = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DomainValidationError("enhanced depth payload is not a safe NPY array") from error
    if depth.dtype != np.float32 or depth.ndim != 2 or depth.size == 0:
        raise DomainValidationError("enhanced depth must be a non-empty float32 image")
    if not bool(np.isfinite(depth).all()) or not bool((depth > 0.0).all()):
        raise DomainValidationError("enhanced depth must contain finite positive metres")
    if float(depth.max()) > 65.535:
        raise DomainValidationError("enhanced depth exceeds uint16 millimetre encoding range")
    millimetres = np.rint(depth * 1_000.0).astype(">u2")
    height, width = millimetres.shape
    rows = b"".join(b"\x00" + millimetres[index].tobytes() for index in range(height))
    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _fixed_zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def package_artgs_input(
    source: str | Path,
    *,
    scene_id: str,
    camera_rig: ArtifactRef,
    enhanced_depth: ArtifactRef,
    depth_frame_ids: Mapping[str, str],
    num_slots: int,
    artifacts: ArtifactStore,
    coordinate_frame: str = "object",
) -> ArtifactRef:
    """Package RGBA/poses and derive ArtGS depth PNGs from validated metric NPY frames."""

    root = Path(source).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DomainValidationError("ArtGS input source must be a directory")
    depth_records = _validated_metric_evidence(camera_rig, enhanced_depth, artifacts)

    required_transforms = [
        root / "transforms_train.json",
        root / "transforms_train_start.json",
        root / "transforms_train_end.json",
    ]
    if any(not item.is_file() for item in required_transforms):
        raise DomainValidationError("ArtGS source is missing required training transforms")
    train_start = _transform_frames(required_transforms[1], "start")
    train_end = _transform_frames(required_transforms[2], "end")
    if len(train_start) != len(train_end):
        raise DomainValidationError("ArtGS training states must have the same view count")

    test_paths = [root / "transforms_test_start.json", root / "transforms_test_end.json"]
    if test_paths[0].is_file() != test_paths[1].is_file():
        raise DomainValidationError("ArtGS validation transforms must contain both states")
    test_start: tuple[Path, ...] = ()
    test_end: tuple[Path, ...] = ()
    if test_paths[0].is_file():
        test_start = _transform_frames(test_paths[0], "start")
        test_end = _transform_frames(test_paths[1], "end")
        if len(test_start) != len(test_end):
            raise DomainValidationError("ArtGS validation states must have the same view count")

    frame_paths = (*train_start, *train_end, *test_start, *test_end)
    normalized_bindings: dict[str, str] = {}
    for image_path, frame_id in depth_frame_ids.items():
        normalized = _safe_relative_path(image_path, "ArtGS depth binding image")
        if normalized in normalized_bindings:
            raise DomainValidationError("ArtGS depth binding image paths must be unique")
        if not isinstance(frame_id, str) or not frame_id:
            raise DomainValidationError("ArtGS enhanced-depth frame IDs must be non-empty")
        normalized_bindings[normalized] = frame_id
    expected_images = {item.as_posix() for item in frame_paths}
    if set(normalized_bindings) != expected_images:
        raise DomainValidationError(
            "ArtGS depth bindings must exactly cover every training and validation image"
        )
    if len(set(normalized_bindings.values())) != len(normalized_bindings):
        raise DomainValidationError("an enhanced-depth frame cannot be reused across ArtGS views")
    missing_depth_frames = sorted(set(normalized_bindings.values()).difference(depth_records))
    if missing_depth_frames:
        raise DomainValidationError(
            "ArtGS depth bindings reference unknown enhanced frames: "
            + ", ".join(missing_depth_frames)
        )

    selected: set[Path] = {item.relative_to(root) for item in required_transforms}
    if test_start:
        selected.update(item.relative_to(root) for item in test_paths)
    for relative in frame_paths:
        image = root / relative
        if image.is_symlink() or not image.is_file():
            raise DomainValidationError(f"ArtGS frame payload is unavailable: {image}")
        selected.add(image.relative_to(root))
        _png_spec(image, depth=False)
    point_cloud = root / "points3d.ply"
    if point_cloud.is_file() and not point_cloud.is_symlink():
        selected.add(point_cloud.relative_to(root))

    with tempfile.TemporaryDirectory(prefix="physrealx-artgs-input-") as temporary:
        temporary_root = Path(temporary)
        depth_archive = temporary_root / "enhanced-depth.zip"
        generated_root = temporary_root / "generated"
        artifacts.materialize(enhanced_depth, depth_archive)
        payload_paths: dict[str, Path] = {}
        bindings: list[ArtGSDepthBinding] = []
        for relative in selected:
            path = (root / relative).resolve(strict=True)
            if root not in path.parents or not path.is_file():
                raise DomainValidationError(f"ArtGS payload escapes its source root: {relative}")
            payload_paths[relative.as_posix()] = path
        with zipfile.ZipFile(depth_archive, "r") as archive:
            for image_relative in frame_paths:
                frame_id = normalized_bindings[image_relative.as_posix()]
                record = depth_records[frame_id]
                payload = archive.read(record.path)
                depth_relative = Path(image_relative.as_posix().replace("/rgba/", "/depth/"))
                depth_path = generated_root / depth_relative
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                depth_path.write_bytes(_metric_depth_png(payload))
                if _png_spec(root / image_relative, depth=False) != _png_spec(
                    depth_path,
                    depth=True,
                ):
                    raise DomainValidationError(
                        "ArtGS RGB and enhanced-depth dimensions must match"
                    )
                payload_paths[depth_relative.as_posix()] = depth_path
                bindings.append(
                    ArtGSDepthBinding(
                        image_path=image_relative.as_posix(),
                        depth_path=depth_relative.as_posix(),
                        enhanced_frame_id=frame_id,
                    )
                )

        files: list[ArtGSInputFile] = []
        total_bytes = 0
        for relative_name, payload_path in sorted(payload_paths.items()):
            size = payload_path.stat().st_size
            total_bytes += size
            files.append(
                ArtGSInputFile(
                    path=relative_name,
                    size_bytes=size,
                    media_type=_media_type(payload_path),
                )
            )
        if len(files) > _MAX_INPUT_FILES or total_bytes > _MAX_INPUT_BYTES:
            raise DomainValidationError("ArtGS input exceeds the platform bundle limit")

        manifest = ArtGSInputManifest(
            scene_id=scene_id,
            coordinate_frame=coordinate_frame,
            source_camera=camera_rig,
            source_depth=enhanced_depth,
            num_slots=num_slots,
            training_frames=len(train_start) + len(train_end),
            validation_frames=len(test_start) + len(test_end),
            depth_bindings=tuple(sorted(bindings, key=lambda item: item.image_path)),
            files=tuple(files),
        )
        archive_path = temporary_root / "artgs-input.zip"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                _fixed_zip_info("bundle-manifest.json"),
                canonical_json_bytes(manifest),
            )
            for item in manifest.files:
                archive.writestr(
                    _fixed_zip_info(item.path),
                    payload_paths[item.path].read_bytes(),
                )
        return artifacts.put_file(archive_path, media_type="application/zip")


def _validated_extract_bundle(archive_path: Path, destination: Path) -> ArtGSInputManifest:
    try:
        archive = zipfile.ZipFile(archive_path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise DomainValidationError("ArtGS input is not a readable ZIP archive") from error
    with archive:
        members = archive.infolist()
        if not members or len(members) > _MAX_INPUT_FILES + 1:
            raise DomainValidationError("ArtGS input has an invalid member count")
        by_name: dict[str, zipfile.ZipInfo] = {}
        portable_names: set[str] = set()
        total = 0
        for member in members:
            try:
                name = _safe_relative_path(member.filename, "ZIP member")
            except ValueError as error:
                raise DomainValidationError(str(error)) from error
            portable = name.casefold()
            mode = member.external_attr >> 16
            if (
                name in by_name
                or portable in portable_names
                or member.is_dir()
                or stat.S_ISLNK(mode)
                or member.flag_bits & 0x1
            ):
                raise DomainValidationError("ArtGS ZIP contains duplicate or unsafe members")
            total += member.file_size
            if total > _MAX_INPUT_BYTES + _MAX_MANIFEST_BYTES:
                raise DomainValidationError("ArtGS ZIP exceeds the uncompressed-size limit")
            by_name[name] = member
            portable_names.add(portable)
        manifest_info = by_name.get("bundle-manifest.json")
        if manifest_info is None or manifest_info.file_size > _MAX_MANIFEST_BYTES:
            raise DomainValidationError("ArtGS ZIP is missing a bounded bundle manifest")
        try:
            manifest = ArtGSInputManifest.model_validate(
                _strict_json(archive.read(manifest_info), "ArtGS bundle manifest")
            )
        except ValueError as error:
            raise DomainValidationError("ArtGS bundle manifest is invalid") from error
        declared = {item.path: item for item in manifest.files}
        if set(by_name).difference({"bundle-manifest.json"}) != set(declared):
            raise DomainValidationError("ArtGS ZIP payload does not match its manifest")
        destination.mkdir(parents=True, exist_ok=False)
        for name, item in declared.items():
            info = by_name[name]
            payload = archive.read(info)
            if len(payload) != item.size_bytes:
                raise DomainValidationError(f"ArtGS ZIP member size is invalid: {name}")
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(payload)
        return manifest


@dataclass(frozen=True)
class _ArtGSRunResult:
    model_root: Path
    mesh_paths: tuple[Path, ...]
    metrics: Mapping[str, float]
    evaluation_scope: Literal["held_out_views", "training_views"]


class _ArtGSRunner(Protocol):
    def probe(self) -> MethodProbe: ...

    def run(
        self,
        dataset_root: Path,
        output_root: Path,
        manifest: ArtGSInputManifest,
        config: ArtGSRunConfig,
    ) -> _ArtGSRunResult: ...


class _SubprocessArtGSRunner:
    """Run the vendored implementation with the active project interpreter."""

    def __init__(self, source_root: Path | None = None) -> None:
        self.source_root = (source_root or artgs_source_root()).resolve(strict=True)

    def _python_paths(self) -> tuple[Path, ...]:
        return (
            self.source_root,
            self.source_root / "submodules" / "diff-gaussian-rasterization",
            self.source_root / "submodules" / "simple-knn",
            self.source_root / "utils" / "pointnet_lib",
        )

    def _environment(self, seed: int) -> dict[str, str]:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        paths = [str(item) for item in self._python_paths()]
        if existing:
            paths.append(existing)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(paths),
                "PYTHONHASHSEED": str(seed),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        return environment

    @staticmethod
    def _nvcc() -> Path | None:
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidate = Path(cuda_home) / "bin" / "nvcc"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        discovered = shutil.which("nvcc")
        return Path(discovered) if discovered else None

    def probe(self) -> MethodProbe:
        if _python_version() != (3, 10):
            return MethodProbe(False, "ArtGS requires the PhysReal-X Python 3.10 environment")
        if self._nvcc() is None:
            return MethodProbe(False, "ArtGS requires nvcc in CUDA_HOME or PATH")
        required_sources = (
            self.source_root / "train.py",
            self.source_root / "submodules" / "diff-gaussian-rasterization" / "setup.py",
            self.source_root / "submodules" / "simple-knn" / "setup.py",
            self.source_root / "utils" / "pointnet_lib" / "setup.py",
        )
        if any(not item.is_file() for item in required_sources):
            return MethodProbe(False, "embedded ArtGS or CUDA extension source is incomplete")
        try:
            _validate_source_snapshot(self.source_root)
        except MethodExecutionError as error:
            return MethodProbe(False, str(error))
        command = [
            sys.executable,
            "-c",
            (
                "import sys;"
                'exec("try:\\n'
                " import torch\\n"
                " if not torch.__version__.startswith('2.4.1'):"
                " raise RuntimeError('Torch 2.4.1 is required')\\n"
                " if torch.version.cuda != '12.1':"
                " raise RuntimeError('the cu121 Torch build is required')\\n"
                " if not torch.cuda.is_available():"
                " raise RuntimeError('no CUDA device is visible to this process')\\n"
                " import open3d,pytorch3d,tinycudann,torchvision\\n"
                " print(torch.cuda.get_device_name(0))\\n"
                "except Exception as error:\\n"
                " print(f'{type(error).__name__}: {error}')\\n"
                ' sys.exit(1)")'
            ),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.source_root,
                env=self._environment(0),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return MethodProbe(False, f"ArtGS dependency probe failed: {error}")
        detail = completed.stdout[-2048:].strip()
        if completed.returncode != 0:
            return MethodProbe(False, detail or "ArtGS dependency probe failed")
        return MethodProbe(
            True,
            f"embedded ArtGS is ready in {sys.executable}; CUDA device: {detail}",
        )

    def _native_imports_ready(self, environment: Mapping[str, str]) -> bool:
        validation_script = """
import importlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
for module_name, relative_root in json.loads(sys.argv[2]):
    module = importlib.import_module(module_name)
    origin_value = getattr(module, "__file__", None)
    if not isinstance(origin_value, str):
        raise RuntimeError(f"{module_name} has no import origin")
    origin = pathlib.Path(origin_value).resolve(strict=True)
    expected = (root / relative_root).resolve(strict=True)
    try:
        origin.relative_to(expected)
    except ValueError as error:
        raise RuntimeError(
            f"{module_name} resolved outside embedded ArtGS: {origin}"
        ) from error
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                validation_script,
                str(self.source_root),
                json.dumps(_NATIVE_IMPORTS, separators=(",", ":")),
            ],
            cwd=self.source_root,
            env=dict(environment),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return completed.returncode == 0

    def _ensure_native_extensions(self, environment: Mapping[str, str]) -> None:
        if self._native_imports_ready(environment):
            return
        lock_path = Path(tempfile.gettempdir()) / "physrealx-artgs-native-build.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._native_imports_ready(environment):
                return
            projects = (
                self.source_root / "submodules" / "diff-gaussian-rasterization",
                self.source_root / "submodules" / "simple-knn",
                self.source_root / "utils" / "pointnet_lib",
            )
            for project in projects:
                completed = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=project,
                    env=dict(environment),
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    text=True,
                )
                if completed.returncode != 0:
                    raise MethodExecutionError(
                        f"failed to build embedded ArtGS CUDA extension in {project.name}: "
                        f"{completed.stdout[-4096:]}"
                    )
            if not self._native_imports_ready(environment):
                raise MethodExecutionError(
                    "embedded ArtGS CUDA extensions failed import validation"
                )

    def _run_stage(
        self,
        arguments: Sequence[str],
        log_path: Path,
        environment: Mapping[str, str],
        timeout: int,
    ) -> None:
        with log_path.open("wb") as log:
            try:
                completed = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=self.source_root,
                    env=dict(environment),
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise MethodExecutionError(f"ArtGS stage timed out: {arguments[0]}") from error
        if completed.returncode != 0:
            tail = log_path.read_bytes()[-8192:].decode("utf-8", errors="replace")
            raise MethodExecutionError(f"ArtGS stage failed: {arguments[0]}\n{tail}")

    @staticmethod
    def _metrics(path: Path, prefix: str) -> dict[str, float]:
        if not path.is_file():
            raise MethodExecutionError(f"ArtGS evaluation result is missing: {path}")
        values: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 2 or row[0] not in {"PSNR", "SSIM", "LPIPS"}:
                    continue
                value = float(row[1])
                if not math.isfinite(value) or value < 0.0:
                    raise MethodExecutionError(f"ArtGS produced an invalid {row[0]} metric")
                values[f"{prefix}_{row[0].lower()}"] = value
        if set(values) != {f"{prefix}_psnr", f"{prefix}_ssim", f"{prefix}_lpips"}:
            raise MethodExecutionError("ArtGS evaluation did not emit PSNR, SSIM and LPIPS")
        return values

    def run(
        self,
        dataset_root: Path,
        output_root: Path,
        manifest: ArtGSInputManifest,
        config: ArtGSRunConfig,
    ) -> _ArtGSRunResult:
        _validate_source_snapshot(self.source_root)
        environment = self._environment(config.seed)
        self._ensure_native_extensions(environment)
        logs = output_root / "logs"
        logs.mkdir(parents=True, exist_ok=False)
        model_root = output_root / "model"
        coarse_root = model_root / "coarse"
        prediction_root = model_root / "prediction"
        reconstruction_root = model_root / "reconstruction"
        common = [
            "--source_path",
            str(dataset_root.parent.parent.parent),
            "--dataset",
            "artgs",
            "--subset",
            "realscan",
            "--scene_name",
            manifest.scene_id,
        ]
        self._run_stage(
            [
                str(self.source_root / "train_coarse.py"),
                *common,
                "--model_path",
                str(coarse_root),
                "--resolution",
                str(config.training_resolution),
                "--num_slots",
                str(manifest.num_slots),
                "--iterations",
                str(config.coarse_iterations),
                "--opacity_reg_weight",
                "0.1",
                "--random_bg_color",
                "--seed",
                str(config.seed),
            ],
            logs / "01-coarse.log",
            environment,
            config.max_stage_seconds,
        )
        self._run_stage(
            [
                str(self.source_root / "train_predict.py"),
                *common,
                "--model_path",
                str(prediction_root),
                "--coarse_model_path",
                str(coarse_root),
                "--eval",
                "--resolution",
                str(config.evaluation_resolution),
                "--num_slots",
                str(manifest.num_slots),
                "--iterations",
                str(config.joint_iterations),
                "--densify_grad_threshold",
                "0.001",
                "--random_bg_color",
                "--seed",
                str(config.seed),
            ],
            logs / "02-joints.log",
            environment,
            config.max_stage_seconds,
        )
        joint_file = prediction_root / "predicted_joint_types.json"
        joint_document = _strict_json(joint_file.read_bytes(), "ArtGS joint prediction")
        joint_types = (
            joint_document.get("joint_types") if isinstance(joint_document, dict) else None
        )
        if not isinstance(joint_types, str) or len(joint_types.split(",")) != manifest.num_slots:
            raise MethodExecutionError("ArtGS joint prediction has an invalid slot count")
        self._run_stage(
            [
                str(self.source_root / "train.py"),
                *common,
                "--model_path",
                str(reconstruction_root),
                "--coarse_model_path",
                str(coarse_root),
                "--eval",
                "--resolution",
                str(config.training_resolution),
                "--joint_types",
                joint_types,
                "--iterations",
                str(config.reconstruction_iterations),
                "--seed",
                str(config.seed),
                "--use_art_type_prior",
                "--random_bg_color",
                "--densify_grad_threshold",
                "0.001",
                "--metric_depth_loss_weight",
                str(config.metric_depth_loss_weight),
                "--mono_depth_loss_weight",
                str(config.mono_depth_loss_weight),
            ],
            logs / "03-reconstruction.log",
            environment,
            config.max_stage_seconds,
        )
        render_arguments = [
            str(self.source_root / "render.py"),
            *common,
            "--model_path",
            str(reconstruction_root),
            "--resolution",
            str(config.evaluation_resolution),
            "--iteration",
            "best",
            "--eval_app",
            "--load2gpu_on_the_fly",
            "--seed",
            str(config.seed),
        ]
        if manifest.validation_frames == 0:
            render_arguments.append("--skip_test")
        self._run_stage(
            render_arguments,
            logs / "04-evaluation.log",
            environment,
            config.max_stage_seconds,
        )

        result_root = reconstruction_root / "train" / "ours_best"
        mesh_paths = tuple(
            result_root / "meshes" / f"start_{index}.ply" for index in range(manifest.num_slots)
        )
        if any(not path.is_file() or path.stat().st_size == 0 for path in mesh_paths):
            raise MethodExecutionError("ArtGS did not export one non-empty visual mesh per slot")
        metrics = self._metrics(result_root / "result.csv", "train")
        evaluation_scope: Literal["held_out_views", "training_views"] = "training_views"
        if manifest.validation_frames > 0:
            metrics.update(
                self._metrics(
                    reconstruction_root / "test" / "ours_best" / "result.csv",
                    "validation",
                )
            )
            evaluation_scope = "held_out_views"
        return _ArtGSRunResult(
            model_root=model_root,
            mesh_paths=mesh_paths,
            metrics=metrics,
            evaluation_scope=evaluation_scope,
        )


def _bundle_outputs(result: _ArtGSRunResult, destination: Path) -> tuple[Path, int]:
    candidates = [
        result.model_root / "reconstruction" / "point_cloud" / "iteration_best" / "point_cloud.ply",
        result.model_root / "reconstruction" / "deform" / "iteration_best" / "deform.pth",
        result.model_root / "reconstruction" / "train" / "ours_best" / "joint_info.json",
        *result.mesh_paths,
    ]
    for candidate in candidates:
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise MethodExecutionError(f"ArtGS required output is missing: {candidate}")
    records: list[JSONValue] = []
    payloads: list[tuple[str, Path]] = []
    for candidate in candidates:
        if candidate in result.mesh_paths:
            name = f"meshes/{candidate.name}"
        elif candidate.suffix == ".pth":
            name = "model/deform.pth"
        elif candidate.name == "joint_info.json":
            name = "model/joint_info.json"
        else:
            name = "model/point_cloud.ply"
        record: dict[str, JSONValue] = {
            "path": name,
            "size_bytes": candidate.stat().st_size,
            "media_type": _media_type(candidate),
        }
        records.append(record)
        payloads.append((name, candidate))
    manifest: dict[str, JSONValue] = {
        "schema": _BUNDLE_SCHEMA,
        "status": "COMPLETED",
        "files": records,
    }
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr(_fixed_zip_info("bundle-manifest.json"), canonical_json_bytes(manifest))
        for name, path in payloads:
            archive.writestr(_fixed_zip_info(name), path.read_bytes())
    return destination, len(result.mesh_paths)


class ArtGSMethodAdapter:
    """Registered `geometry.reconstruct:artgs` implementation."""

    descriptor = ARTGS_DESCRIPTOR

    def __init__(self, runner: _ArtGSRunner | None = None) -> None:
        self._runner = runner or _SubprocessArtGSRunner()

    def probe(self) -> MethodProbe:
        return self._runner.probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.descriptor != self.descriptor:
            raise DomainValidationError("ArtGS operation uses a mismatched method descriptor")
        if len(spec.inputs) != 1 or spec.inputs[0].name != "source":
            raise DomainValidationError("ArtGS requires exactly one source artifact")
        source = spec.inputs[0].artifact
        if source is None or source.media_type != "application/zip" or not artifacts.verify(source):
            raise DomainValidationError("ArtGS source must be an available ZIP artifact")
        try:
            config = ArtGSRunConfig.model_validate(dict(spec.method.config))
        except ValueError as error:
            raise DomainValidationError("ArtGS method configuration is invalid") from error

        with tempfile.TemporaryDirectory(prefix="physrealx-artgs-run-") as temporary:
            root = Path(temporary)
            archive = root / "input.zip"
            artifacts.materialize(source, archive)
            extracted = root / "dataset" / "artgs" / "realscan" / "source"
            manifest = _validated_extract_bundle(archive, extracted)
            if manifest.scene_id != "source":
                target = extracted.parent / manifest.scene_id
                extracted.rename(target)
                extracted = target
            _validated_metric_evidence(
                manifest.source_camera,
                manifest.source_depth,
                artifacts,
            )
            if manifest.validation_frames == 0 and not config.allow_training_view_evaluation:
                raise DomainValidationError(
                    "ArtGS requires held-out views; explicitly enable training-view evaluation "
                    "only for non-promotable experiments"
                )
            output_root = root / "output"
            output_root.mkdir()
            result = self._runner.run(extracted, output_root, manifest, config)
            if (
                result.evaluation_scope == "training_views"
                and not config.allow_training_view_evaluation
            ):
                raise MethodExecutionError("ArtGS runner omitted required held-out evaluation")
            if any(not math.isfinite(float(value)) for value in result.metrics.values()):
                raise MethodExecutionError("ArtGS returned non-finite reconstruction metrics")

            bundle_path, mesh_count = _bundle_outputs(result, root / "visual-geometry.zip")
            bundle_ref = artifacts.put_file(bundle_path, media_type="application/zip")
            mesh_refs = tuple(
                artifacts.put_file(path, media_type="model/ply") for path in result.mesh_paths
            )
            passed = result.evaluation_scope == "held_out_views"
            report = {
                "schema": _REPORT_SCHEMA,
                "status": "PASSED" if passed else "TRAINING_VIEWS_ONLY",
                "unit": "m",
                "coordinate_frame": manifest.coordinate_frame,
                "source_camera_artifact_id": manifest.source_camera.artifact_id,
                "source_depth_artifact_id": manifest.source_depth.artifact_id,
                "method": {
                    "name": self.descriptor.name,
                    "version": self.descriptor.version,
                    "source_revision": self.descriptor.source_revision,
                    "upstream_revision": _validate_source_snapshot(artgs_source_root()),
                },
                "evaluation_scope": result.evaluation_scope,
                "bundle": bundle_ref.model_dump(mode="json"),
                "meshes": [
                    {
                        "link": f"link_{index}",
                        "artifact": reference.model_dump(mode="json"),
                    }
                    for index, reference in enumerate(mesh_refs)
                ],
                "metrics": {key: float(value) for key, value in sorted(result.metrics.items())},
                "gates": {
                    "metric_input": True,
                    "two_state_capture": manifest.training_frames >= 4,
                    "complete_link_meshes": len(mesh_refs) == manifest.num_slots,
                    "held_out_evaluation": passed,
                },
            }
            report_ref = artifacts.put_json(report, media_type="application/json")
            metrics = {key: float(value) for key, value in result.metrics.items()}
            metrics.update(
                {
                    "mesh_count": float(mesh_count),
                    "training_frames": float(manifest.training_frames),
                    "validation_frames": float(manifest.validation_frames),
                }
            )
            return MethodExecutionOutput(
                outputs=(
                    OperationOutput(
                        name="visual_geometry_bundle",
                        contract="VisualGeometryBundle/1",
                        artifact=bundle_ref,
                    ),
                    OperationOutput(
                        name="geometry_reconstruction_report",
                        contract="GeometryReconstructionReport/1",
                        artifact=report_ref,
                    ),
                ),
                metrics=metrics,
            )


__all__ = [
    "ARTGS_DESCRIPTOR",
    "ArtGSInputFile",
    "ArtGSInputManifest",
    "ArtGSMethodAdapter",
    "ArtGSRunConfig",
    "artgs_source_root",
    "package_artgs_input",
]
