"""Offline importers for already captured real-observation packages."""

from __future__ import annotations

import ast
import mimetypes
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from physrealx.contracts import ArtifactWriter
from physrealx.digital_twin import (
    AttachmentRole,
    IssueSeverity,
    ObservationModality,
    ObservationSetManifest,
    ObservationStateRole,
    ObservationStream,
    SourceAttachment,
    SourceInspection,
    SourceIssue,
    SourceStreamSummary,
)
from physrealx.domain import JSONValue, MethodCapability
from physrealx.errors import ArtifactIntegrityError, DomainValidationError, SourcePackageError
from physrealx.methods import builtin_method

_STREAM_LAYOUT: Final = (
    ("real/can/start", "can.start", ObservationStateRole.RECONSTRUCTION),
    ("real/can/end", "can.end", ObservationStateRole.RECONSTRUCTION),
    ("real/cmp/start", "cmp.start", ObservationStateRole.VALIDATION),
    ("real/video", "video", ObservationStateRole.SEQUENCE),
)
_CALIBRATION_MARKERS: Final = ("calib", "intrinsic", "extrinsic", "camera_pose")
_ALLOWED_CONFIG: Final = {"coordinate_frame", "depth_unit", "include_candidates"}
_MAX_NPY_HEADER_BYTES: Final = 65_536
_TAKE_ID: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_PACKED_DEPTH_DTYPE: Final = re.compile(r"^[|<>=]S([1-9]\d*)$")


def _media_type(path: Path) -> str:
    explicit = {
        ".json": "application/json",
        ".mdl": "text/plain",
        ".npy": "application/x-npy",
        ".ply": "model/ply",
        ".usd": "model/vnd.usd",
    }
    return (
        explicit.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


def _npy_sample_count(path: Path) -> int:
    """Validate a packed byte-string NPY without loading its frame payloads."""

    try:
        with path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                raise ValueError("invalid NPY magic")
            version = tuple(stream.read(2))
            if not version:
                raise ValueError("truncated NPY version")
            if version[0] == 1:
                header_size = struct.unpack("<H", stream.read(2))[0]
                preamble_size = 10
            elif version[0] in {2, 3}:
                header_size = struct.unpack("<I", stream.read(4))[0]
                preamble_size = 12
            else:
                raise ValueError(f"unsupported NPY version {version}")
            if not 1 <= header_size <= _MAX_NPY_HEADER_BYTES:
                raise ValueError("NPY header size is outside the accepted range")
            encoded_header = stream.read(header_size)
            if len(encoded_header) != header_size:
                raise ValueError("truncated NPY header")
            header = ast.literal_eval(encoded_header.decode("latin1"))
    except (
        IndexError,
        OSError,
        SyntaxError,
        UnicodeDecodeError,
        ValueError,
        struct.error,
    ) as error:
        raise DomainValidationError(f"cannot read NPY header {path}: {error}") from error
    if not isinstance(header, dict) or not isinstance(header.get("shape"), tuple):
        raise DomainValidationError(f"NPY header has no valid shape: {path}")
    shape = header["shape"]
    if len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0:
        raise DomainValidationError(f"packed depth NPY must have one positive dimension: {path}")
    match = (
        _PACKED_DEPTH_DTYPE.fullmatch(header.get("descr", ""))
        if isinstance(header.get("descr"), str)
        else None
    )
    if match is None or header.get("fortran_order") is not False:
        raise DomainValidationError(
            f"packed depth NPY must contain fixed-width byte strings: {path}"
        )
    sample_count = int(shape[0])
    expected_size = preamble_size + header_size + sample_count * int(match.group(1))
    if path.stat().st_size != expected_size:
        raise DomainValidationError(
            f"packed depth NPY payload size does not match its header: {path}"
        )
    return sample_count


def _source_snapshot(root: Path) -> dict[str, tuple[int, int, int, int]]:
    snapshot: dict[str, tuple[int, int, int, int]] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item.is_symlink():
            continue
        status = item.stat()
        snapshot[item.relative_to(root).as_posix()] = (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
        )
    return snapshot


def _numeric_pngs(directory: Path) -> tuple[list[Path], SourceIssue | None]:
    pngs = [item for item in directory.glob("*.png") if item.is_file() and not item.is_symlink()]
    numeric = [item for item in pngs if item.stem.isdigit()]
    numeric.sort(key=lambda item: int(item.stem))
    if len(numeric) != len(pngs):
        return numeric, SourceIssue(
            severity=IssueSeverity.ERROR,
            code="invalid_rgb_name",
            message="RGB frame names must be zero-based integer PNG names",
        )
    if numeric and [int(item.stem) for item in numeric] != list(range(len(numeric))):
        return numeric, SourceIssue(
            severity=IssueSeverity.ERROR,
            code="non_contiguous_rgb_frames",
            message="RGB frame indices must be contiguous and start at zero",
        )
    return numeric, None


def _state_sources(directory: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    image_root = directory / "img"
    depth_root = directory / "depth"
    image_directories: dict[str, Path] = {}
    depth_files: dict[str, Path] = {}
    if image_root.is_dir() and not image_root.is_symlink():
        image_directories = {
            item.name: item
            for item in image_root.iterdir()
            if item.is_dir() and not item.is_symlink()
        }
    if depth_root.is_dir() and not depth_root.is_symlink():
        depth_files = {
            item.name[: -len("_depth.npy")]: item
            for item in depth_root.glob("*_depth.npy")
            if item.is_file() and not item.is_symlink()
        }
    return image_directories, depth_files


def _stream_name(state: str, take_id: str, modality: ObservationModality) -> str:
    return f"{state}.{take_id}.{modality.value}"


class ArtModelingImporter:
    """Importer for the existing ArtModeling multi-state directory layout."""

    descriptor = builtin_method(
        MethodCapability.OBSERVATION_IMPORT,
        "artmodeling",
        input_contract="ArtModelingCaptureDirectory",
        output_contract="ObservationSetManifest",
    )

    @staticmethod
    def _root(source: Path) -> Path:
        expanded = source.expanduser()
        if expanded.is_symlink():
            raise SourcePackageError(f"source package cannot be a symbolic link: {expanded}")
        try:
            root = expanded.resolve(strict=True)
        except OSError as error:
            raise SourcePackageError(f"source package is not accessible: {source}") from error
        if not root.is_dir():
            raise SourcePackageError(f"source package is not a directory: {root}")
        return root

    def inspect(self, source: Path) -> SourceInspection:
        root = self._root(source)
        files = sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink())
        issues: list[SourceIssue] = []
        streams: list[SourceStreamSummary] = []

        for item in files:
            if item.is_symlink():
                issues.append(
                    SourceIssue(
                        severity=IssueSeverity.ERROR,
                        code="symbolic_link_not_allowed",
                        message="source packages cannot contain symbolic links",
                        path=item.relative_to(root).as_posix(),
                    )
                )

        for relative, state, role in _STREAM_LAYOUT:
            directory = root / relative
            required = state in {"can.start", "can.end"}
            if not directory.is_dir() or directory.is_symlink():
                if required:
                    issues.append(
                        SourceIssue(
                            severity=IssueSeverity.ERROR,
                            code="missing_required_state",
                            message=f"required reconstruction state {state!r} is missing",
                            path=relative,
                        )
                    )
                continue

            image_directories, depth_files = _state_sources(directory)
            take_ids = sorted(set(image_directories) | set(depth_files))
            if not take_ids:
                issues.append(
                    SourceIssue(
                        severity=IssueSeverity.ERROR,
                        code="missing_state_observations",
                        message=f"state {state!r} has no paired image/depth take",
                        path=relative,
                    )
                )
                continue

            for take_id in take_ids:
                take_path = f"{relative}/img/{take_id}"
                if _TAKE_ID.fullmatch(take_id) is None:
                    issues.append(
                        SourceIssue(
                            severity=IssueSeverity.ERROR,
                            code="invalid_take_id",
                            message="take IDs must use lowercase letters, digits, '.', '_' or '-'",
                            path=take_path,
                        )
                    )
                    continue
                image_directory = image_directories.get(take_id)
                depth_file = depth_files.get(take_id)
                if image_directory is None or depth_file is None:
                    issues.append(
                        SourceIssue(
                            severity=IssueSeverity.ERROR,
                            code="unpaired_rgb_depth_take",
                            message=f"take {take_id!r} requires both img/ and depth/ inputs",
                            path=relative,
                        )
                    )
                    continue
                rgb, rgb_issue = _numeric_pngs(image_directory)
                if rgb_issue is not None:
                    issues.append(rgb_issue.model_copy(update={"path": take_path}))
                if not rgb:
                    issues.append(
                        SourceIssue(
                            severity=IssueSeverity.ERROR,
                            code="missing_rgb_frames",
                            message=f"take {state!r}/{take_id!r} contains no RGB frames",
                            path=take_path,
                        )
                    )
                    continue
                try:
                    depth_count = _npy_sample_count(depth_file)
                except DomainValidationError as error:
                    issues.append(
                        SourceIssue(
                            severity=IssueSeverity.ERROR,
                            code="invalid_depth_container",
                            message=str(error),
                            path=depth_file.relative_to(root).as_posix(),
                        )
                    )
                    continue
                streams.extend(
                    (
                        SourceStreamSummary(
                            name=_stream_name(state, take_id, ObservationModality.RGB),
                            modality=ObservationModality.RGB,
                            state=state,
                            role=role,
                            frame_count=len(rgb),
                        ),
                        SourceStreamSummary(
                            name=_stream_name(state, take_id, ObservationModality.DEPTH),
                            modality=ObservationModality.DEPTH,
                            state=state,
                            role=role,
                            frame_count=depth_count,
                        ),
                    )
                )
                if len(rgb) != depth_count:
                    issues.append(
                        SourceIssue(
                            severity=IssueSeverity.ERROR,
                            code="rgb_depth_count_mismatch",
                            message=(
                                f"take {state!r}/{take_id!r} has {len(rgb)} RGB and "
                                f"{depth_count} depth frames"
                            ),
                            path=relative,
                        )
                    )

        lower_names = tuple(item.name.lower() for item in files)
        if not any(marker in name for name in lower_names for marker in _CALIBRATION_MARKERS):
            issues.append(
                SourceIssue(
                    severity=IssueSeverity.WARNING,
                    code="missing_camera_calibration",
                    message="camera intrinsics/extrinsics are absent; a recovery stage is required",
                )
            )
        issues.append(
            SourceIssue(
                severity=IssueSeverity.WARNING,
                code="unspecified_depth_unit",
                message=(
                    "depth scale and invalid-value convention must be supplied during ingestion"
                ),
            )
        )
        regular_files = [item for item in files if item.is_file() and not item.is_symlink()]
        return SourceInspection(
            importer=self.descriptor.name,
            importer_version=self.descriptor.version,
            source_uri=root.as_uri(),
            total_files=len(regular_files),
            total_bytes=sum(item.stat().st_size for item in regular_files),
            streams=tuple(streams),
            issues=tuple(issues),
        )

    def ingest(
        self,
        source: Path,
        *,
        resource_id: str,
        version: str,
        display_name: str,
        config: Mapping[str, JSONValue],
        artifacts: ArtifactWriter,
    ) -> ObservationSetManifest:
        unknown = set(config) - _ALLOWED_CONFIG
        if unknown:
            raise SourcePackageError(f"unknown ArtModeling importer options: {sorted(unknown)}")
        depth_unit = config.get("depth_unit")
        coordinate_frame = config.get("coordinate_frame")
        include_candidates = config.get("include_candidates", True)
        if not isinstance(depth_unit, str) or not depth_unit.strip():
            raise SourcePackageError("ArtModeling ingestion requires an explicit depth_unit")
        if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
            raise SourcePackageError("ArtModeling ingestion requires an explicit coordinate_frame")
        if not isinstance(include_candidates, bool):
            raise SourcePackageError("include_candidates must be a boolean")

        root = self._root(source)
        inspection = self.inspect(root)
        if not inspection.ingestible:
            codes = sorted(
                item.code for item in inspection.issues if item.severity is IssueSeverity.ERROR
            )
            raise SourcePackageError(
                "ArtModeling source package failed inspection",
                details={"issue_codes": cast(JSONValue, codes)},
            )
        source_snapshot = _source_snapshot(root)

        used: set[Path] = set()
        streams: list[ObservationStream] = []
        for relative, state, role in _STREAM_LAYOUT:
            directory = root / relative
            if not directory.is_dir():
                continue
            image_directories, depth_files = _state_sources(directory)
            for take_id, image_directory in sorted(image_directories.items()):
                depth_file = depth_files[take_id]
                rgb, _ = _numeric_pngs(image_directory)
                used.update(rgb)
                used.add(depth_file)
                rgb_artifacts = tuple(
                    artifacts.put_file(item, media_type=_media_type(item)) for item in rgb
                )
                depth_artifact = artifacts.put_file(
                    depth_file,
                    media_type=_media_type(depth_file),
                )
                depth_count = _npy_sample_count(depth_file)
                streams.extend(
                    (
                        ObservationStream(
                            name=_stream_name(state, take_id, ObservationModality.RGB),
                            modality=ObservationModality.RGB,
                            state=state,
                            role=role,
                            encoding="png",
                            frame_count=len(rgb),
                            artifacts=rgb_artifacts,
                            coordinate_frame=coordinate_frame,
                            metadata={
                                "take_id": take_id,
                                "relative_directory": image_directory.relative_to(root).as_posix(),
                            },
                        ),
                        ObservationStream(
                            name=_stream_name(state, take_id, ObservationModality.DEPTH),
                            modality=ObservationModality.DEPTH,
                            state=state,
                            role=role,
                            encoding="npy.packed_png16",
                            frame_count=depth_count,
                            artifacts=(depth_artifact,),
                            unit=depth_unit,
                            coordinate_frame=coordinate_frame,
                            metadata={
                                "take_id": take_id,
                                "relative_path": depth_file.relative_to(root).as_posix(),
                                "invalid_value_policy": "declared_by_recipe",
                            },
                        ),
                    )
                )

        attachments: list[SourceAttachment] = []
        for item in sorted(path for path in root.rglob("*") if path.is_file()):
            if item in used or item.is_symlink():
                continue
            relative = item.relative_to(root).as_posix()
            if relative.startswith("real/recon/"):
                attachment_role = AttachmentRole.RECONSTRUCTION_CANDIDATE
            elif relative.startswith("sim/"):
                attachment_role = AttachmentRole.SIMULATION_CANDIDATE
            elif any(marker in item.name.lower() for marker in _CALIBRATION_MARKERS):
                attachment_role = AttachmentRole.CALIBRATION
            else:
                attachment_role = AttachmentRole.METADATA
            if include_candidates or attachment_role not in {
                AttachmentRole.RECONSTRUCTION_CANDIDATE,
                AttachmentRole.SIMULATION_CANDIDATE,
            }:
                attachments.append(
                    SourceAttachment(
                        path=relative,
                        role=attachment_role,
                        artifact=artifacts.put_file(item, media_type=_media_type(item)),
                    )
                )

        if _source_snapshot(root) != source_snapshot:
            raise ArtifactIntegrityError("source package changed during ingestion")

        return ObservationSetManifest(
            resource_id=resource_id,
            version=version,
            display_name=display_name,
            importer=self.descriptor.name,
            importer_version=self.descriptor.version,
            source_uri=root.as_uri(),
            streams=tuple(streams),
            attachments=tuple(attachments),
            issues=tuple(
                item for item in inspection.issues if item.code != "unspecified_depth_unit"
            ),
            metadata={
                "coordinate_frame": coordinate_frame,
                "depth_unit": depth_unit,
                "source_file_count": inspection.total_files,
                "source_total_bytes": inspection.total_bytes,
            },
        )


__all__ = ["ArtModelingImporter"]
