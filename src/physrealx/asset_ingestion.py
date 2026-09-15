"""Offline importers for already captured real-observation packages."""

from __future__ import annotations

import mimetypes
from collections.abc import Iterator, Mapping
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
    ViewAlignmentSpec,
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
_ALLOWED_CONFIG: Final = {
    "states",
    "coordinate_frame",
    "depth_unit",
    "include_candidates",
    "view_alignment",
}


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


def _validate_depth_frame(path: Path) -> None:
    import numpy as np

    try:
        depth = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DomainValidationError(f"cannot read depth frame {path}: {error}") from error
    if depth.dtype != np.uint16 or depth.ndim != 2 or 0 in depth.shape:
        raise DomainValidationError(f"depth frame must be a nonempty uint16 image: {path}")


def _view_alignment_specs(
    value: JSONValue | None,
    inspection: SourceInspection,
) -> dict[str, ViewAlignmentSpec]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SourcePackageError("view_alignment must map exact RGB stream names to policies")
    valid_streams = {
        stream.name for stream in inspection.streams if stream.modality is ObservationModality.RGB
    }
    unknown = sorted(set(value) - valid_streams)
    if unknown:
        raise SourcePackageError(
            "view_alignment contains unknown or non-RGB streams",
            details={"stream_names": cast(JSONValue, unknown)},
        )
    result: dict[str, ViewAlignmentSpec] = {}
    for name, document in value.items():
        if not isinstance(document, dict):
            raise SourcePackageError(f"view_alignment[{name!r}] must be an object")
        try:
            result[name] = ViewAlignmentSpec.model_validate(document)
        except ValueError as error:
            raise SourcePackageError(f"view_alignment[{name!r}] is invalid: {error}") from error
    return result


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


def _numeric_frames(
    directory: Path, *, suffix: str, modality: str
) -> tuple[list[Path], SourceIssue | None]:
    if not directory.is_dir() or directory.is_symlink():
        return [], None
    files = [item for item in directory.iterdir() if item.is_file() and not item.is_symlink()]
    numeric = [item for item in files if item.suffix.lower() == suffix and item.stem.isdigit()]
    numeric.sort(key=lambda item: int(item.stem))
    if len(numeric) != len(files):
        return numeric, SourceIssue(
            severity=IssueSeverity.ERROR,
            code=f"invalid_{modality}_name",
            message=(f"{modality.upper()} frames must use integer names and {suffix} files"),
        )
    return numeric, None


def _state_sources(directory: Path) -> tuple[Path, Path, Path]:
    return directory / "img", directory / "depth", directory / "mask"


def _stream_name(state: str, modality: ObservationModality) -> str:
    return f"{state}.{modality.value}"


def _state_layout(root: Path) -> Iterator[tuple[str, str, ObservationStateRole]]:
    """Resolve nested sequence states without flattening their identities."""
    for relative, state, role in _STREAM_LAYOUT:
        directory = root / relative
        nested = (
            sorted(
                p
                for p in directory.iterdir()
                if p.is_dir() and not p.is_symlink() and (p / "depth").is_dir()
            )
            if role is ObservationStateRole.SEQUENCE and directory.is_dir()
            else []
        )
        if nested and not (directory / "depth").is_dir():
            for child in nested:
                yield child.relative_to(root).as_posix(), f"{state}.{child.name}", role
        else:
            yield relative, state, role


def _selected_layout(
    root: Path, config: Mapping[str, JSONValue] | None
) -> list[tuple[str, str, ObservationStateRole]]:
    layout = list(_state_layout(root))
    selected = (config or {}).get("states")
    if selected is None:
        return layout
    if (
        not isinstance(selected, (list, tuple))
        or not selected
        or any(not isinstance(s, str) for s in selected)
        or len(set(selected)) != len(selected)
        or not set(selected) <= {row[1] for row in layout}
    ):
        raise SourcePackageError("states must name unique existing capture states")
    return [row for row in layout if row[1] in selected]


class ArtModelingImporter:
    """Importer for the canonical single-stream ArtModeling state layout."""

    descriptor = builtin_method(
        MethodCapability.OBSERVATION_IMPORT,
        "artmodeling",
        input_contract="ArtModelingCaptureDirectory/3",
        output_contract="ObservationSetManifest/1",
        version="3.0.0",
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

    def inspect(
        self, source: Path, *, config: Mapping[str, JSONValue] | None = None
    ) -> SourceInspection:
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

        for relative, state, role in _selected_layout(root, config):
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

            image_directory, depth_directory, mask_directory = _state_sources(directory)
            rgb, rgb_issue = _numeric_frames(image_directory, suffix=".png", modality="rgb")
            depth, depth_issue = _numeric_frames(depth_directory, suffix=".npy", modality="depth")
            masks, mask_issue = _numeric_frames(mask_directory, suffix=".png", modality="mask")
            if not rgb and not depth and not masks:
                issues.append(
                    SourceIssue(
                        severity=IssueSeverity.ERROR,
                        code="missing_state_observations",
                        message=f"state {state!r} has no RGB/depth/mask frames",
                        path=relative,
                    )
                )
                continue
            for issue in (rgb_issue, depth_issue, mask_issue):
                if issue is not None:
                    issues.append(issue.model_copy(update={"path": relative}))
            for path in depth:
                try:
                    _validate_depth_frame(path)
                except DomainValidationError as error:
                    issues.append(
                        SourceIssue(
                            severity=IssueSeverity.ERROR,
                            code="invalid_depth_frame",
                            message=str(error),
                            path=path.relative_to(root).as_posix(),
                        )
                    )
                    break
            for modality, frames in (
                (ObservationModality.RGB, rgb),
                (ObservationModality.DEPTH, depth),
                (ObservationModality.MASK, masks),
            ):
                if frames:
                    streams.append(
                        SourceStreamSummary(
                            name=_stream_name(state, modality),
                            modality=modality,
                            state=state,
                            role=role,
                            frame_count=len(frames),
                        )
                    )
            frame_ids = [path.stem for path in rgb]
            if frame_ids != [path.stem for path in depth] or frame_ids != [
                path.stem for path in masks
            ]:
                issues.append(
                    SourceIssue(
                        severity=IssueSeverity.ERROR,
                        code="rgb_depth_mask_frame_mismatch",
                        message=(
                            f"state {state!r} requires matching RGB, depth and mask frame IDs"
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
        inspection = self.inspect(root, config=config)
        if not inspection.ingestible:
            codes = sorted(
                item.code for item in inspection.issues if item.severity is IssueSeverity.ERROR
            )
            raise SourcePackageError(
                "ArtModeling source package failed inspection",
                details={"issue_codes": cast(JSONValue, codes)},
            )
        view_alignments = _view_alignment_specs(config.get("view_alignment"), inspection)
        source_snapshot = _source_snapshot(root)

        used: set[Path] = set()
        streams: list[ObservationStream] = []
        for relative, state, role in _selected_layout(root, config):
            directory = root / relative
            if not directory.is_dir():
                continue
            image_directory, depth_directory, mask_directory = _state_sources(directory)
            rgb, _ = _numeric_frames(image_directory, suffix=".png", modality="rgb")
            depth, _ = _numeric_frames(depth_directory, suffix=".npy", modality="depth")
            masks, _ = _numeric_frames(mask_directory, suffix=".png", modality="mask")
            used.update(rgb)
            used.update(depth)
            used.update(masks)
            rgb_artifacts = tuple(
                artifacts.put_file(item, media_type=_media_type(item)) for item in rgb
            )
            depth_artifacts = tuple(
                artifacts.put_file(item, media_type="application/x-npy") for item in depth
            )
            mask_artifacts = tuple(
                artifacts.put_file(item, media_type="image/png") for item in masks
            )
            rgb_stream_name = _stream_name(state, ObservationModality.RGB)
            streams.extend(
                (
                    ObservationStream(
                        name=rgb_stream_name,
                        modality=ObservationModality.RGB,
                        state=state,
                        role=role,
                        encoding="png",
                        frame_count=len(rgb),
                        artifacts=rgb_artifacts,
                        coordinate_frame=coordinate_frame,
                        view_alignment=view_alignments.get(
                            rgb_stream_name,
                            ViewAlignmentSpec(),
                        ),
                        metadata={
                            "relative_directory": image_directory.relative_to(root).as_posix(),
                            "frame_ids": [item.stem for item in rgb],
                        },
                    ),
                    ObservationStream(
                        name=_stream_name(state, ObservationModality.DEPTH),
                        modality=ObservationModality.DEPTH,
                        state=state,
                        role=role,
                        encoding="npy.depth.uint16",
                        frame_count=len(depth),
                        artifacts=depth_artifacts,
                        unit=depth_unit,
                        coordinate_frame=coordinate_frame,
                        metadata={
                            "relative_directory": depth_directory.relative_to(root).as_posix(),
                            "frame_ids": [item.stem for item in depth],
                            "invalid_value_policy": "declared_by_recipe",
                        },
                    ),
                    ObservationStream(
                        name=_stream_name(state, ObservationModality.MASK),
                        modality=ObservationModality.MASK,
                        state=state,
                        role=role,
                        encoding="png.mask.uint8",
                        frame_count=len(masks),
                        artifacts=mask_artifacts,
                        coordinate_frame=coordinate_frame,
                        metadata={
                            "relative_directory": mask_directory.relative_to(root).as_posix(),
                            "frame_ids": [item.stem for item in masks],
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
                "import_config": dict(config),
                "source_total_bytes": inspection.total_bytes,
            },
        )


__all__ = ["ArtModelingImporter"]
