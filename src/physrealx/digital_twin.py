"""Contracts for building digital twins from already captured observations.

The contracts describe immutable inputs, versioned processing recipes and
auditable outputs.  They deliberately contain no camera drivers, filesystem
logic or reconstruction framework objects.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    ResourceType,
    StrEnum,
    VersionedRef,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_VERSION = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_RELATIVE_PATH = re.compile(r"^[^\\\x00]+$")
_URI = re.compile(r"^[a-z][a-z0-9+.-]*://\S+$")
_UNIT = re.compile(r"^(?:1|[A-Za-zµ°%][A-Za-z0-9µ°%*/^()._-]{0,62})$")
_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_./:-]{0,126}$")


def _identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{label} must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_' or '-'"
        )
    return value


def _semver(value: str) -> str:
    if _VERSION.fullmatch(value) is None:
        raise ValueError("version must be an explicit semantic version such as '1.0.0'")
    return value


def _relative_path(value: str) -> str:
    if (
        _RELATIVE_PATH.fullmatch(value) is None
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("path must be a normalized relative POSIX path without traversal")
    return value


def _uri(value: str) -> str:
    if _URI.fullmatch(value) is None:
        raise ValueError("URI must be absolute and include a scheme")
    return value


class IssueSeverity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class SourceIssue(DomainModel):
    severity: IssueSeverity
    code: str
    message: str = Field(min_length=1, max_length=1024)
    path: str | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _identifier(value, "issue code")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return _relative_path(value) if value is not None else None


class ObservationModality(StrEnum):
    RGB = "rgb"
    DEPTH = "depth"
    EVENT = "event"
    POINT_CLOUD = "point_cloud"
    FORCE_TORQUE = "force_torque"
    CAMERA_POSE = "camera_pose"
    ROBOT_STATE = "robot_state"


class ObservationStateRole(StrEnum):
    RECONSTRUCTION = "reconstruction"
    VALIDATION = "validation"
    SEQUENCE = "sequence"


class SourceStreamSummary(DomainModel):
    name: str
    modality: ObservationModality
    state: str
    role: ObservationStateRole
    frame_count: int = Field(gt=0)

    @field_validator("name", "state")
    @classmethod
    def validate_names(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))


class SourceInspection(DomainModel):
    importer: str
    importer_version: str
    source_uri: str = Field(min_length=1)
    total_files: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    streams: tuple[SourceStreamSummary, ...] = ()
    issues: tuple[SourceIssue, ...] = ()

    @field_validator("importer")
    @classmethod
    def validate_importer(cls, value: str) -> str:
        return _identifier(value, "importer")

    @field_validator("importer_version")
    @classmethod
    def validate_importer_version(cls, value: str) -> str:
        return _semver(value)

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str) -> str:
        return _uri(value)

    @model_validator(mode="after")
    def validate_streams(self) -> SourceInspection:
        names = [item.name for item in self.streams]
        if len(names) != len(set(names)):
            raise ValueError("inspection stream names must be unique")
        return self

    @property
    def ingestible(self) -> bool:
        return not any(item.severity is IssueSeverity.ERROR for item in self.issues)


class ObservationStream(DomainModel):
    name: str
    modality: ObservationModality
    state: str
    role: ObservationStateRole
    encoding: str = Field(min_length=1, max_length=127)
    frame_count: int = Field(gt=0)
    artifacts: tuple[ArtifactRef, ...]
    unit: str | None = Field(default=None, min_length=1, max_length=31)
    coordinate_frame: str | None = Field(default=None, min_length=1, max_length=127)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name", "state")
    @classmethod
    def validate_names(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, value: str) -> str:
        return _identifier(value, "encoding")

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str | None) -> str | None:
        if value is not None and _UNIT.fullmatch(value) is None:
            raise ValueError("unit must use a compact machine-readable symbol")
        return value

    @field_validator("coordinate_frame")
    @classmethod
    def validate_coordinate_frame(cls, value: str | None) -> str | None:
        if value is not None and (_FRAME.fullmatch(value) is None or ".." in value):
            raise ValueError("coordinate_frame must be a logical frame without traversal")
        return value

    @model_validator(mode="after")
    def validate_artifacts(self) -> ObservationStream:
        if not self.artifacts:
            raise ValueError("an observation stream requires at least one artifact")
        if len(self.artifacts) not in {1, self.frame_count}:
            raise ValueError("a stream must use one packed artifact or one artifact per frame")
        if self.modality is ObservationModality.DEPTH and self.unit is None:
            raise ValueError("a depth stream requires an explicit unit")
        return self


class AttachmentRole(StrEnum):
    CALIBRATION = "calibration"
    METADATA = "metadata"
    RECONSTRUCTION_CANDIDATE = "reconstruction_candidate"
    SIMULATION_CANDIDATE = "simulation_candidate"


class SourceAttachment(DomainModel):
    path: str
    role: AttachmentRole
    artifact: ArtifactRef

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative_path(value)


class ObservationSetManifest(DomainModel):
    schema_version: Literal[1] = 1
    resource_id: str
    version: str
    display_name: str = Field(min_length=1, max_length=255)
    importer: str
    importer_version: str
    source_uri: str = Field(min_length=1)
    streams: tuple[ObservationStream, ...]
    attachments: tuple[SourceAttachment, ...] = ()
    issues: tuple[SourceIssue, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("resource_id", "importer")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("version", "importer_version")
    @classmethod
    def validate_versions(cls, value: str) -> str:
        return _semver(value)

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str) -> str:
        return _uri(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> ObservationSetManifest:
        if not self.streams:
            raise ValueError("an observation set requires at least one stream")
        names = [item.name for item in self.streams]
        if len(names) != len(set(names)):
            raise ValueError("observation stream names must be unique")
        attachment_paths = [item.path for item in self.attachments]
        if len(attachment_paths) != len(set(attachment_paths)):
            raise ValueError("source attachment paths must be unique")
        if any(item.severity is IssueSeverity.ERROR for item in self.issues):
            raise ValueError("an observation-set manifest cannot publish source errors")
        return self

    def reference(self, artifact_id: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.OBSERVATION_SET,
            resource_id=self.resource_id,
            version=self.version,
            artifact_id=artifact_id,
        )


class AssetValidationProfile(StrEnum):
    SCHEMA = "schema"
    SIMULATION = "simulation"
    DIGITAL_TWIN = "digital_twin"
    TRAINABLE = "trainable"


class AssetValidationIssue(DomainModel):
    severity: IssueSeverity
    code: str
    message: str = Field(min_length=1, max_length=1024)
    path: str | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _identifier(value, "validation issue code")


class AssetValidationReport(DomainModel):
    profile: AssetValidationProfile
    issues: tuple[AssetValidationIssue, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metrics(self) -> AssetValidationReport:
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("validation metrics must be finite")
        return self

    @property
    def passed(self) -> bool:
        return not any(item.severity is IssueSeverity.ERROR for item in self.issues)


__all__ = [
    "AssetValidationIssue",
    "AssetValidationProfile",
    "AssetValidationReport",
    "AttachmentRole",
    "IssueSeverity",
    "ObservationModality",
    "ObservationSetManifest",
    "ObservationStateRole",
    "ObservationStream",
    "SourceAttachment",
    "SourceInspection",
    "SourceIssue",
    "SourceStreamSummary",
]
