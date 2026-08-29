"""Contracts for building digital twins from already captured observations.

The contracts describe immutable inputs, versioned processing recipes and
auditable outputs.  They deliberately contain no camera drivers, filesystem
logic or reconstruction framework objects.
"""

from __future__ import annotations

import math
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from physrealx.domain import (
    ArtifactRef,
    AssetSpec,
    DomainModel,
    MethodCapability,
    MethodSpec,
    ResourceType,
    VersionedRef,
)

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Python 3.8-3.10 equivalent of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return str.__str__(self)


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

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.OBSERVATION_SET,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class DigitalTwinStage(StrEnum):
    NORMALIZATION = "normalization"
    CAMERA_RECOVERY = "camera_recovery"
    DEPTH_ENHANCEMENT = "depth_enhancement"
    GEOMETRY_RECONSTRUCTION = "geometry_reconstruction"
    ARTICULATION_RECOVERY = "articulation_recovery"
    COLLISION_BUILD = "collision_build"
    PHYSICS_CALIBRATION = "physics_calibration"
    ASSET_PACKAGING = "asset_packaging"
    QUALITY_VALIDATION = "quality_validation"


_STAGE_ORDER = {stage: index for index, stage in enumerate(DigitalTwinStage)}

_STAGE_CAPABILITY = {
    DigitalTwinStage.NORMALIZATION: MethodCapability.ASSET_NORMALIZE,
    DigitalTwinStage.CAMERA_RECOVERY: MethodCapability.CAMERA_RECOVER,
    DigitalTwinStage.DEPTH_ENHANCEMENT: MethodCapability.DEPTH_ENHANCE,
    DigitalTwinStage.GEOMETRY_RECONSTRUCTION: MethodCapability.GEOMETRY_RECONSTRUCT,
    DigitalTwinStage.ARTICULATION_RECOVERY: MethodCapability.ARTICULATION_RECOVER,
    DigitalTwinStage.COLLISION_BUILD: MethodCapability.COLLISION_BUILD,
    DigitalTwinStage.PHYSICS_CALIBRATION: MethodCapability.PHYSICS_CALIBRATE,
    DigitalTwinStage.ASSET_PACKAGING: MethodCapability.ASSET_PACKAGE,
    DigitalTwinStage.QUALITY_VALIDATION: MethodCapability.ASSET_VALIDATE,
}


class DigitalTwinStep(DomainModel):
    stage: DigitalTwinStage
    method: MethodSpec

    @model_validator(mode="after")
    def validate_method_capability(self) -> DigitalTwinStep:
        expected = _STAGE_CAPABILITY[self.stage]
        if self.method.capability is not expected:
            raise ValueError(
                f"stage {self.stage.value!r} requires method capability {expected.value!r}"
            )
        return self


class DigitalTwinRecipe(DomainModel):
    schema_version: Literal[1] = 1
    name: str
    version: str
    pipeline: str
    pipeline_version: str
    steps: tuple[DigitalTwinStep, ...]

    @field_validator("name", "pipeline")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("version", "pipeline_version")
    @classmethod
    def validate_versions(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_steps(self) -> DigitalTwinRecipe:
        stages = [item.stage for item in self.steps]
        if len(stages) != len(set(stages)):
            raise ValueError("digital-twin recipe stages must be unique")
        if stages != sorted(stages, key=_STAGE_ORDER.__getitem__):
            raise ValueError("digital-twin recipe stages must follow execution order")
        required = {
            DigitalTwinStage.NORMALIZATION,
            DigitalTwinStage.CAMERA_RECOVERY,
            DigitalTwinStage.GEOMETRY_RECONSTRUCTION,
            DigitalTwinStage.ARTICULATION_RECOVERY,
            DigitalTwinStage.COLLISION_BUILD,
            DigitalTwinStage.PHYSICS_CALIBRATION,
            DigitalTwinStage.ASSET_PACKAGING,
            DigitalTwinStage.QUALITY_VALIDATION,
        }
        missing = sorted(item.value for item in required - set(stages))
        if missing:
            raise ValueError(f"digital-twin recipe is missing required stages: {missing}")
        return self


class DigitalTwinBuildSpec(DomainModel):
    schema_version: Literal[1] = 1
    resource_id: str
    version: str
    source: VersionedRef
    recipe: DigitalTwinRecipe

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_source(self) -> DigitalTwinBuildSpec:
        if (
            self.source.resource_type is not ResourceType.OBSERVATION_SET
            or not self.source.resolved
        ):
            raise ValueError("digital-twin build requires a resolved observation-set reference")
        return self


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


class StageExecutionRecord(DomainModel):
    stage: DigitalTwinStage
    method: MethodSpec
    outputs: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_record(self) -> StageExecutionRecord:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("stage timestamps must include a UTC offset")
        if self.completed_at < self.started_at:
            raise ValueError("stage completion cannot precede its start")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("stage metrics must be finite")
        if len({item.digest for item in self.outputs}) != len(self.outputs):
            raise ValueError("stage outputs must be unique")
        return self


class DigitalTwinBuildManifest(DomainModel):
    schema_version: Literal[1] = 1
    resource_id: str
    version: str
    source: VersionedRef
    recipe: DigitalTwinRecipe
    asset: VersionedRef
    stages: tuple[StageExecutionRecord, ...]
    validation: AssetValidationReport
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> DigitalTwinBuildManifest:
        if (
            self.source.resource_type is not ResourceType.OBSERVATION_SET
            or not self.source.resolved
        ):
            raise ValueError("build manifest source must be a resolved observation set")
        if self.asset.resource_type is not ResourceType.ASSET or not self.asset.resolved:
            raise ValueError("build manifest asset must be a resolved asset")
        if not self.validation.passed:
            raise ValueError("a failed candidate cannot be published as a build manifest")
        expected = tuple(item.stage for item in self.recipe.steps)
        actual = tuple(item.stage for item in self.stages)
        if actual != expected:
            raise ValueError("stage records must exactly match the recipe")
        for step, record in zip(self.recipe.steps, self.stages):
            if record.method != step.method:
                raise ValueError("stage record method must match the recipe")
        return self

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.ASSET_BUILD,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class DigitalTwinBuildResult(DomainModel):
    asset: VersionedRef
    build: VersionedRef
    validation: AssetValidationReport

    @model_validator(mode="after")
    def validate_result(self) -> DigitalTwinBuildResult:
        if self.asset.resource_type is not ResourceType.ASSET or not self.asset.resolved:
            raise ValueError("build result asset must be resolved")
        if self.build.resource_type is not ResourceType.ASSET_BUILD or not self.build.resolved:
            raise ValueError("build result manifest must be resolved")
        if not self.validation.passed:
            raise ValueError("build result requires a passing validation report")
        return self


def ensure_pipeline_output_identity(spec: DigitalTwinBuildSpec, asset: AssetSpec) -> None:
    """Reject plugins that return an asset outside the requested identity."""

    if asset.resource_id != spec.resource_id or asset.version != spec.version:
        raise ValueError("digital-twin pipeline returned an asset outside the requested identity")


__all__ = [
    "AssetValidationIssue",
    "AssetValidationProfile",
    "AssetValidationReport",
    "AttachmentRole",
    "DigitalTwinBuildManifest",
    "DigitalTwinBuildResult",
    "DigitalTwinBuildSpec",
    "DigitalTwinRecipe",
    "DigitalTwinStage",
    "DigitalTwinStep",
    "IssueSeverity",
    "ObservationModality",
    "ObservationSetManifest",
    "ObservationStateRole",
    "ObservationStream",
    "SourceAttachment",
    "SourceInspection",
    "SourceIssue",
    "SourceStreamSummary",
    "StageExecutionRecord",
    "ensure_pipeline_output_identity",
]
