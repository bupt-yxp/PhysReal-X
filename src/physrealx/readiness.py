"""Deterministic readiness gates for the complete PhysReal-X lifecycle.

Readiness is deliberately narrower than successful execution.  A stage is
``READY`` only when its exact method implementation is executable *now* and
every required input is backed by concrete evidence.  Planned upstream output
is never treated as evidence.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from physrealx.digital_twin import ObservationSetManifest
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    MethodCapability,
    MethodSpec,
    ResourceType,
    VersionedRef,
)
from physrealx.errors import PhysRealXError, ResourceNotFoundError
from physrealx.robot_agent import RobotInterfaceSpec

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class _StrEnum(str, Enum):
    """Python 3.8-compatible string enum local to this domain."""

    def __str__(self) -> str:
        return str.__str__(self)


class LifecycleStage(_StrEnum):
    INGESTION = "ingestion"
    CAMERA = "camera"
    METRIC_ALIGNMENT = "metric_alignment"
    DEPTH = "depth"
    GEOMETRY = "geometry"
    ARTICULATION = "articulation"
    COLLISION = "collision"
    PHYSICS = "physics"
    ASSEMBLY = "assembly"
    USD = "usd"
    PHYSICAL_VALIDATION = "physical_validation"
    DATASET = "dataset"
    POLICY = "policy"
    WORLD_MODEL = "world_model"
    BENCHMARK = "benchmark"
    ROS_DEPLOY = "ros_deploy"
    FAILURE = "failure"
    FEEDBACK = "feedback"


class LifecycleEvidenceKey(_StrEnum):
    CAPTURE = "capture"
    OBSERVATIONS = "observations"
    CAMERA_RECONSTRUCTION = "camera_reconstruction"
    METRIC_ALIGNMENT_EVIDENCE = "metric_alignment_evidence"
    CAMERA_SOLUTION = "camera_solution"
    ENHANCED_DEPTH = "enhanced_depth"
    VISUAL_GEOMETRY = "visual_geometry"
    ARTICULATION = "articulation"
    COLLISION_GEOMETRY = "collision_geometry"
    PHYSICS_EVIDENCE = "physics_evidence"
    BODY_PHYSICS = "body_physics"
    IDENTIFIED_JOINT_PHYSICS = "identified_joint_physics"
    ASSET_ASSEMBLY = "asset_assembly"
    USD_ASSET = "usd_asset"
    VALIDATION_EVIDENCE = "validation_evidence"
    VALIDATED_ASSET = "validated_asset"
    EPISODES = "episodes"
    DATASET = "dataset"
    POLICY_MODEL = "policy_model"
    WORLD_MODEL = "world_model"
    BENCHMARK_SUITE = "benchmark_suite"
    BENCHMARK_REPORT = "benchmark_report"
    ROBOT_TARGET = "robot_target"
    DEPLOYMENT = "deployment"
    TELEMETRY = "telemetry"
    FAILURE_REPORT = "failure_report"
    FEEDBACK_DATASET = "feedback_dataset"


class ReadinessStatus(_StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class ReadinessBlockerCode(_StrEnum):
    METHOD_NOT_INSTALLED = "method_not_installed"
    METHOD_IDENTITY_MISMATCH = "method_identity_mismatch"
    METHOD_NOT_EXECUTABLE = "method_not_executable"
    METHOD_NOT_READY = "method_not_ready"
    CHECKPOINT_UNAVAILABLE = "checkpoint_unavailable"
    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    EVIDENCE_GATE_FAILED = "evidence_gate_failed"


class ReadinessModel(DomainModel):
    """Strict, immutable base for persisted preflight results."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class LifecycleStageBinding(ReadinessModel):
    """One canonical lifecycle stage bound to one immutable method selection."""

    stage: LifecycleStage
    capability: MethodCapability
    method: MethodSpec
    requires: tuple[LifecycleEvidenceKey, ...]
    produces: tuple[LifecycleEvidenceKey, ...]

    @model_validator(mode="after")
    def validate_canonical_contract(self) -> LifecycleStageBinding:
        contract = _STAGE_CONTRACTS[self.stage]
        if self.capability is not contract.capability:
            raise ValueError(
                f"stage {self.stage.value!r} requires capability {contract.capability.value!r}"
            )
        if self.method.capability is not self.capability:
            raise ValueError(
                f"bound method capability must be {self.capability.value!r}, got "
                f"{self.method.capability.value!r}"
            )
        if self.requires != contract.requires:
            raise ValueError(f"stage {self.stage.value!r} has non-canonical evidence requirements")
        if self.produces != contract.produces:
            raise ValueError(f"stage {self.stage.value!r} has non-canonical outputs")
        return self


class LifecyclePlan(ReadinessModel):
    """A complete, ordered and immutable execution intent."""

    stages: tuple[LifecycleStageBinding, ...]

    @model_validator(mode="after")
    def validate_complete_order(self) -> LifecyclePlan:
        actual = tuple(item.stage for item in self.stages)
        expected = tuple(LifecycleStage)
        if actual != expected:
            raise ValueError(
                "lifecycle plan must contain every stage exactly once in canonical order"
            )
        return self


class LifecycleEvidence(ReadinessModel):
    """A concrete path, immutable artifact or registered immutable resource."""

    key: LifecycleEvidenceKey
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    artifact: ArtifactRef | None = None
    resource: VersionedRef | None = None

    @field_validator("path")
    @classmethod
    def validate_path_text(cls, value: str | None) -> str | None:
        if value is not None and "\x00" in value:
            raise ValueError("evidence path must not contain a NUL byte")
        return value

    @model_validator(mode="after")
    def validate_single_location(self) -> LifecycleEvidence:
        locations = sum(item is not None for item in (self.path, self.artifact, self.resource))
        if locations != 1:
            raise ValueError("lifecycle evidence requires exactly one location")
        if self.resource is not None and not self.resource.resolved:
            raise ValueError("lifecycle resource evidence must include an immutable digest")
        return self


class ReadinessBlocker(ReadinessModel):
    code: ReadinessBlockerCode
    message: str = Field(min_length=1, max_length=1024)


class LifecycleStageReadiness(ReadinessModel):
    binding: LifecycleStageBinding
    status: ReadinessStatus
    satisfied_evidence: tuple[LifecycleEvidenceKey, ...]
    blockers: tuple[ReadinessBlocker, ...]

    @model_validator(mode="after")
    def validate_status(self) -> LifecycleStageReadiness:
        if (self.status is ReadinessStatus.READY) != (not self.blockers):
            raise ValueError("READY requires no blockers and BLOCKED requires at least one blocker")
        return self


class LifecycleReadinessReport(ReadinessModel):
    """Deterministic snapshot; intentionally contains no wall-clock timestamp."""

    status: ReadinessStatus
    stages: tuple[LifecycleStageReadiness, ...]
    supplied_evidence: tuple[LifecycleEvidenceKey, ...]

    @model_validator(mode="after")
    def validate_aggregate(self) -> LifecycleReadinessReport:
        expected_status = (
            ReadinessStatus.READY
            if self.stages and all(item.status is ReadinessStatus.READY for item in self.stages)
            else ReadinessStatus.BLOCKED
        )
        if self.status is not expected_status:
            raise ValueError("lifecycle readiness status does not match its stage results")
        if tuple(item.binding.stage for item in self.stages) != tuple(LifecycleStage):
            raise ValueError("readiness report must cover every lifecycle stage in canonical order")
        return self


class _StageContract:
    def __init__(
        self,
        capability: MethodCapability,
        requires: tuple[LifecycleEvidenceKey, ...],
        produces: tuple[LifecycleEvidenceKey, ...],
    ) -> None:
        self.capability = capability
        self.requires = requires
        self.produces = produces


@dataclass(frozen=True)
class _EvidenceGate:
    schemas: frozenset[str]
    statuses: frozenset[str] = frozenset()
    required_values: tuple[tuple[str, object], ...] = ()


_VERSIONED_SCHEMA = re.compile(r"^physrealx\.[a-z][a-z0-9-]*/[1-9][0-9]*$")
_MAX_EVIDENCE_JSON_BYTES = 64 * 1024 * 1024


_EVIDENCE_GATES: Mapping[LifecycleEvidenceKey, _EvidenceGate] = {
    LifecycleEvidenceKey.CAMERA_RECONSTRUCTION: _EvidenceGate(
        frozenset({"physrealx.camera-rig-calibration/1"}),
        required_values=(("scale_status", "unresolved"),),
    ),
    LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE: _EvidenceGate(
        frozenset({"physrealx.metric-alignment-evidence/1"}),
        required_values=(("unit", "m"),),
    ),
    LifecycleEvidenceKey.CAMERA_SOLUTION: _EvidenceGate(
        frozenset({"physrealx.camera-rig-calibration/1"}),
        required_values=(("scale_status", "resolved"),),
    ),
    LifecycleEvidenceKey.ENHANCED_DEPTH: _EvidenceGate(
        frozenset(
            {
                "physrealx.depth-enhancement-report/1",
                "physrealx.enhanced-depth-bundle/1",
            }
        ),
        frozenset({"PASSED"}),
    ),
    LifecycleEvidenceKey.VISUAL_GEOMETRY: _EvidenceGate(
        frozenset({"physrealx.geometry-reconstruction-report/1"}),
        frozenset({"PASSED"}),
        required_values=(("unit", "m"),),
    ),
    LifecycleEvidenceKey.ARTICULATION: _EvidenceGate(
        frozenset({"physrealx.articulation-model/1"}),
        frozenset({"IDENTIFIED", "VALIDATED"}),
    ),
    LifecycleEvidenceKey.COLLISION_GEOMETRY: _EvidenceGate(
        frozenset({"physrealx.collision-build-report/1"}),
        frozenset({"PASSED"}),
        required_values=(("unit", "m"),),
    ),
    LifecycleEvidenceKey.BODY_PHYSICS: _EvidenceGate(
        frozenset({"physrealx.identified-body-physics/1"}),
        frozenset({"IDENTIFIED", "VALIDATED"}),
        required_values=(("evidence_state", "identified"),),
    ),
    LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS: _EvidenceGate(
        frozenset({"physrealx.identified-joint-physics/1"}),
        frozenset({"IDENTIFIED", "VALIDATED"}),
        required_values=(("evidence_state", "identified"),),
    ),
    LifecycleEvidenceKey.ASSET_ASSEMBLY: _EvidenceGate(
        frozenset({"physrealx.asset-assembly-bundle/1"}),
    ),
    LifecycleEvidenceKey.USD_ASSET: _EvidenceGate(
        frozenset({"physrealx.asset-package-manifest/1"}),
        required_values=(("roundtrip_status", "PASSED"), ("self_contained", True)),
    ),
    LifecycleEvidenceKey.VALIDATED_ASSET: _EvidenceGate(
        frozenset(
            {
                "physrealx.asset-validation-report/1",
                "physrealx.usd-roundtrip-validation-report/1",
            }
        ),
        frozenset({"PASSED"}),
    ),
    LifecycleEvidenceKey.EPISODES: _EvidenceGate(
        frozenset({"physrealx.verified-episode-collection/1"}),
        frozenset({"FINALIZED"}),
    ),
    LifecycleEvidenceKey.DATASET: _EvidenceGate(
        frozenset({"physrealx.dataset-snapshot/1"}),
        frozenset({"VALIDATED"}),
    ),
    LifecycleEvidenceKey.POLICY_MODEL: _EvidenceGate(
        frozenset({"physrealx.policy-model-bundle/1"}),
        frozenset({"TRAINED", "VALIDATED"}),
    ),
    LifecycleEvidenceKey.WORLD_MODEL: _EvidenceGate(
        frozenset({"physrealx.world-model-bundle/1"}),
        frozenset({"TRAINED", "VALIDATED"}),
    ),
    LifecycleEvidenceKey.BENCHMARK_SUITE: _EvidenceGate(
        frozenset({"physrealx.benchmark-suite/1"}),
        frozenset({"FROZEN", "VALIDATED"}),
    ),
    LifecycleEvidenceKey.BENCHMARK_REPORT: _EvidenceGate(
        frozenset({"physrealx.benchmark-report/1"}),
        frozenset({"PASSED"}),
    ),
    LifecycleEvidenceKey.ROBOT_TARGET: _EvidenceGate(
        frozenset({"physrealx.robot-interface-spec/1"}),
        frozenset({"DECLARED"}),
    ),
    LifecycleEvidenceKey.DEPLOYMENT: _EvidenceGate(
        frozenset({"physrealx.deployment-report/1"}),
        frozenset({"ACTIVE", "COMPLETED"}),
    ),
    LifecycleEvidenceKey.TELEMETRY: _EvidenceGate(
        frozenset({"physrealx.telemetry-manifest/1"}),
        frozenset({"COMPLETE"}),
    ),
    LifecycleEvidenceKey.FAILURE_REPORT: _EvidenceGate(
        frozenset({"physrealx.failure-report/1"}),
        frozenset({"ANALYZED", "CONFIRMED"}),
    ),
    LifecycleEvidenceKey.FEEDBACK_DATASET: _EvidenceGate(
        frozenset({"physrealx.feedback-dataset/1"}),
        frozenset({"VALIDATED"}),
    ),
}


_E = LifecycleEvidenceKey
_STAGE_CONTRACTS: Mapping[LifecycleStage, _StageContract] = {
    LifecycleStage.INGESTION: _StageContract(
        MethodCapability.OBSERVATION_IMPORT, (_E.CAPTURE,), (_E.OBSERVATIONS,)
    ),
    LifecycleStage.CAMERA: _StageContract(
        MethodCapability.CAMERA_RECOVER,
        (_E.OBSERVATIONS,),
        (_E.CAMERA_RECONSTRUCTION,),
    ),
    LifecycleStage.METRIC_ALIGNMENT: _StageContract(
        MethodCapability.CAMERA_ALIGN_METRIC,
        (_E.CAMERA_RECONSTRUCTION, _E.METRIC_ALIGNMENT_EVIDENCE),
        (_E.CAMERA_SOLUTION,),
    ),
    LifecycleStage.DEPTH: _StageContract(
        MethodCapability.DEPTH_ENHANCE,
        (_E.OBSERVATIONS, _E.CAMERA_SOLUTION),
        (_E.ENHANCED_DEPTH,),
    ),
    LifecycleStage.GEOMETRY: _StageContract(
        MethodCapability.GEOMETRY_RECONSTRUCT,
        (_E.CAMERA_SOLUTION, _E.ENHANCED_DEPTH),
        (_E.VISUAL_GEOMETRY,),
    ),
    LifecycleStage.ARTICULATION: _StageContract(
        MethodCapability.ARTICULATION_RECOVER,
        (_E.CAMERA_SOLUTION, _E.ENHANCED_DEPTH, _E.VISUAL_GEOMETRY),
        (_E.ARTICULATION,),
    ),
    LifecycleStage.COLLISION: _StageContract(
        MethodCapability.COLLISION_BUILD,
        (_E.VISUAL_GEOMETRY, _E.ARTICULATION),
        (_E.COLLISION_GEOMETRY,),
    ),
    LifecycleStage.PHYSICS: _StageContract(
        MethodCapability.PHYSICS_CALIBRATE,
        (_E.ARTICULATION, _E.COLLISION_GEOMETRY, _E.PHYSICS_EVIDENCE),
        (_E.IDENTIFIED_JOINT_PHYSICS,),
    ),
    LifecycleStage.ASSEMBLY: _StageContract(
        MethodCapability.ASSET_ASSEMBLE,
        (
            _E.CAMERA_SOLUTION,
            _E.VISUAL_GEOMETRY,
            _E.COLLISION_GEOMETRY,
            _E.ARTICULATION,
            _E.BODY_PHYSICS,
            _E.IDENTIFIED_JOINT_PHYSICS,
        ),
        (_E.ASSET_ASSEMBLY,),
    ),
    LifecycleStage.USD: _StageContract(
        MethodCapability.ASSET_PACKAGE,
        (_E.ASSET_ASSEMBLY,),
        (_E.USD_ASSET,),
    ),
    LifecycleStage.PHYSICAL_VALIDATION: _StageContract(
        MethodCapability.ASSET_VALIDATE,
        (_E.USD_ASSET, _E.VALIDATION_EVIDENCE),
        (_E.VALIDATED_ASSET,),
    ),
    LifecycleStage.DATASET: _StageContract(
        MethodCapability.DATASET_INGEST,
        (_E.VALIDATED_ASSET, _E.EPISODES),
        (_E.DATASET,),
    ),
    LifecycleStage.POLICY: _StageContract(
        MethodCapability.POLICY_TRAIN, (_E.DATASET,), (_E.POLICY_MODEL,)
    ),
    LifecycleStage.WORLD_MODEL: _StageContract(
        MethodCapability.WORLD_MODEL_TRAIN, (_E.DATASET,), (_E.WORLD_MODEL,)
    ),
    LifecycleStage.BENCHMARK: _StageContract(
        MethodCapability.EVALUATION_RUN,
        (_E.POLICY_MODEL, _E.WORLD_MODEL, _E.BENCHMARK_SUITE),
        (_E.BENCHMARK_REPORT,),
    ),
    LifecycleStage.ROS_DEPLOY: _StageContract(
        MethodCapability.DEPLOYMENT_STAGE,
        (_E.POLICY_MODEL, _E.BENCHMARK_REPORT, _E.ROBOT_TARGET),
        (_E.DEPLOYMENT,),
    ),
    LifecycleStage.FAILURE: _StageContract(
        MethodCapability.FAILURE_ANALYZE,
        (_E.DEPLOYMENT, _E.TELEMETRY),
        (_E.FAILURE_REPORT,),
    ),
    LifecycleStage.FEEDBACK: _StageContract(
        MethodCapability.FEEDBACK_BUILD_DATASET,
        (_E.DATASET, _E.FAILURE_REPORT),
        (_E.FEEDBACK_DATASET,),
    ),
}

EvidenceInput = Union[str, Path, ArtifactRef, VersionedRef]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = member
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    if len(payload) > _MAX_EVIDENCE_JSON_BYTES:
        raise ValueError(f"{label} exceeds the evidence JSON size limit")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is not unambiguous finite UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    pending: list[object] = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return document


class ReadinessService:
    """Compile and evaluate lifecycle plans without running any stage."""

    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def plan(
        self,
        methods: Mapping[LifecycleStage | str, MethodSpec],
    ) -> LifecyclePlan:
        normalized: dict[LifecycleStage, MethodSpec] = {}
        for raw_stage, method in methods.items():
            stage = (
                raw_stage if isinstance(raw_stage, LifecycleStage) else LifecycleStage(raw_stage)
            )
            if stage in normalized:
                raise ValueError(f"duplicate lifecycle method for stage {stage.value!r}")
            normalized[stage] = method
        missing = tuple(stage.value for stage in LifecycleStage if stage not in normalized)
        extra = tuple(stage.value for stage in normalized if stage not in _STAGE_CONTRACTS)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={','.join(missing)}")
            if extra:
                details.append(f"extra={','.join(extra)}")
            raise ValueError("lifecycle method map must be complete: " + "; ".join(details))
        return LifecyclePlan(
            stages=tuple(
                LifecycleStageBinding(
                    stage=stage,
                    capability=_STAGE_CONTRACTS[stage].capability,
                    method=normalized[stage],
                    requires=_STAGE_CONTRACTS[stage].requires,
                    produces=_STAGE_CONTRACTS[stage].produces,
                )
                for stage in LifecycleStage
            )
        )

    def preflight(
        self,
        plan: LifecyclePlan,
        *,
        roots: Mapping[LifecycleEvidenceKey | str, EvidenceInput],
    ) -> LifecycleReadinessReport:
        evidence = self._normalize_evidence(roots)
        availability = {key: self._evidence_blocker(item) for key, item in evidence.items()}
        availability = self._cross_evidence_availability(evidence, availability)
        results = tuple(
            self._stage_readiness(binding, evidence, availability) for binding in plan.stages
        )
        status = (
            ReadinessStatus.READY
            if all(item.status is ReadinessStatus.READY for item in results)
            else ReadinessStatus.BLOCKED
        )
        return LifecycleReadinessReport(
            status=status,
            stages=results,
            supplied_evidence=tuple(sorted(evidence, key=lambda item: item.value)),
        )

    @staticmethod
    def _evidence_digest(evidence: LifecycleEvidence) -> str | None:
        if evidence.artifact is not None:
            return evidence.artifact.digest
        if evidence.resource is not None:
            return evidence.resource.digest
        return None

    def _cross_evidence_availability(
        self,
        evidence: Mapping[LifecycleEvidenceKey, LifecycleEvidence],
        availability: Mapping[LifecycleEvidenceKey, ReadinessBlocker | None],
    ) -> dict[LifecycleEvidenceKey, ReadinessBlocker | None]:
        """Bind successful stage evidence to the exact supplied upstream artifacts."""

        result = dict(availability)

        def usable(*keys: LifecycleEvidenceKey) -> bool:
            return all(key in evidence and result.get(key) is None for key in keys)

        if usable(
            LifecycleEvidenceKey.VISUAL_GEOMETRY,
            LifecycleEvidenceKey.CAMERA_SOLUTION,
            LifecycleEvidenceKey.ENHANCED_DEPTH,
        ):
            geometry_evidence = evidence[LifecycleEvidenceKey.VISUAL_GEOMETRY]
            geometry = self._semantic_document(geometry_evidence)
            camera_digest = self._evidence_digest(evidence[LifecycleEvidenceKey.CAMERA_SOLUTION])
            depth_digest = self._evidence_digest(evidence[LifecycleEvidenceKey.ENHANCED_DEPTH])
            if (
                geometry.get("source_camera_digest") != camera_digest
                or geometry.get("source_depth_digest") != depth_digest
            ):
                result[LifecycleEvidenceKey.VISUAL_GEOMETRY] = self._gate_failure(
                    geometry_evidence,
                    "geometry source digests do not match camera_solution and enhanced_depth",
                )

        if usable(
            LifecycleEvidenceKey.COLLISION_GEOMETRY,
            LifecycleEvidenceKey.VISUAL_GEOMETRY,
        ):
            collision_evidence = evidence[LifecycleEvidenceKey.COLLISION_GEOMETRY]
            collision = self._semantic_document(collision_evidence)
            geometry = self._semantic_document(evidence[LifecycleEvidenceKey.VISUAL_GEOMETRY])
            mesh_digests = {
                raw["artifact"]["digest"]
                for raw in geometry["meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("digest"), str)
            }
            source = collision.get("source_geometry")
            source_digest = source.get("digest") if isinstance(source, dict) else None
            if source_digest not in mesh_digests:
                result[LifecycleEvidenceKey.COLLISION_GEOMETRY] = self._gate_failure(
                    collision_evidence,
                    "collision source mesh is not part of the supplied visual geometry",
                )

        assembly_dependencies = (
            LifecycleEvidenceKey.ASSET_ASSEMBLY,
            LifecycleEvidenceKey.CAMERA_SOLUTION,
            LifecycleEvidenceKey.VISUAL_GEOMETRY,
            LifecycleEvidenceKey.COLLISION_GEOMETRY,
            LifecycleEvidenceKey.ARTICULATION,
            LifecycleEvidenceKey.BODY_PHYSICS,
            LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS,
        )
        if usable(*assembly_dependencies):
            assembly_evidence = evidence[LifecycleEvidenceKey.ASSET_ASSEMBLY]
            assembly = self._semantic_document(assembly_evidence)
            geometry = self._semantic_document(evidence[LifecycleEvidenceKey.VISUAL_GEOMETRY])
            collision = self._semantic_document(evidence[LifecycleEvidenceKey.COLLISION_GEOMETRY])
            supplied_visual = {
                raw["artifact"]["digest"]
                for raw in geometry["meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("digest"), str)
            }
            assembled_visual = {
                raw["artifact"]["digest"]
                for raw in assembly["visual_meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("digest"), str)
            }
            assembled_collision = {
                raw["artifact"]["digest"]
                for raw in assembly["collision_meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("digest"), str)
            }
            collision_output = collision.get("collision_geometry")
            collision_digest = (
                collision_output.get("digest") if isinstance(collision_output, dict) else None
            )
            articulation_digest = self._evidence_digest(evidence[LifecycleEvidenceKey.ARTICULATION])
            assembly_articulation = assembly.get("articulation_model")
            evidence_closure = {
                raw["digest"]
                for raw in assembly["evidence"]
                if isinstance(raw, dict) and isinstance(raw.get("digest"), str)
            }
            required_closure = {
                self._evidence_digest(evidence[key])
                for key in (
                    LifecycleEvidenceKey.CAMERA_SOLUTION,
                    LifecycleEvidenceKey.BODY_PHYSICS,
                    LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS,
                )
            }
            if (
                supplied_visual != assembled_visual
                or collision_digest not in assembled_collision
                or not isinstance(assembly_articulation, dict)
                or assembly_articulation.get("digest") != articulation_digest
                or not required_closure.issubset(evidence_closure)
            ):
                result[LifecycleEvidenceKey.ASSET_ASSEMBLY] = self._gate_failure(
                    assembly_evidence,
                    "asset assembly is not closed over the supplied metric geometry, collision, "
                    "articulation and physics evidence",
                )
        return result

    def _normalize_evidence(
        self,
        roots: Mapping[LifecycleEvidenceKey | str, EvidenceInput],
    ) -> dict[LifecycleEvidenceKey, LifecycleEvidence]:
        normalized: dict[LifecycleEvidenceKey, LifecycleEvidence] = {}
        for raw_key, value in roots.items():
            key = (
                raw_key
                if isinstance(raw_key, LifecycleEvidenceKey)
                else LifecycleEvidenceKey(raw_key)
            )
            if key in normalized:
                raise ValueError(f"duplicate lifecycle evidence for {key.value!r}")
            if isinstance(value, ArtifactRef):
                item = LifecycleEvidence(key=key, artifact=value)
            elif isinstance(value, VersionedRef):
                item = LifecycleEvidence(key=key, resource=value)
            elif isinstance(value, (str, Path)):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = self._platform.workspace / path
                item = LifecycleEvidence(key=key, path=str(path.resolve(strict=False)))
            else:
                raise TypeError("lifecycle evidence must be a path, ArtifactRef or VersionedRef")
            normalized[key] = item
        return normalized

    def _evidence_blocker(self, evidence: LifecycleEvidence) -> ReadinessBlocker | None:
        if evidence.path is not None:
            path = Path(evidence.path)
            try:
                if path.is_file():
                    available = path.stat().st_size > 0
                    detail = "path is an empty file"
                elif path.is_dir():
                    available = next(path.iterdir(), None) is not None
                    detail = "path is an empty directory"
                else:
                    available = False
                    detail = "path does not exist as a regular file or directory"
            except OSError as error:
                available = False
                detail = f"path cannot be inspected: {error}"
            if not available:
                return ReadinessBlocker(
                    code=ReadinessBlockerCode.EVIDENCE_UNAVAILABLE,
                    message=f"evidence {evidence.key.value!r} is unavailable: {detail}",
                )
            return self._semantic_evidence_blocker(evidence)
        if evidence.artifact is not None:
            if not self._platform.artifacts.verify(evidence.artifact):
                return ReadinessBlocker(
                    code=ReadinessBlockerCode.EVIDENCE_UNAVAILABLE,
                    message=(
                        f"artifact evidence {evidence.key.value!r} failed integrity verification"
                    ),
                )
            return self._semantic_evidence_blocker(evidence)
        assert evidence.resource is not None
        try:
            registered = self._platform.repository.get_resource(
                evidence.resource.resource_type.value,
                evidence.resource.resource_id,
                evidence.resource.version,
            )
        except PhysRealXError as error:
            return ReadinessBlocker(
                code=ReadinessBlockerCode.EVIDENCE_UNAVAILABLE,
                message=f"resource evidence {evidence.key.value!r} is unavailable: {error}",
            )
        if registered.digest != evidence.resource.digest:
            return ReadinessBlocker(
                code=ReadinessBlockerCode.EVIDENCE_UNAVAILABLE,
                message=f"resource evidence {evidence.key.value!r} has a registry digest mismatch",
            )
        return self._semantic_evidence_blocker(evidence)

    def _semantic_evidence_blocker(
        self,
        evidence: LifecycleEvidence,
    ) -> ReadinessBlocker | None:
        if evidence.key is LifecycleEvidenceKey.OBSERVATIONS:
            return self._observation_evidence_blocker(evidence)
        gate = _EVIDENCE_GATES.get(evidence.key)
        if gate is None:
            return None
        if evidence.path is not None:
            return self._gate_failure(
                evidence,
                "critical lifecycle evidence must be an integrity-verified artifact or resource",
            )
        try:
            document = self._semantic_document(evidence)
        except (OSError, PhysRealXError, ValueError, zipfile.BadZipFile) as error:
            return self._gate_failure(
                evidence,
                f"cannot verify its semantic contract: {error}",
            )
        schema = document.get("schema")
        if not isinstance(schema, str) or _VERSIONED_SCHEMA.fullmatch(schema) is None:
            return self._gate_failure(
                evidence,
                "schema must be a recognizable versioned physrealx.*/* identifier",
            )
        if schema not in gate.schemas:
            return self._gate_failure(
                evidence,
                f"schema {schema!r} is not allowed; expected {sorted(gate.schemas)!r}",
            )
        if gate.statuses:
            status = document.get("status")
            if status not in gate.statuses:
                return self._gate_failure(
                    evidence,
                    f"status {status!r} is not successful; expected {sorted(gate.statuses)!r}",
                )
        for field, expected in gate.required_values:
            actual = document.get(field)
            if actual != expected or type(actual) is not type(expected):
                return self._gate_failure(
                    evidence,
                    f"field {field!r} must equal {expected!r}, got {actual!r}",
                )
        if evidence.key is LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE:
            detail = self._metric_alignment_evidence_error(document)
            if detail is not None:
                return self._gate_failure(evidence, detail)
        try:
            if evidence.key is LifecycleEvidenceKey.VISUAL_GEOMETRY:
                self._validate_geometry_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.COLLISION_GEOMETRY:
                self._validate_collision_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.BODY_PHYSICS:
                self._validate_body_physics_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS:
                self._validate_joint_physics_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.ASSET_ASSEMBLY:
                self._validate_asset_assembly(document)
            elif evidence.key is LifecycleEvidenceKey.ROBOT_TARGET:
                self._validate_robot_target(document)
        except ValueError as error:
            return self._gate_failure(evidence, str(error))
        return None

    @staticmethod
    def _artifact_reference(value: object, label: str) -> ArtifactRef:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an ArtifactRef object")
        try:
            return ArtifactRef.model_validate(value)
        except ValueError as error:
            raise ValueError(f"{label} must be a valid ArtifactRef") from error

    @staticmethod
    def _digest(value: object, label: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a SHA-256 digest")
        try:
            return ArtifactRef(
                digest=value,
                size_bytes=0,
                media_type="application/octet-stream",
            ).digest
        except ValueError as error:
            raise ValueError(f"{label} must be a SHA-256 digest") from error

    def _require_available_artifacts(
        self,
        references: list[ArtifactRef],
        label: str,
    ) -> None:
        missing = sorted(
            reference.digest
            for reference in references
            if not self._platform.artifacts.verify(reference)
        )
        if missing:
            raise ValueError(f"{label} has unavailable artifacts: {', '.join(missing)}")

    def _validate_geometry_evidence(self, document: Mapping[str, Any]) -> None:
        self._digest(document.get("source_camera_digest"), "source_camera_digest")
        self._digest(document.get("source_depth_digest"), "source_depth_digest")
        method = document.get("method")
        if not isinstance(method, dict) or any(
            not isinstance(method.get(field), str) or not method[field].strip()
            for field in ("name", "version", "source_revision")
        ):
            raise ValueError("geometry report method identity must be complete")
        gates = document.get("gates")
        if (
            not isinstance(gates, dict)
            or not gates
            or any(value is not True for value in gates.values())
        ):
            raise ValueError("geometry report must pass every declared gate")
        metrics = document.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise ValueError("geometry report metrics must be non-empty and finite")
        raw_meshes = document.get("meshes")
        if not isinstance(raw_meshes, list) or not raw_meshes:
            raise ValueError("geometry report meshes must be a non-empty array")
        links: set[str] = set()
        references: list[ArtifactRef] = []
        for index, raw in enumerate(raw_meshes):
            if not isinstance(raw, dict):
                raise ValueError(f"meshes[{index}] must be an object")
            link = raw.get("link")
            if not isinstance(link, str) or not link.strip() or link in links:
                raise ValueError("geometry report mesh links must be non-empty and unique")
            links.add(link)
            reference = self._artifact_reference(raw.get("artifact"), f"meshes[{index}].artifact")
            if not reference.media_type.startswith("model/"):
                raise ValueError("geometry report meshes must reference model/* artifacts")
            references.append(reference)
        if len({reference.digest for reference in references}) != len(references):
            raise ValueError("geometry report mesh artifact digests must be unique")
        self._require_available_artifacts(references, "geometry report")

    def _validate_collision_evidence(self, document: Mapping[str, Any]) -> None:
        link = document.get("link")
        if not isinstance(link, str) or not link.strip():
            raise ValueError("collision report link must be non-empty")
        source = self._artifact_reference(
            document.get("source_geometry"),
            "collision report source_geometry",
        )
        collision = self._artifact_reference(
            document.get("collision_geometry"),
            "collision report collision_geometry",
        )
        if not source.media_type.startswith("model/"):
            raise ValueError("collision report source_geometry must be a model/* artifact")
        if collision.media_type != "model/gltf-binary":
            raise ValueError("collision report collision_geometry must be a GLB artifact")
        if source.digest == collision.digest:
            raise ValueError("collision geometry must be distinct from its visual source mesh")
        method = document.get("method")
        if not isinstance(method, dict) or any(
            not isinstance(method.get(field), str) or not method[field].strip()
            for field in ("name", "version", "source_revision")
        ):
            raise ValueError("collision report method identity must be complete")
        gates = document.get("gates")
        if (
            not isinstance(gates, dict)
            or not gates
            or any(value is not True for value in gates.values())
        ):
            raise ValueError("collision report must pass every declared gate")
        metrics = document.get("metrics")
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        ):
            raise ValueError("collision report metrics must be non-empty and finite")
        self._require_available_artifacts([source, collision], "collision report")

    def _identified_parameter(
        self,
        value: object,
        label: str,
        *,
        unit: str,
        minimum: float,
        maximum: float | None = None,
    ) -> tuple[float, str]:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an identified parameter object")
        raw_number = value.get("value")
        if (
            isinstance(raw_number, bool)
            or not isinstance(raw_number, (int, float))
            or not math.isfinite(float(raw_number))
        ):
            raise ValueError(f"{label}.value must be finite")
        number = float(raw_number)
        if number < minimum or (maximum is not None and number > maximum):
            raise ValueError(f"{label}.value is outside its physical range")
        if value.get("unit") != unit:
            raise ValueError(f"{label}.unit must be {unit!r}")
        evidence_digest = self._digest(value.get("evidence_digest"), f"{label}.evidence_digest")
        return number, evidence_digest

    @staticmethod
    def _finite_vector(value: object, label: str, length: int) -> tuple[float, ...]:
        if (
            not isinstance(value, list)
            or len(value) != length
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value
            )
        ):
            raise ValueError(f"{label} must contain {length} finite values")
        return tuple(float(item) for item in value)

    def _validate_body_physics_evidence(self, document: Mapping[str, Any]) -> None:
        raw_bodies = document.get("bodies")
        if not isinstance(raw_bodies, list) or not raw_bodies:
            raise ValueError("body physics must contain at least one identified body")
        links: set[str] = set()
        for index, body in enumerate(raw_bodies):
            if not isinstance(body, dict):
                raise ValueError(f"bodies[{index}] must be an object")
            link = body.get("link")
            if not isinstance(link, str) or not link.strip() or link in links:
                raise ValueError("body physics links must be non-empty and unique")
            links.add(link)
            label = f"bodies[{index}]"
            body_digest = self._digest(body.get("evidence_digest"), f"{label}.evidence_digest")
            _, mass_digest = self._identified_parameter(
                body.get("mass_kg"),
                f"{label}.mass_kg",
                unit="kg",
                minimum=1e-12,
            )
            self._finite_vector(body.get("center_of_mass_m"), f"{label}.center_of_mass_m", 3)
            inertia_rows = body.get("inertia_kg_m2")
            if not isinstance(inertia_rows, list) or len(inertia_rows) != 3:
                raise ValueError(f"{label}.inertia_kg_m2 must be a 3x3 matrix")
            inertia = tuple(
                self._finite_vector(row, f"{label}.inertia_kg_m2[{row_index}]", 3)
                for row_index, row in enumerate(inertia_rows)
            )
            if any(
                abs(inertia[row][column] - inertia[column][row]) > 1e-9
                for row in range(3)
                for column in range(3)
            ):
                raise ValueError(f"{label}.inertia_kg_m2 must be symmetric")
            first_minor = inertia[0][0]
            second_minor = inertia[0][0] * inertia[1][1] - inertia[0][1] * inertia[1][0]
            determinant = (
                inertia[0][0] * (inertia[1][1] * inertia[2][2] - inertia[1][2] * inertia[2][1])
                - inertia[0][1] * (inertia[1][0] * inertia[2][2] - inertia[1][2] * inertia[2][0])
                + inertia[0][2] * (inertia[1][0] * inertia[2][1] - inertia[1][1] * inertia[2][0])
            )
            if first_minor <= 0.0 or second_minor <= 0.0 or determinant <= 0.0:
                raise ValueError(f"{label}.inertia_kg_m2 must be positive definite")
            static_friction, static_digest = self._identified_parameter(
                body.get("static_friction"),
                f"{label}.static_friction",
                unit="1",
                minimum=0.0,
            )
            dynamic_friction, dynamic_digest = self._identified_parameter(
                body.get("dynamic_friction"),
                f"{label}.dynamic_friction",
                unit="1",
                minimum=0.0,
            )
            _, restitution_digest = self._identified_parameter(
                body.get("restitution"),
                f"{label}.restitution",
                unit="1",
                minimum=0.0,
                maximum=1.0,
            )
            if dynamic_friction > static_friction:
                raise ValueError(f"{label} dynamic friction cannot exceed static friction")
            if {
                mass_digest,
                static_digest,
                dynamic_digest,
                restitution_digest,
            } != {body_digest}:
                raise ValueError(f"{label} parameters must share its measured evidence digest")
            if not self._platform.artifacts.verify(body_digest):
                raise ValueError(f"{label} measured evidence digest is unavailable")

    def _validate_joint_physics_evidence(self, document: Mapping[str, Any]) -> None:
        joint = document.get("joint")
        parameters = document.get("parameters")
        if not isinstance(joint, dict):
            raise ValueError("joint physics must identify one joint")
        if not isinstance(joint.get("name"), str) or not joint["name"].strip():
            raise ValueError("joint physics joint.name must be non-empty")
        if joint.get("type") not in {"hinge", "slide"}:
            raise ValueError("joint physics joint.type must be hinge or slide")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("joint physics parameters must be non-empty")
        evidence_digest = self._digest(
            document.get("evidence_digest"),
            "joint physics evidence_digest",
        )
        if not self._platform.artifacts.verify(evidence_digest):
            raise ValueError("joint physics measured evidence digest is unavailable")
        expected_units = (
            {
                "effective_inertia": "kg",
                "viscous_damping": "N*s/m",
                "coulomb_friction": "N",
            }
            if joint["type"] == "slide"
            else {
                "effective_inertia": "kg*m^2",
                "viscous_damping": "N*m*s/rad",
                "coulomb_friction": "N*m",
            }
        )
        if set(parameters) != set(expected_units):
            raise ValueError("joint physics must identify inertia, damping and Coulomb friction")
        for name, unit in expected_units.items():
            parameter = parameters[name]
            if not isinstance(parameter, dict):
                raise ValueError(f"joint physics parameter {name!r} must be an object")
            raw_value = parameter.get("value")
            raw_spread = parameter.get("multi_start_spread")
            confidence = parameter.get("confidence_interval")
            minimum = 1e-12 if name == "effective_inertia" else 0.0
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
                or float(raw_value) < minimum
                or parameter.get("unit") != unit
                or isinstance(raw_spread, bool)
                or not isinstance(raw_spread, (int, float))
                or not math.isfinite(float(raw_spread))
                or float(raw_spread) < 0.0
                or not isinstance(confidence, dict)
            ):
                raise ValueError(f"joint physics parameter {name!r} is not a finite estimate")
            lower = confidence.get("lower")
            upper = confidence.get("upper")
            level = confidence.get("level")
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in (lower, upper, level)
            ):
                raise ValueError(f"joint physics parameter {name!r} confidence interval is invalid")
            assert isinstance(lower, (int, float)) and not isinstance(lower, bool)
            assert isinstance(upper, (int, float)) and not isinstance(upper, bool)
            assert isinstance(level, (int, float)) and not isinstance(level, bool)
            if (
                not 0.0 < float(level) < 1.0
                or float(lower) > float(raw_value)
                or float(raw_value) > float(upper)
            ):
                raise ValueError(f"joint physics parameter {name!r} confidence interval is invalid")

    def _validate_asset_assembly(self, document: Mapping[str, Any]) -> None:
        for field in ("resource_id", "version", "root_link"):
            if not isinstance(document.get(field), str) or not document[field].strip():
                raise ValueError(f"asset assembly {field} must be non-empty")
        if document.get("up_axis") not in {"Y", "Z"}:
            raise ValueError("asset assembly up_axis must be Y or Z")
        metric_scale = document.get("metric_scale")
        if not isinstance(metric_scale, dict) or metric_scale.get("status") != "resolved":
            raise ValueError("asset assembly metric_scale must be resolved")
        meters_per_unit = metric_scale.get("meters_per_unit")
        if (
            isinstance(meters_per_unit, bool)
            or not isinstance(meters_per_unit, (int, float))
            or not math.isfinite(float(meters_per_unit))
            or float(meters_per_unit) <= 0.0
        ):
            raise ValueError("asset assembly meters_per_unit must be positive and finite")
        scale_digest = self._digest(
            metric_scale.get("evidence_digest"),
            "asset assembly metric scale evidence_digest",
        )
        references: list[ArtifactRef] = []
        geometry_links: dict[str, set[str]] = {"visual_meshes": set(), "collision_meshes": set()}
        for field in geometry_links:
            records = document.get(field)
            if not isinstance(records, list) or not records:
                raise ValueError(f"asset assembly {field} must be non-empty")
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(f"asset assembly {field}[{index}] must be an object")
                link = record.get("link")
                if not isinstance(link, str) or not link.strip() or link in geometry_links[field]:
                    raise ValueError(f"asset assembly {field} links must be non-empty and unique")
                geometry_links[field].add(link)
                if record.get("units") != "meters":
                    raise ValueError(f"asset assembly {field}[{index}] units must be meters")
                references.append(
                    self._artifact_reference(
                        record.get("artifact"),
                        f"asset assembly {field}[{index}].artifact",
                    )
                )
        if geometry_links["visual_meshes"] != geometry_links["collision_meshes"]:
            raise ValueError("every assembled link requires visual and collision geometry")
        articulation_ref = self._artifact_reference(
            document.get("articulation_model"),
            "asset assembly articulation_model",
        )
        physics_ref = self._artifact_reference(
            document.get("identified_physics"),
            "asset assembly identified_physics",
        )
        raw_evidence = document.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError("asset assembly evidence closure must be non-empty")
        evidence_refs = [
            self._artifact_reference(item, f"asset assembly evidence[{index}]")
            for index, item in enumerate(raw_evidence)
        ]
        if scale_digest not in {item.digest for item in evidence_refs}:
            raise ValueError("metric scale evidence is outside the asset assembly closure")
        references.extend((articulation_ref, physics_ref, *evidence_refs))
        if len({reference.digest for reference in references}) != len(references):
            raise ValueError("asset assembly artifact digests must be unique")
        self._require_available_artifacts(references, "asset assembly")
        articulation = _strict_json_object(
            self._platform.artifacts.get_bytes(articulation_ref),
            "asset assembly articulation model",
        )
        physics = _strict_json_object(
            self._platform.artifacts.get_bytes(physics_ref),
            "asset assembly identified physics",
        )
        if articulation.get("schema") != "physrealx.articulation-model/1" or articulation.get(
            "status"
        ) not in {"IDENTIFIED", "VALIDATED"}:
            raise ValueError("asset assembly articulation model is not identified")
        links = articulation.get("links")
        joints = articulation.get("joints")
        if not isinstance(links, list) or not links or len(set(links)) != len(links):
            raise ValueError("asset assembly articulation links must be non-empty and unique")
        if not isinstance(joints, list):
            raise ValueError("asset assembly articulation joints must be an array")
        joint_names = {
            joint.get("name")
            for joint in joints
            if isinstance(joint, dict) and isinstance(joint.get("name"), str)
        }
        if not joint_names or len(joint_names) != len(joints):
            raise ValueError("asset assembly articulation joints must be non-empty and named")
        if set(links) != geometry_links["visual_meshes"]:
            raise ValueError("asset assembly geometry does not cover every articulation link")
        if (
            physics.get("schema") != "physrealx.identified-physics/1"
            or physics.get("status") not in {"IDENTIFIED", "VALIDATED"}
            or physics.get("evidence_state") not in {"identified", "validated"}
        ):
            raise ValueError("asset assembly full physics is not identified")
        bodies = physics.get("bodies")
        physics_joints = physics.get("joints")
        body_links = (
            {
                body.get("link")
                for body in bodies
                if isinstance(body, dict) and isinstance(body.get("link"), str)
            }
            if isinstance(bodies, list)
            else set()
        )
        physics_joint_names = (
            {
                joint.get("name")
                for joint in physics_joints
                if isinstance(joint, dict) and isinstance(joint.get("name"), str)
            }
            if isinstance(physics_joints, list)
            else set()
        )
        if body_links != set(links) or physics_joint_names != joint_names:
            raise ValueError("asset assembly physics must cover every body and joint exactly")

    def _validate_robot_target(self, document: Mapping[str, Any]) -> None:
        try:
            target = RobotInterfaceSpec.model_validate_json(
                json.dumps(document, allow_nan=False, sort_keys=True)
            )
        except ValueError as error:
            raise ValueError("robot target does not satisfy RobotInterfaceSpec/1") from error
        self._require_available_artifacts(
            [target.urdf, target.controller_configuration],
            "robot target",
        )

    @staticmethod
    def _metric_alignment_evidence_error(document: Mapping[str, Any]) -> str | None:
        """Reject marker detections that do not carry independent metric metrology."""

        digest = document.get("camera_rig_digest")
        try:
            if not isinstance(digest, str):
                raise ValueError
            ArtifactRef(
                digest=digest,
                size_bytes=0,
                media_type="application/json",
            )
        except ValueError:
            return "camera_rig_digest must bind the exact unresolved camera artifact"
        for field in ("component_id", "source_frame", "target_frame"):
            if not isinstance(document.get(field), str) or not document[field].strip():
                return f"field {field!r} must be a non-empty string"
        if document["source_frame"] == document["target_frame"]:
            return "source_frame and target_frame must differ"
        reference = document.get("metric_reference")
        provenance = document.get("provenance")
        if not isinstance(reference, dict) or not isinstance(provenance, dict):
            return "metric_reference and provenance must be objects"
        for field in ("kind", "identifier", "source"):
            if not isinstance(reference.get(field), str) or not reference[field].strip():
                return f"metric_reference.{field} must be a non-empty string"
        value_m = reference.get("value_m")
        if (
            isinstance(value_m, bool)
            or not isinstance(value_m, (int, float))
            or not math.isfinite(float(value_m))
            or float(value_m) <= 0.0
        ):
            return "metric_reference.value_m must be a positive finite metric value"
        for field in ("capture_id", "measurement_method", "source_uri"):
            if not isinstance(provenance.get(field), str) or not provenance[field].strip():
                return f"provenance.{field} must be a non-empty string"
        if "://" not in provenance["source_uri"]:
            return "provenance.source_uri must be absolute"
        marker_evidence = "marker" in reference["kind"].lower() or any(
            token in provenance["measurement_method"].lower() for token in ("apriltag", "aruco")
        )
        if marker_evidence:
            for field in ("marker_family", "layout", "layout_source_uri"):
                if not isinstance(reference.get(field), str) or not reference[field].strip():
                    return f"metric_reference.{field} is required for marker evidence"
            marker_size = reference.get("marker_size_m")
            if (
                isinstance(marker_size, bool)
                or not isinstance(marker_size, (int, float))
                or not math.isfinite(float(marker_size))
                or float(marker_size) <= 0.0
                or not math.isclose(float(marker_size), float(value_m), rel_tol=0.0, abs_tol=1e-12)
            ):
                return "marker_size_m must be positive and equal metric_reference.value_m"
            if "://" not in reference["layout_source_uri"]:
                return "metric_reference.layout_source_uri must be absolute"
        identifiers: dict[str, set[str]] = {}
        for split, minimum in (("train_correspondences", 4), ("holdout_correspondences", 3)):
            records = document.get(split)
            if not isinstance(records, list) or len(records) < minimum:
                return f"{split} requires at least {minimum} measured 3D correspondences"
            split_ids: set[str] = set()
            for record in records:
                if not isinstance(record, dict):
                    return f"{split} entries must be objects"
                identifier = record.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    return f"{split} entries require non-empty ids"
                for field in ("source_xyz", "target_xyz_m"):
                    point = record.get(field)
                    if (
                        not isinstance(point, list)
                        or len(point) != 3
                        or any(
                            isinstance(item, bool)
                            or not isinstance(item, (int, float))
                            or not math.isfinite(float(item))
                            for item in point
                        )
                    ):
                        return f"{split}.{field} must contain three finite coordinates"
                split_ids.add(identifier)
            if len(split_ids) != len(records):
                return f"{split} ids must be unique"
            identifiers[split] = split_ids
        if identifiers["train_correspondences"] & identifiers["holdout_correspondences"]:
            return "training and holdout correspondence ids must be disjoint"
        train_records = document["train_correspondences"]
        holdout_records = document["holdout_correspondences"]
        assert isinstance(train_records, list) and isinstance(holdout_records, list)
        for coordinate in ("source_xyz", "target_xyz_m"):
            train_points = {
                tuple(record[coordinate]) for record in train_records if isinstance(record, dict)
            }
            holdout_points = {
                tuple(record[coordinate]) for record in holdout_records if isinstance(record, dict)
            }
            if train_points & holdout_points:
                return f"training and holdout {coordinate} coordinates must be disjoint"
        return None

    def _observation_evidence_blocker(
        self,
        evidence: LifecycleEvidence,
    ) -> ReadinessBlocker | None:
        resource = evidence.resource
        if resource is None or resource.resource_type is not ResourceType.OBSERVATION_SET:
            return self._gate_failure(
                evidence,
                "observations must be a registered immutable observation_set resource",
            )
        try:
            payload = self._platform.artifacts.get_bytes(resource.digest or "")
            document = _strict_json_object(payload, "observation resource")
            manifest = ObservationSetManifest.model_validate(document)
        except (OSError, PhysRealXError, ValueError) as error:
            return self._gate_failure(
                evidence,
                f"observation resource manifest is invalid or unavailable: {error}",
            )
        if (manifest.resource_id, manifest.version) != (resource.resource_id, resource.version):
            return self._gate_failure(
                evidence,
                "observation resource identity differs from its manifest",
            )
        referenced_artifacts = tuple(
            artifact for stream in manifest.streams for artifact in stream.artifacts
        ) + tuple(attachment.artifact for attachment in manifest.attachments)
        unavailable = tuple(
            artifact.digest
            for artifact in referenced_artifacts
            if not self._platform.artifacts.verify(artifact)
        )
        if unavailable:
            return self._gate_failure(
                evidence,
                "observation resource has unavailable referenced artifacts: "
                + ", ".join(unavailable),
            )
        return None

    def _semantic_document(self, evidence: LifecycleEvidence) -> dict[str, Any]:
        if evidence.artifact is not None:
            artifact = evidence.artifact
            if evidence.key is LifecycleEvidenceKey.ENHANCED_DEPTH and artifact.media_type in {
                "application/zip",
                "application/x-zip-compressed",
            }:
                return self._enhanced_depth_zip_manifest(artifact)
            if not (
                artifact.media_type == "application/json" or artifact.media_type.endswith("+json")
            ):
                raise ValueError(
                    f"binary media type {artifact.media_type!r} has no semantic evidence policy"
                )
            payload = self._platform.artifacts.get_bytes(artifact)
        else:
            resource = evidence.resource
            if resource is None or resource.digest is None:
                raise ValueError("evidence has no immutable JSON location")
            payload = self._platform.artifacts.get_bytes(resource.digest)
        return _strict_json_object(payload, f"evidence {evidence.key.value!r}")

    def _enhanced_depth_zip_manifest(self, artifact: ArtifactRef) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="physrealx-readiness-") as directory:
            archive_path = Path(directory) / "enhanced-depth.zip"
            self._platform.artifacts.materialize(artifact, archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                manifests = [
                    item for item in archive.infolist() if item.filename == "manifest.json"
                ]
                if len(manifests) != 1:
                    raise ValueError("enhanced-depth ZIP must contain exactly one manifest.json")
                manifest = manifests[0]
                if manifest.is_dir() or manifest.flag_bits & 0x1:
                    raise ValueError("enhanced-depth ZIP manifest is not a plain file")
                if manifest.file_size > _MAX_EVIDENCE_JSON_BYTES:
                    raise ValueError("enhanced-depth ZIP manifest exceeds the size limit")
                if manifest.file_size / max(manifest.compress_size, 1) > 200.0:
                    raise ValueError("enhanced-depth ZIP manifest exceeds the compression limit")
                payload = archive.read(manifest)
        return _strict_json_object(payload, "enhanced-depth ZIP manifest")

    @staticmethod
    def _gate_failure(evidence: LifecycleEvidence, detail: str) -> ReadinessBlocker:
        return ReadinessBlocker(
            code=ReadinessBlockerCode.EVIDENCE_GATE_FAILED,
            message=f"evidence {evidence.key.value!r} failed its semantic gate: {detail}",
        )

    def _stage_readiness(
        self,
        binding: LifecycleStageBinding,
        evidence: Mapping[LifecycleEvidenceKey, LifecycleEvidence],
        availability: Mapping[LifecycleEvidenceKey, ReadinessBlocker | None],
    ) -> LifecycleStageReadiness:
        blockers: list[ReadinessBlocker] = []
        try:
            installation = self._platform.plugins.installation(
                binding.capability, binding.method.name
            )
        except ResourceNotFoundError:
            blockers.append(
                ReadinessBlocker(
                    code=ReadinessBlockerCode.METHOD_NOT_INSTALLED,
                    message=(
                        f"method {binding.capability.value}:{binding.method.name!r} "
                        "is not installed"
                    ),
                )
            )
        else:
            if installation.descriptor != binding.method.descriptor:
                blockers.append(
                    ReadinessBlocker(
                        code=ReadinessBlockerCode.METHOD_IDENTITY_MISMATCH,
                        message=(
                            f"installed method {binding.capability.value}:"
                            f"{binding.method.name!r} does not match the plan identity"
                        ),
                    )
                )
            elif not installation.executable:
                blockers.append(
                    ReadinessBlocker(
                        code=ReadinessBlockerCode.METHOD_NOT_EXECUTABLE,
                        message=installation.detail or "method has no executable adapter",
                    )
                )
            elif not installation.ready:
                blockers.append(
                    ReadinessBlocker(
                        code=ReadinessBlockerCode.METHOD_NOT_READY,
                        message=installation.detail or "method readiness probe failed",
                    )
                )
        checkpoint = binding.method.checkpoint
        if checkpoint is not None and not self._platform.artifacts.verify(checkpoint):
            blockers.append(
                ReadinessBlocker(
                    code=ReadinessBlockerCode.CHECKPOINT_UNAVAILABLE,
                    message=f"checkpoint for method {binding.method.name!r} is unavailable",
                )
            )

        satisfied: list[LifecycleEvidenceKey] = []
        for requirement in binding.requires:
            if requirement not in evidence:
                blockers.append(
                    ReadinessBlocker(
                        code=ReadinessBlockerCode.EVIDENCE_MISSING,
                        message=f"required evidence {requirement.value!r} was not supplied",
                    )
                )
            elif (evidence_blocker := availability[requirement]) is not None:
                blockers.append(evidence_blocker)
            else:
                satisfied.append(requirement)
        if (
            binding.stage is LifecycleStage.METRIC_ALIGNMENT
            and LifecycleEvidenceKey.CAMERA_RECONSTRUCTION in satisfied
            and LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE in satisfied
        ):
            camera_evidence = evidence[LifecycleEvidenceKey.CAMERA_RECONSTRUCTION]
            metric_evidence = evidence[LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE]
            camera_digest = (
                camera_evidence.artifact.digest
                if camera_evidence.artifact is not None
                else camera_evidence.resource.digest
                if camera_evidence.resource is not None
                else None
            )
            metric_document = self._semantic_document(metric_evidence)
            if camera_digest is None or metric_document.get("camera_rig_digest") != camera_digest:
                blockers.append(
                    self._gate_failure(
                        metric_evidence,
                        "camera_rig_digest does not match camera_reconstruction evidence",
                    )
                )
                satisfied.remove(LifecycleEvidenceKey.METRIC_ALIGNMENT_EVIDENCE)
        return LifecycleStageReadiness(
            binding=binding,
            status=ReadinessStatus.BLOCKED if blockers else ReadinessStatus.READY,
            satisfied_evidence=tuple(satisfied),
            blockers=tuple(blockers),
        )


__all__ = [
    "EvidenceInput",
    "LifecycleEvidence",
    "LifecycleEvidenceKey",
    "LifecyclePlan",
    "LifecycleReadinessReport",
    "LifecycleStage",
    "LifecycleStageBinding",
    "LifecycleStageReadiness",
    "ReadinessBlocker",
    "ReadinessBlockerCode",
    "ReadinessService",
    "ReadinessStatus",
]
