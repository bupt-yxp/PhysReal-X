"""Deterministic readiness gates for the complete PhysReal-X lifecycle.

Readiness is deliberately narrower than successful execution.  A stage is
``READY`` only when its exact method implementation is executable *now* and
every required input is backed by concrete evidence.  Planned upstream output
is never treated as evidence.
"""

from __future__ import annotations

import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
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
    StrEnum,
    VersionedRef,
)
from physrealx.errors import PhysRealXError, ResourceNotFoundError
from physrealx.evidence import ScientificEvidenceValidator, strict_json_object

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class LifecycleStage(StrEnum):
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


class LifecycleEvidenceKey(StrEnum):
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


class ReadinessStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class ReadinessBlockerCode(StrEnum):
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
            raise ValueError("lifecycle resource evidence must include an immutable artifact ID")
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
        frozenset({"physrealx.asset-validation-report/1"}),
        frozenset({"PASSED"}),
        required_values=(
            ("physical_validation", True),
            ("validation_scope", "independent_physical_trajectory"),
        ),
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


class ReadinessService:
    """Compile and evaluate lifecycle plans without running any stage."""

    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform
        self._evidence = ScientificEvidenceValidator(platform.artifacts)

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
    def _evidence_artifact_id(evidence: LifecycleEvidence) -> str | None:
        if evidence.artifact is not None:
            return evidence.artifact.artifact_id
        if evidence.resource is not None:
            return evidence.resource.artifact_id
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
            camera_artifact_id = self._evidence_artifact_id(
                evidence[LifecycleEvidenceKey.CAMERA_SOLUTION]
            )
            depth_artifact_id = self._evidence_artifact_id(
                evidence[LifecycleEvidenceKey.ENHANCED_DEPTH]
            )
            if (
                geometry.get("source_camera_artifact_id") != camera_artifact_id
                or geometry.get("source_depth_artifact_id") != depth_artifact_id
            ):
                result[LifecycleEvidenceKey.VISUAL_GEOMETRY] = self._gate_failure(
                    geometry_evidence,
                    "geometry source artifact IDs do not match camera_solution and enhanced_depth",
                )

        if usable(
            LifecycleEvidenceKey.COLLISION_GEOMETRY,
            LifecycleEvidenceKey.VISUAL_GEOMETRY,
        ):
            collision_evidence = evidence[LifecycleEvidenceKey.COLLISION_GEOMETRY]
            collision = self._semantic_document(collision_evidence)
            geometry = self._semantic_document(evidence[LifecycleEvidenceKey.VISUAL_GEOMETRY])
            mesh_artifact_ids = {
                raw["artifact"]["artifact_id"]
                for raw in geometry["meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("artifact_id"), str)
            }
            source = collision.get("source_geometry")
            source_artifact_id = source.get("artifact_id") if isinstance(source, dict) else None
            if source_artifact_id not in mesh_artifact_ids:
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
                raw["artifact"]["artifact_id"]
                for raw in geometry["meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("artifact_id"), str)
            }
            assembled_visual = {
                raw["artifact"]["artifact_id"]
                for raw in assembly["visual_meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("artifact_id"), str)
            }
            assembled_collision = {
                raw["artifact"]["artifact_id"]
                for raw in assembly["collision_meshes"]
                if isinstance(raw, dict)
                and isinstance(raw.get("artifact"), dict)
                and isinstance(raw["artifact"].get("artifact_id"), str)
            }
            collision_output = collision.get("collision_geometry")
            collision_artifact_id = (
                collision_output.get("artifact_id") if isinstance(collision_output, dict) else None
            )
            articulation_artifact_id = self._evidence_artifact_id(
                evidence[LifecycleEvidenceKey.ARTICULATION]
            )
            assembly_articulation = assembly.get("articulation_model")
            evidence_closure = {
                raw["artifact_id"]
                for raw in assembly["evidence"]
                if isinstance(raw, dict) and isinstance(raw.get("artifact_id"), str)
            }
            required_closure = {
                self._evidence_artifact_id(evidence[key])
                for key in (
                    LifecycleEvidenceKey.CAMERA_SOLUTION,
                    LifecycleEvidenceKey.BODY_PHYSICS,
                    LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS,
                )
            }
            if (
                supplied_visual != assembled_visual
                or collision_artifact_id not in assembled_collision
                or not isinstance(assembly_articulation, dict)
                or assembly_articulation.get("artifact_id") != articulation_artifact_id
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
        if registered.artifact_id != evidence.resource.artifact_id:
            return ReadinessBlocker(
                code=ReadinessBlockerCode.EVIDENCE_UNAVAILABLE,
                message=(
                    f"resource evidence {evidence.key.value!r} has a registry artifact ID mismatch"
                ),
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
            detail = self._evidence.metric_alignment_evidence_error(document)
            if detail is not None:
                return self._gate_failure(evidence, detail)
        try:
            if evidence.key is LifecycleEvidenceKey.VISUAL_GEOMETRY:
                self._evidence.validate_geometry_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.COLLISION_GEOMETRY:
                self._evidence.validate_collision_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.BODY_PHYSICS:
                self._evidence.validate_body_physics_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.IDENTIFIED_JOINT_PHYSICS:
                self._evidence.validate_joint_physics_evidence(document)
            elif evidence.key is LifecycleEvidenceKey.ASSET_ASSEMBLY:
                self._evidence.assembled_asset(document)
            elif evidence.key is LifecycleEvidenceKey.ROBOT_TARGET:
                self._evidence.validate_robot_target(document)
        except ValueError as error:
            return self._gate_failure(evidence, str(error))
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
            payload = self._platform.artifacts.get_bytes(resource.artifact_id or "")
            document = strict_json_object(payload, "observation resource")
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
            artifact.artifact_id
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
            if resource is None or resource.artifact_id is None:
                raise ValueError("evidence has no immutable JSON location")
            payload = self._platform.artifacts.get_bytes(resource.artifact_id)
        return strict_json_object(payload, f"evidence {evidence.key.value!r}")

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
        return strict_json_object(payload, "enhanced-depth ZIP manifest")

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
            camera_artifact_id = (
                camera_evidence.artifact.artifact_id
                if camera_evidence.artifact is not None
                else camera_evidence.resource.artifact_id
                if camera_evidence.resource is not None
                else None
            )
            metric_document = self._semantic_document(metric_evidence)
            if (
                camera_artifact_id is None
                or metric_document.get("camera_rig_artifact_id") != camera_artifact_id
            ):
                blockers.append(
                    self._gate_failure(
                        metric_evidence,
                        "camera_rig_artifact_id does not match camera_reconstruction evidence",
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
