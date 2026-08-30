"""Evidence-first contracts for training, evaluation, and feedback.

The classes in this module are deliberately algorithm-agnostic.  They validate
outputs produced by separately installed workers; they do not pretend that
LeRobot, DINO-WM, LIBERO, ROS 2, or a robot was executed by the platform core.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import TypeAlias

from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    FailureCategory,
    MethodCapability,
    MethodSpec,
    ResourceType,
    VersionedRef,
    canonical_json_bytes,
    sha256_digest,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class _StrEnum(str, Enum):
    """Python 3.8-compatible string enumeration."""

    def __str__(self) -> str:
        return str.__str__(self)


def _identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase machine identifier")
    return value


def _digest(value: str, label: str) -> str:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise ValueError(f"{label} must be a UUID") from error


def _require_resolved(value: VersionedRef, kind: ResourceType, label: str) -> None:
    if value.resource_type is not kind or not value.resolved:
        raise ValueError(f"{label} must be a resolved {kind.value} reference")


def _require_nonempty_artifact(value: ArtifactRef, label: str) -> None:
    if value.size_bytes <= 0:
        raise ValueError(f"{label} must be a non-empty immutable artifact")


def _require_unique_artifacts(values: tuple[ArtifactRef, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must contain at least one artifact")
    if len({item.digest for item in values}) != len(values):
        raise ValueError(f"{label} artifact digests must be unique")
    for value in values:
        _require_nonempty_artifact(value, label)


def _require_finite_metrics(values: dict[str, float], label: str) -> None:
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"{label} metrics must be finite")


def _method_digest(value: MethodSpec) -> str:
    return sha256_digest(canonical_json_bytes(value))


class CodePin(DomainModel):
    """Exact source tree and revision used to create a dataset or model."""

    source_uri: str = Field(min_length=1, max_length=1024)
    revision: str = Field(min_length=1, max_length=128)
    source_tree: ArtifactRef

    @model_validator(mode="after")
    def validate_pin(self) -> CodePin:
        if "://" not in self.source_uri:
            raise ValueError("code source_uri must be absolute")
        if self.revision.lower() in {"head", "latest", "main", "master"}:
            raise ValueError("code revision must be immutable")
        _require_nonempty_artifact(self.source_tree, "source tree")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class EnvironmentPin(DomainModel):
    """Exact runtime lock and materialized environment identity."""

    runtime: str = Field(min_length=1, max_length=255)
    lockfile: ArtifactRef
    environment_digest: str

    @field_validator("environment_digest")
    @classmethod
    def validate_environment_digest(cls, value: str) -> str:
        return _digest(value, "environment_digest")

    @model_validator(mode="after")
    def validate_pin(self) -> EnvironmentPin:
        _require_nonempty_artifact(self.lockfile, "environment lockfile")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class EpisodeArtifact(DomainModel):
    """Content-addressed payload for one complete episode."""

    episode_id: str
    artifact: ArtifactRef

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        return _uuid(value, "episode_id")

    @model_validator(mode="after")
    def validate_artifact(self) -> EpisodeArtifact:
        _require_nonempty_artifact(self.artifact, "episode")
        return self


class DatasetSplit(DomainModel):
    """Frozen train/validation/test membership."""

    train: tuple[str, ...]
    validation: tuple[str, ...] = ()
    test: tuple[str, ...]

    @field_validator("train", "validation", "test")
    @classmethod
    def validate_episode_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_uuid(item, "split episode id") for item in value)

    @model_validator(mode="after")
    def validate_split(self) -> DatasetSplit:
        if not self.train or not self.test:
            raise ValueError("dataset split requires non-empty train and test partitions")
        combined = self.train + self.validation + self.test
        if len(combined) != len(set(combined)):
            raise ValueError("dataset split partitions must be disjoint")
        return self

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class SnapshotLineage(DomainModel):
    """Evidence that a snapshot was derived without mutating its parent."""

    parent_snapshot_digest: str
    method: MethodSpec
    evidence: tuple[ArtifactRef, ...]

    @field_validator("parent_snapshot_digest")
    @classmethod
    def validate_parent_digest(cls, value: str) -> str:
        return _digest(value, "parent_snapshot_digest")

    @model_validator(mode="after")
    def validate_lineage(self) -> SnapshotLineage:
        if self.method.capability not in {
            MethodCapability.DATASET_SNAPSHOT,
            MethodCapability.FEEDBACK_BUILD_DATASET,
        }:
            raise ValueError("snapshot lineage requires a dataset snapshot or feedback method")
        _require_unique_artifacts(self.evidence, "snapshot lineage evidence")
        return self


class DatasetSnapshot(DomainModel):
    """An immutable, fully reproducible dataset input to training."""

    schema_name: Literal["physrealx.dataset-snapshot/1"] = Field(
        default="physrealx.dataset-snapshot/1", alias="schema", serialization_alias="schema"
    )
    status: Literal["VALIDATED"] = "VALIDATED"
    schema_version: Literal[1] = 1
    dataset: VersionedRef
    manifest: ArtifactRef
    episodes: tuple[EpisodeArtifact, ...]
    code: CodePin
    environment: EnvironmentPin
    seed: int = Field(ge=0, le=4_294_967_295)
    split: DatasetSplit
    lineage: SnapshotLineage | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_snapshot(self) -> DatasetSnapshot:
        _require_resolved(self.dataset, ResourceType.DATASET, "snapshot dataset")
        _require_nonempty_artifact(self.manifest, "dataset manifest")
        if not self.episodes:
            raise ValueError("dataset snapshot requires episode artifacts")
        episode_ids = tuple(item.episode_id for item in self.episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("dataset snapshot episode ids must be unique")
        if len({item.artifact.digest for item in self.episodes}) != len(self.episodes):
            raise ValueError("dataset snapshot episode artifacts must be unique")
        if set(episode_ids) != set(self.split.episode_ids):
            raise ValueError("dataset split must cover every snapshot episode exactly once")
        if self.lineage is not None and self.lineage.parent_snapshot_digest == self.digest:
            raise ValueError("dataset snapshot cannot be its own parent")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class ModelKind(_StrEnum):
    POLICY = "policy"
    WORLD_MODEL = "world_model"


class PolicyTrainingPlan(DomainModel):
    """Pinned request for a separately installed policy-training worker."""

    kind: Literal["policy"] = "policy"
    model_id: str
    version: str
    dataset: DatasetSnapshot
    trainer: MethodSpec
    inference: MethodSpec
    code: CodePin
    environment: EnvironmentPin
    seed: int = Field(ge=0, le=4_294_967_295)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _identifier(value, "model_id")

    @model_validator(mode="after")
    def validate_methods(self) -> PolicyTrainingPlan:
        VersionedRef(
            resource_type=ResourceType.MODEL,
            resource_id=self.model_id,
            version=self.version,
        )
        if self.trainer.capability is not MethodCapability.POLICY_TRAIN:
            raise ValueError("policy plan trainer requires policy.train capability")
        if self.inference.capability is not MethodCapability.POLICY_INFERENCE:
            raise ValueError("policy plan inference requires policy.inference capability")
        return self


class WorldModelTrainingPlan(DomainModel):
    """Pinned request for a separately installed world-model worker."""

    kind: Literal["world_model"] = "world_model"
    model_id: str
    version: str
    dataset: DatasetSnapshot
    trainer: MethodSpec
    rollout: MethodSpec
    code: CodePin
    environment: EnvironmentPin
    seed: int = Field(ge=0, le=4_294_967_295)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _identifier(value, "model_id")

    @model_validator(mode="after")
    def validate_methods(self) -> WorldModelTrainingPlan:
        VersionedRef(
            resource_type=ResourceType.MODEL,
            resource_id=self.model_id,
            version=self.version,
        )
        if self.trainer.capability is not MethodCapability.WORLD_MODEL_TRAIN:
            raise ValueError("world-model plan trainer requires world_model.train capability")
        if self.rollout.capability is not MethodCapability.WORLD_MODEL_ROLLOUT:
            raise ValueError("world-model plan rollout requires world_model.rollout capability")
        return self


TrainingPlan: TypeAlias = Union[PolicyTrainingPlan, WorldModelTrainingPlan]


class TrainingExecutionResult(DomainModel):
    """Untrusted worker result accepted only after all pins are checked."""

    run_id: str
    kind: ModelKind
    model: VersionedRef
    dataset_snapshot_digest: str
    trainer_digest: str
    code_digest: str
    environment_digest: str
    split_digest: str
    seed: int = Field(ge=0, le=4_294_967_295)
    checkpoint: ArtifactRef
    auxiliary_artifacts: tuple[ArtifactRef, ...] = ()
    evidence: tuple[ArtifactRef, ...]
    metrics: dict[str, float]
    started_at: datetime
    completed_at: datetime

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _uuid(value, "training run_id")

    @field_validator(
        "dataset_snapshot_digest",
        "trainer_digest",
        "code_digest",
        "environment_digest",
        "split_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: object) -> str:
        return _digest(value, getattr(info, "field_name", "digest"))

    @model_validator(mode="after")
    def validate_result(self) -> TrainingExecutionResult:
        _require_resolved(self.model, ResourceType.MODEL, "training output model")
        _require_nonempty_artifact(self.checkpoint, "training checkpoint")
        _require_unique_artifacts(self.evidence, "training evidence")
        if self.auxiliary_artifacts:
            _require_unique_artifacts(self.auxiliary_artifacts, "training auxiliary outputs")
        _require_finite_metrics(self.metrics, "training")
        if self.completed_at < self.started_at:
            raise ValueError("training completed_at cannot precede started_at")
        return self


class TrainingReport(DomainModel):
    """Verified training record; its existence is not inferred from a checkpoint alone."""

    schema_name: Literal["physrealx.training-report/1"] = Field(
        default="physrealx.training-report/1", alias="schema", serialization_alias="schema"
    )
    status: Literal["TRAINED"] = "TRAINED"
    schema_version: Literal[1] = 1
    run_id: str
    kind: ModelKind
    model: VersionedRef
    dataset_snapshot_digest: str
    trainer_digest: str
    code_digest: str
    environment_digest: str
    split_digest: str
    seed: int = Field(ge=0, le=4_294_967_295)
    checkpoint: ArtifactRef
    auxiliary_artifacts: tuple[ArtifactRef, ...]
    evidence: tuple[ArtifactRef, ...]
    metrics: dict[str, float]
    started_at: datetime
    completed_at: datetime

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return _uuid(value, "training report run_id")

    @field_validator(
        "dataset_snapshot_digest",
        "trainer_digest",
        "code_digest",
        "environment_digest",
        "split_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: object) -> str:
        return _digest(value, getattr(info, "field_name", "digest"))

    @model_validator(mode="after")
    def validate_report(self) -> TrainingReport:
        _require_resolved(self.model, ResourceType.MODEL, "training report model")
        _require_nonempty_artifact(self.checkpoint, "training report checkpoint")
        _require_unique_artifacts(self.evidence, "training report evidence")
        if self.auxiliary_artifacts:
            _require_unique_artifacts(self.auxiliary_artifacts, "training report outputs")
        _require_finite_metrics(self.metrics, "training report")
        if self.completed_at < self.started_at:
            raise ValueError("training report completed_at cannot precede started_at")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class PolicyModelBundle(DomainModel):
    """A policy checkpoint paired only with a policy inference method."""

    schema_name: Literal["physrealx.policy-model-bundle/1"] = Field(
        default="physrealx.policy-model-bundle/1",
        alias="schema",
        serialization_alias="schema",
    )
    status: Literal["TRAINED"] = "TRAINED"
    schema_version: Literal[1] = 1
    kind: Literal["policy"] = "policy"
    model: VersionedRef
    dataset: DatasetSnapshot
    trainer: MethodSpec
    inference: MethodSpec
    code: CodePin
    environment: EnvironmentPin
    report: TrainingReport

    @model_validator(mode="after")
    def validate_bundle(self) -> PolicyModelBundle:
        _validate_common_bundle(self, ModelKind.POLICY)
        if self.trainer.capability is not MethodCapability.POLICY_TRAIN:
            raise ValueError("policy bundle trainer requires policy.train capability")
        if self.inference.capability is not MethodCapability.POLICY_INFERENCE:
            raise ValueError("policy bundle inference requires policy.inference capability")
        return self


class WorldModelBundle(DomainModel):
    """A world-model checkpoint paired only with a rollout method."""

    schema_name: Literal["physrealx.world-model-bundle/1"] = Field(
        default="physrealx.world-model-bundle/1",
        alias="schema",
        serialization_alias="schema",
    )
    status: Literal["TRAINED"] = "TRAINED"
    schema_version: Literal[1] = 1
    kind: Literal["world_model"] = "world_model"
    model: VersionedRef
    dataset: DatasetSnapshot
    trainer: MethodSpec
    rollout: MethodSpec
    code: CodePin
    environment: EnvironmentPin
    report: TrainingReport

    @model_validator(mode="after")
    def validate_bundle(self) -> WorldModelBundle:
        _validate_common_bundle(self, ModelKind.WORLD_MODEL)
        if self.trainer.capability is not MethodCapability.WORLD_MODEL_TRAIN:
            raise ValueError("world-model bundle trainer requires world_model.train capability")
        if self.rollout.capability is not MethodCapability.WORLD_MODEL_ROLLOUT:
            raise ValueError("world-model bundle rollout requires world_model.rollout capability")
        return self


ModelBundle: TypeAlias = Union[PolicyModelBundle, WorldModelBundle]


def _validate_common_bundle(
    bundle: PolicyModelBundle | WorldModelBundle,
    kind: ModelKind,
) -> None:
    _require_resolved(bundle.model, ResourceType.MODEL, "model bundle")
    report = bundle.report
    if report.kind is not kind or report.model != bundle.model:
        raise ValueError("model bundle and training report identity must match")
    if report.dataset_snapshot_digest != bundle.dataset.digest:
        raise ValueError("model bundle dataset snapshot does not match training report")
    if report.trainer_digest != _method_digest(bundle.trainer):
        raise ValueError("model bundle trainer does not match training report")
    if report.code_digest != bundle.code.digest:
        raise ValueError("model bundle code does not match training report")
    if report.environment_digest != bundle.environment.digest:
        raise ValueError("model bundle environment does not match training report")
    if report.split_digest != bundle.dataset.split.digest:
        raise ValueError("model bundle split does not match training report")


class TrainingLifecycle:
    """Accept successful external-worker evidence and construct typed bundles."""

    @staticmethod
    def finalize_policy(
        plan: PolicyTrainingPlan,
        result: TrainingExecutionResult,
    ) -> PolicyModelBundle:
        TrainingLifecycle._validate_result(plan, result, ModelKind.POLICY)
        report = TrainingLifecycle._report(result)
        return PolicyModelBundle(
            model=result.model,
            dataset=plan.dataset,
            trainer=plan.trainer,
            inference=plan.inference,
            code=plan.code,
            environment=plan.environment,
            report=report,
        )

    @staticmethod
    def finalize_world_model(
        plan: WorldModelTrainingPlan,
        result: TrainingExecutionResult,
    ) -> WorldModelBundle:
        TrainingLifecycle._validate_result(plan, result, ModelKind.WORLD_MODEL)
        report = TrainingLifecycle._report(result)
        return WorldModelBundle(
            model=result.model,
            dataset=plan.dataset,
            trainer=plan.trainer,
            rollout=plan.rollout,
            code=plan.code,
            environment=plan.environment,
            report=report,
        )

    @staticmethod
    def _validate_result(
        plan: TrainingPlan,
        result: TrainingExecutionResult,
        expected_kind: ModelKind,
    ) -> None:
        if result.kind is not expected_kind:
            raise ValueError("training worker returned the wrong model kind")
        if (result.model.resource_id, result.model.version) != (plan.model_id, plan.version):
            raise ValueError("training worker returned an unrequested model identity")
        expected = {
            "dataset_snapshot_digest": plan.dataset.digest,
            "trainer_digest": _method_digest(plan.trainer),
            "code_digest": plan.code.digest,
            "environment_digest": plan.environment.digest,
            "split_digest": plan.dataset.split.digest,
            "seed": plan.seed,
        }
        for field_name, value in expected.items():
            if getattr(result, field_name) != value:
                raise ValueError(f"training worker result changed pinned {field_name}")

    @staticmethod
    def _report(result: TrainingExecutionResult) -> TrainingReport:
        return TrainingReport.model_validate(result.model_dump(round_trip=True))


class BenchmarkEpisode(DomainModel):
    """One immutable test episode and its fixed evaluation seed."""

    episode_id: str
    seed: int = Field(ge=0, le=4_294_967_295)
    input: ArtifactRef

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        return _uuid(value, "benchmark episode_id")

    @model_validator(mode="after")
    def validate_input(self) -> BenchmarkEpisode:
        _require_nonempty_artifact(self.input, "benchmark episode input")
        return self


class BenchmarkSuite(DomainModel):
    """Fixed test membership, inputs, seeds, method, and timeout."""

    schema_name: Literal["physrealx.benchmark-suite/1"] = Field(
        default="physrealx.benchmark-suite/1", alias="schema", serialization_alias="schema"
    )
    status: Literal["FROZEN"] = "FROZEN"
    schema_version: Literal[1] = 1
    suite_id: str
    version: str
    dataset: DatasetSnapshot
    evaluator: MethodSpec
    global_seed: int = Field(ge=0, le=4_294_967_295)
    timeout_s: float = Field(gt=0, allow_inf_nan=False)
    episodes: tuple[BenchmarkEpisode, ...]

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str) -> str:
        return _identifier(value, "suite_id")

    @model_validator(mode="after")
    def validate_suite(self) -> BenchmarkSuite:
        VersionedRef(
            resource_type=ResourceType.EVALUATION,
            resource_id=self.suite_id,
            version=self.version,
        )
        if self.evaluator.capability is not MethodCapability.EVALUATION_RUN:
            raise ValueError("benchmark evaluator requires evaluation.run capability")
        if not self.episodes:
            raise ValueError("benchmark suite requires episodes")
        ids = tuple(item.episode_id for item in self.episodes)
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark episode ids must be unique")
        if ids != self.dataset.split.test:
            raise ValueError("benchmark episodes must exactly match the frozen test split order")
        source_by_id = {item.episode_id: item.artifact for item in self.dataset.episodes}
        if any(source_by_id[item.episode_id] != item.input for item in self.episodes):
            raise ValueError("benchmark inputs must match frozen dataset episode artifacts")
        return self

    @classmethod
    def from_snapshot(
        cls,
        *,
        suite_id: str,
        version: str,
        dataset: DatasetSnapshot,
        evaluator: MethodSpec,
        global_seed: int,
        timeout_s: float,
    ) -> BenchmarkSuite:
        source_by_id = {item.episode_id: item.artifact for item in dataset.episodes}
        episodes = tuple(
            BenchmarkEpisode(
                episode_id=episode_id,
                seed=_episode_seed(global_seed, episode_id),
                input=source_by_id[episode_id],
            )
            for episode_id in dataset.split.test
        )
        return cls(
            suite_id=suite_id,
            version=version,
            dataset=dataset,
            evaluator=evaluator,
            global_seed=global_seed,
            timeout_s=timeout_s,
            episodes=episodes,
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self)


def _episode_seed(global_seed: int, episode_id: str) -> int:
    payload = f"physrealx-eval-v1:{global_seed}:{episode_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class EpisodeExecutionResult(DomainModel):
    """Raw external evaluator result before deadline and aggregate checks."""

    episode_id: str
    seed: int = Field(ge=0, le=4_294_967_295)
    elapsed_s: float = Field(ge=0, allow_inf_nan=False)
    objective_satisfied: bool
    timed_out: bool = False
    error_code: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    evidence: ArtifactRef

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        return _uuid(value, "evaluation episode_id")

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str | None) -> str | None:
        return _identifier(value, "error_code") if value is not None else None

    @model_validator(mode="after")
    def validate_result(self) -> EpisodeExecutionResult:
        _require_nonempty_artifact(self.evidence, "evaluation episode evidence")
        _require_finite_metrics(self.metrics, "evaluation episode")
        return self


class EvaluationOutcome(_StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class EpisodeEvaluationEvidence(DomainModel):
    """Normalized decision for one and only one benchmark episode."""

    episode_id: str
    seed: int = Field(ge=0, le=4_294_967_295)
    outcome: EvaluationOutcome
    elapsed_s: float = Field(ge=0, allow_inf_nan=False)
    metrics: dict[str, float]
    evidence: ArtifactRef
    error_code: str | None = None

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        return _uuid(value, "evaluation evidence episode_id")

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str | None) -> str | None:
        return _identifier(value, "error_code") if value is not None else None

    @model_validator(mode="after")
    def validate_evidence(self) -> EpisodeEvaluationEvidence:
        _require_nonempty_artifact(self.evidence, "evaluation episode evidence")
        _require_finite_metrics(self.metrics, "evaluation episode evidence")
        if self.outcome is EvaluationOutcome.SUCCESS and self.error_code is not None:
            raise ValueError("successful evaluation evidence cannot contain an error")
        if self.outcome is EvaluationOutcome.ERROR and self.error_code is None:
            raise ValueError("error evaluation evidence requires an error_code")
        return self

    @property
    def succeeded(self) -> bool:
        return self.outcome is EvaluationOutcome.SUCCESS


class WilsonInterval(DomainModel):
    confidence_percent: Literal[95] = 95
    lower: float = Field(ge=0, le=1, allow_inf_nan=False)
    upper: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_interval(self) -> WilsonInterval:
        if self.lower > self.upper:
            raise ValueError("Wilson interval lower bound cannot exceed upper bound")
        return self


class EvaluationReport(DomainModel):
    """Fixed-suite report with evidence for every episode and timeout-as-failure."""

    schema_name: Literal["physrealx.benchmark-report/1"] = Field(
        default="physrealx.benchmark-report/1", alias="schema", serialization_alias="schema"
    )
    status: Literal["PASSED"] = "PASSED"
    schema_version: Literal[1] = 1
    report_id: str
    version: str
    model: VersionedRef
    model_kind: ModelKind
    suite: BenchmarkSuite
    episodes: tuple[EpisodeEvaluationEvidence, ...]
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    success_rate_ci: WilsonInterval
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, value: str) -> str:
        return _identifier(value, "report_id")

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        VersionedRef(
            resource_type=ResourceType.EVALUATION,
            resource_id=self.report_id,
            version=self.version,
        )
        _require_resolved(self.model, ResourceType.MODEL, "evaluated model")
        expected_ids = tuple(item.episode_id for item in self.suite.episodes)
        if tuple(item.episode_id for item in self.episodes) != expected_ids:
            raise ValueError("evaluation report requires evidence for every suite episode in order")
        counts = {
            EvaluationOutcome.SUCCESS: 0,
            EvaluationOutcome.FAILURE: 0,
            EvaluationOutcome.TIMEOUT: 0,
            EvaluationOutcome.ERROR: 0,
        }
        for item in self.episodes:
            counts[item.outcome] += 1
        total = len(self.episodes)
        if (
            self.success_count != counts[EvaluationOutcome.SUCCESS]
            or self.failure_count != total - counts[EvaluationOutcome.SUCCESS]
            or self.timeout_count != counts[EvaluationOutcome.TIMEOUT]
            or self.error_count != counts[EvaluationOutcome.ERROR]
        ):
            raise ValueError("evaluation aggregate counts do not match episode evidence")
        expected_rate = self.success_count / total
        if not math.isclose(self.success_rate, expected_rate, abs_tol=1e-12):
            raise ValueError("evaluation success_rate does not match episode evidence")
        lower, upper = _wilson_interval(self.success_count, total)
        if not (
            math.isclose(self.success_rate_ci.lower, lower, abs_tol=1e-12)
            and math.isclose(self.success_rate_ci.upper, upper, abs_tol=1e-12)
        ):
            raise ValueError("evaluation Wilson interval does not match episode evidence")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    z = 1.959963984540054
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total) / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


class EvaluationLifecycle:
    """Normalize external episode results and compute the fixed aggregate."""

    @staticmethod
    def finalize(
        *,
        report_id: str,
        version: str,
        model: ModelBundle,
        suite: BenchmarkSuite,
        results: tuple[EpisodeExecutionResult, ...],
    ) -> EvaluationReport:
        by_id = {item.episode_id: item for item in results}
        if len(by_id) != len(results):
            raise ValueError("evaluation results contain duplicate episode ids")
        expected = {item.episode_id for item in suite.episodes}
        if set(by_id) != expected:
            raise ValueError("evaluation results must cover the fixed suite exactly")
        normalized: list[EpisodeEvaluationEvidence] = []
        for benchmark in suite.episodes:
            result = by_id[benchmark.episode_id]
            if result.seed != benchmark.seed:
                raise ValueError("evaluation worker changed a fixed episode seed")
            timed_out = result.timed_out or result.elapsed_s >= suite.timeout_s
            if timed_out:
                outcome = EvaluationOutcome.TIMEOUT
            elif result.error_code is not None:
                outcome = EvaluationOutcome.ERROR
            elif result.objective_satisfied:
                outcome = EvaluationOutcome.SUCCESS
            else:
                outcome = EvaluationOutcome.FAILURE
            normalized.append(
                EpisodeEvaluationEvidence(
                    episode_id=result.episode_id,
                    seed=result.seed,
                    outcome=outcome,
                    elapsed_s=result.elapsed_s,
                    metrics=result.metrics,
                    evidence=result.evidence,
                    error_code=result.error_code,
                )
            )
        success_count = sum(item.succeeded for item in normalized)
        failure_count = len(normalized) - success_count
        timeout_count = sum(item.outcome is EvaluationOutcome.TIMEOUT for item in normalized)
        error_count = sum(item.outcome is EvaluationOutcome.ERROR for item in normalized)
        lower, upper = _wilson_interval(success_count, len(normalized))
        return EvaluationReport(
            report_id=report_id,
            version=version,
            model=model.model,
            model_kind=ModelKind(model.kind),
            suite=suite,
            episodes=tuple(normalized),
            success_count=success_count,
            failure_count=failure_count,
            timeout_count=timeout_count,
            error_count=error_count,
            success_rate=success_count / len(normalized),
            success_rate_ci=WilsonInterval(lower=lower, upper=upper),
        )


class FailureRecord(DomainModel):
    """Evidence-backed failure; ``unknown`` remains valid until confirmed."""

    schema_name: Literal["physrealx.failure-record/1"] = Field(
        default="physrealx.failure-record/1", alias="schema", serialization_alias="schema"
    )
    schema_version: Literal[1] = 1
    failure_id: str
    evaluation_report_digest: str
    model: VersionedRef
    suite_digest: str
    episode_id: str
    category: FailureCategory = FailureCategory.UNKNOWN
    message: str = Field(min_length=1, max_length=2048)
    evidence: tuple[ArtifactRef, ...]
    confirmed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("failure_id", "episode_id")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        return _uuid(value, getattr(info, "field_name", "id"))

    @field_validator("evaluation_report_digest", "suite_digest")
    @classmethod
    def validate_record_digest(cls, value: str, info: object) -> str:
        return _digest(value, getattr(info, "field_name", "digest"))

    @model_validator(mode="after")
    def validate_record(self) -> FailureRecord:
        _require_resolved(self.model, ResourceType.MODEL, "failure model")
        _require_unique_artifacts(self.evidence, "failure evidence")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class FailureReport(DomainModel):
    """Complete failure analysis for one frozen benchmark report."""

    schema_name: Literal["physrealx.failure-report/1"] = Field(
        default="physrealx.failure-report/1", alias="schema", serialization_alias="schema"
    )
    status: Literal["ANALYZED", "CONFIRMED"]
    evaluation_report_digest: str
    records: tuple[FailureRecord, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evaluation_report_digest")
    @classmethod
    def validate_evaluation_digest(cls, value: str) -> str:
        return _digest(value, "evaluation_report_digest")

    @model_validator(mode="after")
    def validate_report(self) -> FailureReport:
        if len({item.failure_id for item in self.records}) != len(self.records):
            raise ValueError("failure report record ids must be unique")
        if any(
            item.evaluation_report_digest != self.evaluation_report_digest for item in self.records
        ):
            raise ValueError("failure report records must reference one evaluation report")
        confirmed = all(item.confirmed for item in self.records)
        if (self.status == "CONFIRMED") != confirmed:
            raise ValueError("failure report is CONFIRMED exactly when every record is confirmed")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class FailureLifecycle:
    """Create records only for non-successful evaluated episodes."""

    @staticmethod
    def record(
        report: EvaluationReport,
        *,
        episode_id: str,
        category: FailureCategory = FailureCategory.UNKNOWN,
        message: str,
        evidence: tuple[ArtifactRef, ...],
        confirmed: bool = False,
    ) -> FailureRecord:
        try:
            episode = next(item for item in report.episodes if item.episode_id == episode_id)
        except StopIteration as error:
            raise ValueError("failure episode is absent from the evaluation report") from error
        if episode.succeeded:
            raise ValueError("a successful episode cannot become a failure record")
        if episode.evidence.digest not in {item.digest for item in evidence}:
            raise ValueError("failure record must retain the evaluation episode evidence")
        return FailureRecord(
            failure_id=str(uuid.uuid4()),
            evaluation_report_digest=report.digest,
            model=report.model,
            suite_digest=report.suite.digest,
            episode_id=episode_id,
            category=category,
            message=message,
            evidence=evidence,
            confirmed=confirmed,
        )

    @staticmethod
    def finalize(
        report: EvaluationReport,
        records: tuple[FailureRecord, ...],
    ) -> FailureReport:
        """Require exactly one analysis record for every failed benchmark episode."""

        failed_episode_ids = tuple(
            item.episode_id for item in report.episodes if not item.succeeded
        )
        if tuple(item.episode_id for item in records) != failed_episode_ids:
            raise ValueError(
                "failure analysis must cover every failed episode exactly once in report order"
            )
        if any(item.evaluation_report_digest != report.digest for item in records):
            raise ValueError("failure analysis record is bound to another evaluation report")
        return FailureReport(
            status=("CONFIRMED" if all(item.confirmed for item in records) else "ANALYZED"),
            evaluation_report_digest=report.digest,
            records=records,
        )


class FeedbackDatasetSpec(DomainModel):
    """Immutable plan for a new dataset version derived from failures."""

    schema_version: Literal[1] = 1
    base: DatasetSnapshot
    target: VersionedRef
    failures: tuple[FailureRecord, ...]
    method: MethodSpec
    code: CodePin
    environment: EnvironmentPin
    seed: int = Field(ge=0, le=4_294_967_295)

    @model_validator(mode="after")
    def validate_spec(self) -> FeedbackDatasetSpec:
        if self.target.resource_type is not ResourceType.DATASET or self.target.resolved:
            raise ValueError("feedback target must be an unresolved dataset reference")
        if self.target.resource_id != self.base.dataset.resource_id:
            raise ValueError("feedback target must preserve the dataset resource id")
        if self.target.version == self.base.dataset.version:
            raise ValueError("feedback must create a new dataset version")
        if self.method.capability is not MethodCapability.FEEDBACK_BUILD_DATASET:
            raise ValueError("feedback method requires feedback.build_dataset capability")
        if not self.failures:
            raise ValueError("feedback dataset requires at least one failure")
        ids = tuple(item.failure_id for item in self.failures)
        if len(ids) != len(set(ids)):
            raise ValueError("feedback failure ids must be unique")
        return self


class FeedbackDatasetManifest(DomainModel):
    """Validated immutable dataset revision produced from confirmed failure evidence."""

    schema_name: Literal["physrealx.feedback-dataset/1"] = Field(
        default="physrealx.feedback-dataset/1", alias="schema", serialization_alias="schema"
    )
    status: Literal["VALIDATED"] = "VALIDATED"
    snapshot: DatasetSnapshot
    parent_snapshot_digest: str
    failure_record_digests: tuple[str, ...]
    method_digest: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("parent_snapshot_digest", "method_digest")
    @classmethod
    def validate_digest_fields(cls, value: str, info: object) -> str:
        return _digest(value, getattr(info, "field_name", "digest"))

    @field_validator("failure_record_digests")
    @classmethod
    def validate_failure_digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("feedback manifest requires failure records")
        normalized = tuple(_digest(item, "failure_record_digest") for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("feedback failure record digests must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_lineage(self) -> FeedbackDatasetManifest:
        lineage = self.snapshot.lineage
        if lineage is None or lineage.parent_snapshot_digest != self.parent_snapshot_digest:
            raise ValueError("feedback manifest snapshot must preserve parent lineage")
        if _method_digest(lineage.method) != self.method_digest:
            raise ValueError("feedback manifest method must match snapshot lineage")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class FeedbackLifecycle:
    """Accept a derived snapshot while proving the parent was not overwritten."""

    @staticmethod
    def finalize(
        spec: FeedbackDatasetSpec,
        produced: DatasetSnapshot,
    ) -> FeedbackDatasetManifest:
        if (produced.dataset.resource_id, produced.dataset.version) != (
            spec.target.resource_id,
            spec.target.version,
        ):
            raise ValueError("feedback worker returned an unrequested dataset identity")
        if produced.dataset.digest == spec.base.dataset.digest:
            raise ValueError("feedback output cannot reuse the parent dataset digest")
        if produced.lineage is None:
            raise ValueError("feedback output requires explicit snapshot lineage")
        if produced.lineage.parent_snapshot_digest != spec.base.digest:
            raise ValueError("feedback output lineage does not identify the immutable parent")
        if produced.lineage.method != spec.method:
            raise ValueError("feedback output lineage changed the requested method")
        if produced.code != spec.code or produced.environment != spec.environment:
            raise ValueError("feedback output changed pinned code or environment")
        if produced.seed != spec.seed:
            raise ValueError("feedback output changed the pinned seed")
        return FeedbackDatasetManifest(
            snapshot=produced,
            parent_snapshot_digest=spec.base.digest,
            failure_record_digests=tuple(item.digest for item in spec.failures),
            method_digest=_method_digest(spec.method),
        )


__all__ = [
    "BenchmarkEpisode",
    "BenchmarkSuite",
    "CodePin",
    "DatasetSnapshot",
    "DatasetSplit",
    "EnvironmentPin",
    "EpisodeArtifact",
    "EpisodeEvaluationEvidence",
    "EpisodeExecutionResult",
    "EvaluationLifecycle",
    "EvaluationOutcome",
    "EvaluationReport",
    "FailureLifecycle",
    "FailureRecord",
    "FailureReport",
    "FeedbackDatasetManifest",
    "FeedbackDatasetSpec",
    "FeedbackLifecycle",
    "ModelBundle",
    "ModelKind",
    "PolicyModelBundle",
    "PolicyTrainingPlan",
    "SnapshotLineage",
    "TrainingExecutionResult",
    "TrainingLifecycle",
    "TrainingPlan",
    "TrainingReport",
    "WilsonInterval",
    "WorldModelBundle",
    "WorldModelTrainingPlan",
]
