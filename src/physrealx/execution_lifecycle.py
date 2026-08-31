"""Evidence contracts for scene execution, rendering, and episode packing.

This module does not run a simulator, renderer, policy, or episode converter.
It accepts versioned evidence produced by registered in-process methods and
finalizes an episode only when every immutable pin and every frame agrees.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator, model_validator

from physrealx.domain import (
    ArtifactRef,
    DataValue,
    DomainModel,
    Episode,
    EpisodeSource,
    MethodCapability,
    MethodSpec,
    ResourceType,
    SignalDType,
    VersionedRef,
    canonical_json_bytes,
    canonical_record_id,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_RECORD_ID = re.compile(r"^rec_[0-9a-f]{32}$")
_CONTRACT = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}/[1-9][0-9]*$")
_MAX_SEED = 2**63 - 1


def _identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase machine identifier")
    return value


def _record_id(value: str, label: str) -> str:
    if _RECORD_ID.fullmatch(value) is None:
        raise ValueError(f"{label} must use the platform record-ID format")
    return value


def _uuid(value: str, label: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise ValueError(f"{label} must be a UUID") from error


def _require_resolved(value: VersionedRef, kind: ResourceType, label: str) -> None:
    if value.resource_type is not kind or not value.resolved:
        raise ValueError(f"{label} must be a resolved {kind.value} reference")


def _require_artifact(value: ArtifactRef, label: str) -> None:
    if value.size_bytes <= 0:
        raise ValueError(f"{label} must be a non-empty immutable artifact")


def _require_evidence(values: tuple[ArtifactRef, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} requires at least one evidence artifact")
    if len({item.artifact_id for item in values}) != len(values):
        raise ValueError(f"{label} evidence artifact identifiers must be unique")
    for item in values:
        _require_artifact(item, label)


def _method_record_id(value: MethodSpec) -> str:
    return canonical_record_id(value)


def _seconds_to_nanoseconds(value: float, label: str) -> int:
    scaled = Decimal(str(value)) * Decimal(1_000_000_000)
    if scaled != scaled.to_integral_value():
        raise ValueError(f"{label} must be exactly representable in integer nanoseconds")
    return int(scaled)


def _value_schema(value: DataValue) -> tuple[SignalDType, tuple[int, ...], str, str]:
    return value.dtype, value.shape, value.unit, value.frame


def _require_timestamp(value: DataValue, timestamp_ns: int, label: str) -> None:
    if value.timestamp_seconds is None:
        raise ValueError(f"{label} requires an explicit timestamp")
    if _seconds_to_nanoseconds(value.timestamp_seconds, label) != timestamp_ns:
        raise ValueError(f"{label} timestamp does not match its frame")


class ExecutionInput(DomainModel):
    """One named immutable input pinned into an execution plan."""

    schema_version: Literal[1] = 1
    name: str
    contract: str
    resource: VersionedRef | None = None
    artifact: ArtifactRef | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "execution input name")

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, value: str) -> str:
        if _CONTRACT.fullmatch(value) is None:
            raise ValueError("execution input contract must be explicitly versioned")
        return value

    @model_validator(mode="after")
    def validate_input(self) -> ExecutionInput:
        if (self.resource is None) == (self.artifact is None):
            raise ValueError("execution input requires exactly one resource or artifact")
        if self.resource is not None and not self.resource.resolved:
            raise ValueError("execution input resources must be resolved")
        if self.artifact is not None:
            _require_artifact(self.artifact, "execution input")
        return self


def _input_set_record_id(values: tuple[ExecutionInput, ...]) -> str:
    return canonical_record_id(
        {
            "schema": "physrealx.execution-input-set/1",
            "inputs": [item.model_dump(mode="json", by_alias=True) for item in values],
        }
    )


class ExecutionPlan(DomainModel):
    """Fully pinned request; methods may not substitute any field."""

    schema_name: Literal["physrealx.execution-plan/1"] = Field(
        default="physrealx.execution-plan/1", alias="schema", serialization_alias="schema"
    )
    schema_version: Literal[1] = 1
    status: Literal["FROZEN"] = "FROZEN"
    run_id: str
    episode_id: str
    episode_index: int = Field(ge=0)
    scene: VersionedRef
    task: VersionedRef
    simulator: MethodSpec
    policy: MethodSpec
    renderer: MethodSpec
    packer: MethodSpec
    seed: int = Field(ge=0, le=_MAX_SEED)
    inputs: tuple[ExecutionInput, ...]
    modalities: tuple[str, ...]

    @field_validator("run_id", "episode_id")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        return _uuid(value, getattr(info, "field_name", "execution id"))

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("execution plan requires rendered modalities")
        for value in values:
            _identifier(value, "rendered modality")
        if len(values) != len(set(values)):
            raise ValueError("rendered modalities must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("rendered modalities must use canonical lexical order")
        return values

    @model_validator(mode="after")
    def validate_plan(self) -> ExecutionPlan:
        _require_resolved(self.scene, ResourceType.SCENE, "execution scene")
        _require_resolved(self.task, ResourceType.TASK, "execution task")
        capabilities = (
            (self.simulator, MethodCapability.SIMULATION_RUN, "simulator"),
            (self.policy, MethodCapability.POLICY_INFERENCE, "policy"),
            (self.renderer, MethodCapability.RENDER_GENERATE, "renderer"),
            (self.packer, MethodCapability.EPISODE_PACK, "packer"),
        )
        for method, capability, label in capabilities:
            if method.capability is not capability:
                raise ValueError(f"execution {label} requires {capability.value} capability")
        if not self.inputs:
            raise ValueError("execution plan requires at least one immutable input")
        names = tuple(item.name for item in self.inputs)
        if len(names) != len(set(names)):
            raise ValueError("execution input names must be unique")
        if names != tuple(sorted(names)):
            raise ValueError("execution inputs must use canonical lexical name order")
        return self

    @property
    def input_record_id(self) -> str:
        return _input_set_record_id(self.inputs)

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


class StateFrameEvidence(DomainModel):
    """One nanosecond-indexed physics state and the action leading onward."""

    schema_version: Literal[1] = 1
    index: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    state: dict[str, DataValue]
    action: DataValue | None

    @model_validator(mode="after")
    def validate_frame(self) -> StateFrameEvidence:
        if not self.state:
            raise ValueError("state frame requires at least one state signal")
        for name, value in self.state.items():
            _identifier(name, "state signal name")
            _require_timestamp(value, self.timestamp_ns, f"state signal {name!r}")
        if self.action is not None:
            _require_timestamp(self.action, self.timestamp_ns, "state-frame action")
        return self

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


class StateTrajectoryEvidence(DomainModel):
    """Untrusted simulator output with every plan pin repeated for comparison."""

    schema_name: Literal["physrealx.state-trajectory-evidence/1"] = Field(
        default="physrealx.state-trajectory-evidence/1",
        alias="schema",
        serialization_alias="schema",
    )
    schema_version: Literal[1] = 1
    plan_record_id: str
    scene: VersionedRef
    task: VersionedRef
    simulator_record_id: str
    policy_record_id: str
    seed: int = Field(ge=0, le=_MAX_SEED)
    input_record_id: str
    trajectory: ArtifactRef
    frames: tuple[StateFrameEvidence, ...]
    evidence: tuple[ArtifactRef, ...]
    started_at: datetime
    completed_at: datetime

    @field_validator("plan_record_id", "simulator_record_id", "policy_record_id", "input_record_id")
    @classmethod
    def validate_record_ids(cls, value: str, info: object) -> str:
        return _record_id(value, getattr(info, "field_name", "trajectory record ID"))

    @model_validator(mode="after")
    def validate_trajectory(self) -> StateTrajectoryEvidence:
        _require_resolved(self.scene, ResourceType.SCENE, "trajectory scene")
        _require_resolved(self.task, ResourceType.TASK, "trajectory task")
        _require_artifact(self.trajectory, "state trajectory")
        _require_evidence(self.evidence, "state trajectory")
        if len(self.frames) < 2:
            raise ValueError("state trajectory requires at least two frames")
        if tuple(item.index for item in self.frames) != tuple(range(len(self.frames))):
            raise ValueError("state frame indices must be contiguous from zero")
        timestamps = tuple(item.timestamp_ns for item in self.frames)
        if timestamps[0] != 0:
            raise ValueError("state trajectory must use a zero-based relative timeline")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("state frame timestamps must be strictly increasing")
        if self.frames[-1].action is not None or any(
            item.action is None for item in self.frames[:-1]
        ):
            raise ValueError("only the final state frame may omit an action")
        state_names = set(self.frames[0].state)
        state_schema = {name: _value_schema(value) for name, value in self.frames[0].state.items()}
        action_schema = _value_schema(self.frames[0].action)  # type: ignore[arg-type]
        for frame in self.frames:
            if set(frame.state) != state_names:
                raise ValueError("state signal sets must remain constant across frames")
            if any(
                _value_schema(value) != state_schema[name] for name, value in frame.state.items()
            ):
                raise ValueError("state signal schemas must remain constant across frames")
            if frame.action is not None and _value_schema(frame.action) != action_schema:
                raise ValueError("action schema must remain constant across frames")
        if self.completed_at < self.started_at:
            raise ValueError("trajectory completed_at cannot precede started_at")
        return self

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


class RenderedModality(DomainModel):
    """One rendered signal backed by an immutable artifact."""

    schema_version: Literal[1] = 1
    name: str
    value: DataValue

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "rendered modality name")

    @model_validator(mode="after")
    def validate_value(self) -> RenderedModality:
        if self.value.artifact is None:
            raise ValueError("rendered modalities must use immutable artifact payloads")
        _require_artifact(self.value.artifact, "rendered modality")
        return self


class RenderedFrameEvidence(DomainModel):
    """Rendered modalities tied to one exact state-frame record."""

    schema_version: Literal[1] = 1
    index: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    state_frame_id: str
    modalities: tuple[RenderedModality, ...]

    @field_validator("state_frame_id")
    @classmethod
    def validate_state_frame_id(cls, value: str) -> str:
        return _record_id(value, "state_frame_id")

    @model_validator(mode="after")
    def validate_frame(self) -> RenderedFrameEvidence:
        if not self.modalities:
            raise ValueError("rendered frame requires modalities")
        names = tuple(item.name for item in self.modalities)
        if len(names) != len(set(names)):
            raise ValueError("rendered modality names must be unique per frame")
        if names != tuple(sorted(names)):
            raise ValueError("rendered modalities must use canonical lexical order")
        for item in self.modalities:
            _require_timestamp(item.value, self.timestamp_ns, f"rendered modality {item.name!r}")
        return self


class RenderStreamEvidence(DomainModel):
    """Renderer output whose frames must replay the exact physics timeline."""

    schema_name: Literal["physrealx.render-stream-evidence/1"] = Field(
        default="physrealx.render-stream-evidence/1",
        alias="schema",
        serialization_alias="schema",
    )
    schema_version: Literal[1] = 1
    plan_record_id: str
    trajectory_evidence_id: str
    scene: VersionedRef
    task: VersionedRef
    renderer_record_id: str
    seed: int = Field(ge=0, le=_MAX_SEED)
    input_record_id: str
    stream: ArtifactRef
    frames: tuple[RenderedFrameEvidence, ...]
    evidence: tuple[ArtifactRef, ...]
    started_at: datetime
    completed_at: datetime

    @field_validator(
        "plan_record_id", "trajectory_evidence_id", "renderer_record_id", "input_record_id"
    )
    @classmethod
    def validate_record_ids(cls, value: str, info: object) -> str:
        return _record_id(value, getattr(info, "field_name", "render record ID"))

    @model_validator(mode="after")
    def validate_stream(self) -> RenderStreamEvidence:
        _require_resolved(self.scene, ResourceType.SCENE, "render scene")
        _require_resolved(self.task, ResourceType.TASK, "render task")
        _require_artifact(self.stream, "render stream")
        _require_evidence(self.evidence, "render stream")
        if len(self.frames) < 2:
            raise ValueError("render stream requires at least two frames")
        if tuple(item.index for item in self.frames) != tuple(range(len(self.frames))):
            raise ValueError("rendered frame indices must be contiguous from zero")
        timestamps = tuple(item.timestamp_ns for item in self.frames)
        if timestamps[0] != 0:
            raise ValueError("render stream must use a zero-based relative timeline")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("rendered frame timestamps must be strictly increasing")
        names = tuple(item.name for item in self.frames[0].modalities)
        schema = {item.name: _value_schema(item.value) for item in self.frames[0].modalities}
        for frame in self.frames:
            if tuple(item.name for item in frame.modalities) != names:
                raise ValueError("rendered modality sets must remain constant across frames")
            if any(_value_schema(item.value) != schema[item.name] for item in frame.modalities):
                raise ValueError("rendered modality schemas must remain constant across frames")
        if self.completed_at < self.started_at:
            raise ValueError("render completed_at cannot precede started_at")
        return self

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


class EpisodePackResult(DomainModel):
    """Untrusted episode-packer result, accepted only after closure checks."""

    schema_name: Literal["physrealx.episode-pack-result/1"] = Field(
        default="physrealx.episode-pack-result/1",
        alias="schema",
        serialization_alias="schema",
    )
    schema_version: Literal[1] = 1
    plan_record_id: str
    trajectory_evidence_id: str
    render_evidence_id: str
    packer_record_id: str
    input_record_id: str
    scene: VersionedRef
    task: VersionedRef
    seed: int = Field(ge=0, le=_MAX_SEED)
    episode_payload_id: str
    episode: Episode
    artifact: ArtifactRef
    evidence: tuple[ArtifactRef, ...]
    started_at: datetime
    completed_at: datetime

    @field_validator(
        "plan_record_id",
        "trajectory_evidence_id",
        "render_evidence_id",
        "packer_record_id",
        "input_record_id",
        "episode_payload_id",
    )
    @classmethod
    def validate_record_ids(cls, value: str, info: object) -> str:
        return _record_id(value, getattr(info, "field_name", "episode-pack record ID"))

    @model_validator(mode="after")
    def validate_result(self) -> EpisodePackResult:
        _require_resolved(self.scene, ResourceType.SCENE, "packed episode scene")
        _require_resolved(self.task, ResourceType.TASK, "packed episode task")
        _require_artifact(self.artifact, "packed episode")
        _require_evidence(self.evidence, "episode packing")
        if self.episode_payload_id != self.episode.record_id:
            raise ValueError("episode_payload_id does not match the canonical episode")
        if self.completed_at < self.started_at:
            raise ValueError("episode packing completed_at cannot precede started_at")
        return self

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


class ExecutionValidationReport(DomainModel):
    """Machine-verifiable record that every execution-chain check passed."""

    schema_name: Literal["physrealx.execution-validation-report/1"] = Field(
        default="physrealx.execution-validation-report/1",
        alias="schema",
        serialization_alias="schema",
    )
    schema_version: Literal[1] = 1
    status: Literal["VALIDATED"] = "VALIDATED"
    plan_record_id: str
    input_record_id: str
    trajectory_evidence_id: str
    render_evidence_id: str
    pack_result_id: str
    episode_record_id: str
    checks: tuple[str, ...]
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "plan_record_id",
        "input_record_id",
        "trajectory_evidence_id",
        "render_evidence_id",
        "pack_result_id",
        "episode_record_id",
    )
    @classmethod
    def validate_record_ids(cls, value: str, info: object) -> str:
        return _record_id(value, getattr(info, "field_name", "validation record ID"))

    @field_validator("checks")
    @classmethod
    def validate_checks(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("execution validation report requires checks")
        for value in values:
            _identifier(value, "execution validation check")
        if len(values) != len(set(values)):
            raise ValueError("execution validation checks must be unique")
        return values

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


_VALIDATION_CHECKS = (
    "artifact_evidence",
    "episode_binding",
    "input_binding",
    "method_binding",
    "resource_binding",
    "timeline_alignment",
)


def _require_common_binding(
    plan: ExecutionPlan,
    *,
    plan_record_id: str,
    scene: VersionedRef,
    task: VersionedRef,
    seed: int,
    input_record_id: str,
    label: str,
) -> None:
    expected = {
        "plan_record_id": plan.record_id,
        "scene": plan.scene,
        "task": plan.task,
        "seed": plan.seed,
        "input_record_id": plan.input_record_id,
    }
    actual = {
        "plan_record_id": plan_record_id,
        "scene": scene,
        "task": task,
        "seed": seed,
        "input_record_id": input_record_id,
    }
    for field_name, expected_value in expected.items():
        if actual[field_name] != expected_value:
            raise ValueError(f"{label} changed pinned {field_name}")


def _validate_chain(
    plan: ExecutionPlan,
    trajectory: StateTrajectoryEvidence,
    rendering: RenderStreamEvidence,
    packed: EpisodePackResult,
) -> None:
    _require_common_binding(
        plan,
        plan_record_id=trajectory.plan_record_id,
        scene=trajectory.scene,
        task=trajectory.task,
        seed=trajectory.seed,
        input_record_id=trajectory.input_record_id,
        label="trajectory",
    )
    if trajectory.simulator_record_id != _method_record_id(plan.simulator):
        raise ValueError("trajectory changed pinned simulator")
    if trajectory.policy_record_id != _method_record_id(plan.policy):
        raise ValueError("trajectory changed pinned policy")

    _require_common_binding(
        plan,
        plan_record_id=rendering.plan_record_id,
        scene=rendering.scene,
        task=rendering.task,
        seed=rendering.seed,
        input_record_id=rendering.input_record_id,
        label="rendering",
    )
    if rendering.trajectory_evidence_id != trajectory.record_id:
        raise ValueError("rendering does not consume the exact trajectory evidence")
    if rendering.renderer_record_id != _method_record_id(plan.renderer):
        raise ValueError("rendering changed pinned renderer")
    if len(rendering.frames) != len(trajectory.frames):
        raise ValueError("rendering frame count does not match the state trajectory")
    for state_frame, rendered_frame in zip(trajectory.frames, rendering.frames):
        if (
            rendered_frame.index,
            rendered_frame.timestamp_ns,
            rendered_frame.state_frame_id,
        ) != (state_frame.index, state_frame.timestamp_ns, state_frame.record_id):
            raise ValueError("rendering timeline does not exactly replay the state trajectory")
        if tuple(item.name for item in rendered_frame.modalities) != plan.modalities:
            raise ValueError("rendering modalities do not match the execution plan")

    _require_common_binding(
        plan,
        plan_record_id=packed.plan_record_id,
        scene=packed.scene,
        task=packed.task,
        seed=packed.seed,
        input_record_id=packed.input_record_id,
        label="episode packer",
    )
    if packed.trajectory_evidence_id != trajectory.record_id:
        raise ValueError("episode packer did not consume the exact trajectory evidence")
    if packed.render_evidence_id != rendering.record_id:
        raise ValueError("episode packer did not consume the exact render evidence")
    if packed.packer_record_id != _method_record_id(plan.packer):
        raise ValueError("episode packer changed pinned method")

    episode = packed.episode
    if episode.source is not EpisodeSource.SIMULATOR:
        raise ValueError("simulation execution can only finalize simulator episodes")
    episode_binding = (
        episode.run_id,
        episode.episode_id,
        episode.index,
        episode.scene,
        episode.task,
        episode.backend,
        episode.policy,
        episode.seed,
    )
    plan_binding = (
        plan.run_id,
        plan.episode_id,
        plan.episode_index,
        plan.scene,
        plan.task,
        plan.simulator,
        plan.policy,
        plan.seed,
    )
    if episode_binding != plan_binding:
        raise ValueError("packed episode changed its execution-plan binding")
    if len(episode.steps) != len(trajectory.frames):
        raise ValueError("packed episode step count does not match the execution timeline")
    for step, state_frame, rendered_frame in zip(
        episode.steps, trajectory.frames, rendering.frames
    ):
        if (
            step.index != state_frame.index
            or _seconds_to_nanoseconds(step.timestamp_seconds, "episode step timestamp")
            != state_frame.timestamp_ns
        ):
            raise ValueError("packed episode timeline does not match the state trajectory")
        if step.state != state_frame.state or step.action != state_frame.action:
            raise ValueError("packed episode state/action does not match the state trajectory")
        observations = {item.name: item.value for item in rendered_frame.modalities}
        if step.observations != observations:
            raise ValueError("packed episode observations do not match rendered artifacts")


class VerifiedEpisodeBundle(DomainModel):
    """A finalized Episode plus the complete evidence chain that justified it."""

    schema_name: Literal["physrealx.verified-episode-bundle/1"] = Field(
        default="physrealx.verified-episode-bundle/1",
        alias="schema",
        serialization_alias="schema",
    )
    schema_version: Literal[1] = 1
    status: Literal["FINALIZED"] = "FINALIZED"
    plan: ExecutionPlan
    trajectory: StateTrajectoryEvidence
    rendering: RenderStreamEvidence
    packed: EpisodePackResult
    validation: ExecutionValidationReport

    @model_validator(mode="after")
    def validate_bundle(self) -> VerifiedEpisodeBundle:
        _validate_chain(self.plan, self.trajectory, self.rendering, self.packed)
        expected = (
            self.plan.record_id,
            self.plan.input_record_id,
            self.trajectory.record_id,
            self.rendering.record_id,
            self.packed.record_id,
            self.packed.episode.record_id,
            _VALIDATION_CHECKS,
        )
        actual = (
            self.validation.plan_record_id,
            self.validation.input_record_id,
            self.validation.trajectory_evidence_id,
            self.validation.render_evidence_id,
            self.validation.pack_result_id,
            self.validation.episode_record_id,
            self.validation.checks,
        )
        if actual != expected:
            raise ValueError("execution validation report does not match the evidence chain")
        return self

    @property
    def episode(self) -> Episode:
        return self.packed.episode

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


class VerifiedEpisodeArtifact(DomainModel):
    """A finalized bundle paired with its actually materialized canonical JSON."""

    schema_name: Literal["physrealx.verified-episode-artifact/1"] = Field(
        default="physrealx.verified-episode-artifact/1",
        alias="schema",
        serialization_alias="schema",
    )
    schema_version: Literal[1] = 1
    episode_id: str
    bundle: VerifiedEpisodeBundle
    artifact: ArtifactRef

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        return _uuid(value, "verified episode artifact id")

    @model_validator(mode="after")
    def validate_materialization(self) -> VerifiedEpisodeArtifact:
        if self.episode_id != self.bundle.episode.episode_id:
            raise ValueError("verified episode artifact id does not match its bundle")
        payload = canonical_json_bytes(self.bundle)
        if self.artifact.size_bytes != len(payload):
            raise ValueError("verified episode artifact size does not match canonical JSON")
        if self.artifact.media_type != "application/json":
            raise ValueError("verified episode artifacts must use application/json")
        return self


class VerifiedEpisodeCollection(DomainModel):
    """Deterministically ordered manifest of materialized, verified episodes."""

    schema_name: Literal["physrealx.verified-episode-collection/1"] = Field(
        default="physrealx.verified-episode-collection/1",
        alias="schema",
        serialization_alias="schema",
    )
    schema_version: Literal[1] = 1
    status: Literal["FINALIZED"] = "FINALIZED"
    episodes: tuple[VerifiedEpisodeArtifact, ...]

    @model_validator(mode="after")
    def validate_collection(self) -> VerifiedEpisodeCollection:
        if not self.episodes:
            raise ValueError("verified episode collection cannot be empty")
        episode_ids = tuple(item.episode_id for item in self.episodes)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("verified episode collection ids must be unique")
        if episode_ids != tuple(sorted(episode_ids)):
            raise ValueError("verified episodes must use canonical episode-id order")
        artifact_ids = tuple(item.artifact.artifact_id for item in self.episodes)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("verified episode artifacts must be unique")
        return self

    @property
    def record_id(self) -> str:
        return canonical_record_id(self)


class ExecutionLifecycle:
    """Validate registered execution evidence; never execute an algorithm itself."""

    @staticmethod
    def validate(
        plan: ExecutionPlan,
        trajectory: StateTrajectoryEvidence,
        rendering: RenderStreamEvidence,
        packed: EpisodePackResult,
    ) -> ExecutionValidationReport:
        _validate_chain(plan, trajectory, rendering, packed)
        return ExecutionValidationReport(
            plan_record_id=plan.record_id,
            input_record_id=plan.input_record_id,
            trajectory_evidence_id=trajectory.record_id,
            render_evidence_id=rendering.record_id,
            pack_result_id=packed.record_id,
            episode_record_id=packed.episode.record_id,
            checks=_VALIDATION_CHECKS,
        )

    @staticmethod
    def finalize(
        plan: ExecutionPlan,
        trajectory: StateTrajectoryEvidence,
        rendering: RenderStreamEvidence,
        packed: EpisodePackResult,
    ) -> VerifiedEpisodeBundle:
        validation = ExecutionLifecycle.validate(plan, trajectory, rendering, packed)
        return VerifiedEpisodeBundle(
            plan=plan,
            trajectory=trajectory,
            rendering=rendering,
            packed=packed,
            validation=validation,
        )

    @staticmethod
    def collect(
        episodes: tuple[VerifiedEpisodeArtifact, ...],
    ) -> VerifiedEpisodeCollection:
        """Create a canonical manifest without pretending to persist bundle bytes."""

        return VerifiedEpisodeCollection(
            episodes=tuple(sorted(episodes, key=lambda item: item.episode_id))
        )
