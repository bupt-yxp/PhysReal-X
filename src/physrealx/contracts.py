"""Narrow runtime ports implemented by simulation and learning adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from physrealx.digital_twin import (
    ObservationSetManifest,
    SourceInspection,
)
from physrealx.domain import (
    ArtifactRef,
    AssetSpec,
    DatasetManifest,
    DataValue,
    Episode,
    EpisodeSummary,
    JSONValue,
    MethodDescriptor,
    ModelManifest,
    OperationOutput,
    OperationSpec,
    RunRecord,
    RunResult,
    RunSpec,
    RunStatus,
    SceneSpec,
    TaskSpec,
    TrainingSpec,
    VersionedRef,
)

API_VERSION = "1"


@dataclass(frozen=True)
class BackendCapabilities:
    rigid_body: bool = True
    articulation: bool = True
    deformable: bool = False
    contact_force: bool = False
    batch_execution: bool = False
    deterministic_reset: bool = True
    observation_keys: tuple[str, ...] = ()
    observation_patterns: tuple[str, ...] = ()
    action_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class Frame:
    observations: Mapping[str, DataValue]
    state: Mapping[str, DataValue]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    frame: Frame
    reward: float
    terminated: bool
    truncated: bool = False
    events: tuple[str, ...] = ()


@runtime_checkable
class SimulationSession(Protocol):
    @property
    def dt_seconds(self) -> float: ...

    def reset(self, seed: int) -> Frame: ...

    def step(self, action: DataValue) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class SimulationBackend(Protocol):
    descriptor: MethodDescriptor

    def capabilities(self) -> BackendCapabilities: ...

    def create_session(
        self,
        scene: SceneSpec,
        assets: Mapping[str, AssetSpec],
        task: TaskSpec,
        config: Mapping[str, JSONValue],
    ) -> SimulationSession: ...


@runtime_checkable
class Policy(Protocol):
    descriptor: MethodDescriptor

    def reset(self, seed: int) -> None: ...

    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue: ...

    def close(self) -> None: ...


@runtime_checkable
class Trainer(Protocol):
    descriptor: MethodDescriptor

    def train(
        self,
        spec: TrainingSpec,
        dataset: DatasetManifest,
        episodes: Iterable[Episode],
        artifacts: ArtifactWriter,
    ) -> ModelManifest: ...


@runtime_checkable
class ObservationImporter(Protocol):
    """Adapter for one immutable, already captured source-package format."""

    descriptor: MethodDescriptor

    def inspect(self, source: Path) -> SourceInspection: ...

    def ingest(
        self,
        source: Path,
        *,
        resource_id: str,
        version: str,
        display_name: str,
        config: Mapping[str, JSONValue],
        artifacts: ArtifactWriter,
    ) -> ObservationSetManifest: ...


@dataclass(frozen=True)
class MethodExecutionOutput:
    """Unpublished outputs returned by one capability adapter."""

    outputs: tuple[OperationOutput, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class MethodProbe:
    """Side-effect-free local readiness result for one installed method."""

    ready: bool
    detail: str


@runtime_checkable
class MethodAdapter(Protocol):
    """Executable adapter for one and only one ``MethodCapability``."""

    descriptor: MethodDescriptor

    def execute(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput: ...


@runtime_checkable
class ArtifactWriter(Protocol):
    def put_bytes(self, payload: bytes, *, media_type: str) -> ArtifactRef: ...

    def put_file(self, source: Path, *, media_type: str) -> ArtifactRef: ...


@runtime_checkable
class ArtifactReader(Protocol):
    def get_bytes(self, reference: ArtifactRef | str) -> bytes: ...

    def materialize(self, reference: ArtifactRef | str, destination: Path) -> None: ...


@runtime_checkable
class ArtifactStore(ArtifactWriter, ArtifactReader, Protocol):
    def put_json(self, value: object, *, media_type: str = "application/json") -> ArtifactRef: ...

    def verify(self, reference: ArtifactRef | str) -> bool: ...


@runtime_checkable
class Repository(Protocol):
    def register_resource(self, reference: VersionedRef) -> VersionedRef: ...

    def register_resources(
        self,
        references: Iterable[VersionedRef],
    ) -> tuple[VersionedRef, ...]: ...

    def get_resource(self, resource_type: str, resource_id: str, version: str) -> VersionedRef: ...

    def list_resources(
        self,
        *,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> list[VersionedRef]: ...

    def create_run(self, spec: RunSpec) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord: ...

    def transition_run(
        self,
        run_id: str,
        target: RunStatus,
        *,
        result: RunResult | None = None,
    ) -> RunRecord: ...

    def publish_episode(self, summary: EpisodeSummary) -> EpisodeSummary: ...

    def list_episodes(self, run_id: str) -> list[EpisodeSummary]: ...

    def register_dataset(self, reference: VersionedRef) -> VersionedRef: ...

    def get_dataset(self, resource_id: str, version: str) -> VersionedRef: ...

    def register_model(self, reference: VersionedRef) -> VersionedRef: ...

    def get_model(self, resource_id: str, version: str) -> VersionedRef: ...

    def register_evaluation(self, reference: VersionedRef) -> VersionedRef: ...

    def get_evaluation(self, resource_id: str, version: str) -> VersionedRef: ...

    def finalize_success(
        self,
        run_id: str,
        dataset: VersionedRef,
        result: RunResult,
    ) -> RunRecord: ...
