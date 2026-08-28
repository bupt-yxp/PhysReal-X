"""Domain-oriented application services exposed by the local platform kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physrealx.domain import (
    AssetSpec,
    DatasetManifest,
    Episode,
    EvaluationReport,
    ModelManifest,
    ResourceType,
    RunRecord,
    RunResult,
    RunSpec,
    SceneSpec,
    TaskSpec,
    TrainingSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class AssetsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: AssetSpec) -> VersionedRef:
        return self._platform._register_asset(spec)

    def get(self, reference: VersionedRef) -> AssetSpec:
        return self._platform._load_registered(reference, AssetSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.ASSET)


class ScenesService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: SceneSpec) -> VersionedRef:
        return self._platform._register_scene(spec)

    def get(self, reference: VersionedRef) -> SceneSpec:
        return self._platform._load_registered(reference, SceneSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.SCENE)


class TasksService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: TaskSpec) -> VersionedRef:
        return self._platform._register_task(spec)

    def get(self, reference: VersionedRef) -> TaskSpec:
        return self._platform._load_registered(reference, TaskSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.TASK)


class RunsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def submit(self, spec: RunSpec) -> RunRecord:
        return self._platform._executor.execute(spec)

    def get(self, run: str | RunRecord) -> RunRecord:
        run_id = run if isinstance(run, str) else run.run_id
        return self._platform.repository.get_run(run_id)

    def wait(self, run: str | RunRecord) -> RunResult:
        record = self.get(run)
        if record.result is None:
            raise DomainValidationError(f"run {record.run_id!r} has not reached a terminal state")
        return record.result

    def episodes(self, run: str | RunRecord) -> tuple[Episode, ...]:
        run_id = run if isinstance(run, str) else run.run_id
        return tuple(
            self._platform._executor.load_episode(summary)
            for summary in self._platform.repository.list_episodes(run_id)
        )


class DatasetsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def get(self, reference: VersionedRef) -> DatasetManifest:
        return self._platform._load_registered(reference, DatasetManifest)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.DATASET)


class TrainingService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def submit(self, spec: TrainingSpec) -> VersionedRef:
        return self._platform._train(spec)

    def get_model(self, reference: VersionedRef) -> ModelManifest:
        return self._platform._load_registered(reference, ModelManifest)

    def list_models(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.MODEL)


class EvaluationsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def submit(
        self,
        *,
        resource_id: str,
        version: str,
        model: VersionedRef,
        run: RunSpec,
        episode_count: int = 16,
        base_seed: int = 100_000,
    ) -> VersionedRef:
        return self._platform._evaluate(
            resource_id=resource_id,
            version=version,
            model=model,
            run=run,
            episode_count=episode_count,
            base_seed=base_seed,
        )

    def get_report(self, reference: VersionedRef) -> EvaluationReport:
        return self._platform._load_registered(reference, EvaluationReport)

    def list_reports(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.EVALUATION)


__all__ = [
    "AssetsService",
    "DatasetsService",
    "EvaluationsService",
    "RunsService",
    "ScenesService",
    "TasksService",
    "TrainingService",
]
