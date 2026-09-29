"""Physics execution and synchronized observation generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from physrealx.domain import (
    ArtifactRef,
    Episode,
    JSONValue,
    MethodCapability,
    MethodSpec,
    OperationManifest,
    RunRecord,
    RunResult,
    RunSpec,
    VersionedRef,
)
from physrealx.errors import (
    DomainValidationError,
)

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class SimulationsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def run(
        self,
        scene: VersionedRef,
        task: VersionedRef,
        *,
        backend: str | MethodSpec,
        policy: MethodSpec,
        dataset_id: str,
        dataset_version: str = "1.0.0",
        config: Mapping[str, JSONValue] | None = None,
        episode_count: int = 1,
        base_seed: int = 0,
        stop_on_success: bool = True,
        record_signals: tuple[str, ...] | None = None,
    ) -> RunRecord:
        """Execute a backend through the same lifecycle as runs.submit(RunSpec)."""
        if isinstance(backend, MethodSpec) and config is not None:
            raise DomainValidationError("config is already bound in the backend specification")
        selected = (
            self._platform.methods.bind(MethodCapability.SIMULATION_RUN, backend, config=config)
            if isinstance(backend, str)
            else backend
        )
        signals = (
            tuple(item.name for item in self._platform.tasks.get(task).observations)
            if record_signals is None
            else record_signals
        )
        return self._platform.runs.submit(
            RunSpec(
                scene=scene,
                task=task,
                backend=selected,
                policy=policy,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                episode_count=episode_count,
                base_seed=base_seed,
                stop_on_success=stop_on_success,
                record_signals=signals,
            )
        )


class RendersService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def generate(
        self,
        trajectory: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.RENDER_GENERATE,
            method,
            inputs={"state_trajectory": trajectory},
            config=config,
        )


class EventsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def synthesize(
        self,
        rgb_stream: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.EVENT_SYNTHESIZE,
            method,
            inputs={"rgb_stream": rgb_stream},
            config=config,
        )


class PointCloudsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def project(
        self,
        depth: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.POINT_CLOUD_PROJECT,
            method,
            inputs={"depth": depth},
            config=config,
        )


class EpisodesService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def pack(
        self,
        run: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.EPISODE_PACK,
            method,
            inputs={"run": run},
            config=config,
        )


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
