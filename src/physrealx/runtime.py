"""Application service for compiling and executing reproducible local runs."""

from __future__ import annotations

import math
import uuid
from contextlib import ExitStack
from fnmatch import fnmatchcase
from typing import TypeVar

from pydantic import BaseModel

from physrealx.contracts import (
    ArtifactStore,
    Frame,
    Policy,
    Repository,
    SimulationBackend,
    SimulationSession,
)
from physrealx.domain import (
    AssetSpec,
    DatasetManifest,
    DataValue,
    Episode,
    EpisodeSource,
    EpisodeStep,
    EpisodeSummary,
    Predicate,
    ResourceType,
    RunRecord,
    RunResult,
    RunSpec,
    RunStatus,
    SceneSpec,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError, PhysRealXError, ResourceConflictError
from physrealx.plugins import PluginRegistry

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def load_episode(store: ArtifactStore, summary: EpisodeSummary) -> Episode:
    """Load one immutable episode and verify the identity and label in its summary."""
    if summary.artifact is None:
        raise DomainValidationError("episode summary has no published artifact")
    episode = Episode.model_validate_json(store.get_bytes(summary.artifact))
    if (
        episode.episode_id,
        episode.run_id,
        episode.index,
        episode.success,
        len(episode.steps),
        episode.failure_reason,
    ) != (
        summary.episode_id,
        summary.run_id,
        summary.index,
        summary.success,
        summary.step_count,
        summary.failure_reason,
    ):
        raise DomainValidationError("episode summary differs from its immutable payload")
    return episode


class RunExecutor:
    def __init__(
        self,
        repository: Repository,
        artifacts: ArtifactStore,
        plugins: PluginRegistry,
    ) -> None:
        self._repository = repository
        self._artifacts = artifacts
        self._plugins = plugins

    def compile(self, spec: RunSpec) -> tuple[RunSpec, SceneSpec, dict[str, AssetSpec], TaskSpec]:
        scene_ref, scene = self._resolve(spec.scene, SceneSpec)
        task_ref, task = self._resolve(spec.task, TaskSpec)
        if task.scene.resource_id != scene.resource_id or task.scene.version != scene.version:
            raise DomainValidationError("task scene does not match the run scene")
        if task.scene.artifact_id is not None and task.scene.artifact_id != scene_ref.artifact_id:
            raise DomainValidationError("task scene artifact does not match the catalog scene")
        assets = self._resolve_scene_assets(scene)
        if task.subject_entity not in assets:
            raise DomainValidationError("task subject_entity does not exist in the scene")
        known_observations = {item.name for item in task.observations}
        unknown = set(spec.record_signals) - known_observations
        if unknown:
            raise DomainValidationError(
                f"run requests undeclared observation signals: {sorted(unknown)!r}"
            )
        self._plugins.backend(spec.backend.name)
        self._plugins.resolve_method(spec.backend)
        self._plugins.policy_descriptor(spec.policy.name)
        self._plugins.resolve_method(spec.policy)
        for method in (spec.backend, spec.policy):
            if method.checkpoint is not None and not self._artifacts.verify(method.checkpoint):
                raise DomainValidationError(
                    f"method checkpoint is unavailable for {method.capability.value}:{method.name}"
                )
        resolved = spec.model_copy(
            update={
                "scene": scene_ref,
                "task": task_ref,
            }
        )
        return resolved, scene, assets, task

    def execute(self, spec: RunSpec) -> RunRecord:
        resolved, scene, assets, task = self.compile(spec)
        existing = self._repository.create_run(resolved)
        if existing.status is RunStatus.SUCCEEDED:
            return existing
        if existing.status is not RunStatus.PENDING:
            raise ResourceConflictError(
                f"run {resolved.run_id!r} cannot be executed from {existing.status.value}"
            )
        self._repository.transition_run(resolved.run_id, RunStatus.RUNNING)
        summaries: list[EpisodeSummary] = []
        try:
            backend = self._plugins.backend(resolved.backend.name)
            self._preflight(backend.capabilities(), task)
            with ExitStack() as resources:
                session = backend.create_session(scene, assets, task, resolved.backend.config)
                resources.callback(session.close)
                policy = self._plugins.policy(resolved.policy.name, resolved.policy.config)
                resources.callback(policy.close)
                for index in range(resolved.episode_count):
                    episode = self._rollout(
                        resolved,
                        task,
                        index,
                        backend=backend,
                        session=session,
                        policy=policy,
                    )
                    artifact = self._artifacts.put_json(
                        episode,
                        media_type="application/vnd.physrealx.episode+json",
                    )
                    summary = EpisodeSummary(
                        episode_id=episode.episode_id,
                        run_id=episode.run_id,
                        index=episode.index,
                        success=episode.success,
                        step_count=len(episode.steps),
                        artifact=artifact,
                        failure_reason=episode.failure_reason,
                    )
                    self._repository.publish_episode(summary)
                    summaries.append(summary)

            dataset = DatasetManifest(
                resource_id=resolved.dataset_id,
                version=resolved.dataset_version,
                episodes=tuple(summaries),
                source_runs=(resolved.run_id,),
                observation_schema=tuple(
                    item for item in task.observations if item.name in resolved.record_signals
                ),
                action_schema=task.action.signal,
                total_steps=sum(item.step_count for item in summaries),
                metadata={
                    "scene_artifact_id": resolved.scene.artifact_id or "",
                    "task_artifact_id": resolved.task.artifact_id or "",
                    "backend_method": resolved.backend.model_dump(mode="json"),
                    "policy_method": resolved.policy.model_dump(mode="json"),
                },
            )
            dataset_artifact = self._artifacts.put_json(
                dataset,
                media_type="application/vnd.physrealx.dataset+json",
            )
            dataset_ref = dataset.reference(dataset_artifact.artifact_id)
            result = RunResult(
                run_id=resolved.run_id,
                status=RunStatus.SUCCEEDED,
                episode_count=len(summaries),
                dataset=dataset_ref,
            )
            return self._repository.finalize_success(
                resolved.run_id,
                dataset_ref,
                result,
            )
        except BaseException as error:
            if isinstance(error, PhysRealXError):
                code = error.code
            elif isinstance(error, Exception):
                code = "execution_error"
            else:
                code = "execution_interrupted"
            result = RunResult(
                run_id=resolved.run_id,
                status=RunStatus.FAILED,
                episode_count=len(summaries),
                error_code=code,
                error_message=str(error) or type(error).__name__,
            )
            current = self._repository.get_run(resolved.run_id)
            if current.status is RunStatus.RUNNING:
                self._repository.transition_run(
                    resolved.run_id,
                    RunStatus.FAILED,
                    result=result,
                )
            raise

    def load_episode(self, summary: EpisodeSummary) -> Episode:
        return load_episode(self._artifacts, summary)

    def load_dataset(self, reference: VersionedRef) -> DatasetManifest:
        if reference.resource_type is not ResourceType.DATASET or reference.artifact_id is None:
            raise DomainValidationError("load_dataset requires a resolved dataset reference")
        return DatasetManifest.model_validate_json(self._artifacts.get_bytes(reference.artifact_id))

    def _resolve(
        self, reference: VersionedRef, model: type[_ModelT]
    ) -> tuple[VersionedRef, _ModelT]:
        registered = self._repository.get_resource(
            reference.resource_type.value,
            reference.resource_id,
            reference.version,
        )
        if reference.artifact_id is not None and reference.artifact_id != registered.artifact_id:
            raise ResourceConflictError(
                f"requested {reference.resource_type.value}/{reference.resource_id}@"
                f"{reference.version} has a different catalog artifact"
            )
        if registered.artifact_id is None:
            raise DomainValidationError("catalog returned an unresolved resource")
        resolved = model.model_validate_json(self._artifacts.get_bytes(registered.artifact_id))
        resource_id = getattr(resolved, "resource_id", None)
        version = getattr(resolved, "version", None)
        if resource_id != registered.resource_id or version != registered.version:
            raise DomainValidationError("catalog reference and resource payload disagree")
        return registered, resolved

    def _resolve_scene_assets(self, scene: SceneSpec) -> dict[str, AssetSpec]:
        assets: dict[str, AssetSpec] = {}
        for entity in scene.entities:
            asset_ref, asset = self._resolve(entity.asset, AssetSpec)
            if (
                entity.asset.artifact_id is not None
                and entity.asset.artifact_id != asset_ref.artifact_id
            ):
                raise ResourceConflictError(
                    f"scene entity {entity.name!r} asset does not match the catalog"
                )
            assets[entity.name] = asset
        return assets

    @staticmethod
    def _preflight(capabilities: object, task: TaskSpec) -> None:
        action_kinds = set(getattr(capabilities, "action_kinds", ()))
        if task.action.kind not in action_kinds:
            raise DomainValidationError(
                f"backend does not support action kind {task.action.kind!r}"
            )
        observation_keys = set(getattr(capabilities, "observation_keys", ()))
        patterns = getattr(capabilities, "observation_patterns", ())
        missing = {
            item.name
            for item in task.observations
            if item.name not in observation_keys
            and not any(fnmatchcase(item.name, pattern) for pattern in patterns)
        }
        if missing:
            raise DomainValidationError(
                f"backend cannot produce required observations: {sorted(missing)!r}"
            )

    def _rollout(
        self,
        spec: RunSpec,
        task: TaskSpec,
        episode_index: int,
        *,
        backend: SimulationBackend,
        session: SimulationSession,
        policy: Policy,
    ) -> Episode:
        seed = spec.base_seed + episode_index
        steps: list[EpisodeStep] = []
        success = False
        final_events: tuple[str, ...] = ()
        final_terminal = False
        dt_seconds = session.dt_seconds
        if not math.isfinite(dt_seconds) or dt_seconds <= 0:
            raise DomainValidationError("simulation session dt_seconds must be finite and positive")
        policy.reset(seed)
        current = session.reset(seed)
        self._validate_frame(current, task)
        if not self._predicates_match(current, task.initial_conditions):
            raise DomainValidationError("simulation reset does not satisfy task initial conditions")
        for step_index in range(task.max_steps):
            action = policy.act(current, task, step_index)
            self._validate_action(action, task)
            result = session.step(action)
            steps.append(
                EpisodeStep(
                    index=step_index,
                    timestamp_seconds=step_index * dt_seconds,
                    observations=self._recorded(current, spec),
                    state=dict(current.state),
                    action=action,
                    reward=result.reward,
                    discount=task.discount,
                    is_first=step_index == 0,
                    events=result.events,
                )
            )
            self._validate_frame(result.frame, task)
            current = result.frame
            success = self._predicates_match(current, task.success_conditions)
            task_terminated = bool(task.termination_conditions) and self._predicates_match(
                current, task.termination_conditions
            )
            final_events = result.events
            final_terminal = (
                result.terminated or (success and spec.stop_on_success) or task_terminated
            )
            if final_terminal or result.truncated:
                break
        steps.append(
            EpisodeStep(
                index=len(steps),
                timestamp_seconds=len(steps) * dt_seconds,
                observations=self._recorded(current, spec),
                state=dict(current.state),
                action=None,
                reward=None,
                discount=None,
                is_last=True,
                is_terminal=final_terminal or success,
                events=final_events,
            )
        )

        episode_id = str(uuid.uuid5(uuid.UUID(spec.run_id), f"episode:{episode_index}"))
        return Episode(
            episode_id=episode_id,
            run_id=spec.run_id,
            index=episode_index,
            source=EpisodeSource.SIMULATOR,
            seed=seed,
            scene=spec.scene,
            task=spec.task,
            backend=spec.backend,
            policy=spec.policy,
            steps=tuple(steps),
            success=success,
            failure_reason=(
                None if success else "termination_condition" if final_terminal else "task_timeout"
            ),
            metadata={
                "dt_seconds": dt_seconds,
                "tags": list(spec.tags),
                "stop_on_success": spec.stop_on_success,
                "sensor_metadata": dict(current.metadata),
            },
        )

    @staticmethod
    def _validate_frame(frame: Frame, task: TaskSpec) -> None:
        for signal in task.observations:
            try:
                value = frame.observations[signal.name]
            except KeyError as error:
                raise DomainValidationError(
                    f"frame is missing observation {signal.name!r}"
                ) from error
            try:
                value.assert_matches(signal)
            except ValueError as error:
                raise DomainValidationError(str(error)) from error

    @staticmethod
    def _recorded(frame: Frame, spec: RunSpec) -> dict[str, DataValue]:
        return {name: frame.observations[name] for name in spec.record_signals}

    @staticmethod
    def _validate_action(action: DataValue, task: TaskSpec) -> None:
        try:
            action.assert_matches(task.action.signal)
        except ValueError as error:
            raise DomainValidationError(str(error)) from error
        if action.inline is None:
            raise DomainValidationError("bounded actions must use an inline numeric payload")
        for value, lower, upper in zip(
            action.inline,
            task.action.lower,
            task.action.upper,
        ):
            numeric = float(value)
            if not lower <= numeric <= upper:
                raise DomainValidationError("policy action is outside the declared bounds")

    @staticmethod
    def _predicates_match(frame: Frame, predicates: tuple[Predicate, ...]) -> bool:
        if not predicates:
            return True
        scalars: dict[str, float] = {}
        for name, value in frame.observations.items():
            if value.inline is not None and len(value.inline) == 1:
                scalars[name] = float(value.inline[0])
        try:
            return all(item.evaluate(scalars) for item in predicates)
        except KeyError as error:
            raise DomainValidationError(
                f"task predicate references unavailable observation {error.args[0]!r}"
            ) from error


__all__ = ["RunExecutor"]
