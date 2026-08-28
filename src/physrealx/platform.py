"""High-level application facade for a local PhysReal-X workspace."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from physrealx.contracts import ArtifactStore, Repository
from physrealx.domain import (
    AssetSpec,
    EvaluationReport,
    FailureCase,
    FailureCategory,
    ResourceType,
    RunSpec,
    SceneEntitySpec,
    SceneSpec,
    TaskSpec,
    TrainingSpec,
    VersionedRef,
    sha256_digest,
)
from physrealx.errors import DomainValidationError, ResourceConflictError
from physrealx.plugins import PluginDescriptor, PluginRegistry
from physrealx.reference import ReferenceDrawerBackend, ScriptedDrawerPolicy
from physrealx.runtime import RunExecutor
from physrealx.services import (
    AssetsService,
    DatasetsService,
    EvaluationsService,
    RunsService,
    ScenesService,
    TasksService,
    TrainingService,
)
from physrealx.storage import LocalArtifactStore, SQLiteRepository
from physrealx.training import TorchBehaviorCloningPolicy, TorchBehaviorCloningTrainer

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class PhysRealX:
    """A process-local facade backed by durable workspace state."""

    def __init__(
        self,
        workspace: Path,
        repository: Repository,
        artifacts: ArtifactStore,
        plugins: PluginRegistry,
    ) -> None:
        self.workspace = workspace
        self.repository = repository
        self.artifacts = artifacts
        self.plugins = plugins
        self._executor = RunExecutor(repository, artifacts, plugins)
        self.assets = AssetsService(self)
        self.scenes = ScenesService(self)
        self.tasks = TasksService(self)
        self.runs = RunsService(self)
        self.datasets = DatasetsService(self)
        self.training = TrainingService(self)
        self.evaluations = EvaluationsService(self)

    @classmethod
    def open(cls, workspace: str | Path, *, discover_plugins: bool = True) -> PhysRealX:
        root = Path(workspace).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        artifacts = LocalArtifactStore(root / "artifacts")
        repository = SQLiteRepository(root / "registry.sqlite3")
        plugins = cls._default_plugins(artifacts)
        if discover_plugins:
            plugins.discover()
        return cls(root, repository, artifacts, plugins)

    @staticmethod
    def _default_plugins(artifacts: ArtifactStore) -> PluginRegistry:
        plugins = PluginRegistry()
        plugins.register_backend(ReferenceDrawerBackend())
        plugins.register_policy(
            PluginDescriptor(
                kind="policy",
                name=ScriptedDrawerPolicy.name,
                version=ScriptedDrawerPolicy.plugin_version,
                api_version=ScriptedDrawerPolicy.api_version,
            ),
            ScriptedDrawerPolicy,
        )
        plugins.register_trainer(TorchBehaviorCloningTrainer())
        plugins.register_policy(
            PluginDescriptor(
                kind="policy",
                name=TorchBehaviorCloningPolicy.name,
                version=TorchBehaviorCloningPolicy.plugin_version,
                api_version=TorchBehaviorCloningPolicy.api_version,
            ),
            lambda config: TorchBehaviorCloningPolicy(config, artifacts),
        )
        return plugins

    def _register_asset(self, spec: AssetSpec) -> VersionedRef:
        return self._register(spec, spec.reference(), "application/vnd.physrealx.asset+json")

    def _register_scene(self, spec: SceneSpec) -> VersionedRef:
        resolved_entities: list[SceneEntitySpec] = []
        for entity in spec.entities:
            asset_ref = self.repository.get_resource(
                ResourceType.ASSET.value,
                entity.asset.resource_id,
                entity.asset.version,
            )
            if entity.asset.digest is not None and entity.asset.digest != asset_ref.digest:
                raise ResourceConflictError(
                    f"scene entity {entity.name!r} asset digest does not match the registry"
                )
            resolved_entities.append(entity.model_copy(update={"asset": asset_ref}))
        resolved = spec.model_copy(update={"entities": tuple(resolved_entities)})
        return self._register(
            resolved,
            resolved.reference(),
            "application/vnd.physrealx.scene+json",
        )

    def _register_task(self, spec: TaskSpec) -> VersionedRef:
        scene_ref = self.repository.get_resource(
            ResourceType.SCENE.value,
            spec.scene.resource_id,
            spec.scene.version,
        )
        if spec.scene.digest is not None and spec.scene.digest != scene_ref.digest:
            raise ResourceConflictError("task scene digest does not match the registered scene")
        scene = self._load_registered(scene_ref, SceneSpec)
        if spec.subject_entity not in {entity.name for entity in scene.entities}:
            raise DomainValidationError("task subject_entity does not exist in the scene")
        resolved = spec.model_copy(update={"scene": scene_ref})
        return self._register(
            resolved,
            resolved.reference(),
            "application/vnd.physrealx.task+json",
        )

    def _train(self, spec: TrainingSpec) -> VersionedRef:
        registered = self.repository.get_dataset(spec.dataset.resource_id, spec.dataset.version)
        if registered.digest != spec.dataset.digest:
            raise ResourceConflictError("training dataset digest does not match the registry")
        dataset = self.datasets.get(registered)
        episodes = (self._executor.load_episode(summary) for summary in dataset.episodes)
        trainer = self.plugins.trainer(spec.trainer)
        manifest = trainer.train(spec, dataset, episodes, self.artifacts)
        if (
            manifest.resource_id != spec.resource_id
            or manifest.version != spec.version
            or manifest.dataset != registered
        ):
            raise DomainValidationError("trainer returned a model outside the requested lineage")
        if manifest.trainer != trainer.name or manifest.trainer_version != trainer.plugin_version:
            raise DomainValidationError("trainer returned mismatched plugin identity")
        policy = self.plugins.policy_descriptor(manifest.policy)
        if manifest.policy_version != policy.version:
            raise DomainValidationError("trainer returned an unavailable inference policy version")
        if manifest.action_signal != dataset.action_schema:
            raise DomainValidationError("trainer changed the dataset action schema")
        if not self.artifacts.verify(manifest.weights):
            raise DomainValidationError("trainer returned unpublished or corrupt model weights")
        return self._register(
            manifest,
            manifest.reference(),
            "application/vnd.physrealx.model+json",
            specialized="model",
        )

    def _evaluate(
        self,
        *,
        resource_id: str,
        version: str,
        model: VersionedRef,
        run: RunSpec,
        episode_count: int = 16,
        base_seed: int = 100_000,
    ) -> VersionedRef:
        if model.resource_type is not ResourceType.MODEL or not model.resolved:
            raise DomainValidationError("evaluation requires a resolved model reference")
        registered_model = self.repository.get_model(model.resource_id, model.version)
        if model.digest != registered_model.digest:
            raise ResourceConflictError("evaluation model digest does not match the registry")
        model_manifest = self.training.get_model(registered_model)
        task = self.tasks.get(run.task)
        task_observations = {item.name: item for item in task.observations}
        incompatible = {
            item.name
            for item in model_manifest.observation_signals
            if task_observations.get(item.name) != item
        }
        if incompatible or model_manifest.action_signal != task.action.signal:
            raise DomainValidationError(
                "model observation/action schema is incompatible with the evaluation task"
            )
        evaluation_payload = run.model_dump(mode="json")
        evaluation_payload.update(
            {
                "policy": model_manifest.policy,
                "policy_version": model_manifest.policy_version,
                "policy_config": {"model_digest": registered_model.digest or ""},
                "episode_count": episode_count,
                "base_seed": base_seed,
                "dataset_id": f"{resource_id}.episodes",
                "dataset_version": version,
            }
        )
        evaluation_payload.pop("run_id", None)
        evaluation_key = sha256_digest(evaluation_payload)
        evaluation_payload["run_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, evaluation_key))
        evaluation_run = RunSpec.model_validate(evaluation_payload)
        record = self.runs.submit(evaluation_run)
        if record.result is None or record.result.dataset is None:
            raise DomainValidationError("evaluation run completed without a dataset")
        dataset = self.datasets.get(record.result.dataset)
        failures = tuple(
            FailureCase(
                episode_id=item.episode_id,
                category=(
                    FailureCategory.TIMEOUT
                    if item.failure_reason == "task_timeout"
                    else FailureCategory.UNKNOWN
                ),
                message=item.failure_reason or "task failed",
            )
            for item in dataset.episodes
            if not item.success
        )
        successes = sum(item.success for item in dataset.episodes)
        report = EvaluationReport(
            resource_id=resource_id,
            version=version,
            model=registered_model,
            dataset=record.result.dataset,
            success_rate=successes / len(dataset.episodes),
            metrics={
                "success_count": float(successes),
                "episode_count": float(len(dataset.episodes)),
            },
            failures=failures,
            created_at=dataset.created_at,
        )
        return self._register(
            report,
            report.reference(),
            "application/vnd.physrealx.evaluation+json",
            specialized="evaluation",
        )

    def _list_resources(self, resource_type: ResourceType) -> list[VersionedRef]:
        return self.repository.list_resources(resource_type=resource_type.value)

    def _register(
        self,
        model: _ModelT,
        unresolved: VersionedRef,
        media_type: str,
        *,
        specialized: str | None = None,
    ) -> VersionedRef:
        artifact = self.artifacts.put_json(model, media_type=media_type)
        if artifact.digest != sha256_digest(model):
            raise DomainValidationError("artifact store changed canonical resource bytes")
        reference = unresolved.model_copy(update={"digest": artifact.digest})
        if specialized == "model":
            return self.repository.register_model(reference)
        if specialized == "evaluation":
            return self.repository.register_evaluation(reference)
        return self.repository.register_resource(reference)

    def _load_registered(self, reference: VersionedRef, model: type[_ModelT]) -> _ModelT:
        registered = self.repository.get_resource(
            reference.resource_type.value,
            reference.resource_id,
            reference.version,
        )
        if reference.digest is not None and reference.digest != registered.digest:
            raise ResourceConflictError("resource digest does not match the registry")
        if registered.digest is None:
            raise DomainValidationError("registered resource is unresolved")
        value = model.model_validate_json(self.artifacts.get_bytes(registered.digest))
        if (
            getattr(value, "resource_id", None) != registered.resource_id
            or getattr(value, "version", None) != registered.version
        ):
            raise DomainValidationError("resource payload identity does not match the registry")
        if sha256_digest(value) != registered.digest:
            raise DomainValidationError("resource bytes do not match the registered digest")
        return value


__all__ = ["PhysRealX"]
