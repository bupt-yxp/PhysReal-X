"""High-level application facade for a local PhysReal-X workspace."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from physrealx.asset_ingestion import ArtModelingImporter
from physrealx.asset_validation import validate_asset
from physrealx.contracts import ArtifactStore, MethodExecutionOutput, Repository
from physrealx.digital_twin import (
    AssetValidationProfile,
    AssetValidationReport,
    ObservationSetManifest,
    SourceInspection,
)
from physrealx.domain import (
    AssetSpec,
    EvaluationReport,
    FailureCase,
    FailureCategory,
    JSONValue,
    OperationManifest,
    OperationOutput,
    OperationSpec,
    ResourceType,
    RunSpec,
    SceneEntitySpec,
    SceneSpec,
    TaskSpec,
    TrainingSpec,
    VersionedRef,
    canonical_record_id,
)
from physrealx.errors import (
    DomainValidationError,
    MethodExecutionError,
    PhysRealXError,
    ResourceConflictError,
    ResourceNotFoundError,
    SourcePackageError,
)
from physrealx.integrations.act import ACTPolicy, ACTTrainer
from physrealx.integrations.artgs import ArtGSMethodAdapter
from physrealx.integrations.colmap import create_colmap_adapter
from physrealx.integrations.physical import register_physical_methods
from physrealx.operation_validation import validate_operation_output
from physrealx.plugins import PluginRegistry
from physrealx.readiness import ReadinessService
from physrealx.recipe_executor import RecipeExecutor
from physrealx.reference import ReferenceDrawerBackend, ScriptedDrawerPolicy
from physrealx.runtime import RunExecutor
from physrealx.services import (
    AssetsService,
    DatasetsService,
    DeploymentsService,
    EpisodesService,
    EvaluationsService,
    EventsService,
    FailuresService,
    FeedbackService,
    MethodsService,
    ModelsService,
    OperationsService,
    PointCloudsService,
    RendersService,
    RunsService,
    ScenesService,
    SimulationsService,
    TasksService,
    TelemetryService,
    TrainingService,
    TrajectoriesService,
    WorldModelsService,
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
        self.methods = MethodsService(self)
        self.readiness = ReadinessService(self)
        self.recipes = RecipeExecutor(self)
        self.operations = OperationsService(self)
        self.assets = AssetsService(self)
        self.scenes = ScenesService(self)
        self.tasks = TasksService(self)
        self.trajectories = TrajectoriesService(self)
        self.simulations = SimulationsService(self)
        self.renders = RendersService(self)
        self.events = EventsService(self)
        self.pointclouds = PointCloudsService(self)
        self.episodes = EpisodesService(self)
        self.runs = RunsService(self)
        self.datasets = DatasetsService(self)
        self.training = TrainingService(self)
        self.models = ModelsService(self)
        self.world_models = WorldModelsService(self)
        self.evaluations = EvaluationsService(self)
        self.deployments = DeploymentsService(self)
        self.telemetry = TelemetryService(self)
        self.failures = FailuresService(self)
        self.feedback = FeedbackService(self)

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
        plugins.register_importer(ArtModelingImporter())
        plugins.register_adapter(ArtGSMethodAdapter())
        colmap = create_colmap_adapter()
        if colmap is not None:
            plugins.register_adapter(colmap)
        register_physical_methods(plugins, artifacts)
        plugins.register_backend(ReferenceDrawerBackend())
        plugins.register_policy(
            ScriptedDrawerPolicy.descriptor,
            ScriptedDrawerPolicy,
        )
        if find_spec("torch") is not None and find_spec("safetensors") is not None:
            plugins.register_trainer(TorchBehaviorCloningTrainer())
            plugins.register_policy(
                TorchBehaviorCloningPolicy.descriptor,
                lambda config: TorchBehaviorCloningPolicy(config, artifacts),
            )
            if find_spec("torchvision") is not None and find_spec("einops") is not None:
                plugins.register_trainer(ACTTrainer(artifacts))
                plugins.register_policy(
                    ACTPolicy.descriptor, lambda config: ACTPolicy(config, artifacts)
                )
        return plugins

    def _register_asset(self, spec: AssetSpec) -> VersionedRef:
        return self._register(spec, spec.reference(), "application/vnd.physrealx.asset+json")

    def _inspect_asset_source(self, source: Path, importer: str) -> SourceInspection:
        adapter = self.plugins.importer(importer)
        try:
            return adapter.inspect(source)
        except PhysRealXError:
            raise
        except Exception as error:
            raise SourcePackageError(
                f"observation importer {adapter.descriptor.name!r} failed during inspection"
            ) from error

    def _ingest_asset_source(
        self,
        source: Path,
        *,
        importer: str,
        resource_id: str,
        version: str,
        display_name: str,
        config: Mapping[str, JSONValue],
    ) -> VersionedRef:
        try:
            existing = self.repository.get_resource(
                ResourceType.OBSERVATION_SET.value, resource_id, version
            )
        except ResourceNotFoundError:
            pass
        else:
            manifest = self._load_registered(existing, ObservationSetManifest)
            if (
                manifest.source_uri != source.expanduser().resolve().as_uri()
                or manifest.importer != importer
                or manifest.metadata.get("import_config") != dict(config)
            ):
                raise ResourceConflictError(
                    "observation version already exists with a different or unaudited import "
                    "configuration; use a new explicit version"
                )
            return existing
        adapter = self.plugins.importer(importer)
        try:
            manifest = adapter.ingest(
                source,
                resource_id=resource_id,
                version=version,
                display_name=display_name,
                config=config,
                artifacts=self.artifacts,
            )
        except PhysRealXError:
            raise
        except Exception as error:
            raise SourcePackageError(
                f"observation importer {adapter.descriptor.name!r} failed during ingestion"
            ) from error
        if (
            manifest.resource_id != resource_id
            or manifest.version != version
            or manifest.importer != adapter.descriptor.name
            or manifest.importer_version != adapter.descriptor.version
        ):
            raise DomainValidationError("observation importer returned mismatched identity")
        referenced = tuple(
            artifact for stream in manifest.streams for artifact in stream.artifacts
        ) + tuple(item.artifact for item in manifest.attachments)
        if any(not self.artifacts.verify(item) for item in referenced):
            raise DomainValidationError("observation importer returned an unpublished artifact")
        return self._register(
            manifest,
            manifest.reference(),
            "application/vnd.physrealx.observation-set+json",
        )

    def _validate_asset(
        self,
        asset: AssetSpec,
        profile: AssetValidationProfile,
    ) -> AssetValidationReport:
        return validate_asset(asset, profile, self.artifacts)

    def _register_scene(self, spec: SceneSpec) -> VersionedRef:
        resolved_entities: list[SceneEntitySpec] = []
        for entity in spec.entities:
            asset_ref = self.repository.get_resource(
                ResourceType.ASSET.value,
                entity.asset.resource_id,
                entity.asset.version,
            )
            if (
                entity.asset.artifact_id is not None
                and entity.asset.artifact_id != asset_ref.artifact_id
            ):
                raise ResourceConflictError(
                    f"scene entity {entity.name!r} asset does not match the registry"
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
        if spec.scene.artifact_id is not None and spec.scene.artifact_id != scene_ref.artifact_id:
            raise ResourceConflictError("task scene does not match the registered scene")
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
        if registered.artifact_id != spec.dataset.artifact_id:
            raise ResourceConflictError("training dataset does not match the registry")
        dataset = self.datasets.get(registered)
        episodes = (self._executor.load_episode(summary) for summary in dataset.episodes)
        trainer = self.plugins.trainer(spec.trainer.name)
        self.plugins.resolve_method(spec.trainer)
        if spec.trainer.checkpoint is not None and not self.artifacts.verify(
            spec.trainer.checkpoint
        ):
            raise DomainValidationError("trainer checkpoint is not present in artifact storage")
        manifest = trainer.train(spec, dataset, episodes, self.artifacts)
        if (
            manifest.resource_id != spec.resource_id
            or manifest.version != spec.version
            or manifest.dataset != registered
        ):
            raise DomainValidationError("trainer returned a model outside the requested lineage")
        if manifest.trainer != spec.trainer:
            raise DomainValidationError("trainer returned mismatched method identity")
        self.plugins.policy_descriptor(manifest.policy.name)
        self.plugins.resolve_method(manifest.policy)
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
        if model.artifact_id != registered_model.artifact_id:
            raise ResourceConflictError("evaluation model does not match the registry")
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
                "policy": model_manifest.policy.model_copy(
                    update={
                        "config": {
                            **model_manifest.policy.config,
                            "model_artifact_id": registered_model.artifact_id or "",
                        }
                    }
                ).model_dump(mode="json"),
                "episode_count": episode_count,
                "base_seed": base_seed,
                "dataset_id": f"{resource_id}.episodes",
                "dataset_version": version,
            }
        )
        evaluation_payload.pop("run_id", None)
        evaluation_key = canonical_record_id(evaluation_payload)
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

    def _execute_operation(self, spec: OperationSpec) -> OperationManifest:
        """Execute and atomically publish one capability-method invocation."""

        try:
            existing_ref = self.repository.get_resource(
                ResourceType.OPERATION.value,
                spec.resource_id,
                spec.version,
            )
        except ResourceNotFoundError:
            pass
        else:
            existing = self._load_registered(existing_ref, OperationManifest)
            if existing.spec != spec:
                raise ResourceConflictError(
                    "operation identity already exists for a different specification"
                )
            return existing

        resolved_inputs = []
        for item in spec.inputs:
            if item.resource is not None:
                registered = self.repository.get_resource(
                    item.resource.resource_type.value,
                    item.resource.resource_id,
                    item.resource.version,
                )
                if registered.artifact_id != item.resource.artifact_id:
                    raise ResourceConflictError(
                        f"operation input {item.name!r} does not match the registry"
                    )
                resolved_inputs.append(item.model_copy(update={"resource": registered}))
            else:
                if item.artifact is None or not self.artifacts.verify(item.artifact):
                    raise DomainValidationError(
                        f"operation input artifact {item.name!r} is unavailable or corrupt"
                    )
                resolved_inputs.append(item)

        resolved = spec.model_copy(update={"inputs": tuple(resolved_inputs)})
        adapter = self.plugins.adapter(resolved.method)
        checkpoint = resolved.method.checkpoint
        if checkpoint is not None and not self.artifacts.verify(checkpoint):
            raise DomainValidationError("method checkpoint is unavailable or corrupt")

        started_at = datetime.now(timezone.utc)
        try:
            result = adapter.execute(resolved, self.artifacts)
        except PhysRealXError:
            raise
        except Exception as error:
            raise MethodExecutionError(
                f"method {resolved.method.capability.value}:{resolved.method.name!r} failed"
            ) from error
        completed_at = datetime.now(timezone.utc)
        if not isinstance(result, MethodExecutionOutput) or any(
            not isinstance(item, OperationOutput) for item in result.outputs
        ):
            raise MethodExecutionError("method adapter returned an incompatible result")
        expected_outputs = tuple((item.name, item.contract) for item in resolved.outputs)
        actual_outputs = tuple((item.name, item.contract) for item in result.outputs)
        if actual_outputs != expected_outputs:
            raise MethodExecutionError(
                "method adapter output names or contracts do not match the operation"
            )
        if any(not self.artifacts.verify(item.artifact) for item in result.outputs):
            raise MethodExecutionError("method adapter returned unpublished or invalid artifacts")
        for requirement, output in zip(resolved.outputs, result.outputs):
            validate_operation_output(self.artifacts, output, requirement)

        try:
            manifest = OperationManifest(
                resource_id=resolved.resource_id,
                version=resolved.version,
                spec=resolved,
                outputs=result.outputs,
                metrics=dict(result.metrics),
                started_at=started_at,
                completed_at=completed_at,
            )
        except ValueError as error:
            raise MethodExecutionError(
                "method adapter returned invalid execution evidence"
            ) from error
        self._register(
            manifest,
            manifest.reference(),
            "application/vnd.physrealx.operation+json",
        )
        return manifest

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
        try:
            existing = self.repository.get_resource(
                unresolved.resource_type.value,
                unresolved.resource_id,
                unresolved.version,
            )
        except ResourceNotFoundError:
            existing = None
        if existing is not None:
            current = type(model).model_validate_json(
                self.artifacts.get_bytes(existing.artifact_id or "")
            )
            if current != model:
                raise ResourceConflictError(
                    f"{unresolved.resource_type.value}/{unresolved.resource_id}@"
                    f"{unresolved.version} already contains a different resource"
                )
            return existing
        reference = self._store_resource(model, unresolved, media_type)
        if specialized == "model":
            return self.repository.register_model(reference)
        if specialized == "evaluation":
            return self.repository.register_evaluation(reference)
        return self.repository.register_resource(reference)

    def _store_resource(
        self,
        model: _ModelT,
        unresolved: VersionedRef,
        media_type: str,
    ) -> VersionedRef:
        """Store canonical resource bytes without making the resource visible."""

        artifact = self.artifacts.put_json(model, media_type=media_type)
        return unresolved.model_copy(update={"artifact_id": artifact.artifact_id})

    def _load_registered(self, reference: VersionedRef, model: type[_ModelT]) -> _ModelT:
        registered = self.repository.get_resource(
            reference.resource_type.value,
            reference.resource_id,
            reference.version,
        )
        if reference.artifact_id is not None and reference.artifact_id != registered.artifact_id:
            raise ResourceConflictError("resource artifact does not match the registry")
        if registered.artifact_id is None:
            raise DomainValidationError("registered resource is unresolved")
        value = model.model_validate_json(self.artifacts.get_bytes(registered.artifact_id))
        if (
            getattr(value, "resource_id", None) != registered.resource_id
            or getattr(value, "version", None) != registered.version
        ):
            raise DomainValidationError("resource payload identity does not match the registry")
        return value


__all__ = ["PhysRealX"]
