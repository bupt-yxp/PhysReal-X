"""High-level application facade for a local PhysReal-X workspace."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from physrealx.asset_ingestion import ArtModelingImporter
from physrealx.contracts import ArtifactStore, Repository
from physrealx.domain import (
    ResourceType,
    VersionedRef,
)
from physrealx.errors import (
    DomainValidationError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from physrealx.integrations.act import ACTPolicy, ACTTrainer
from physrealx.integrations.artgs import ArtGSMethodAdapter
from physrealx.integrations.colmap import create_colmap_adapter
from physrealx.integrations.dtart import DTArtMethodAdapter
from physrealx.integrations.iql import IQLPolicy, IQLTrainer
from physrealx.integrations.physical import register_physical_methods
from physrealx.plugins import PluginRegistry
from physrealx.readiness import ReadinessService
from physrealx.recipe_executor import RecipeExecutor
from physrealx.reference import ReferenceDrawerBackend, ScriptedDrawerPolicy
from physrealx.runtime import RunExecutor
from physrealx.services.assets import AssetsService
from physrealx.services.datasets import DatasetsService
from physrealx.services.feedback import (
    DeploymentsService,
    EvaluationsService,
    FailuresService,
    FeedbackService,
    TelemetryService,
)
from physrealx.services.learning import ModelsService, TrainingService, WorldModelsService
from physrealx.services.methods import MethodsService
from physrealx.services.operations import OperationsService
from physrealx.services.scenes import ScenesService, TasksService, TrajectoriesService
from physrealx.services.simulation import (
    EpisodesService,
    EventsService,
    PointCloudsService,
    RendersService,
    RunsService,
    SimulationsService,
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
        plugins.register_adapter(DTArtMethodAdapter())
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
            plugins.register_trainer(IQLTrainer(artifacts))
            plugins.register_policy(
                IQLPolicy.descriptor,
                lambda config: IQLPolicy(config, artifacts),
            )
            if find_spec("torchvision") is not None and find_spec("einops") is not None:
                plugins.register_trainer(ACTTrainer(artifacts))
                plugins.register_policy(
                    ACTPolicy.descriptor, lambda config: ACTPolicy(config, artifacts)
                )
        return plugins

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
