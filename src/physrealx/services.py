"""Domain-oriented application services exposed by the local platform kernel."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from physrealx.digital_twin import (
    AssetValidationProfile,
    AssetValidationReport,
    DigitalTwinBuildManifest,
    DigitalTwinBuildResult,
    DigitalTwinBuildSpec,
    ObservationSetManifest,
    SourceInspection,
)
from physrealx.domain import (
    ArtifactRef,
    AssetSpec,
    DatasetManifest,
    Episode,
    EvaluationReport,
    JSONValue,
    MethodCapability,
    MethodDescriptor,
    MethodSpec,
    ModelManifest,
    OperationInput,
    OperationManifest,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
    RunRecord,
    RunResult,
    RunSpec,
    SceneSpec,
    TaskSpec,
    TrainingSpec,
    VersionedRef,
    sha256_digest,
)
from physrealx.errors import DomainValidationError
from physrealx.plugins import MethodInstallation

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

    def inspect_source(
        self,
        source: str | Path,
        *,
        importer: str = "artmodeling",
    ) -> SourceInspection:
        return self._platform._inspect_asset_source(Path(source), importer)

    def ingest(
        self,
        source: str | Path,
        *,
        resource_id: str,
        version: str,
        display_name: str,
        importer: str = "artmodeling",
        config: Mapping[str, JSONValue] | None = None,
    ) -> VersionedRef:
        return self._platform._ingest_asset_source(
            Path(source),
            importer=importer,
            resource_id=resource_id,
            version=version,
            display_name=display_name,
            config=config or {},
        )

    def get_observations(self, reference: VersionedRef) -> ObservationSetManifest:
        return self._platform._load_registered(reference, ObservationSetManifest)

    def list_observations(self) -> builtins.list[VersionedRef]:
        return self._platform._list_resources(ResourceType.OBSERVATION_SET)

    def build(self, spec: DigitalTwinBuildSpec) -> DigitalTwinBuildResult:
        return self._platform._build_digital_twin(spec)

    def get_build(self, reference: VersionedRef) -> DigitalTwinBuildManifest:
        return self._platform._load_registered(reference, DigitalTwinBuildManifest)

    def list_builds(self) -> builtins.list[VersionedRef]:
        return self._platform._list_resources(ResourceType.ASSET_BUILD)

    def validate(
        self,
        asset: AssetSpec | VersionedRef,
        *,
        profile: AssetValidationProfile | str = AssetValidationProfile.DIGITAL_TWIN,
    ) -> AssetValidationReport:
        target = asset if isinstance(asset, AssetSpec) else self.get(asset)
        try:
            selected_profile = AssetValidationProfile(profile)
        except ValueError as error:
            raise DomainValidationError(f"unknown asset validation profile: {profile!r}") from error
        return self._platform._validate_asset(target, selected_profile)

    def articulate(
        self,
        source: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.ARTICULATION_RECOVER,
            method,
            inputs={"source": source},
            config=config,
            outputs={"articulation": "ArticulationModel"},
        )

    def build_collision(
        self,
        asset: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.COLLISION_BUILD,
            method,
            inputs={"asset": asset},
            config=config,
            outputs={"collision_geometry": "CollisionGeometry"},
        )

    def calibrate_physics(
        self,
        asset: VersionedRef | ArtifactRef,
        evidence: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.PHYSICS_CALIBRATE,
            method,
            inputs={"asset": asset, "evidence": evidence},
            config=config,
            outputs={"identified_physics": "IdentifiedPhysics"},
        )

    def package(
        self,
        asset: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.ASSET_PACKAGE,
            method,
            inputs={"asset": asset},
            config=config,
            outputs={
                "asset_package": "PackagedAsset",
                "asset_manifest": "AssetPackageManifest",
            },
        )


class MethodsService:
    """Inspect and bind the exact implementations installed in this workspace."""

    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, descriptor: MethodDescriptor) -> None:
        self._platform.plugins.register_method(descriptor)

    def get(self, capability: MethodCapability, name: str) -> MethodDescriptor:
        return self._platform.plugins.method_descriptor(capability, name)

    def list(
        self,
        capability: MethodCapability | None = None,
    ) -> tuple[MethodDescriptor, ...]:
        return self._platform.plugins.descriptors(capability)

    def inspect(self, capability: MethodCapability, name: str) -> MethodInstallation:
        return self._platform.plugins.installation(capability, name)

    def installations(
        self,
        capability: MethodCapability | None = None,
    ) -> tuple[MethodInstallation, ...]:
        return self._platform.plugins.installations(capability)

    def bind(
        self,
        capability: MethodCapability,
        name: str,
        *,
        checkpoint: ArtifactRef | None = None,
        config: Mapping[str, JSONValue] | None = None,
    ) -> MethodSpec:
        descriptor = self.get(capability, name)
        if checkpoint is not None and not self._platform.artifacts.verify(checkpoint):
            raise DomainValidationError("method checkpoint is not present in artifact storage")
        return descriptor.bind(checkpoint=checkpoint, config=config)

    def execute(
        self,
        capability: MethodCapability,
        method: str | MethodSpec,
        *,
        inputs: Mapping[str, VersionedRef | ArtifactRef] | None = None,
        config: Mapping[str, JSONValue] | None = None,
        arguments: Mapping[str, JSONValue] | None = None,
        outputs: Mapping[str, str],
        resource_id: str | None = None,
        version: str = "1.0.0",
    ) -> OperationManifest:
        """Bind, validate and execute one capability method with immutable inputs."""

        if isinstance(method, MethodSpec) and config is not None:
            raise DomainValidationError(
                "config must be bound in MethodSpec when an explicit MethodSpec is supplied"
            )
        selected = (
            self.bind(capability, method, config=config) if isinstance(method, str) else method
        )
        if selected.capability is not capability:
            raise DomainValidationError(
                f"service requires capability {capability.value!r}, got "
                f"{selected.capability.value!r}"
            )
        self._platform.plugins.resolve_method(selected)
        operation_inputs = tuple(
            OperationInput(
                name=name,
                resource=value if isinstance(value, VersionedRef) else None,
                artifact=value if isinstance(value, ArtifactRef) else None,
            )
            for name, value in sorted((inputs or {}).items())
        )
        output_requirements = tuple(
            OperationOutputRequirement(name=name, contract=contract)
            for name, contract in sorted(outputs.items())
        )
        identity_payload: dict[str, JSONValue] = {
            "method": selected.model_dump(mode="json"),
            "inputs": [item.model_dump(mode="json") for item in operation_inputs],
            "arguments": dict(arguments or {}),
            "outputs": [item.model_dump(mode="json") for item in output_requirements],
        }
        identity = resource_id or (
            f"operation.{capability.value}.{sha256_digest(identity_payload)[-16:]}"
        )
        return self._platform._execute_operation(
            OperationSpec(
                resource_id=identity,
                version=version,
                method=selected,
                inputs=operation_inputs,
                arguments=dict(arguments or {}),
                outputs=output_requirements,
            )
        )


class OperationsService:
    """Read immutable execution evidence emitted by capability services."""

    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def get(self, reference: VersionedRef) -> OperationManifest:
        if reference.resource_type is not ResourceType.OPERATION:
            raise DomainValidationError("operations.get requires an operation reference")
        return self._platform._load_registered(reference, OperationManifest)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.OPERATION)


class ScenesService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: SceneSpec) -> VersionedRef:
        return self._platform._register_scene(spec)

    def get(self, reference: VersionedRef) -> SceneSpec:
        return self._platform._load_registered(reference, SceneSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.SCENE)

    def generate(
        self,
        request: Mapping[str, JSONValue],
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.SCENE_GENERATE,
            method,
            config=config,
            arguments=request,
            outputs={"scene_draft": "SceneDraft"},
        )

    def compose(
        self,
        scene: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.SCENE_COMPOSE,
            method,
            inputs={"scene": scene},
            config=config,
            outputs={"scene_spec": "SceneSpec", "constraint_report": "ConstraintReport"},
        )


class TasksService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: TaskSpec) -> VersionedRef:
        return self._platform._register_task(spec)

    def get(self, reference: VersionedRef) -> TaskSpec:
        return self._platform._load_registered(reference, TaskSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.TASK)

    def compile(
        self,
        instruction: str,
        scene: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TASK_COMPILE,
            method,
            inputs={"scene": scene},
            config=config,
            arguments={"instruction": instruction},
            outputs={"task_spec": "TaskSpec"},
        )

    def validate(
        self,
        task: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TASK_VALIDATE,
            method,
            inputs={"task": task},
            config=config,
            outputs={"validation_report": "TaskValidationReport"},
        )


class TrajectoriesService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def generate(
        self,
        instance: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TRAJECTORY_GENERATE,
            method,
            inputs={"instance": instance},
            config=config,
            outputs={"trajectory": "Trajectory", "planning_report": "PlanningReport"},
        )


class SimulationsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def run(
        self,
        scene: VersionedRef,
        task: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.SIMULATION_RUN,
            method,
            inputs={"scene": scene, "task": task},
            config=config,
            outputs={
                "state_trajectory": "StateTrajectory",
                "simulation_report": "SimulationReport",
            },
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
            outputs={
                "render_streams": "RenderStreams",
                "camera_metadata": "CameraMetadata",
            },
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
            outputs={"event_stream": "EventStream"},
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
            outputs={"point_cloud": "PointCloud"},
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
            outputs={
                "episode_bundle": "EpisodeBundle",
                "episode_manifest": "EpisodeManifest",
            },
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


class DatasetsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def get(self, reference: VersionedRef) -> DatasetManifest:
        return self._platform._load_registered(reference, DatasetManifest)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.DATASET)

    def ingest(
        self,
        source: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.DATASET_INGEST,
            method,
            inputs={"source": source},
            config=config,
            outputs={"dataset": "DatasetBundle", "quality_report": "DataQualityReport"},
        )

    def snapshot(
        self,
        dataset: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.DATASET_SNAPSHOT,
            method,
            inputs={"dataset": dataset},
            config=config,
            outputs={"snapshot": "DatasetSnapshot", "snapshot_manifest": "SnapshotManifest"},
        )


class TrainingService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def submit(self, spec: TrainingSpec) -> VersionedRef:
        return self._platform._train(spec)

    def get_model(self, reference: VersionedRef) -> ModelManifest:
        return self._platform._load_registered(reference, ModelManifest)

    def list_models(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.MODEL)


class ModelsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(
        self,
        model: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.MODEL_REGISTER,
            method,
            inputs={"model": model},
            config=config,
            outputs={"model_registration": "ModelRegistration"},
        )


class WorldModelsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def rollout(
        self,
        model: VersionedRef | ArtifactRef,
        context: VersionedRef | ArtifactRef,
        actions: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.WORLD_MODEL_ROLLOUT,
            method,
            inputs={"actions": actions, "context": context, "model": model},
            config=config,
            outputs={"rollout": "WorldModelRollout", "uncertainty": "UncertaintyReport"},
        )


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

    def run(
        self,
        model: VersionedRef | ArtifactRef,
        suite: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.EVALUATION_RUN,
            method,
            inputs={"model": model, "suite": suite},
            config=config,
            outputs={
                "evaluation_report": "EvaluationReport",
                "episode_evidence": "EpisodeEvidence",
            },
        )


class DeploymentsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def stage(
        self,
        model: VersionedRef | ArtifactRef,
        robot: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.DEPLOYMENT_STAGE,
            method,
            inputs={"model": model, "robot": robot},
            config=config,
            outputs={
                "deployment_plan": "DeploymentPlan",
                "safety_report": "SafetyReport",
            },
        )


class TelemetryService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def record(
        self,
        run: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TELEMETRY_RECORD,
            method,
            inputs={"run": run},
            config=config,
            outputs={
                "telemetry": "TelemetryBundle",
                "telemetry_manifest": "TelemetryManifest",
            },
        )


class FailuresService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def analyze(
        self,
        run: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.FAILURE_ANALYZE,
            method,
            inputs={"run": run},
            config=config,
            outputs={"failure_report": "FailureReport"},
        )


class FeedbackService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def build_dataset(
        self,
        query: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.FEEDBACK_BUILD_DATASET,
            method,
            inputs={"query": query},
            config=config,
            outputs={"dataset": "DatasetBundle", "selection_report": "SelectionReport"},
        )


__all__ = [
    "AssetsService",
    "DatasetsService",
    "DeploymentsService",
    "EpisodesService",
    "EvaluationsService",
    "EventsService",
    "FailuresService",
    "FeedbackService",
    "MethodsService",
    "ModelsService",
    "OperationsService",
    "PointCloudsService",
    "RendersService",
    "RunsService",
    "ScenesService",
    "SimulationsService",
    "TasksService",
    "TelemetryService",
    "TrainingService",
    "TrajectoriesService",
    "WorldModelsService",
]
