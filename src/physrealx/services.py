"""Domain-oriented application services exposed by the local platform kernel."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from physrealx.digital_twin import (
    AssetValidationProfile,
    AssetValidationReport,
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
from physrealx.readiness import ReadinessService

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


_MIB = 1024 * 1024
_GIB = 1024 * _MIB


def _output(
    name: str,
    contract: str,
    *,
    content_kind: Literal["auto", "binary", "json", "zip"] = "auto",
    media_types: tuple[str, ...] = (),
    max_size_bytes: int | None = None,
    schema_discriminator: str | None = None,
) -> OperationOutputRequirement:
    """Build one immutable service-owned output declaration."""

    return OperationOutputRequirement(
        name=name,
        contract=contract,
        content_kind=content_kind,
        media_types=media_types,
        max_size_bytes=max_size_bytes,
        schema_discriminator=schema_discriminator,
    )


@dataclass(frozen=True)
class _ServiceMethodContract:
    """Canonical wire contract for one public capability service.

    A service owns these names and schemas.  Callers select an implementation,
    but cannot silently redefine what the capability consumes or publishes.
    """

    input_contract: str
    input_names: tuple[str, ...]
    outputs: tuple[OperationOutputRequirement, ...]
    input_kinds: tuple[Literal["resource", "artifact", "either"], ...] = ()

    def __post_init__(self) -> None:
        if len(self.input_names) != len(set(self.input_names)):
            raise ValueError("service method input names must be unique")
        if self.input_kinds and len(self.input_kinds) != len(self.input_names):
            raise ValueError("service method input kinds must match its input names")
        if not self.outputs or len(self.outputs) != len({item.name for item in self.outputs}):
            raise ValueError("service method outputs must be non-empty and uniquely named")


_SERVICE_METHOD_CONTRACTS: dict[MethodCapability, _ServiceMethodContract] = {
    MethodCapability.CAMERA_RECOVER: _ServiceMethodContract(
        "ObservationSetManifest/1",
        ("observations",),
        (
            _output(
                "camera_rig_calibration",
                "CameraRigCalibration/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=64 * _MIB,
                schema_discriminator="physrealx.camera-rig-calibration/1",
            ),
            _output(
                "sparse_reconstruction_bundle",
                "SparseReconstructionBundle/1",
                content_kind="zip",
                media_types=("application/zip",),
                max_size_bytes=8 * _GIB,
            ),
            _output(
                "camera_recovery_report",
                "CameraRecoveryReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=16 * _MIB,
                schema_discriminator="physrealx.camera-recovery-report/1",
            ),
        ),
        input_kinds=("resource",),
    ),
    MethodCapability.CAMERA_ALIGN_METRIC: _ServiceMethodContract(
        "CameraRigCalibration/1+MetricAlignmentEvidence/1",
        ("camera_rig", "evidence"),
        (
            _output(
                "camera_rig_calibration",
                "CameraRigCalibration/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=64 * _MIB,
                schema_discriminator="physrealx.camera-rig-calibration/1",
            ),
            _output(
                "metric_alignment_report",
                "MetricAlignmentReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=16 * _MIB,
                schema_discriminator="physrealx.metric-alignment-report/1",
            ),
        ),
        input_kinds=("artifact", "artifact"),
    ),
    MethodCapability.DEPTH_ENHANCE: _ServiceMethodContract(
        "CalibratedRGBDBundle/1",
        ("rgbd",),
        (
            _output(
                "enhanced_depth",
                "EnhancedDepthBundle/1",
                content_kind="zip",
                media_types=("application/zip",),
                max_size_bytes=32 * _GIB,
            ),
            _output(
                "enhancement_report",
                "DepthEnhancementReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=32 * _MIB,
                schema_discriminator="physrealx.depth-enhancement-report/1",
            ),
        ),
        input_kinds=("resource",),
    ),
    MethodCapability.ARTICULATION_RECOVER: _ServiceMethodContract(
        "PairedInteractionPointCloudBundle/1",
        ("pair",),
        (
            _output(
                "articulation_model",
                "ArticulationModel/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=32 * _MIB,
                schema_discriminator="physrealx.articulation-model/1",
            ),
            _output(
                "articulation_recovery_report",
                "ArticulationRecoveryReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=16 * _MIB,
                schema_discriminator="physrealx.articulation-recovery-report/1",
            ),
        ),
        input_kinds=("resource",),
    ),
    MethodCapability.COLLISION_BUILD: _ServiceMethodContract(
        "VisualMesh/1",
        ("asset",),
        (
            _output(
                "collision_geometry",
                "CollisionGeometry/1",
                content_kind="binary",
                media_types=("model/gltf-binary",),
                max_size_bytes=8 * _GIB,
            ),
            _output(
                "collision_build_report",
                "CollisionBuildReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=16 * _MIB,
                schema_discriminator="physrealx.collision-build-report/1",
            ),
        ),
        input_kinds=("artifact",),
    ),
    MethodCapability.PHYSICS_CALIBRATE: _ServiceMethodContract(
        "JointSystemIdentificationEvidence/1",
        ("evidence",),
        (
            _output(
                "identified_joint_physics",
                "IdentifiedJointPhysics/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=4 * _MIB,
                schema_discriminator="physrealx.identified-joint-physics/1",
            ),
            _output(
                "identification_report",
                "PhysicsIdentificationReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=8 * _MIB,
                schema_discriminator="physrealx.physics-identification-report/1",
            ),
        ),
        input_kinds=("resource",),
    ),
    MethodCapability.ASSET_PACKAGE: _ServiceMethodContract(
        "AssetAssemblyBundle/1",
        ("asset",),
        (
            _output(
                "usd_asset_bundle",
                "UsdAssetBundle/1",
                content_kind="binary",
                media_types=("model/vnd.usd+binary",),
                max_size_bytes=8 * _GIB,
            ),
            _output(
                "asset_package_manifest",
                "AssetPackageManifest/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=8 * _MIB,
                schema_discriminator="physrealx.asset-package-manifest/1",
            ),
            _output(
                "roundtrip_validation_report",
                "UsdRoundTripValidationReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=16 * _MIB,
                schema_discriminator="physrealx.usd-roundtrip-validation-report/1",
            ),
        ),
        input_kinds=("resource",),
    ),
    MethodCapability.SCENE_GENERATE: _ServiceMethodContract(
        "SceneGenerationRequest/1",
        (),
        (_output("scene_draft", "SceneDraft/1"),),
    ),
    MethodCapability.SCENE_COMPOSE: _ServiceMethodContract(
        "SceneDraft/1",
        ("scene",),
        (
            _output("scene_spec", "SceneSpec/1"),
            _output("constraint_report", "ConstraintReport/1"),
        ),
    ),
    MethodCapability.TASK_COMPILE: _ServiceMethodContract(
        "SceneSpec/1+Instruction/1",
        ("scene",),
        (_output("task_spec", "TaskSpec/1"),),
    ),
    MethodCapability.TASK_VALIDATE: _ServiceMethodContract(
        "TaskSpec/1",
        ("task",),
        (_output("validation_report", "TaskValidationReport/1"),),
    ),
    MethodCapability.TRAJECTORY_GENERATE: _ServiceMethodContract(
        "TaskInstance/1",
        ("instance",),
        (
            _output("trajectory", "Trajectory/1"),
            _output("planning_report", "PlanningReport/1"),
        ),
    ),
    MethodCapability.SIMULATION_RUN: _ServiceMethodContract(
        "SceneSpec/1+TaskSpec/1",
        ("scene", "task"),
        (
            _output("state_trajectory", "StateTrajectory/1"),
            _output("simulation_report", "SimulationReport/1"),
        ),
    ),
    MethodCapability.RENDER_GENERATE: _ServiceMethodContract(
        "StateTrajectory/1",
        ("state_trajectory",),
        (
            _output("render_streams", "RenderStreams/1"),
            _output("camera_metadata", "CameraMetadata/1"),
        ),
    ),
    MethodCapability.EVENT_SYNTHESIZE: _ServiceMethodContract(
        "RGBStream/1",
        ("rgb_stream",),
        (_output("event_stream", "EventStream/1"),),
    ),
    MethodCapability.POINT_CLOUD_PROJECT: _ServiceMethodContract(
        "DepthStream/1",
        ("depth",),
        (_output("point_cloud", "PointCloud/1"),),
    ),
    MethodCapability.EPISODE_PACK: _ServiceMethodContract(
        "RunEvidence/1",
        ("run",),
        (
            _output("episode_bundle", "EpisodeBundle/1"),
            _output("episode_manifest", "EpisodeManifest/1"),
        ),
    ),
    MethodCapability.DATASET_INGEST: _ServiceMethodContract(
        "EpisodeBundle/1",
        ("source",),
        (
            _output("dataset", "DatasetBundle/1"),
            _output("quality_report", "DataQualityReport/1"),
        ),
    ),
    MethodCapability.DATASET_SNAPSHOT: _ServiceMethodContract(
        "DatasetBundle/1",
        ("dataset",),
        (
            _output("snapshot", "DatasetSnapshot/1"),
            _output("snapshot_manifest", "SnapshotManifest/1"),
        ),
    ),
    MethodCapability.MODEL_REGISTER: _ServiceMethodContract(
        "ModelBundle/1",
        ("model",),
        (_output("model_registration", "ModelRegistration/1"),),
    ),
    MethodCapability.WORLD_MODEL_ROLLOUT: _ServiceMethodContract(
        "ModelBundle/1+ContextWindow/1+ActionSequence/1",
        ("model", "context", "actions"),
        (
            _output("rollout", "WorldModelRollout/1"),
            _output("uncertainty", "UncertaintyReport/1"),
        ),
    ),
    MethodCapability.EVALUATION_RUN: _ServiceMethodContract(
        "ModelBundle/1+BenchmarkSuite/1",
        ("model", "suite"),
        (
            _output("evaluation_report", "EvaluationReport/1"),
            _output("episode_evidence", "EpisodeEvidence/1"),
        ),
    ),
    MethodCapability.DEPLOYMENT_STAGE: _ServiceMethodContract(
        "ModelBundle/1+RobotTarget/1",
        ("model", "robot"),
        (
            _output("deployment_plan", "DeploymentPlan/1"),
            _output("safety_report", "SafetyReport/1"),
        ),
    ),
    MethodCapability.TELEMETRY_RECORD: _ServiceMethodContract(
        "DeploymentRun/1",
        ("run",),
        (
            _output("telemetry", "TelemetryBundle/1"),
            _output("telemetry_manifest", "TelemetryManifest/1"),
        ),
    ),
    MethodCapability.FAILURE_ANALYZE: _ServiceMethodContract(
        "TelemetryBundle/1",
        ("run",),
        (_output("failure_report", "FailureReport/1"),),
    ),
    MethodCapability.FEEDBACK_BUILD_DATASET: _ServiceMethodContract(
        "FailureQuery/1",
        ("query",),
        (
            _output("dataset", "DatasetBundle/1"),
            _output("selection_report", "SelectionReport/1"),
        ),
    ),
}


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

    def recover_cameras(
        self,
        observations: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Recover camera intrinsics, poses and an unscaled sparse reconstruction."""

        if observations.resource_type is not ResourceType.OBSERVATION_SET:
            raise DomainValidationError(
                "assets.recover_cameras requires an observation-set reference"
            )
        return self._platform.methods.execute(
            MethodCapability.CAMERA_RECOVER,
            method,
            inputs={"observations": observations},
            config=config,
        )

    def align_cameras_metric(
        self,
        camera_rig: ArtifactRef,
        evidence: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Resolve a recovered camera rig into metres from held-out metrology evidence."""

        return self._platform.methods.execute(
            MethodCapability.CAMERA_ALIGN_METRIC,
            method,
            inputs={"camera_rig": camera_rig, "evidence": evidence},
            config=config,
        )

    def enhance_depth(
        self,
        rgbd: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Complete calibrated metric RGB-D while retaining held-out quality evidence."""

        return self._platform.methods.execute(
            MethodCapability.DEPTH_ENHANCE,
            method,
            inputs={"rgbd": rgbd},
            config=config,
        )

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
        pair: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.ARTICULATION_RECOVER,
            method,
            inputs={"pair": pair},
            config=config,
        )

    def build_collision(
        self,
        asset: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.COLLISION_BUILD,
            method,
            inputs={"asset": asset},
            config=config,
        )

    def calibrate_physics(
        self,
        evidence: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.PHYSICS_CALIBRATE,
            method,
            inputs={"evidence": evidence},
            config=config,
        )

    def package(
        self,
        asset: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.ASSET_PACKAGE,
            method,
            inputs={"asset": asset},
            config=config,
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
        resource_id: str | None = None,
        version: str = "1.0.0",
    ) -> OperationManifest:
        """Execute the canonical contract owned by one public capability service."""

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
        try:
            service_contract = _SERVICE_METHOD_CONTRACTS[capability]
        except KeyError as error:
            raise DomainValidationError(
                f"capability {capability.value!r} has no public service contract; "
                "execute it through an explicit external recipe"
            ) from error
        descriptor = selected.descriptor
        if descriptor.input_contract != service_contract.input_contract:
            raise DomainValidationError(
                f"method {capability.value}:{selected.name!r} declares input contract "
                f"{descriptor.input_contract!r}; the service requires "
                f"{service_contract.input_contract!r}"
            )
        declared_outputs = tuple(descriptor.output_contract.split("+"))
        required_outputs = tuple(item.contract for item in service_contract.outputs)
        if sorted(declared_outputs) != sorted(required_outputs):
            raise DomainValidationError(
                f"method {capability.value}:{selected.name!r} declares output contracts "
                f"{declared_outputs!r}; the service requires {required_outputs!r}"
            )
        supplied = inputs or {}
        supplied_inputs = tuple(supplied)
        if set(supplied_inputs) != set(service_contract.input_names):
            raise DomainValidationError(
                f"service {capability.value!r} requires inputs "
                f"{service_contract.input_names!r}, got {supplied_inputs!r}"
            )
        for name, value, expected_kind in zip(
            service_contract.input_names,
            (supplied[name] for name in service_contract.input_names),
            service_contract.input_kinds,
        ):
            actual_kind = "resource" if isinstance(value, VersionedRef) else "artifact"
            if expected_kind != "either" and actual_kind != expected_kind:
                raise DomainValidationError(
                    f"service input {name!r} must be an immutable {expected_kind}, "
                    f"got {actual_kind}"
                )
        self._platform.plugins.resolve_method(selected)
        operation_inputs = tuple(
            OperationInput(
                name=name,
                resource=supplied[name] if isinstance(supplied[name], VersionedRef) else None,
                artifact=supplied[name] if isinstance(supplied[name], ArtifactRef) else None,
            )
            for name in service_contract.input_names
        )
        output_requirements = service_contract.outputs
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
            inputs={"model": model, "context": context, "actions": actions},
            config=config,
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
    "ReadinessService",
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
