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
    canonical_record_id,
)
from physrealx.errors import DomainValidationError, ResourceConflictError, ResourceNotFoundError
from physrealx.plugins import MethodInstallation
from physrealx.readiness import ReadinessService

if TYPE_CHECKING:
    from physrealx.integrations.sensors import CameraSpec
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
        input_kinds=("artifact",),
    ),
    MethodCapability.GEOMETRY_RECONSTRUCT: _ServiceMethodContract(
        "MetricMultiStateRGBDBundle/1",
        ("source",),
        (
            _output(
                "visual_geometry_bundle",
                "VisualGeometryBundle/1",
                content_kind="zip",
                media_types=("application/zip",),
                max_size_bytes=32 * _GIB,
            ),
            _output(
                "geometry_reconstruction_report",
                "GeometryReconstructionReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=32 * _MIB,
                schema_discriminator="physrealx.geometry-reconstruction-report/1",
            ),
        ),
        input_kinds=("artifact",),
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
        input_kinds=("artifact",),
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
        input_kinds=("artifact",),
    ),
    MethodCapability.ASSET_ASSEMBLE: _ServiceMethodContract(
        "AssetAssemblyInput/1",
        ("evidence",),
        (
            _output(
                "asset_assembly",
                "AssetAssemblyBundle/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=64 * _MIB,
                schema_discriminator="physrealx.asset-assembly-bundle/1",
            ),
        ),
        input_kinds=("artifact",),
    ),
    MethodCapability.ASSET_VALIDATE: _ServiceMethodContract(
        "AssetPackageManifest/1+PhysicalValidationEvidence/1",
        ("package", "evidence"),
        (
            _output(
                "physical_validation_report",
                "PhysicalAssetValidationReport/1",
                content_kind="json",
                media_types=("application/json",),
                max_size_bytes=64 * _MIB,
                schema_discriminator="physrealx.asset-validation-report/1",
            ),
        ),
        input_kinds=("artifact", "artifact"),
    ),
    MethodCapability.ASSET_PACKAGE: _ServiceMethodContract(
        "AssetAssemblyBundle/1",
        ("asset",),
        (
            _output(
                "usd_asset_bundle",
                "UsdAssetBundle/1",
                content_kind="binary",
                media_types=("model/vnd.usdz+zip",),
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
        input_kinds=("artifact",),
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
    MethodCapability.WORLD_MODEL_TRAIN: _ServiceMethodContract(
        "DatasetSnapshot/1+PretrainedEncoder/1",
        ("dataset", "encoder"),
        (
            _output(
                "model",
                "ModelBundle/1",
                content_kind="json",
                schema_discriminator="physrealx.world-model-bundle/1",
            ),
            _output(
                "training_report",
                "TrainingReport/1",
                content_kind="json",
                schema_discriminator="physrealx.training-report/1",
            ),
        ),
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
        "DeploymentEvidence/1",
        ("run",),
        (
            _output("telemetry", "TelemetryBundle/2"),
            _output("telemetry_manifest", "TelemetryManifest/2"),
        ),
    ),
    MethodCapability.FAILURE_ANALYZE: _ServiceMethodContract(
        "TelemetryBundle/2",
        ("run",),
        (_output("failure_report", "FailureReport/2"),),
    ),
    MethodCapability.FEEDBACK_BUILD_DATASET: _ServiceMethodContract(
        "FailureQuery/2",
        ("query",),
        (
            _output("dataset", "DatasetBundle/1"),
            _output("selection_report", "SelectionReport/2"),
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
        config: Mapping[str, JSONValue] | None = None,
    ) -> SourceInspection:
        return self._platform._inspect_asset_source(Path(source), importer, config)

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

    def import_partnet_mobility(
        self,
        catalog: str | Path,
        *,
        model_id: str,
        resource_id: str,
        version: str = "1.0.0",
        display_name: str | None = None,
        target_extent_m: float,
    ) -> VersionedRef:
        """Register one PartNet-Mobility URDF as a metric canonical asset."""
        for reference in self.list():
            if (reference.resource_id, reference.version) != (resource_id, version):
                continue
            asset = self.get(reference)
            if (
                asset.metadata.get("source") != "PartNet-Mobility"
                or asset.metadata.get("model_id") != model_id
                or asset.metadata.get("target_extent_m") != target_extent_m
            ):
                raise DomainValidationError(
                    "registered PartNet-Mobility asset identity has different import settings"
                )
            return reference
        from physrealx.integrations.partnet_mobility import import_partnet_mobility_asset

        return self.register(
            import_partnet_mobility_asset(
                self._platform.artifacts,
                catalog,
                model_id=model_id,
                resource_id=resource_id,
                version=version,
                display_name=display_name,
                target_extent_m=target_extent_m,
            )
        )

    def select_partnet_mobility(
        self,
        catalog: str | Path,
        *,
        scene_type: str,
        seed: int = 0,
        version: str = "1.0.0",
    ) -> dict[str, VersionedRef]:
        """Select and register the canonical assets required by a scene type."""
        from typing import cast

        from physrealx.integrations.partnet_mobility import (
            PartNetSceneType,
            select_partnet_scene_assets,
        )

        selections = select_partnet_scene_assets(
            catalog,
            scene_type=cast(PartNetSceneType, scene_type),
            seed=seed,
        )
        return {
            item.name: self.import_partnet_mobility(
                catalog,
                model_id=item.model_id,
                resource_id=f"partnet.{item.name}.{item.model_id}",
                version=version,
                display_name=item.category,
                target_extent_m=item.target_extent_m,
            )
            for item in selections
        }

    def import_mujoco_model(
        self,
        source: str | Path,
        *,
        resource_id: str,
        version: str = "1.0.0",
        display_name: str,
        source_name: str,
        source_revision: str,
        license_name: str,
    ) -> VersionedRef:
        """Register a local open-source MJCF robot as a canonical asset."""
        identity = {
            "source": source_name,
            "source_revision": source_revision,
            "source_model": Path(source).name,
            "license": license_name,
        }
        for reference in self.list():
            if (reference.resource_id, reference.version) != (resource_id, version):
                continue
            asset = self.get(reference)
            if any(asset.metadata.get(key) != value for key, value in identity.items()):
                raise DomainValidationError(
                    "registered MuJoCo asset identity has different source provenance"
                )
            return reference
        from physrealx.integrations.mujoco_model import import_mujoco_model_asset

        return self.register(
            import_mujoco_model_asset(
                self._platform.artifacts,
                source,
                resource_id=resource_id,
                version=version,
                display_name=display_name,
                source_name=source_name,
                source_revision=source_revision,
                license_name=license_name,
            )
        )

    def create_room_shell(
        self,
        *,
        resource_id: str,
        version: str = "1.0.0",
        width_m: float,
        depth_m: float,
        height_m: float,
    ) -> VersionedRef:
        """Register a metric floor and two-wall room environment."""
        from physrealx.integrations.scene_assets import room_shell_asset

        return self.register(
            room_shell_asset(
                resource_id=resource_id,
                version=version,
                width_m=width_m,
                depth_m=depth_m,
                height_m=height_m,
            )
        )

    def get_observations(self, reference: VersionedRef) -> ObservationSetManifest:
        return self._platform._load_registered(reference, ObservationSetManifest)

    def list_observations(self) -> builtins.list[VersionedRef]:
        return self._platform._list_resources(ResourceType.OBSERVATION_SET)

    def import_reference(
        self,
        observations: VersionedRef,
        *,
        usd_path: str,
        root_link: str | None = None,
        moving_links: tuple[str, ...] | None = None,
    ) -> ArtifactRef:
        """Import supplied USD geometry and kinematics with explicit model provenance."""
        from physrealx.integrations.usd_reference import import_usd_reference

        return import_usd_reference(
            self.get_observations(observations),
            self._platform.artifacts,
            usd_path=usd_path,
            root_link=root_link,
            moving_links=moving_links,
        )

    def prepare_rgbd(
        self,
        observations: VersionedRef,
        camera_rig: ArtifactRef,
        *,
        stride: int = 4,
        max_frames_per_stream: int | None = 8,
    ) -> ArtifactRef:
        """Decode registered RGB-D views; None selects all within the 512-view safety limit."""
        from physrealx.integrations.capture_rgbd import prepare_rgbd_frames

        return prepare_rgbd_frames(
            self.get_observations(observations),
            camera_rig,
            self._platform.artifacts,
            stride=stride,
            max_frames_per_stream=max_frames_per_stream,
        )

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
        rgbd: ArtifactRef,
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

    def reconstruct_geometry(
        self,
        source: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Reconstruct metric per-link visual geometry from a validated source bundle."""

        return self._platform.methods.execute(
            MethodCapability.GEOMETRY_RECONSTRUCT,
            method,
            inputs={"source": source},
            config=config,
        )

    def evaluate_reconstruction(self, source: ArtifactRef, *, seed: int = 2026) -> ArtifactRef:
        """Evaluate any normalized reconstruction with the fixed DTArt metric protocol."""
        from physrealx.integrations._artifacts import read_json
        from physrealx.integrations.reconstruction_evaluation import (
            ReconstructionEvaluationConfig,
            ReconstructionEvaluationInput,
            evaluate_reconstruction,
        )

        request = ReconstructionEvaluationInput.model_validate(
            read_json(
                self._platform.artifacts, source, "physrealx.reconstruction-evaluation-input/1"
            )
        )
        report = evaluate_reconstruction(
            self._platform.artifacts, request, ReconstructionEvaluationConfig(seed=seed)
        )
        return self._platform.artifacts.put_json(report)

    def preview_reconstruction(
        self, evaluation: ArtifactRef, *, frames: int = 24, fps: int = 12
    ) -> ArtifactRef:
        """Animate appearance, reconstructed geometry and reference geometry together."""
        from physrealx.integrations.reconstruction_preview import preview_reconstruction

        return preview_reconstruction(self._platform.artifacts, evaluation, frames=frames, fps=fps)

    def preview_physical_asset(
        self, assembly: ArtifactRef, *, frames: int = 24, fps: int = 12
    ) -> ArtifactRef:
        """Animate assembled visual and collision geometry through declared joint limits."""
        from physrealx.integrations.physical_preview import preview_physical_asset

        return preview_physical_asset(self._platform.artifacts, assembly, frames=frames, fps=fps)

    def register_reconstruction_geometry(
        self, reconstruction: ArtifactRef, registration: ArtifactRef
    ) -> ArtifactRef:
        """Bind real reconstructed parts to registered articulated link rest frames."""
        from physrealx.integrations.reconstructed_asset import (
            register_reconstruction_geometry,
        )

        return register_reconstruction_geometry(
            self._platform.artifacts, reconstruction, registration
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

    def preview_physics_report(
        self,
        prior: ArtifactRef,
        feasibility: ArtifactRef,
        readiness: ArtifactRef,
    ) -> ArtifactRef:
        from physrealx.integrations.physical_preview import preview_physics_report

        return preview_physics_report(self._platform.artifacts, prior, feasibility, readiness)

    def preview_physics_video(
        self, directory: Path, *, frames: int = 24, fps: int = 8
    ) -> ArtifactRef:
        """Animate synchronized RGB, depth and manually reviewed object/link masks."""
        from physrealx.integrations.physical_preview import preview_physics_video

        return preview_physics_video(
            self._platform.artifacts, directory, frames=frames, fps=fps
        )

    def articulate(
        self,
        pair: ArtifactRef,
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
        evidence: ArtifactRef,
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

    def predict_physics(
        self,
        reference: ArtifactRef,
        registration: ArtifactRef,
        geometry: ArtifactRef,
        images: tuple[ArtifactRef, ...],
        *,
        python: str,
        model: str,
        revision: str,
    ) -> ArtifactRef:
        """Initialize per-body and joint priors with a pinned local VLM implementation."""
        from physrealx.integrations.physical_estimation import predict_physics

        return predict_physics(
            self._platform.artifacts,
            reference,
            registration,
            geometry,
            images,
            python=python,
            model=model,
            revision=revision,
        )

    def refine_video_geometry(
        self,
        evidence: ArtifactRef,
        *,
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Refine observed surfaces without changing the start/end joint range."""
        from physrealx.integrations.video_geometry import (
            GeometryRefinementConfig,
            refine_video_geometry,
        )

        return refine_video_geometry(
            self._platform.artifacts,
            evidence,
            config=GeometryRefinementConfig.model_validate(config or {}),
        )

    def check_physics_feasibility(
        self,
        assembly: ArtifactRef,
        *,
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Audit collision clearance and dynamics; does not certify measured physical truth."""
        from physrealx.integrations.physical_feasibility import FeasibilityConfig, check_feasibility

        return check_feasibility(
            self._platform.artifacts,
            assembly,
            config=FeasibilityConfig.model_validate(config or {}),
        )

    def validate_physics(
        self,
        package: ArtifactRef,
        evidence: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Validate packaged physics against an independent measured trajectory."""
        return self._platform.methods.execute(
            MethodCapability.ASSET_VALIDATE,
            method,
            inputs={"package": package, "evidence": evidence},
            config=config,
        )

    def assemble(
        self,
        evidence: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        """Assemble all links and joints from explicit immutable measured evidence."""
        return self._platform.methods.execute(
            MethodCapability.ASSET_ASSEMBLE,
            method,
            inputs={"evidence": evidence},
            config=config,
        )

    def package(
        self,
        asset: ArtifactRef,
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
                "execute it through an explicit method recipe"
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
            f"operation.{capability.value}.{canonical_record_id(identity_payload)[-16:]}"
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

    def place_robot_for_joint_task(
        self,
        scene: SceneSpec,
        *,
        entity: str,
        joint: str,
        robot: VersionedRef,
        name: str,
        end_effector_link: str,
        initial_position: float,
        target_position: float,
        clearance_m: float,
        resource_id: str,
    ) -> SceneSpec:
        """Place a canonical articulated robot and solve target-link endpoint poses."""
        from physrealx.integrations.scenes import place_robot_for_joint_task

        target = next((item for item in scene.entities if item.name == entity), None)
        if target is None:
            raise DomainValidationError(f"scene entity not found: {entity}")
        return place_robot_for_joint_task(
            scene,
            self._platform.assets.get(target.asset),
            self._platform.assets.get(robot),
            self._platform.artifacts,
            entity=entity,
            joint=joint,
            robot=robot,
            name=name,
            end_effector_link=end_effector_link,
            initial_position=initial_position,
            target_position=target_position,
            clearance_m=clearance_m,
            resource_id=resource_id,
        )

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

    def preview_task(
        self,
        scene: VersionedRef,
        task: VersionedRef,
        trajectory: ArtifactRef,
        *,
        frames: int = 16,
        fps: int = 8,
    ) -> ArtifactRef:
        """Render one Blender scene-execution preview from recorded states."""
        from physrealx.integrations.task_preview import preview_task_execution

        return preview_task_execution(
            self._platform.artifacts,
            scene,
            task,
            trajectory,
            frames=frames,
            fps=fps,
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

    def with_sensors(
        self,
        task: TaskSpec,
        scene: VersionedRef,
        cameras: Mapping[str, CameraSpec],
        *,
        resource_id: str,
        frequency_hz: float,
    ) -> TaskSpec:
        from physrealx.integrations.sensors import with_sensor_signals

        if task.scene != scene:
            raise DomainValidationError("sensor task and scene references must match")
        layout = self._platform.scenes.get(scene)
        link_count = sum(len(self._platform.assets.get(e.asset).links) for e in layout.entities)
        return with_sensor_signals(
            task,
            dict(cameras),
            link_count=link_count,
            resource_id=resource_id,
            frequency_hz=frequency_hz,
        )

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


class DatasetsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def get(self, reference: VersionedRef) -> DatasetManifest:
        return self._platform._load_registered(reference, DatasetManifest)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.DATASET)

    def register(self, manifest: DatasetManifest) -> VersionedRef:
        """Publish an assembled or selected dataset after checking its actual episodes."""
        from physrealx.integrations.datasets import validate_dataset

        validate_dataset(self._platform.artifacts, manifest)
        return self._platform._register(
            manifest, manifest.reference(), "application/vnd.physrealx.dataset+json"
        )

    def preview_episode(
        self,
        dataset: VersionedRef,
        *,
        episode_index: int = 0,
        camera: str | None = None,
        frames: int = 12,
        fps: int = 6,
        sensor_source: str = "blender",
    ) -> ArtifactRef:
        """Render one recorded episode with synchronized physics and sensor views."""
        from physrealx.integrations.simulation_preview import preview_simulation_episode

        return preview_simulation_episode(
            self._platform.artifacts,
            dataset,
            episode_index=episode_index,
            camera=camera,
            frames=frames,
            fps=fps,
            sensor_source=sensor_source,
        )

    def preview_snapshot(
        self,
        snapshot: ArtifactRef,
        *,
        camera: str = "front",
    ) -> ArtifactRef:
        """Render modalities, sequences, labels and split membership from a snapshot."""
        from physrealx.integrations.learning_preview import preview_dataset_snapshot

        return preview_dataset_snapshot(self._platform.artifacts, snapshot, camera=camera)

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

    def training_subset(
        self,
        snapshot: ArtifactRef,
        *,
        resource_id: str,
        version: str = "1.0.0",
        augmentations: tuple[VersionedRef, ...] = (),
        excluded_datasets: tuple[VersionedRef, ...] = (),
        required_tag: str | None = None,
        selection: Literal["successful_demonstrations", "all_episodes"] = (
            "successful_demonstrations"
        ),
    ) -> VersionedRef:
        from physrealx.integrations.datasets import training_subset

        selected = training_subset(
            self._platform.artifacts,
            snapshot,
            resource_id=resource_id,
            version=version,
            augmentations=augmentations,
            excluded_datasets=excluded_datasets,
            required_tag=required_tag,
            selection=selection,
        )
        try:
            existing_ref = self._platform.repository.get_dataset(resource_id, version)
        except ResourceNotFoundError:
            existing_ref = None
        if existing_ref is not None:
            existing = self.get(existing_ref)
            if existing.model_dump(exclude={"created_at"}) == selected.model_dump(
                exclude={"created_at"}
            ):
                return existing_ref
            raise ResourceConflictError(
                f"dataset/{resource_id}@{version} already uses a different training selection"
            )
        return self.register(selected)


class TrainingService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def submit(self, spec: TrainingSpec) -> VersionedRef:
        return self._platform._train(spec)

    def get_model(self, reference: VersionedRef) -> ModelManifest:
        return self._platform._load_registered(reference, ModelManifest)

    def list_models(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.MODEL)

    def preview_policy(
        self,
        model: VersionedRef,
        evaluation: VersionedRef,
    ) -> ArtifactRef:
        """Render policy optimization and closed-loop evaluation evidence."""
        from physrealx.integrations.learning_preview import preview_policy_training

        return preview_policy_training(
            self._platform.artifacts,
            model,
            self.get_model(model),
            self._platform.evaluations.get_report(evaluation),
        )

    def prepare_interface(
        self,
        snapshot: ArtifactRef,
        *,
        method: Literal["openvla_oft", "td_mpc2"],
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Validate a snapshot hand-off to one pinned official training runtime."""
        from physrealx.integrations.training_interfaces import prepare_training_interface

        return prepare_training_interface(
            self._platform.artifacts,
            snapshot,
            method=method,
            config=config,
        )

    def preview_interfaces(self, interfaces: tuple[ArtifactRef, ...]) -> ArtifactRef:
        """Show validated external interfaces and their execution status."""
        from physrealx.integrations.training_interfaces import preview_training_interfaces

        return preview_training_interfaces(self._platform.artifacts, interfaces)


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

    def prepare_encoder(
        self, *, cache: Path | None = None, weights_path: Path | None = None
    ) -> ArtifactRef:
        """Prepare DINOv2 from verified cache, local weights or the official source."""
        from physrealx.integrations.world_model import prepare_dinov2_encoder

        return prepare_dinov2_encoder(
            self._platform.artifacts,
            cache
            if cache is not None
            else self._platform.workspace / "pretrained/dinov2-encoder.json",
            weights_path=weights_path,
        )

    def train(
        self,
        dataset: ArtifactRef,
        encoder: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.WORLD_MODEL_TRAIN,
            method,
            inputs={"dataset": dataset, "encoder": encoder},
            config=config,
        )

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

    def evaluate_rollout(
        self,
        rollout: ArtifactRef,
        reference: Episode,
        *,
        start_index: int = 0,
    ) -> ArtifactRef:
        """Compare predicted states with a time-aligned episode and persistence."""
        from physrealx.integrations.world_model import evaluate_world_model_rollout

        return evaluate_world_model_rollout(
            self._platform.artifacts,
            rollout,
            reference,
            start_index=start_index,
        )

    def preview_rollout(
        self,
        rollout: ArtifactRef,
        evaluation: ArtifactRef,
        reference: Episode,
        *,
        camera: str = "front",
        frames: int = 8,
        fps: int = 4,
    ) -> ArtifactRef:
        """Animate world-model RGB and state predictions against aligned truth."""
        from physrealx.integrations.learning_preview import preview_world_model_rollout

        return preview_world_model_rollout(
            self._platform.artifacts,
            rollout,
            evaluation,
            reference,
            camera=camera,
            frames=frames,
            fps=fps,
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

    def preview_demonstrations(self, datasets: tuple[VersionedRef, ...]) -> ArtifactRef:
        from physrealx.integrations.feedback_preview import preview_demonstrations

        for reference in datasets:
            self._platform.datasets.get(reference)
        return preview_demonstrations(self._platform.artifacts, datasets)

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

    def plan_physics_calibration(
        self,
        telemetry: ArtifactRef,
        failure_report: ArtifactRef,
    ) -> ArtifactRef:
        """Decide whether verified deployment evidence can support physics calibration."""
        from physrealx.integrations.feedback import build_physics_calibration_plan

        return build_physics_calibration_plan(
            self._platform.artifacts,
            telemetry,
            failure_report,
        )

    def analyze_policy_intervention(self, evidence: ArtifactRef) -> ArtifactRef:
        """Attribute failures using same-condition reruns that change only the policy."""
        from physrealx.integrations.feedback import analyze_policy_intervention

        return analyze_policy_intervention(self._platform.artifacts, evidence)

    def assess_policy_update(
        self,
        initial: VersionedRef,
        updated: VersionedRef,
        *,
        reference_dataset: VersionedRef | None = None,
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Gate a candidate policy with paired closed-loop evaluation episodes."""
        from physrealx.integrations.feedback import assess_policy_update

        self._platform.evaluations.get_report(initial)
        self._platform.evaluations.get_report(updated)
        if reference_dataset is not None:
            self._platform.datasets.get(reference_dataset)
        return assess_policy_update(
            self._platform.artifacts,
            initial,
            updated,
            config,
            reference_dataset=reference_dataset,
        )

    def prepare_physics_calibration(
        self,
        telemetry: ArtifactRef,
        *,
        joint_name: str,
        joint_type: Literal["slide", "hinge"],
        joint_index: int,
        prior_parameters: tuple[float, float, float],
    ) -> ArtifactRef:
        """Prepare existing inverse-dynamics calibration from attested real episodes."""
        from physrealx.integrations.feedback import prepare_joint_identification_evidence

        return prepare_joint_identification_evidence(
            self._platform.artifacts,
            telemetry,
            joint_name=joint_name,
            joint_type=joint_type,
            joint_index=joint_index,
            prior_parameters=prior_parameters,
        )

    def preview_report(
        self,
        initial: VersionedRef,
        updated: VersionedRef,
        failure_report: ArtifactRef,
        causal_report: ArtifactRef,
        calibration_plan: ArtifactRef,
        policy_update: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Render closed-loop metrics, attribution evidence, and calibration status."""
        from physrealx.integrations.feedback_preview import preview_feedback_report

        return preview_feedback_report(
            self._platform.artifacts,
            self._platform.evaluations.get_report(initial),
            self._platform.evaluations.get_report(updated),
            failure_report,
            causal_report,
            calibration_plan,
            policy_update,
        )

    def preview_cycle(
        self,
        failed_episode: ArtifactRef,
        recovery_episode: ArtifactRef,
        updated_episode: ArtifactRef,
        *,
        frames: int = 8,
        fps: int = 3,
    ) -> ArtifactRef:
        """Animate a same-condition failure, expert recovery, and updated policy replay."""
        from physrealx.integrations.feedback_preview import preview_feedback_cycle

        return preview_feedback_cycle(
            self._platform.artifacts,
            failed_episode,
            recovery_episode,
            updated_episode,
            frames=frames,
            fps=fps,
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
