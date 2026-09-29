"""Canonical input and output contracts owned by capability services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physrealx.domain import (
    MethodCapability,
    OperationOutputRequirement,
)

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


SERVICE_METHOD_CONTRACTS: dict[MethodCapability, _ServiceMethodContract] = {
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
