"""Composition root for the optional physical pipeline; imports do not load native engines."""

from __future__ import annotations

from physrealx.contracts import ArtifactStore
from physrealx.integrations.articulation import ArticulationAdapter
from physrealx.integrations.assembly import AssetAssemblyAdapter
from physrealx.integrations.asset_prior import ReferenceAssemblyAdapter
from physrealx.integrations.capture_rgbd import ModelScaleAdapter
from physrealx.integrations.collision import CollisionAdapter
from physrealx.integrations.datasets import DatasetSnapshotAdapter, EpisodeDatasetAdapter
from physrealx.integrations.depth import DepthEnhancementAdapter
from physrealx.integrations.feedback import (
    FailureAnalysisAdapter,
    FailureDatasetAdapter,
    TelemetryArchiveAdapter,
)
from physrealx.integrations.gaussian_render import RegisteredGaussianRenderer
from physrealx.integrations.metric import MetricAlignmentAdapter
from physrealx.integrations.mujoco import MujocoBackend
from physrealx.integrations.physics import JointPhysicsAdapter
from physrealx.integrations.registration import ReferenceRegistrationAdapter
from physrealx.integrations.rgbd_fusion import RGBDFusionAdapter
from physrealx.integrations.scenes import (
    PhySceneGuidanceAdapter,
    SceneCompositionAdapter,
    SceneGenerationAdapter,
    TaskTemplateAdapter,
    TaskValidationAdapter,
)
from physrealx.integrations.trajectories import JointTrajectoryAdapter
from physrealx.integrations.usd import UsdPackageAdapter
from physrealx.integrations.validation import PhysicalValidationAdapter
from physrealx.integrations.world_model import WorldModelRolloutAdapter, WorldModelTrainAdapter
from physrealx.plugins import PluginRegistry
from physrealx.policies import JointSequencePolicy, JointTargetPolicy


def register_physical_methods(registry: PluginRegistry, artifacts: ArtifactStore) -> None:
    for adapter in (
        MetricAlignmentAdapter(),
        ModelScaleAdapter(),
        ReferenceRegistrationAdapter(),
        RegisteredGaussianRenderer(),
        DepthEnhancementAdapter(),
        RGBDFusionAdapter(),
        ArticulationAdapter(),
        CollisionAdapter(),
        JointPhysicsAdapter(),
        AssetAssemblyAdapter(),
        ReferenceAssemblyAdapter(),
        UsdPackageAdapter(),
        PhysicalValidationAdapter(),
        SceneGenerationAdapter(),
        PhySceneGuidanceAdapter(),
        SceneCompositionAdapter(),
        TaskTemplateAdapter(),
        TaskValidationAdapter(),
        JointTrajectoryAdapter(),
        EpisodeDatasetAdapter(),
        DatasetSnapshotAdapter(),
        WorldModelTrainAdapter(),
        WorldModelRolloutAdapter(),
        TelemetryArchiveAdapter(),
        FailureAnalysisAdapter(),
        FailureDatasetAdapter(),
    ):
        registry.register_adapter(adapter)
    registry.register_backend(MujocoBackend(artifacts))
    registry.register_policy(JointTargetPolicy.descriptor, JointTargetPolicy)
    registry.register_policy(JointSequencePolicy.descriptor, JointSequencePolicy)
