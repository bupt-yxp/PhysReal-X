from __future__ import annotations

import importlib.util
from pathlib import Path

import physrealx
from physrealx.domain import EvaluationReport as DomainEvaluationReport
from physrealx.training_lifecycle import BenchmarkReport


def test_public_api_is_explicit_and_stable() -> None:
    assert physrealx.__all__ == [
        "ActionSpec",
        "ArtGSInputManifest",
        "ArtGSRunConfig",
        "ArtifactRef",
        "AssetEvidenceState",
        "AssetSpec",
        "AssetValidationProfile",
        "AssetValidationReport",
        "BenchmarkEpisode",
        "BenchmarkReport",
        "BenchmarkSuite",
        "CodePin",
        "CommandDisposition",
        "CompletedStepOutput",
        "DataValue",
        "DatasetManifest",
        "DatasetSnapshot",
        "DatasetSplit",
        "DeploymentLease",
        "DeploymentMode",
        "DeploymentPlan",
        "EnvironmentPin",
        "Episode",
        "EpisodeArtifact",
        "EpisodeEvaluationEvidence",
        "EpisodeExecutionResult",
        "EpisodePackResult",
        "EpisodeStep",
        "EvaluationLifecycle",
        "EvaluationOutcome",
        "EvaluationReport",
        "ExecutionInput",
        "ExecutionLifecycle",
        "ExecutionPlan",
        "ExecutionValidationReport",
        "FailureLifecycle",
        "FailureRecord",
        "FailureReport",
        "FeedbackDatasetManifest",
        "FeedbackDatasetSpec",
        "FeedbackLifecycle",
        "InitialRecipeInput",
        "JointSafetyLimit",
        "LifecycleEvidence",
        "LifecycleEvidenceKey",
        "LifecyclePlan",
        "LifecycleReadinessReport",
        "LifecycleStage",
        "LifecycleStageBinding",
        "LifecycleStageReadiness",
        "MethodAccelerator",
        "MethodCapability",
        "MethodDescriptor",
        "MethodExecutionMode",
        "MethodMaturity",
        "MethodRecipe",
        "MethodRuntime",
        "MethodSpec",
        "ModelKind",
        "ModelManifest",
        "ObservationSetManifest",
        "OperationInput",
        "OperationManifest",
        "OperationOutput",
        "OperationOutputRequirement",
        "OperationSpec",
        "PhysRealX",
        "PolicyModelBundle",
        "PolicyTrainingPlan",
        "ReadinessBlocker",
        "ReadinessBlockerCode",
        "ReadinessStatus",
        "RecipeExecutionManifest",
        "RecipeExecutionResult",
        "RecipeExecutionSpec",
        "RecipeExecutor",
        "RecipeFinalOutput",
        "RecipeInputBinding",
        "RecipeOperationRecord",
        "RecipeStep",
        "RenderStreamEvidence",
        "RenderedFrameEvidence",
        "RenderedModality",
        "RobotCommand",
        "RobotExecutionEvidence",
        "RobotInterfaceSpec",
        "RobotObservation",
        "RunRecord",
        "RunSpec",
        "SafetyDecision",
        "SafetyFault",
        "SafetyPolicy",
        "SceneEntityRole",
        "SceneEntitySpec",
        "SceneSpec",
        "SignalSpec",
        "SnapshotLineage",
        "SourceInspection",
        "StateFrameEvidence",
        "StateTrajectoryEvidence",
        "TaskSpec",
        "TelemetryManifest",
        "TrainingExecutionResult",
        "TrainingLifecycle",
        "TrainingReport",
        "TrainingSpec",
        "VerifiedEpisodeArtifact",
        "VerifiedEpisodeBundle",
        "VerifiedEpisodeCollection",
        "VersionedRef",
        "WilsonInterval",
        "WorkspaceBounds",
        "WorldModelBundle",
        "WorldModelTrainingPlan",
        "__version__",
        "package_artgs_input",
    ]


def test_evaluation_contract_names_are_unambiguous() -> None:
    assert physrealx.EvaluationReport is DomainEvaluationReport
    assert physrealx.BenchmarkReport is BenchmarkReport
    assert physrealx.BenchmarkReport is not physrealx.EvaluationReport


def test_public_protocol_and_lifecycle_entrypoints_are_callable(tmp_path: Path) -> None:
    platform = physrealx.PhysRealX.open(tmp_path, discover_plugins=False)

    assert callable(platform.readiness.plan)
    assert callable(platform.readiness.preflight)
    assert callable(platform.recipes.execute)
    assert callable(physrealx.TrainingLifecycle.finalize_policy)
    assert callable(physrealx.TrainingLifecycle.finalize_world_model)
    assert callable(physrealx.EvaluationLifecycle.finalize)
    assert callable(physrealx.FailureLifecycle.record)
    assert callable(physrealx.FeedbackLifecycle.finalize)
    assert callable(physrealx.ExecutionLifecycle.validate)
    assert callable(physrealx.ExecutionLifecycle.finalize)
    assert callable(physrealx.ExecutionLifecycle.collect)


def test_no_legacy_import_aliases_exist() -> None:
    assert importlib.util.find_spec("physreal") is None
    assert importlib.util.find_spec("physreal_x") is None


def test_removed_composite_asset_pipeline_api_is_not_exposed(tmp_path: Path) -> None:
    platform = physrealx.PhysRealX.open(tmp_path, discover_plugins=False)

    assert not hasattr(platform.assets, "build")
    assert not hasattr(platform.assets, "get_build")
    assert not hasattr(platform.assets, "list_builds")
    assert not hasattr(platform.plugins, "register_asset_pipeline")
    assert not hasattr(platform.plugins, "asset_pipeline")
    assert all(
        not hasattr(physrealx, name)
        for name in (
            "DigitalTwinBuildResult",
            "DigitalTwinBuildSpec",
            "DigitalTwinRecipe",
            "DigitalTwinStage",
            "DigitalTwinStep",
        )
    )


def test_facade_exposes_domain_services_without_flat_legacy_methods(tmp_path: Path) -> None:
    platform = physrealx.PhysRealX.open(tmp_path, discover_plugins=False)

    assert all(
        hasattr(platform, name)
        for name in (
            "assets",
            "datasets",
            "deployments",
            "episodes",
            "evaluations",
            "events",
            "failures",
            "feedback",
            "models",
            "operations",
            "pointclouds",
            "readiness",
            "recipes",
            "renders",
            "runs",
            "scenes",
            "simulations",
            "tasks",
            "telemetry",
            "training",
            "trajectories",
            "world_models",
        )
    )
    assert not any(
        hasattr(platform, name)
        for name in ("register_asset", "register_task", "execute", "train", "evaluate")
    )
    assert hasattr(platform, "methods")
