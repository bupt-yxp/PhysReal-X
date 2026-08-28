from __future__ import annotations

import importlib.util
from pathlib import Path

import physrealx


def test_public_api_is_explicit_and_stable() -> None:
    assert physrealx.__all__ == [
        "ActionSpec",
        "AlgorithmSpec",
        "ArtifactRef",
        "AssetSpec",
        "AssetValidationProfile",
        "AssetValidationReport",
        "DataValue",
        "DatasetManifest",
        "DigitalTwinBuildResult",
        "DigitalTwinBuildSpec",
        "DigitalTwinRecipe",
        "DigitalTwinStage",
        "DigitalTwinStep",
        "Episode",
        "EpisodeStep",
        "EvaluationReport",
        "ModelManifest",
        "ObservationSetManifest",
        "PhysRealX",
        "RunRecord",
        "RunSpec",
        "SceneEntityRole",
        "SceneEntitySpec",
        "SceneSpec",
        "SignalSpec",
        "SourceInspection",
        "TaskSpec",
        "TrainingSpec",
        "VersionedRef",
        "__version__",
    ]


def test_no_legacy_import_aliases_exist() -> None:
    assert importlib.util.find_spec("physreal") is None
    assert importlib.util.find_spec("physreal_x") is None


def test_facade_exposes_domain_services_without_flat_legacy_methods(tmp_path: Path) -> None:
    platform = physrealx.PhysRealX.open(tmp_path, discover_plugins=False)

    assert all(
        hasattr(platform, name)
        for name in ("assets", "scenes", "tasks", "runs", "datasets", "training", "evaluations")
    )
    assert not any(
        hasattr(platform, name)
        for name in ("register_asset", "register_task", "execute", "train", "evaluate")
    )
