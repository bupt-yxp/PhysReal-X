from __future__ import annotations

import importlib.util

import physrealx


def test_public_api_is_explicit_and_stable() -> None:
    assert physrealx.__all__ == [
        "ActionSpec",
        "ArtifactRef",
        "AssetSpec",
        "DataValue",
        "DatasetManifest",
        "Episode",
        "EpisodeStep",
        "EvaluationReport",
        "ModelManifest",
        "PhysRealX",
        "RunRecord",
        "RunSpec",
        "SignalSpec",
        "TaskSpec",
        "TrainingSpec",
        "VersionedRef",
        "__version__",
    ]


def test_no_legacy_import_aliases_exist() -> None:
    assert importlib.util.find_spec("physreal") is None
    assert importlib.util.find_spec("physreal_x") is None
