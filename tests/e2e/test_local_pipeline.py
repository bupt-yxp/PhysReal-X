from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.contracts import SimulationSession
from physrealx.domain import (
    AssetSpec,
    Episode,
    JSONValue,
    RunStatus,
    SceneSpec,
    TaskSpec,
    TrainingSpec,
)
from physrealx.errors import DomainValidationError, ResourceConflictError
from physrealx.platform import PhysRealX
from physrealx.reference import reference_asset, reference_run, reference_scene, reference_task
from physrealx.training import TorchBehaviorCloningTrainer


def test_reference_run_publishes_replayable_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    backend = platform.plugins.backend("reference_drawer")
    original_create_session = backend.create_session
    created_sessions = 0

    def counted_create_session(
        scene_spec: SceneSpec,
        assets: Mapping[str, AssetSpec],
        task_spec: TaskSpec,
        config: Mapping[str, JSONValue],
    ) -> SimulationSession:
        nonlocal created_sessions
        created_sessions += 1
        return original_create_session(scene_spec, assets, task_spec, config)

    monkeypatch.setattr(backend, "create_session", counted_create_session)
    asset = reference_asset()
    scene = reference_scene(asset)
    task = reference_task(scene)
    asset_ref = platform.assets.register(asset)
    scene_ref = platform.scenes.register(scene)
    task_ref = platform.tasks.register(task)
    run = reference_run(scene, task, episode_count=4, base_seed=11)

    record = platform.runs.submit(run)

    assert record.status is RunStatus.SUCCEEDED
    assert created_sessions == 1
    assert record.spec.backend.version == "1.0.0"
    assert record.spec.policy.version == "1.0.0"
    assert record.result is not None
    assert record.result.dataset is not None
    assert record.result.episode_count == 4
    assert record.spec.scene == scene_ref
    assert platform.scenes.get(scene_ref).entities[0].asset == asset_ref
    assert platform.tasks.get(task_ref).scene == scene_ref
    dataset = platform.datasets.get(record.result.dataset)
    assert len(dataset.episodes) == 4
    assert all(item.success for item in dataset.episodes)
    assert platform.assets.get(asset_ref) == asset
    assert dataset.metadata["scene_artifact_id"] == scene_ref.artifact_id
    assert dataset.metadata["task_artifact_id"] == task_ref.artifact_id
    assert dataset.metadata["backend_method"]["descriptor"]["version"] == "1.0.0"
    assert dataset.metadata["policy_method"]["descriptor"]["version"] == "1.0.0"
    episodes = platform.runs.episodes(run.run_id)
    assert tuple(item.record_id for item in episodes) == tuple(
        item.record_id for item in platform.runs.episodes(run.run_id)
    )
    invalid_episode = episodes[0].model_dump(mode="json")
    invalid_episode["steps"][-1]["observations"]["drawer.position"]["unit"] = "cm"
    with pytest.raises(ValidationError, match="schemas must remain constant"):
        Episode.model_validate(invalid_episode)

    # Submitting an identical immutable RunSpec is idempotent.
    assert platform.runs.submit(run) == record
    conflicting = run.model_copy(update={"run_id": str(uuid.uuid4())})
    with pytest.raises(ResourceConflictError, match="already exists"):
        platform.runs.submit(conflicting)
    assert created_sessions == 1
    assert platform.runs.get(conflicting.run_id).status is RunStatus.FAILED


@pytest.mark.integration
def test_dataset_trains_and_evaluates_torch_policy(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()
    scene = reference_scene(asset)
    task = reference_task(scene)
    platform.assets.register(asset)
    platform.scenes.register(scene)
    platform.tasks.register(task)
    base_run = reference_run(scene, task, episode_count=8, base_seed=31).model_copy(
        update={"tags": ("expert_demonstration", "domain_randomized")}
    )
    record = platform.runs.submit(base_run)
    assert record.result is not None and record.result.dataset is not None

    model = platform.training.submit(
        TrainingSpec(
            resource_id="drawer.bc.test",
            version="1.0.0",
            dataset=record.result.dataset,
            trainer=TorchBehaviorCloningTrainer.descriptor.bind(
                config={"device": "cpu", "epochs": 80, "seed": 3}
            ),
        )
    )
    manifest = platform.training.get_model(model)
    assert manifest.metrics["training_mse"] < 0.01
    assert manifest.policy.name == "torch_bc"
    assert manifest.policy.version == "1.0.0"
    assert platform.artifacts.verify(manifest.weights)

    evaluation_ref = platform.evaluations.submit(
        resource_id="drawer.bc.test.eval",
        version="1.0.0",
        model=model,
        run=base_run,
        episode_count=4,
        base_seed=50_000,
    )
    report = platform.evaluations.get_report(evaluation_ref)
    assert report.success_rate >= 0.75
    assert report.metrics["episode_count"] == 4.0
    assert report.dataset is not None
    from physrealx.runtime import load_episode

    for summary in platform.datasets.get(report.dataset).episodes:
        tags = load_episode(platform.artifacts, summary).metadata["tags"]
        assert "expert_demonstration" not in tags
        assert "policy_evaluation" in tags and "domain_randomized" in tags

    # The evaluation identity includes the requested output identity, not only
    # the model and base task, so independent reports cannot collide.
    second_evaluation = platform.evaluations.submit(
        resource_id="drawer.bc.test.eval.other",
        version="1.0.0",
        model=model,
        run=base_run,
        episode_count=1,
        base_seed=50_000,
    )
    assert second_evaluation.artifact_id != evaluation_ref.artifact_id

    incompatible_task_payload = task.model_dump(mode="json")
    incompatible_task_payload["version"] = "1.1.0"
    incompatible_task_payload["observations"][0]["unit"] = "cm"
    incompatible_task = TaskSpec.model_validate(incompatible_task_payload)
    platform.tasks.register(incompatible_task)
    with pytest.raises(DomainValidationError, match="schema is incompatible"):
        platform.evaluations.submit(
            resource_id="drawer.bc.test.eval.incompatible",
            version="1.0.0",
            model=model,
            run=reference_run(scene, incompatible_task, episode_count=1),
            episode_count=1,
        )


def test_fixed_horizon_recording_has_the_same_semantics_on_reference_backend(tmp_path: Path):
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()
    scene = reference_scene(asset)
    task = reference_task(scene)
    platform.assets.register(asset)
    platform.scenes.register(scene)
    platform.tasks.register(task)
    request = reference_run(scene, task, episode_count=1).model_copy(
        update={"stop_on_success": False}
    )
    record = platform.runs.submit(request)
    episode = platform.runs.episodes(record)[0]
    assert episode.success
    assert len(episode.steps) == task.max_steps + 1
