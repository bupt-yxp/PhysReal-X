from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.contracts import SimulationSession
from physrealx.domain import AssetSpec, Episode, JSONValue, RunStatus, TaskSpec, TrainingSpec
from physrealx.errors import DomainValidationError
from physrealx.platform import PhysRealX
from physrealx.reference import reference_asset, reference_run, reference_task


def test_reference_run_publishes_replayable_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    backend = platform.plugins.backend("reference_drawer")
    original_create_session = backend.create_session
    created_sessions = 0

    def counted_create_session(
        asset_spec: AssetSpec,
        task_spec: TaskSpec,
        config: Mapping[str, JSONValue],
    ) -> SimulationSession:
        nonlocal created_sessions
        created_sessions += 1
        return original_create_session(asset_spec, task_spec, config)

    monkeypatch.setattr(backend, "create_session", counted_create_session)
    asset = reference_asset()
    task = reference_task(asset)
    asset_ref = platform.register_asset(asset)
    task_ref = platform.register_task(task)
    run = reference_run(asset, task, episode_count=4, base_seed=11)

    record = platform.execute(run)

    assert record.status is RunStatus.SUCCEEDED
    assert created_sessions == 1
    assert record.spec.backend_version == "1.0.0"
    assert record.spec.policy_version == "1.0.0"
    assert record.result is not None
    assert record.result.dataset is not None
    assert record.result.episode_count == 4
    dataset = platform.load_dataset(record.result.dataset)
    assert len(dataset.episodes) == 4
    assert all(item.success for item in dataset.episodes)
    assert dataset.metadata["asset_digest"] == asset_ref.digest
    assert dataset.metadata["task_digest"] == task_ref.digest
    assert dataset.metadata["backend_version"] == "1.0.0"
    assert dataset.metadata["policy_version"] == "1.0.0"
    episodes = platform.episodes(run.run_id)
    assert tuple(item.digest for item in episodes) == tuple(
        item.digest for item in platform.episodes(run.run_id)
    )
    invalid_episode = episodes[0].model_dump(mode="json")
    invalid_episode["steps"][-1]["observations"]["drawer.position"]["unit"] = "cm"
    with pytest.raises(ValidationError, match="schemas must remain constant"):
        Episode.model_validate(invalid_episode)

    # Submitting an identical immutable RunSpec is idempotent.
    assert platform.execute(run) == record


@pytest.mark.integration
def test_dataset_trains_and_evaluates_torch_policy(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    asset = reference_asset()
    task = reference_task(asset)
    platform.register_asset(asset)
    platform.register_task(task)
    base_run = reference_run(asset, task, episode_count=8, base_seed=31)
    record = platform.execute(base_run)
    assert record.result is not None and record.result.dataset is not None

    model = platform.train(
        TrainingSpec(
            resource_id="drawer.bc.test",
            version="1.0.0",
            dataset=record.result.dataset,
            trainer="torch_bc",
            config={"device": "cpu", "epochs": 80, "seed": 3},
        )
    )
    manifest = platform.load_model(model)
    assert manifest.metrics["training_mse"] < 0.01
    assert manifest.policy == "torch_bc"
    assert manifest.policy_version == "1.0.0"
    assert platform.artifacts.verify(manifest.weights)

    evaluation_ref = platform.evaluate(
        resource_id="drawer.bc.test.eval",
        version="1.0.0",
        model=model,
        run=base_run,
        episode_count=4,
        base_seed=50_000,
    )
    report = platform.load_evaluation(evaluation_ref)
    assert report.success_rate >= 0.75
    assert report.metrics["episode_count"] == 4.0

    # The evaluation identity includes the requested output identity, not only
    # the model and base task, so independent reports cannot collide.
    second_evaluation = platform.evaluate(
        resource_id="drawer.bc.test.eval.other",
        version="1.0.0",
        model=model,
        run=base_run,
        episode_count=1,
        base_seed=50_000,
    )
    assert second_evaluation.digest != evaluation_ref.digest

    incompatible_task_payload = task.model_dump(mode="json")
    incompatible_task_payload["version"] = "1.1.0"
    incompatible_task_payload["observations"][0]["unit"] = "cm"
    incompatible_task = TaskSpec.model_validate(incompatible_task_payload)
    platform.register_task(incompatible_task)
    with pytest.raises(DomainValidationError, match="schema is incompatible"):
        platform.evaluate(
            resource_id="drawer.bc.test.eval.incompatible",
            version="1.0.0",
            model=model,
            run=reference_run(asset, incompatible_task, episode_count=1),
            episode_count=1,
        )
