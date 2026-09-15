"""Exercise learning/feedback contracts on explicitly synthetic, bounded episode data."""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import pytest

from physrealx import PhysRealX
from physrealx.contracts import Frame
from physrealx.domain import (
    ActionSpec,
    ArtifactRef,
    DatasetManifest,
    DataValue,
    Episode,
    EpisodeSource,
    EpisodeStep,
    EpisodeSummary,
    MethodCapability,
    Predicate,
    PredicateOperator,
    SignalDType,
    SignalSpec,
    TaskSpec,
    TrainingSpec,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations.datasets import EpisodeCollection, load_snapshot
from physrealx.integrations.feedback import FailureQuery, RunEvidence
from physrealx.methods import builtin_method
from physrealx.reference import reference_asset, reference_scene
from physrealx.training_lifecycle import WorldModelBundle

np = pytest.importorskip("numpy")


def output(operation, name):
    return next(row.artifact for row in operation.outputs if row.name == name)


def read(px, reference):
    return json.loads(px.artifacts.get_bytes(reference))


def array(px, value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return px.artifacts.put_bytes(stream.getvalue(), media_type="application/x-npy")


@pytest.fixture
def episode_data(tmp_path: Path):
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    asset = reference_asset()
    px.assets.register(asset)
    scene = reference_scene(asset)
    scene_ref = px.scenes.register(scene)
    signals = tuple(
        SignalSpec(
            name=name,
            shape=(1,),
            dtype=SignalDType.FLOAT32,
            unit=unit,
            frame="joint",
            frequency_hz=20,
        )
        for name, unit in (
            ("joint.position", "m"),
            ("joint.velocity", "m/s"),
            ("joint.effort", "N"),
            ("contact.force", "N"),
        )
    )
    signals += (
        SignalSpec(
            name="camera.front.rgb",
            shape=(32, 32, 3),
            dtype=SignalDType.UINT8,
            unit="1",
            frame="camera.front",
            frequency_hz=20,
        ),
    )
    action = SignalSpec(
        name="joint.command",
        shape=(1,),
        dtype=SignalDType.FLOAT32,
        unit="m",
        frame="joint",
        frequency_hz=20,
    )
    task = TaskSpec(
        resource_id="learning.task",
        version="1.0.0",
        scene=scene_ref,
        subject_entity=scene.entities[0].name,
        instruction="synthetic fixture: open",
        observations=signals,
        action=ActionSpec(kind="joint_position", signal=action, lower=(0,), upper=(1,)),
        max_steps=20,
        initial_conditions=(),
        success_conditions=(
            Predicate(signal="joint.position", operator=PredicateOperator.GE, value=0.3),
        ),
        metadata={"target_positions": [0.3]},
    )
    task_ref = px.tasks.register(task)
    method = builtin_method(
        MethodCapability.SIMULATION_RUN,
        "synthetic_test",
        input_contract="fixture/1",
        output_contract="fixture/1",
    ).bind()
    policy = builtin_method(
        MethodCapability.POLICY_INFERENCE,
        "synthetic_test",
        input_contract="fixture/1",
        output_contract="fixture/1",
    ).bind()
    summaries, episodes = [], []
    for index in range(8):
        success = index < 6
        steps = []
        for step in range(9):
            timestamp = step * 0.05
            position = step * 0.3 / 8 if success else 0.0
            values = {
                signal.name: DataValue(
                    dtype=signal.dtype,
                    shape=signal.shape,
                    unit=signal.unit,
                    frame=signal.frame,
                    timestamp_seconds=timestamp,
                    inline=(value,),
                )
                for signal, value in zip(signals, (position, 0.75 if success else 0, 0.1, 0.0))
            }
            rgb = np.full((32, 32, 3), 30 + int(position * 400), np.uint8)
            values["camera.front.rgb"] = DataValue(
                dtype=SignalDType.UINT8,
                shape=rgb.shape,
                unit="1",
                frame="camera.front",
                timestamp_seconds=timestamp,
                artifact=array(px, rgb),
            )
            terminal = step == 8
            command = (
                None
                if terminal
                else DataValue(
                    dtype=SignalDType.FLOAT32,
                    shape=(1,),
                    unit="m",
                    frame="joint",
                    inline=(min(0.3, position + 0.0375) if success else 0,),
                )
            )
            steps.append(
                EpisodeStep(
                    index=step,
                    timestamp_seconds=timestamp,
                    observations=values,
                    state=values,
                    action=command,
                    reward=None if terminal else 0,
                    discount=None if terminal else 1,
                    is_first=step == 0,
                    is_last=terminal,
                    is_terminal=terminal and success,
                )
            )
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            run_id=str(uuid.uuid4()),
            index=0,
            source=EpisodeSource.SIMULATOR,
            seed=index,
            scene=scene_ref,
            task=task_ref,
            backend=method,
            policy=policy,
            steps=tuple(steps),
            success=success,
            failure_reason=None if success else "task_timeout",
            metadata={"fixture": "synthetic trajectory; not physics validation"},
        )
        reference = px.artifacts.put_json(episode)
        summaries.append(
            EpisodeSummary(
                episode_id=episode.episode_id,
                run_id=episode.run_id,
                index=0,
                step_count=len(steps),
                success=success,
                failure_reason=episode.failure_reason,
                artifact=reference,
            )
        )
        episodes.append(episode)
    dataset = DatasetManifest(
        resource_id="learning.dataset",
        version="1.0.0",
        episodes=tuple(summaries),
        source_runs=tuple(row.run_id for row in summaries),
        observation_schema=signals,
        action_schema=action,
        total_steps=sum(row.step_count for row in summaries),
    )
    reference = px.datasets.register(dataset)
    collection = px.artifacts.put_json(EpisodeCollection(datasets=(reference,)))
    operation = px.datasets.ingest(
        collection, method="episode_dataset", config={"resource_id": "learning.collection"}
    )
    snapshot = px.datasets.snapshot(
        output(operation, "dataset"), method="episode_split", config={"seed": 42}
    )
    return px, dataset, reference, task, episodes, output(snapshot, "snapshot")


def test_split_pins_sources_and_feedback_keeps_original_failure_labels(episode_data):
    px, dataset, reference, _, _, snapshot_ref = episode_data
    snapshot, _, splits = load_snapshot(px.artifacts, snapshot_ref)
    assert set(splits) == {"train", "validation", "test"}
    assert px.artifacts.verify(snapshot.code.source_tree)
    assert px.artifacts.verify(snapshot.environment.lockfile)
    evidence = px.artifacts.put_json(RunEvidence(dataset=reference))
    archive = px.telemetry.record(evidence, method="episode_telemetry")
    analysis = px.failures.analyze(output(archive, "telemetry"), method="trajectory_diagnostics")
    report = output(analysis, "failure_report")
    assert {row["category"] for row in read(px, report)["failures"]} == {"policy_target_mismatch"}
    query = px.artifacts.put_json(FailureQuery(report=report, resource_id="learning.failures"))
    feedback = px.feedback.build_dataset(query, method="failure_episodes")
    selected = DatasetManifest.model_validate(read(px, output(feedback, "dataset"))["manifest"])
    assert len(selected.episodes) == 2 and all(not row.success for row in selected.episodes)
    assert {row.artifact for row in selected.episodes} <= {row.artifact for row in dataset.episodes}
    tampered = read(px, output(archive, "telemetry"))
    tampered["episodes"][0]["success"] = False
    with pytest.raises(DomainValidationError, match="differ"):
        px.failures.analyze(px.artifacts.put_json(tampered), method="trajectory_diagnostics")


def test_act_official_network_checkpoint_and_action_chunks_roundtrip(episode_data):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("einops")
    pytest.importorskip("safetensors")
    from physrealx.integrations.act import ACTPolicy, ACTTrainer

    px, _, reference, task, episodes, _ = episode_data
    previous = torch.get_num_threads()
    ref = px.training.submit(
        TrainingSpec(
            resource_id="learning.act",
            version="1.0.0",
            dataset=reference,
            trainer=ACTTrainer.descriptor.bind(
                config={"steps": 2, "chunk_size": 3, "batch_size": 2}
            ),
        )
    )
    model = px.training.get_model(ref)
    assert model.policy.config["n_action_steps"] == 3
    assert model.training_config["code"]["source_tree"]["artifact_id"].startswith("art_")
    policy = ACTPolicy({"model_artifact_id": ref.artifact_id}, px.artifacts)
    policy.reset(0)
    action = policy.act(Frame(observations=episodes[0].steps[0].observations, state={}), task, 0)
    assert 0 <= action.inline[0] <= 1
    assert len(policy._commands) == 2
    policy.reset(1)
    assert not policy._commands
    policy.close()
    assert torch.get_num_threads() == previous


def test_dino_wm_official_causal_predictor_and_typed_lifecycle(episode_data, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    pytest.importorskip("safetensors")
    from safetensors.torch import save

    from physrealx.integrations import world_model as module
    from physrealx.integrations.learning_sources import algorithm_module

    # The upstream causal predictor is exercised directly. A tiny encoder is an
    # injected dependency for this data/lifecycle test; the real pretrained DINO
    # encoder is exercised by the opt-in notebook acceptance run.
    predictor = (
        algorithm_module("dino_wm")
        .ViTPredictor(num_patches=4, num_frames=3, dim=12, depth=1, heads=2, dim_head=6, mlp_dim=24)
        .eval()
    )
    x = torch.randn(1, 12, 12)
    changed = x.clone()
    changed[:, 8:] += 10
    with torch.inference_mode():
        torch.testing.assert_close(predictor(x)[:, :8], predictor(changed)[:, :8])

    class FixtureEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchor", torch.tensor(1.0))

        def forward_features(self, rgb):
            features = (
                torch.nn.functional.adaptive_avg_pool2d(rgb, (2, 2)).flatten(2).transpose(1, 2)
            )
            return {"x_norm_patchtokens": features.repeat(1, 1, 128) * self.anchor}

    monkeypatch.setattr(module, "encoder_network", FixtureEncoder)
    px, _, _, _, episodes, snapshot_ref = episode_data
    weights = px.artifacts.put_bytes(
        save(FixtureEncoder().state_dict()), media_type="application/vnd.safetensors"
    )
    encoder = px.artifacts.put_json(module.EncoderSpec(weights=weights))
    trained = px.world_models.train(
        snapshot_ref,
        encoder,
        method="dino_wm",
        config={"steps": 2, "image_size": 28, "history": 2, "depth": 1, "batch_size": 2},
    )
    model_ref = output(trained, "model")
    bundle = WorldModelBundle.model_validate(read(px, model_ref))
    assert bundle.report.checkpoint.media_type == "application/vnd.safetensors"
    assert bundle.dataset.split.record_id == bundle.report.split_record_id
    snapshot, dataset, _ = load_snapshot(px.artifacts, snapshot_ref)
    summary = next(row for row in dataset.episodes if row.episode_id in snapshot.split.test)
    context = px.artifacts.put_json(module.WorldModelContext(episode=summary, start_index=2))
    actions = px.artifacts.put_json(
        module.WorldModelActions(values=((0.2,), (0.3,)), interval_seconds=0.05)
    )
    rollout = px.world_models.rollout(model_ref, context, actions, method="dino_wm")
    document = read(px, output(rollout, "rollout"))
    state = module.read_array(px.artifacts, ArtifactRef.model_validate(document["states"]))
    assert state.shape == (2, 2) and np.isfinite(state).all()
    assert (
        document["source"] == "world_model" and document["independent_physical_validation"] is False
    )
    reference = next(row for row in episodes if row.episode_id == summary.episode_id)
    evaluation = module.WorldModelRolloutEvaluation.model_validate(
        read(
            px,
            px.world_models.evaluate_rollout(output(rollout, "rollout"), reference, start_index=2),
        )
    )
    assert evaluation.horizon == 2
    assert evaluation.reference_episode_id == reference.episode_id
    assert evaluation.reference_states.media_type == "application/x-npy"
    assert set(evaluation.signals) == {"joint.position", "joint.velocity"}
    assert all(
        len(signal.rmse) == signal.column_stop - signal.column_start
        for signal in evaluation.signals.values()
    )
    invalid = px.artifacts.put_json(
        module.WorldModelActions(values=((0.2,),), interval_seconds=0.1)
    )
    with pytest.raises(DomainValidationError, match="frequency"):
        px.world_models.rollout(model_ref, context, invalid, method="dino_wm")


def test_training_augmentation_rejects_heldout_scenarios_and_bad_demonstrations(episode_data):
    px, dataset, _, _, episodes, snapshot_ref = episode_data
    _, _, partitions = load_snapshot(px.artifacts, snapshot_ref)
    selected_ref = px.datasets.training_subset(snapshot_ref, resource_id="training.selected")
    selected = px.datasets.get(selected_ref)
    assert {row.episode_id for row in selected.episodes} == {
        row.episode_id for row in partitions["train"] if row.success
    }

    def augment(source, *, seed=None, tags=None):
        # Regeneration has new record IDs; scenario identity must still be protected.
        episode = source.model_copy(
            update={
                "episode_id": str(uuid.uuid4()),
                "run_id": str(uuid.uuid4()),
                "seed": source.seed if seed is None else seed,
                "metadata": {"tags": [] if tags is None else tags},
            }
        )
        artifact = px.artifacts.put_json(episode)
        summary = EpisodeSummary(
            episode_id=episode.episode_id,
            run_id=episode.run_id,
            index=0,
            step_count=len(episode.steps),
            success=episode.success,
            failure_reason=episode.failure_reason,
            artifact=artifact,
        )
        return px.datasets.register(
            dataset.model_copy(
                update={
                    "resource_id": "augmentation." + episode.episode_id,
                    "episodes": (summary,),
                    "source_runs": (episode.run_id,),
                    "total_steps": summary.step_count,
                }
            )
        )

    heldout = partitions["test"][0]
    with pytest.raises(DomainValidationError, match="held-out"):
        px.datasets.training_subset(
            snapshot_ref, resource_id="leaked", augmentations=(augment(heldout),)
        )
    failed = next(e for e in episodes if not e.success)
    with pytest.raises(DomainValidationError, match="successful demonstrations"):
        px.datasets.training_subset(
            snapshot_ref, resource_id="failed", augmentations=(augment(failed, seed=999),)
        )
    expert = next(e for e in episodes if e.success)
    extra = augment(expert, seed=998, tags=["expert"])
    result = px.datasets.training_subset(
        snapshot_ref, resource_id="augmented", augmentations=(extra,)
    )
    assert len(px.datasets.get(result).episodes) == len(selected.episodes) + 1
    with pytest.raises(DomainValidationError, match="duplicates"):
        px.datasets.training_subset(
            snapshot_ref, resource_id="duplicates", augmentations=(extra, extra)
        )
    with pytest.raises(DomainValidationError, match="required tag"):
        px.datasets.training_subset(
            snapshot_ref,
            resource_id="unlabelled",
            augmentations=(extra,),
            required_tag="verified_expert",
        )
