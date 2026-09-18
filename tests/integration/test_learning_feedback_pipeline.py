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
    ResourceType,
    SignalDType,
    SignalSpec,
    TaskSpec,
    TrainingSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations.datasets import EpisodeCollection, load_snapshot
from physrealx.integrations.feedback import (
    DeploymentEvidence,
    FailureQuery,
    PolicyInterventionEvidence,
)
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
        output(operation, "dataset"),
        method="episode_split",
        config={"seed": 42, "stratify_success": True},
    )
    return px, dataset, reference, task, episodes, output(snapshot, "snapshot")


def test_split_pins_sources_and_feedback_keeps_original_failure_labels(episode_data):
    px, dataset, reference, _, _, snapshot_ref = episode_data
    snapshot, _, splits = load_snapshot(px.artifacts, snapshot_ref)
    assert set(splits) == {"train", "validation", "test"}
    assert {episode.success for episode in splits["train"]} == {False, True}
    assert {episode.success for name in ("validation", "test") for episode in splits[name]} == {
        False,
        True,
    }
    assert px.artifacts.verify(snapshot.code.source_tree)
    assert px.artifacts.verify(snapshot.environment.lockfile)
    preview = px.datasets.preview_snapshot(snapshot_ref)
    assert px.artifacts.get_bytes(preview).startswith(b"\x89PNG\r\n\x1a\n")
    evidence = px.artifacts.put_json(DeploymentEvidence(dataset=reference))
    archive = px.telemetry.record(evidence, method="episode_telemetry")
    analysis = px.failures.analyze(output(archive, "telemetry"), method="trajectory_diagnostics")
    report = output(analysis, "failure_report")
    assert {row["category"] for row in read(px, report)["failures"]} == {"perception"}
    query = px.artifacts.put_json(FailureQuery(report=report, resource_id="learning.failures"))
    feedback = px.feedback.build_dataset(query, method="failure_episodes")
    selected = DatasetManifest.model_validate(read(px, output(feedback, "dataset"))["manifest"])
    assert len(selected.episodes) == 2 and all(not row.success for row in selected.episodes)
    assert {row.artifact for row in selected.episodes} <= {row.artifact for row in dataset.episodes}
    tampered = read(px, output(archive, "telemetry"))
    tampered["episodes"][0]["success"] = False
    with pytest.raises(DomainValidationError, match="differ"):
        px.failures.analyze(px.artifacts.put_json(tampered), method="trajectory_diagnostics")


def test_feedback_validates_policy_intervention_and_renders_evidence(episode_data):
    from physrealx.domain import EvaluationReport

    px, dataset, reference, _, episodes, _ = episode_data
    archive = px.telemetry.record(
        px.artifacts.put_json(DeploymentEvidence(dataset=reference)),
        method="episode_telemetry",
    )
    telemetry = output(archive, "telemetry")
    analysis = px.failures.analyze(telemetry, method="trajectory_diagnostics")
    failure_report = output(analysis, "failure_report")
    failed = [episode for episode in episodes if not episode.success]
    successful = next(episode for episode in episodes if episode.success)
    recovery_refs = []
    recovery_episodes = []
    recovery_policy = builtin_method(
        MethodCapability.POLICY_INFERENCE,
        "recovery_fixture",
        input_contract="fixture/1",
        output_contract="fixture/1",
    ).bind()
    for index, original in enumerate(failed):
        recovery = successful.model_copy(
            update={
                "episode_id": str(uuid.uuid4()),
                "run_id": str(uuid.uuid4()),
                "seed": original.seed,
                "scene": original.scene,
                "task": original.task,
                "backend": original.backend,
                "policy": recovery_policy,
                "metadata": {"tags": ["expert_demonstration", "failure_recovery"]},
            }
        )
        artifact = px.artifacts.put_json(recovery)
        summary = EpisodeSummary(
            episode_id=recovery.episode_id,
            run_id=recovery.run_id,
            index=0,
            success=True,
            step_count=len(recovery.steps),
            artifact=artifact,
        )
        recovery_ref = px.datasets.register(
            dataset.model_copy(
                update={
                    "resource_id": f"recovery.{index}",
                    "episodes": (summary,),
                    "source_runs": (recovery.run_id,),
                    "total_steps": len(recovery.steps),
                }
            )
        )
        recovery_refs.append(recovery_ref)
        recovery_episodes.append(recovery)
    causal = px.feedback.analyze_policy_intervention(
        px.artifacts.put_json(
            PolicyInterventionEvidence(
                failure_report=failure_report,
                recovery_datasets=tuple(recovery_refs),
            )
        )
    )
    causal_document = read(px, causal)
    assert causal_document["validated_count"] == len(failed)
    plan = px.feedback.plan_physics_calibration(telemetry, failure_report)
    assert read(px, plan)["status"] == "AWAITING_REAL_EVIDENCE"

    model_artifact = px.artifacts.put_json({"model": "initial"})
    initial_report = EvaluationReport(
        resource_id="feedback.initial",
        version="1.0.0",
        model=VersionedRef(
            resource_type=ResourceType.MODEL,
            resource_id="feedback.model",
            version="1.0.0",
            artifact_id=model_artifact.artifact_id,
        ),
        dataset=reference,
        success_rate=6 / 8,
        metrics={"episode_count": 8.0, "success_count": 6.0},
        failures=(),
    )
    updated_dataset = px.datasets.register(
        dataset.model_copy(
            update={
                "resource_id": "feedback.updated.dataset",
                "episodes": tuple(
                    EpisodeSummary(
                        episode_id=episode.episode_id,
                        run_id=episode.run_id,
                        index=episode.index,
                        success=episode.success,
                        step_count=len(episode.steps),
                        artifact=px.artifacts.put_json(episode),
                        failure_reason=episode.failure_reason,
                    )
                    for episode in recovery_episodes
                ),
                "source_runs": tuple(episode.run_id for episode in recovery_episodes),
                "total_steps": sum(len(episode.steps) for episode in recovery_episodes),
            }
        )
    )
    updated_report = initial_report.model_copy(
        update={
            "resource_id": "feedback.updated",
            "model": VersionedRef(
                resource_type=ResourceType.MODEL,
                resource_id="feedback.model.updated",
                version="1.0.0",
                artifact_id=px.artifacts.put_json({"model": "updated"}).artifact_id,
            ),
            "dataset": updated_dataset,
            "success_rate": 1.0,
            "metrics": {
                "episode_count": float(len(recovery_episodes)),
                "success_count": float(len(recovery_episodes)),
            },
        }
    )
    from physrealx.integrations.feedback import assess_policy_update
    from physrealx.integrations.feedback_preview import (
        preview_feedback_cycle,
        preview_feedback_report,
    )

    failed_ids = {episode.episode_id for episode in failed}
    initial_paired_dataset = px.datasets.register(
        dataset.model_copy(
            update={
                "resource_id": "feedback.initial.paired.dataset",
                "episodes": tuple(
                    summary for summary in dataset.episodes if summary.episode_id in failed_ids
                ),
                "source_runs": tuple(episode.run_id for episode in failed),
                "total_steps": sum(len(episode.steps) for episode in failed),
            }
        )
    )
    initial_paired = initial_report.model_copy(
        update={
            "resource_id": "feedback.initial.paired",
            "dataset": initial_paired_dataset,
            "success_rate": 0.0,
            "metrics": {
                "episode_count": float(len(failed)),
                "success_count": 0.0,
            },
        }
    )
    assessment = assess_policy_update(px.artifacts, initial_paired, updated_report)
    assert read(px, assessment)["status"] == "PROMOTE"

    dashboard = preview_feedback_report(
        px.artifacts,
        initial_report,
        updated_report,
        failure_report,
        causal,
        plan,
        assessment,
    )
    assert px.artifacts.get_bytes(dashboard).startswith(b"\x89PNG\r\n\x1a\n")
    updated_policy = builtin_method(
        MethodCapability.POLICY_INFERENCE,
        "updated_fixture",
        input_contract="fixture/1",
        output_contract="fixture/1",
    ).bind()
    updated_episode = recovery_episodes[0].model_copy(
        update={
            "episode_id": str(uuid.uuid4()),
            "run_id": str(uuid.uuid4()),
            "policy": updated_policy,
        }
    )
    updated_artifact = px.artifacts.put_json(updated_episode)
    cycle = preview_feedback_cycle(
        px.artifacts,
        next(episode for episode in dataset.episodes if not episode.success).artifact,
        px.datasets.get(recovery_refs[0]).episodes[0].artifact,
        updated_artifact,
        frames=3,
        fps=2,
    )
    assert px.artifacts.get_bytes(cycle).startswith(b"GIF")


def test_real_feedback_requires_deployment_attestation_and_prepares_calibration(episode_data):
    from physrealx.robot_agent import DeploymentMode, RobotExecutionEvidence

    px, dataset, _, _, episodes, _ = episode_data
    summaries = []
    real_episodes = []
    for index, source in enumerate(episodes[:2]):
        episode = source.model_copy(
            update={
                "episode_id": str(uuid.uuid4()),
                "run_id": str(uuid.uuid4()),
                "source": EpisodeSource.REAL,
                "metadata": {"capture_id": f"robot-capture-{index}"},
            }
        )
        artifact = px.artifacts.put_json(episode)
        summaries.append(
            EpisodeSummary(
                episode_id=episode.episode_id,
                run_id=episode.run_id,
                index=episode.index,
                success=episode.success,
                step_count=len(episode.steps),
                artifact=artifact,
                failure_reason=episode.failure_reason,
            )
        )
        real_episodes.append(episode)
    real_dataset = px.datasets.register(
        dataset.model_copy(
            update={
                "resource_id": "learning.real.deployment",
                "episodes": tuple(summaries),
                "source_runs": tuple(episode.run_id for episode in real_episodes),
                "total_steps": sum(len(episode.steps) for episode in real_episodes),
            }
        )
    )
    with pytest.raises(DomainValidationError, match="deployment report"):
        px.telemetry.record(
            px.artifacts.put_json(DeploymentEvidence(dataset=real_dataset)),
            method="episode_telemetry",
        )

    deployment_report = px.artifacts.put_json(
        RobotExecutionEvidence(
            status="COMPLETED",
            deployment_plan=px.artifacts.put_json({"kind": "deployment-plan"}),
            telemetry=px.artifacts.put_bytes(b"mcap", media_type="application/vnd.mcap"),
            safety_events=px.artifacts.put_json({"events": []}),
            final_mode=DeploymentMode.REVOKED,
            started_at_ns=0,
            completed_at_ns=1,
        )
    )
    archive = px.telemetry.record(
        px.artifacts.put_json(
            DeploymentEvidence(dataset=real_dataset, deployment_report=deployment_report)
        ),
        method="episode_telemetry",
    )
    manifest = read(px, output(archive, "telemetry_manifest"))
    assert manifest["live_robot_execution"] is True
    calibration = px.feedback.prepare_physics_calibration(
        output(archive, "telemetry"),
        joint_name="joint-0",
        joint_type="hinge",
        joint_index=0,
        prior_parameters=(1.0, 0.1, 0.05),
    )
    assert read(px, calibration)["schema"] == ("physrealx.joint-system-identification-evidence/1")


def test_act_official_network_checkpoint_and_action_chunks_roundtrip(episode_data):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("einops")
    pytest.importorskip("safetensors")
    from physrealx.integrations.act import ACTPolicy, ACTTrainer

    px, _, reference, task, episodes, _ = episode_data
    previous = torch.get_num_threads()
    spec = TrainingSpec(
        resource_id="learning.act",
        version="1.0.0",
        dataset=reference,
        trainer=ACTTrainer.descriptor.bind(
            config={
                "steps": 2,
                "chunk_size": 3,
                "n_action_steps": 2,
                "batch_size": 2,
            }
        ),
    )
    ref = px.training.submit(spec)
    assert px.training.submit(spec) == ref
    model = px.training.get_model(ref)
    from physrealx.domain import EvaluationReport
    from physrealx.integrations.learning_preview import preview_policy_training

    policy_preview = preview_policy_training(
        px.artifacts,
        ref,
        model,
        EvaluationReport(
            resource_id="learning.act.preview",
            version="1.0.0",
            model=ref,
            success_rate=1.0,
            metrics={"episode_count": 2.0, "success_count": 2.0},
            failures=(),
        ),
    )
    assert px.artifacts.get_bytes(policy_preview).startswith(b"\x89PNG\r\n\x1a\n")
    assert model.policy.config["n_action_steps"] == 2
    assert model.training_config["code"]["source_tree"]["artifact_id"].startswith("art_")
    policy = ACTPolicy({"model_artifact_id": ref.artifact_id, "n_action_steps": 2}, px.artifacts)
    policy.reset(0)
    action = policy.act(Frame(observations=episodes[0].steps[0].observations, state={}), task, 0)
    assert 0 <= action.inline[0] <= 1
    assert len(policy._commands) == 1
    policy.reset(1)
    assert not policy._commands
    policy.close()
    tuned_ref = px.training.submit(
        TrainingSpec(
            resource_id="learning.act.tuned",
            version="1.0.0",
            dataset=reference,
            trainer=ACTTrainer.descriptor.bind(
                checkpoint=model.weights,
                config={
                    "steps": 1,
                    "chunk_size": 3,
                    "n_action_steps": 2,
                    "batch_size": 2,
                },
            ),
        )
    )
    tuned = px.training.get_model(tuned_ref)
    assert tuned.training_config["training_scope"] == "fine_tune_on_selected_episode_split"
    assert tuned.training_config["parent_checkpoint"] == model.weights.model_dump(mode="json")
    assert tuned.trainer.checkpoint == model.weights
    assert torch.get_num_threads() == previous


def test_iql_trains_on_mixed_returns_and_roundtrips_policy(episode_data):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from physrealx.integrations.iql import SOURCE_REVISION, IQLPolicy, IQLTrainer

    px, _, _, task, episodes, snapshot_ref = episode_data
    dataset_ref = px.datasets.training_subset(
        snapshot_ref,
        resource_id="learning.iql.dataset",
        selection="all_episodes",
    )
    dataset = px.datasets.get(dataset_ref)
    assert {row.success for row in dataset.episodes} == {False, True}
    model_ref = px.training.submit(
        TrainingSpec(
            resource_id="learning.iql",
            version="1.0.0",
            dataset=dataset_ref,
            trainer=IQLTrainer.descriptor.bind(
                config={"steps": 12, "batch_size": 16, "hidden_dim": 32}
            ),
        )
    )
    model = px.training.get_model(model_ref)
    assert model.metrics["transitions"] > 0
    assert np.isfinite(tuple(model.metrics.values())).all()
    assert model.training_config["source_revision"] == SOURCE_REVISION
    policy = IQLPolicy({"model_artifact_id": model_ref.artifact_id}, px.artifacts)
    policy.reset(0)
    action = policy.act(Frame(observations=episodes[0].steps[0].observations, state={}), task, 0)
    assert action.inline is not None and 0 <= action.inline[0] <= 1
    policy.close()
    assert torch.get_num_threads() > 0


def test_external_training_interfaces_are_pinned_validated_and_visible(episode_data):
    px, _, _, _, _, snapshot_ref = episode_data
    openvla = px.training.prepare_interface(snapshot_ref, method="openvla_oft")
    tdmpc2 = px.training.prepare_interface(snapshot_ref, method="td_mpc2")
    openvla_document = read(px, openvla)
    tdmpc2_document = read(px, tdmpc2)
    assert openvla_document["status"] == tdmpc2_document["status"] == "PREPARED"
    assert openvla_document["source"]["revision"] == ("e4287e94541f459edc4feabc4e181f537cd569a8")
    assert tdmpc2_document["source"]["revision"] == ("e9f59321933cbc8e11a002b842adc7d4ffae8ff1")
    assert openvla_document["input_mapping"]["full_image"] == ["camera.front.rgb"]
    assert tdmpc2_document["input_mapping"]["reward"] == ["transition.reward"]
    assert all(
        name.startswith("joint.") for name in tdmpc2_document["input_mapping"]["observation"]
    )
    assert openvla_document["train_transitions"] == tdmpc2_document["train_transitions"]
    preview = px.training.preview_interfaces((openvla, tdmpc2))
    assert px.artifacts.get_bytes(preview).startswith(b"\x89PNG\r\n\x1a\n")


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
    evaluation_ref = px.world_models.evaluate_rollout(
        output(rollout, "rollout"), reference, start_index=2
    )
    evaluation = module.WorldModelRolloutEvaluation.model_validate(read(px, evaluation_ref))
    assert evaluation.horizon == 2
    assert evaluation.reference_episode_id == reference.episode_id
    assert evaluation.reference_states.media_type == "application/x-npy"
    assert set(evaluation.signals) == {"joint.position", "joint.velocity"}
    assert all(
        len(signal.rmse) == signal.column_stop - signal.column_start
        for signal in evaluation.signals.values()
    )
    preview = px.world_models.preview_rollout(
        output(rollout, "rollout"),
        evaluation_ref,
        reference,
        frames=2,
        fps=2,
    )
    assert px.artifacts.get_bytes(preview).startswith(b"GIF")
    invalid = px.artifacts.put_json(
        module.WorldModelActions(values=((0.2,),), interval_seconds=0.1)
    )
    with pytest.raises(DomainValidationError, match="frequency"):
        px.world_models.rollout(model_ref, context, invalid, method="dino_wm")


def test_training_augmentation_rejects_heldout_scenarios_and_bad_demonstrations(episode_data):
    px, dataset, _, _, episodes, snapshot_ref = episode_data
    _, _, partitions = load_snapshot(px.artifacts, snapshot_ref)
    selected_ref = px.datasets.training_subset(snapshot_ref, resource_id="training.selected")
    assert (
        px.datasets.training_subset(snapshot_ref, resource_id="training.selected") == selected_ref
    )
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
