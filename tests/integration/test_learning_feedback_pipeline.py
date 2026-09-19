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
    ModelManifest,
    Predicate,
    PredicateOperator,
    SignalDType,
    SignalSpec,
    TaskSpec,
    TrainingSpec,
)
from physrealx.errors import DomainValidationError, ResourceConflictError
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


def fixture_model(px, dataset_ref, resource_id, *, parent=None):
    dataset = px.datasets.get(dataset_ref)
    model = ModelManifest(
        resource_id=resource_id,
        version="1.0.0",
        trainer=builtin_method(
            MethodCapability.POLICY_TRAIN,
            "fixture",
            input_contract="fixture/1",
            output_contract="fixture/1",
        ).bind(checkpoint=parent),
        policy=builtin_method(
            MethodCapability.POLICY_INFERENCE,
            "fixture",
            input_contract="fixture/1",
            output_contract="fixture/1",
        ).bind(),
        dataset=dataset_ref,
        weights=px.artifacts.put_bytes(
            b"synthetic model fixture", media_type="application/octet-stream"
        ),
        observation_signals=dataset.observation_schema,
        action_signal=dataset.action_schema,
        training_config={},
        metrics={},
    )
    artifact = px.artifacts.put_json(model, media_type="application/vnd.physrealx.model+json")
    return model.reference(artifact.artifact_id), artifact


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
                for signal, value in zip(
                    signals, (position, 0.75 if success and step > 0 else 0, 0.1, 0.0)
                )
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
    selected_ref = px.datasets.register(selected)
    copied_report = px.artifacts.put_json(read(px, report))
    assert copied_report.artifact_id != report.artifact_id
    repeated = px.feedback.build_dataset(
        px.artifacts.put_json(FailureQuery(report=copied_report, resource_id="learning.failures")),
        method="failure_episodes",
    )
    repeated_dataset = DatasetManifest.model_validate(
        read(px, output(repeated, "dataset"))["manifest"]
    )
    assert px.datasets.register(repeated_dataset) == selected_ref
    tampered = read(px, output(archive, "telemetry"))
    tampered["episodes"][0]["success"] = False
    with pytest.raises(DomainValidationError, match="differ"):
        px.failures.analyze(px.artifacts.put_json(tampered), method="trajectory_diagnostics")


def test_feedback_validates_policy_intervention_and_renders_evidence(episode_data, monkeypatch):
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
                "steps": tuple(
                    step.model_copy(
                        update={
                            "observations": {
                                **step.observations,
                                "contact.force": step.observations["contact.force"].model_copy(
                                    update={
                                        "inline": (100.0 if 2 <= step.index <= 4 else 0.0,),
                                    }
                                ),
                            },
                        }
                    )
                    for step in successful.steps
                ),
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
    assert all(row["causal_category"] == "policy" for row in causal_document["paired_failures"])
    plan = px.feedback.plan_physics_calibration(telemetry, failure_report)
    assert read(px, plan)["status"] == "AWAITING_REAL_EVIDENCE"

    training_dataset = px.datasets.register(
        dataset.model_copy(
            update={
                "resource_id": "feedback.training",
                "episodes": (dataset.episodes[0],),
                "source_runs": (dataset.episodes[0].run_id,),
                "total_steps": dataset.episodes[0].step_count,
            }
        )
    )
    initial_model, parent = fixture_model(px, training_dataset, "feedback.initial.model")
    updated_model, _ = fixture_model(px, training_dataset, "feedback.updated.model", parent=parent)
    initial_report = EvaluationReport(
        resource_id="feedback.initial",
        version="1.0.0",
        model=initial_model,
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
            "model": updated_model,
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
    initial_paired_ref = initial_paired.reference(px.artifacts.put_json(initial_paired).artifact_id)
    updated_report_ref = updated_report.reference(px.artifacts.put_json(updated_report).artifact_id)
    assessment = assess_policy_update(px.artifacts, initial_paired_ref, updated_report_ref)
    assert read(px, assessment)["initial_evaluation"] == initial_paired_ref.model_dump(mode="json")
    document = read(px, assessment)
    assert document["status"] == "COLLECT_MORE_DATA"
    assert "insufficient_evaluation_episodes" in document["guardrail_violations"]
    assert document["metrics"]["contact_variation_delta_n"] is None
    assert document["metrics"]["quality_pair_count"] == 0

    # Independently executed experts enable comparison when the baseline never succeeds.
    expert_episodes = [
        row.model_copy(
            update={
                "episode_id": str(uuid.uuid4()),
                "run_id": str(uuid.uuid4()),
                "policy": recovery_policy.model_copy(update={"config": {"expert": True}}),
            }
        )
        for row in recovery_episodes
    ]
    expert_summaries = tuple(
        EpisodeSummary(
            episode_id=row.episode_id,
            run_id=row.run_id,
            index=row.index,
            success=row.success,
            step_count=len(row.steps),
            artifact=px.artifacts.put_json(row),
        )
        for row in expert_episodes
    )
    expert_dataset = px.datasets.register(
        dataset.model_copy(
            update={
                "resource_id": "feedback.quality.reference",
                "episodes": expert_summaries,
                "source_runs": tuple(row.run_id for row in expert_episodes),
                "total_steps": sum(row.step_count for row in expert_summaries),
            }
        )
    )
    small_fixture_gate = {"min_episodes": 2, "min_quality_pairs": 2}
    passed = assess_policy_update(
        px.artifacts,
        initial_paired_ref,
        updated_report_ref,
        small_fixture_gate,
        reference_dataset=expert_dataset,
    )
    assert read(px, passed)["status"] == "PROMOTE"
    assert read(px, passed)["metrics"]["quality_pair_count"] == 2
    assert read(px, passed)["metrics"]["contact_variation_delta_n"] == 0
    assert all(
        row["updated_contact_variation_n"] > row["initial_contact_variation_n"]
        for row in read(px, passed)["paired_episodes"]
    )
    insufficient = assess_policy_update(
        px.artifacts, initial_paired_ref, updated_report_ref, reference_dataset=expert_dataset
    )
    assert read(px, insufficient)["guardrail_violations"] == [
        "insufficient_evaluation_episodes",
        "insufficient_successful_quality_pairs",
    ]

    # Same seed alone is insufficient: recorded initial state must also agree.
    from physrealx.integrations.feedback import _validate_paired_conditions

    changed = recovery_episodes[0]
    initial_step = changed.steps[0]
    initial_value = initial_step.observations["joint.position"]
    changed = changed.model_copy(
        update={
            "steps": (
                initial_step.model_copy(
                    update={
                        "observations": {
                            **initial_step.observations,
                            "joint.position": initial_value.model_copy(update={"inline": (0.01,)}),
                        }
                    }
                ),
                *changed.steps[1:],
            )
        }
    )
    with pytest.raises(DomainValidationError, match="initial joint state"):
        _validate_paired_conditions(failed[0], changed)
    with pytest.raises(DomainValidationError, match="separate expert"):
        assess_policy_update(
            px.artifacts,
            initial_paired_ref,
            updated_report_ref,
            small_fixture_gate,
            reference_dataset=updated_dataset,
        )

    # Changing the candidate training set cannot erase its parent's leaked scenario.
    _, checkpoint = fixture_model(px, initial_paired_dataset, "leaked.parent")
    contaminated_model, _ = fixture_model(px, training_dataset, "leaked.child", parent=checkpoint)
    contaminated_report = updated_report.model_copy(update={"model": contaminated_model})
    contaminated_ref = contaminated_report.reference(
        px.artifacts.put_json(contaminated_report).artifact_id
    )
    with pytest.raises(DomainValidationError, match="training ancestry"):
        assess_policy_update(px.artifacts, initial_paired_ref, contaminated_ref)

    preview = px.feedback.preview_demonstrations(tuple(recovery_refs))
    assert px.artifacts.get_bytes(preview).startswith(b"\x89PNG")
    with pytest.raises(DomainValidationError, match="duplicate"):
        px.feedback.preview_demonstrations((recovery_refs[0], recovery_refs[0]))

    dashboard = preview_feedback_report(
        px.artifacts,
        initial_paired,
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
            "steps": tuple(
                step.model_copy(
                    update={
                        "timestamp_seconds": step.timestamp_seconds * 2,
                        "observations": {
                            name: value.model_copy(
                                update={"timestamp_seconds": step.timestamp_seconds * 2}
                            )
                            for name, value in step.observations.items()
                        },
                        "state": {
                            name: value.model_copy(
                                update={"timestamp_seconds": step.timestamp_seconds * 2}
                            )
                            for name, value in step.state.items()
                        },
                    }
                )
                for step in recovery_episodes[0].steps
            ),
        }
    )
    updated_artifact = px.artifacts.put_json(updated_episode)
    from matplotlib.figure import Figure

    captured = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        captured.append(
            {
                "rgb": [np.asarray(axis.images[0].get_array()).copy() for axis in figure.axes[:3]],
                "positions": [
                    np.asarray(axis.lines[0].get_ydata()).copy() for axis in figure.axes[3:6]
                ],
                "limits": [axis.get_ylim() for axis in figure.axes[3:6]],
                "units": figure.axes[3].get_ylabel(),
            }
        )
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    cycle = preview_feedback_cycle(
        px.artifacts,
        next(episode for episode in dataset.episodes if not episode.success).artifact,
        px.datasets.get(recovery_refs[0]).episodes[0].artifact,
        updated_artifact,
        frames=3,
        fps=2,
    )
    assert px.artifacts.get_bytes(cycle).startswith(b"GIF")
    assert len(captured) == 3
    assert len(set(captured[0]["limits"])) == 1
    assert "(m)" in captured[0]["units"]
    assert captured[0]["positions"][1][-1] == pytest.approx(0.3)
    assert captured[1]["rgb"][1][0, 0, 0] > captured[1]["rgb"][2][0, 0, 0]
    with pytest.raises(DomainValidationError, match="differs from policy assessment"):
        preview_feedback_report(
            px.artifacts, initial_report, updated_report, failure_report, causal, plan, assessment
        )


def test_real_feedback_requires_deployment_attestation_and_prepares_calibration(episode_data):
    from physrealx.domain import canonical_record_id
    from physrealx.robot_agent import DeploymentMode, RobotExecutionEvidence, TelemetryManifest

    px, dataset, _, _, episodes, _ = episode_data
    raw = px.artifacts.put_bytes(b"synthetic mcap fixture", media_type="application/vnd.mcap")
    telemetry_manifest = px.artifacts.put_json(
        TelemetryManifest(
            deployment_record_id=canonical_record_id("synthetic deployment"),
            robot_id="test-robot",
            clock_domain="ros_time",
            started_at_ns=0,
            completed_at_ns=1,
            message_count=34,
            topics=("/joint_states",),
            mcap=raw,
            topic_schemas=px.artifacts.put_json({"/joint_states": "sensor_msgs/msg/JointState"}),
            crc_verified=True,
            index_verified=True,
            timestamps_monotonic=True,
        )
    )
    deployment_report = px.artifacts.put_json(
        RobotExecutionEvidence(
            status="COMPLETED",
            deployment_plan=px.artifacts.put_json({"kind": "deployment-plan"}),
            telemetry=telemetry_manifest,
            safety_events=px.artifacts.put_json({"events": []}),
            final_mode=DeploymentMode.REVOKED,
            started_at_ns=0,
            completed_at_ns=1,
        )
    )
    summaries = []
    real_episodes = []
    for index, source in enumerate(episodes[:2]):
        episode = source.model_copy(
            update={
                "episode_id": str(uuid.uuid4()),
                "run_id": str(uuid.uuid4()),
                "source": EpisodeSource.REAL,
                "metadata": {
                    "capture_id": f"robot-capture-{index}",
                    "deployment_report": deployment_report.model_dump(mode="json"),
                    "raw_telemetry": raw.model_dump(mode="json"),
                    "gravity_compensated": True,
                    "isolated_joint": "joint-0",
                },
                "steps": tuple(
                    source.steps[min(i // 2, 8)].model_copy(
                        update={
                            "index": i,
                            "timestamp_seconds": i * 0.025,
                            "is_first": i == 0,
                            "observations": {
                                name: value.model_copy(update={"timestamp_seconds": i * 0.025})
                                for name, value in source.steps[min(i // 2, 8)].observations.items()
                            },
                            "state": {
                                name: value.model_copy(update={"timestamp_seconds": i * 0.025})
                                for name, value in source.steps[min(i // 2, 8)].state.items()
                            },
                        }
                    )
                    for i in range(17)
                ),
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
        joint_type="slide",
        joint_index=0,
        prior_parameters=(1.0, 0.1, 0.05),
    )
    assert read(px, calibration)["schema"] == ("physrealx.joint-system-identification-evidence/1")

    assert read(px, calibration)["gravity_compensated"] is True
    for changes, message in (
        ({"joint_index": -1}, "nonnegative"),
        ({"joint_index": 2}, "joint_index"),
        ({"joint_type": "hinge"}, "units"),
    ):
        with pytest.raises(DomainValidationError, match=message):
            px.feedback.prepare_physics_calibration(
                output(archive, "telemetry"),
                **{
                    "joint_name": "joint-0",
                    "joint_type": "slide",
                    "joint_index": 0,
                    "prior_parameters": (1.0, 0.1, 0.05),
                    **changes,
                },
            )
    unqualified = []
    for index, episode in enumerate(real_episodes):
        without_compensation = episode.model_copy(
            update={
                "episode_id": str(uuid.uuid4()),
                "metadata": {
                    key: value
                    for key, value in episode.metadata.items()
                    if key not in {"gravity_compensated", "isolated_joint"}
                },
            }
        )
        unqualified.append(
            summaries[index].model_copy(
                update={
                    "episode_id": without_compensation.episode_id,
                    "artifact": px.artifacts.put_json(without_compensation),
                }
            )
        )
    unqualified_ref = px.datasets.register(
        px.datasets.get(real_dataset).model_copy(
            update={
                "resource_id": "learning.real.uncompensated",
                "episodes": tuple(unqualified),
            }
        )
    )
    unqualified_archive = px.telemetry.record(
        px.artifacts.put_json(
            DeploymentEvidence(dataset=unqualified_ref, deployment_report=deployment_report)
        ),
        method="episode_telemetry",
    )
    with pytest.raises(DomainValidationError, match="gravity-compensated"):
        px.feedback.prepare_physics_calibration(
            output(unqualified_archive, "telemetry"),
            joint_name="joint-0",
            joint_type="slide",
            joint_index=0,
            prior_parameters=(1.0, 0.1, 0.05),
        )
    unrelated_report = read(px, deployment_report)
    unrelated_manifest = read(px, telemetry_manifest)
    unrelated_manifest["mcap"] = px.artifacts.put_bytes(
        b"unrelated-capture", media_type="application/vnd.mcap"
    ).model_dump(mode="json")
    unrelated_report["telemetry"] = px.artifacts.put_json(unrelated_manifest).model_dump(
        mode="json"
    )
    with pytest.raises(DomainValidationError, match="must link"):
        px.telemetry.record(
            px.artifacts.put_json(
                DeploymentEvidence(
                    dataset=real_dataset, deployment_report=px.artifacts.put_json(unrelated_report)
                )
            ),
            method="episode_telemetry",
        )


def test_act_official_network_checkpoint_and_action_chunks_roundtrip(episode_data, monkeypatch):
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
    policy = ACTPolicy({"model_artifact_id": ref.artifact_id}, px.artifacts)
    policy.reset(0)
    action = policy.act(Frame(observations=episodes[0].steps[0].observations, state={}), task, 0)
    assert 0 <= action.inline[0] <= 1
    assert len(policy._commands) == 1
    policy.reset(1)
    assert not policy._commands
    policy.close()
    parent = px.artifacts.put_json(model, media_type="application/vnd.physrealx.model+json")
    dataset = px.datasets.get(reference)
    summary = dataset.episodes[0]
    shifted_ref = px.datasets.register(
        dataset.model_copy(
            update={
                "resource_id": "learning.act.shifted",
                "episodes": (summary,),
                "source_runs": (summary.run_id,),
                "total_steps": summary.step_count,
            }
        )
    )
    tuned_ref = px.training.submit(
        TrainingSpec(
            resource_id="learning.act.tuned",
            version="1.0.0",
            dataset=shifted_ref,
            trainer=ACTTrainer.descriptor.bind(
                checkpoint=parent,
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
    assert tuned.training_config["parent_checkpoint"] == parent.model_dump(mode="json")
    assert tuned.trainer.checkpoint == parent
    assert torch.get_num_threads() == previous

    for name in ("state_mean", "state_scale", "action_mean", "action_scale"):
        assert tuned.training_config[name] == model.training_config[name]
    with pytest.raises(DomainValidationError, match="complete model manifest"):
        px.training.submit(
            spec.model_copy(
                update={
                    "resource_id": "learning.act.raw",
                    "trainer": spec.trainer.model_copy(update={"checkpoint": model.weights}),
                }
            )
        )
    with pytest.raises(DomainValidationError, match="incompatible"):
        px.training.submit(
            spec.model_copy(
                update={
                    "resource_id": "learning.act.reordered",
                    "trainer": spec.trainer.model_copy(
                        update={
                            "checkpoint": parent,
                            "config": {
                                **spec.trainer.config,
                                "observation_signals": ["joint.velocity", "joint.position"],
                            },
                        }
                    ),
                }
            )
        )
    original_verify = px.artifacts.verify
    with monkeypatch.context() as patch:
        patch.setattr(
            px.artifacts,
            "verify",
            lambda reference: False if reference == model.weights else original_verify(reference),
        )
        with pytest.raises(DomainValidationError, match="corrupt model weights"):
            px.training.submit(spec)


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

    # A matching recipe label is insufficient when the registered episodes differ.
    wrong = dataset.model_copy(
        update={
            "resource_id": "training.wrong-selection",
            "metadata": selected.metadata,
        }
    )
    px.datasets.register(wrong)
    with pytest.raises(ResourceConflictError, match="different training selection"):
        px.datasets.training_subset(snapshot_ref, resource_id=wrong.resource_id)

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
    labelled_evaluation = augment(
        expert, seed=997, tags=["expert_demonstration", "policy_evaluation"]
    )
    with pytest.raises(DomainValidationError, match="required tag"):
        px.datasets.training_subset(
            snapshot_ref,
            resource_id="not-an-expert",
            augmentations=(labelled_evaluation,),
            required_tag="expert_demonstration",
        )
    result = px.datasets.training_subset(
        snapshot_ref, resource_id="augmented", augmentations=(extra,)
    )
    assert len(px.datasets.get(result).episodes) == len(selected.episodes) + 1
    with pytest.raises(DomainValidationError, match="held-out"):
        px.datasets.training_subset(
            snapshot_ref,
            resource_id="external-heldout",
            augmentations=(extra,),
            excluded_datasets=(extra,),
        )
    with pytest.raises(DomainValidationError, match="excluded evaluation"):
        px.datasets.training_subset(
            snapshot_ref, resource_id="leaked-base", excluded_datasets=(selected_ref,)
        )
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
