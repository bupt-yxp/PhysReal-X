"""Evidence-based trajectory diagnostics and traceable failure sample selection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DatasetManifest,
    DomainModel,
    Episode,
    MethodCapability,
    OperationSpec,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import input_artifact, load_resource, publish, read_json
from physrealx.integrations.datasets import validate_dataset
from physrealx.methods import builtin_method
from physrealx.runtime import load_episode


class RunEvidence(DomainModel):
    schema_id: Literal["physrealx.run-evidence/1"] = Field(
        default="physrealx.run-evidence/1", alias="schema"
    )
    dataset: VersionedRef


def episode_metrics(episode: Episode, task: TaskSpec) -> dict[str, Any]:
    import numpy as np

    names = ("joint.position", "joint.velocity", "joint.effort", "contact.force")
    signals: dict[str, Any] = {}
    for name in names:
        rows = [step.observations.get(name) for step in episode.steps]
        if any(value is None or value.inline is None for value in rows):
            raise DomainValidationError(f"trajectory diagnostics require recorded signal {name}")
        signals[name] = np.asarray(
            [value.inline for value in rows if value is not None], dtype=float
        )
    targets = np.asarray(task.metadata.get("target_positions"), dtype=float)
    position = signals["joint.position"]
    if targets.shape != position.shape[1:] or not np.isfinite(targets).all():
        raise DomainValidationError("trajectory task lacks an explicit joint target vector")
    actions = np.asarray(
        [
            step.action.inline
            for step in episode.steps[:-1]
            if step.action is not None and step.action.inline is not None
        ],
        dtype=float,
    )
    if actions.shape != position[:-1].shape or task.action.kind != "joint_position":
        raise DomainValidationError("joint trajectory diagnostics require position commands")
    errors = actions - position[1:]
    goals = []
    for predicate in task.success_conditions:
        observed = episode.steps[-1].observations.get(predicate.signal)
        if observed is None or observed.inline is None or len(observed.inline) != 1:
            raise DomainValidationError("task goal diagnostics require a recorded scalar signal")
        value = float(observed.inline[0])
        goals.append(
            {
                "signal": predicate.signal,
                "value": value,
                "target": predicate.value,
                "satisfied": predicate.evaluate({predicate.signal: value}),
            }
        )
    force = np.abs(signals["contact.force"]).reshape(-1)
    active = force > max(1e-3, float(force.max()) * 0.01)
    intervals = np.diff([step.timestamp_seconds for step in episode.steps])
    return {
        "episode_id": episode.episode_id,
        "source": episode.source.value,
        "success": episode.success,
        "failure_reason": episode.failure_reason,
        "seed": episode.seed,
        "duration_seconds": episode.steps[-1].timestamp_seconds
        - episode.steps[0].timestamp_seconds,
        "control_target_error": float(np.linalg.norm(position[-1] - targets)),
        "task_goals": goals,
        "command_target_error": float(np.linalg.norm(actions[-1] - targets)),
        "tracking_rmse": float(np.sqrt(np.mean(errors**2))),
        "motion_extent": float(np.linalg.norm(position.max(0) - position.min(0))),
        "peak_effort": float(np.abs(signals["joint.effort"]).max()),
        "peak_contact_force_n": float(np.abs(signals["contact.force"]).max()),
        "contact_duration_seconds": float(np.sum(intervals * active[1:])),
        "contact_break_count": int(np.sum(active[:-1] & ~active[1:])),
        "contact_force_variation_rms_n": float(np.sqrt(np.mean(np.diff(force) ** 2))),
        "final_position": position[-1].tolist(),
        "target_positions": targets.tolist(),
    }


class TelemetryArchiveAdapter:
    descriptor = builtin_method(
        MethodCapability.TELEMETRY_RECORD,
        "episode_telemetry",
        version="2.0.0",
        input_contract="DeploymentRun/1",
        output_contract="TelemetryBundle/1+TelemetryManifest/1",
    )

    def probe(self) -> MethodProbe:
        return MethodProbe(True, "Immutable episode telemetry and source validation available")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.config:
            raise DomainValidationError("telemetry archives do not modify their input data")
        source = input_artifact(spec, "run", artifacts)
        evidence = RunEvidence.model_validate(
            read_json(artifacts, source, "physrealx.run-evidence/1")
        )
        dataset = load_resource(artifacts, evidence.dataset, DatasetManifest)
        quality = validate_dataset(artifacts, dataset)
        metrics = []
        for summary in dataset.episodes:
            episode = load_episode(artifacts, summary)
            task = load_resource(artifacts, episode.task, TaskSpec)
            metrics.append(episode_metrics(episode, task))
        return publish(
            spec,
            artifacts,
            {
                "telemetry": {
                    "schema": "physrealx.telemetry-bundle/1",
                    "dataset": evidence.dataset.model_dump(mode="json"),
                    "episodes": metrics,
                    "source_counts": quality["source_counts"],
                },
                "telemetry_manifest": {
                    "schema": "physrealx.telemetry-manifest/1",
                    "execution_status": "COMPLETED",
                    "episodes": len(metrics),
                    "source_counts": quality["source_counts"],
                    "recording_mode": "immutable_episode_archive",
                    "live_robot_execution": False,
                },
            },
        )


class DiagnosticConfig(DomainModel):
    command_error_threshold: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    tracking_error_threshold: float = Field(default=0.03, gt=0, allow_inf_nan=False)
    contact_force_threshold_n: float = Field(default=100, gt=0, allow_inf_nan=False)


class FailureAnalysisAdapter:
    descriptor = builtin_method(
        MethodCapability.FAILURE_ANALYZE,
        "trajectory_diagnostics",
        version="2.0.0",
        input_contract="TelemetryBundle/1",
        output_contract="FailureReport/1",
    )

    def probe(self) -> MethodProbe:
        return TelemetryArchiveAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        config = DiagnosticConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "run", artifacts)
        telemetry = read_json(artifacts, source, "physrealx.telemetry-bundle/1")
        dataset_ref = VersionedRef.model_validate(telemetry["dataset"])
        dataset = load_resource(artifacts, dataset_ref, DatasetManifest)
        failures, all_metrics = [], []
        for summary in dataset.episodes:
            episode = load_episode(artifacts, summary)
            task = load_resource(artifacts, episode.task, TaskSpec)
            metrics = episode_metrics(episode, task)
            all_metrics.append(metrics)
            if episode.success:
                continue
            # These categories describe observed symptoms; causal attribution
            # needs a controlled rerun or independent physical measurements.
            if metrics["peak_contact_force_n"] > config.contact_force_threshold_n:
                category, recommendation = (
                    "contact_constraint",
                    "inspect_collision_and_contact_parameters",
                )
            elif metrics["command_target_error"] > config.command_error_threshold:
                category, recommendation = (
                    "policy_target_mismatch",
                    "generate_expert_recovery_from_failed_initial_conditions",
                )
            elif metrics["tracking_rmse"] > config.tracking_error_threshold:
                category, recommendation = (
                    "dynamics_tracking",
                    "calibrate_actuation_mass_friction_with_independent_trajectories",
                )
            else:
                category, recommendation = (
                    "insufficient_progress",
                    "extend_training_coverage_and_check_task_timing",
                )
            failures.append(
                {
                    **metrics,
                    "category": category,
                    "recommendation": recommendation,
                    "causal_validation": False,
                }
            )
        if all_metrics != telemetry["episodes"]:
            raise DomainValidationError("telemetry metrics differ from their immutable episodes")
        report = {
            "schema": "physrealx.failure-report/1",
            "telemetry": source.model_dump(mode="json"),
            "dataset": dataset_ref.model_dump(mode="json"),
            "failures": failures,
            "episodes": len(dataset.episodes),
            "success_rate": sum(row.success for row in dataset.episodes) / len(dataset.episodes),
            "attribution_scope": "recorded symptoms; causal validation requires intervention",
        }
        return publish(
            spec, artifacts, {"failure_report": report}, metrics={"failures": float(len(failures))}
        )


class FailureQuery(DomainModel):
    schema_id: Literal["physrealx.failure-query/1"] = Field(
        default="physrealx.failure-query/1", alias="schema"
    )
    report: ArtifactRef
    categories: tuple[str, ...] = ()
    resource_id: str
    version: str = "1.0.0"


class FailureDatasetAdapter:
    descriptor = builtin_method(
        MethodCapability.FEEDBACK_BUILD_DATASET,
        "failure_episodes",
        input_contract="FailureQuery/1",
        output_contract="DatasetBundle/1+SelectionReport/1",
    )

    def probe(self) -> MethodProbe:
        return TelemetryArchiveAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.config:
            raise DomainValidationError("failure selection is fully specified by its query")
        query = FailureQuery.model_validate(
            read_json(
                artifacts, input_artifact(spec, "query", artifacts), "physrealx.failure-query/1"
            )
        )
        report = read_json(artifacts, query.report, "physrealx.failure-report/1")
        dataset = load_resource(
            artifacts, VersionedRef.model_validate(report["dataset"]), DatasetManifest
        )
        selected = {
            row["episode_id"]
            for row in report["failures"]
            if not query.categories or row["category"] in query.categories
        }
        summaries = tuple(row for row in dataset.episodes if row.episode_id in selected)
        if (
            not summaries
            or len(summaries) != len(selected)
            or any(row.success for row in summaries)
        ):
            raise DomainValidationError(
                "failure query must select actual, distinct failed episodes"
            )
        values = dataset.model_dump(mode="json")
        values.update(
            resource_id=query.resource_id,
            version=query.version,
            episodes=[row.model_dump(mode="json") for row in summaries],
            total_steps=sum(row.step_count for row in summaries),
            source_runs=list(dict.fromkeys(row.run_id for row in summaries)),
            metadata={
                "failure_report": query.report.model_dump(mode="json"),
                "purpose": "failure_feedback; not expert imitation labels",
            },
        )
        result = DatasetManifest.model_validate(values)
        validate_dataset(artifacts, result)
        seeds = [load_episode(artifacts, row).seed for row in summaries]
        return publish(
            spec,
            artifacts,
            {
                "dataset": {
                    "schema": "physrealx.dataset-bundle/1",
                    "manifest": result.model_dump(mode="json"),
                },
                "selection_report": {
                    "schema": "physrealx.selection-report/1",
                    "selected_episode_ids": sorted(selected),
                    "regeneration_seeds": seeds,
                    "failure_report": query.report.model_dump(mode="json"),
                    "relabelled_success": False,
                },
            },
            metrics={"selected_episodes": float(len(summaries))},
        )
