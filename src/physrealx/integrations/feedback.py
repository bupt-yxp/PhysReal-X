"""Deployment evaluation, controlled failure diagnosis, and feedback data selection."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DatasetManifest,
    DomainModel,
    Episode,
    EpisodeSource,
    EvaluationReport,
    MethodCapability,
    OperationSpec,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import (
    input_artifact,
    load_resource,
    publish,
    read_array,
    read_json,
)
from physrealx.integrations.datasets import validate_dataset
from physrealx.methods import builtin_method
from physrealx.runtime import load_episode


class DeploymentEvidence(DomainModel):
    """Canonical deployment episodes plus hardware attestation when the source is real."""

    schema_id: Literal["physrealx.deployment-evidence/1"] = Field(
        default="physrealx.deployment-evidence/1", alias="schema"
    )
    dataset: VersionedRef
    deployment_report: ArtifactRef | None = None


def _observation_array(
    artifacts: ArtifactStore, episode: Episode, index: int, name: str
) -> Any | None:
    import numpy as np

    value = episode.steps[index].observations.get(name)
    if value is None:
        return None
    if value.artifact is not None:
        return read_array(artifacts, value.artifact)
    if value.inline is not None:
        return np.asarray(value.inline).reshape(value.shape)
    return None


def _perception_metrics(artifacts: ArtifactStore, episode: Episode) -> dict[str, Any]:
    import numpy as np

    names = set(episode.steps[0].observations)
    cameras = sorted(
        {
            name.split(".")[1]
            for name in names
            if name.startswith("camera.") and name.endswith(".rgb") and name.count(".") == 2
        }
    )
    if not cameras:
        return {"available": False}
    indices = sorted({0, len(episode.steps) // 2, len(episode.steps) - 1})
    contrasts: list[float] = []
    rgb_valid: list[float] = []
    depth_valid: list[float] = []
    foreground: list[float] = []
    for camera in cameras:
        for index in indices:
            rgb = _observation_array(artifacts, episode, index, f"camera.{camera}.rgb")
            if rgb is not None and rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb_float = np.asarray(rgb, dtype=float)
                rgb_valid.append(float(np.isfinite(rgb_float).mean()))
                contrasts.append(float(np.nanstd(rgb_float) / 255.0))
            depth = _observation_array(artifacts, episode, index, f"camera.{camera}.depth")
            if depth is not None:
                values = np.asarray(depth, dtype=float)
                depth_valid.append(float((np.isfinite(values) & (values > 0)).mean()))
            mask = _observation_array(artifacts, episode, index, f"camera.{camera}.mask")
            if mask is not None:
                foreground.append(float((np.asarray(mask) > 0).mean()))
    return {
        "available": True,
        "camera_count": len(cameras),
        "rgb_valid_fraction": min(rgb_valid, default=0.0),
        "rgb_contrast": min(contrasts, default=0.0),
        "depth_valid_fraction": min(depth_valid, default=1.0),
        "mask_foreground_fraction": min(foreground, default=0.0),
    }


def episode_metrics(artifacts: ArtifactStore, episode: Episode, task: TaskSpec) -> dict[str, Any]:
    """Measure one immutable deployment episode without inferring hidden causes."""
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
    force = np.linalg.norm(signals["contact.force"], axis=-1).reshape(-1)
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
        "peak_contact_force_n": float(force.max()),
        "contact_duration_seconds": float(np.sum(intervals * active[1:])),
        "contact_break_count": int(np.sum(active[:-1] & ~active[1:])),
        "contact_force_variation_rms_n": float(np.sqrt(np.mean(np.diff(force) ** 2))),
        "final_position": position[-1].tolist(),
        "target_positions": targets.tolist(),
        "perception": _perception_metrics(artifacts, episode),
    }


def _validate_deployment_evidence(
    artifacts: ArtifactStore, evidence: DeploymentEvidence
) -> tuple[DatasetManifest, dict[str, Any], bool]:
    dataset = load_resource(artifacts, evidence.dataset, DatasetManifest)
    quality = validate_dataset(artifacts, dataset)
    sources = set(quality["source_counts"])
    if len(sources) != 1:
        raise DomainValidationError("one deployment archive cannot mix simulator and real episodes")
    live = sources == {EpisodeSource.REAL.value}
    if live:
        if evidence.deployment_report is None:
            raise DomainValidationError("real episodes require an immutable deployment report")
        from physrealx.robot_agent import RobotExecutionEvidence

        report = RobotExecutionEvidence.model_validate(
            read_json(artifacts, evidence.deployment_report, "physrealx.deployment-report/1"),
            strict=False,
        )
        if any(
            not artifacts.verify(reference)
            for reference in (report.deployment_plan, report.telemetry, report.safety_events)
        ):
            raise DomainValidationError("real deployment report references missing evidence")
    elif evidence.deployment_report is not None:
        raise DomainValidationError("simulator episodes cannot claim a hardware deployment report")
    return dataset, quality, live


class TelemetryArchiveAdapter:
    descriptor = builtin_method(
        MethodCapability.TELEMETRY_RECORD,
        "episode_telemetry",
        version="3.0.0",
        input_contract="DeploymentEvidence/1",
        output_contract="TelemetryBundle/2+TelemetryManifest/2",
    )

    def probe(self) -> MethodProbe:
        return MethodProbe(
            True, "Canonical simulator and attested real deployment episodes supported"
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.config:
            raise DomainValidationError("telemetry archives do not modify their input data")
        source = input_artifact(spec, "run", artifacts)
        evidence = DeploymentEvidence.model_validate(
            read_json(artifacts, source, "physrealx.deployment-evidence/1")
        )
        dataset, quality, live = _validate_deployment_evidence(artifacts, evidence)
        metrics = []
        for summary in dataset.episodes:
            episode = load_episode(artifacts, summary)
            task = load_resource(artifacts, episode.task, TaskSpec)
            metrics.append(episode_metrics(artifacts, episode, task))
        deployment = (
            evidence.deployment_report.model_dump(mode="json")
            if evidence.deployment_report is not None
            else None
        )
        return publish(
            spec,
            artifacts,
            {
                "telemetry": {
                    "schema": "physrealx.telemetry-bundle/2",
                    "dataset": evidence.dataset.model_dump(mode="json"),
                    "deployment_report": deployment,
                    "episodes": metrics,
                    "source_counts": quality["source_counts"],
                },
                "telemetry_manifest": {
                    "schema": "physrealx.telemetry-archive-manifest/2",
                    "execution_status": "COMPLETED",
                    "episodes": len(metrics),
                    "source_counts": quality["source_counts"],
                    "recording_mode": "immutable_episode_archive",
                    "live_robot_execution": live,
                    "deployment_report": deployment,
                },
            },
        )


class DiagnosticConfig(DomainModel):
    command_error_threshold: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    tracking_error_threshold: float = Field(default=0.03, gt=0, allow_inf_nan=False)
    contact_force_threshold_n: float = Field(default=100, gt=0, allow_inf_nan=False)
    minimum_rgb_contrast: float = Field(default=0.015, ge=0, le=1)
    minimum_depth_valid_fraction: float = Field(default=0.5, ge=0, le=1)


class FailureAnalysisAdapter:
    descriptor = builtin_method(
        MethodCapability.FAILURE_ANALYZE,
        "trajectory_diagnostics",
        version="3.0.0",
        input_contract="TelemetryBundle/2",
        output_contract="FailureReport/2",
    )

    def probe(self) -> MethodProbe:
        return TelemetryArchiveAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        config = DiagnosticConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "run", artifacts)
        telemetry = read_json(artifacts, source, "physrealx.telemetry-bundle/2")
        dataset_ref = VersionedRef.model_validate(telemetry["dataset"])
        dataset = load_resource(artifacts, dataset_ref, DatasetManifest)
        failures, all_metrics = [], []
        for summary in dataset.episodes:
            episode = load_episode(artifacts, summary)
            task = load_resource(artifacts, episode.task, TaskSpec)
            metrics = episode_metrics(artifacts, episode, task)
            all_metrics.append(metrics)
            if episode.success:
                continue
            perception = metrics["perception"]
            if perception["available"] and (
                perception["rgb_contrast"] < config.minimum_rgb_contrast
                or perception["depth_valid_fraction"] < config.minimum_depth_valid_fraction
            ):
                category, recommendation = (
                    "perception",
                    "inspect_camera_exposure_depth_validity_and_sensor_registration",
                )
            elif metrics["peak_contact_force_n"] > config.contact_force_threshold_n:
                category, recommendation = (
                    "contact",
                    "inspect_collision_contact_and_force_limits",
                )
            elif metrics["command_target_error"] > config.command_error_threshold:
                category, recommendation = (
                    "planning",
                    "run_a_paired_policy_intervention_from_the_same_initial_condition",
                )
            elif metrics["tracking_rmse"] > config.tracking_error_threshold:
                category, recommendation = (
                    "control",
                    "identify_actuation_inertia_damping_and_friction_from_real_trajectories",
                )
            else:
                category, recommendation = (
                    "timeout",
                    "extend_task_timing_and_collect_a_controlled_rerun",
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
            "schema": "physrealx.failure-report/2",
            "telemetry": source.model_dump(mode="json"),
            "dataset": dataset_ref.model_dump(mode="json"),
            "deployment_report": telemetry.get("deployment_report"),
            "failures": failures,
            "episodes": len(dataset.episodes),
            "success_rate": sum(row.success for row in dataset.episodes) / len(dataset.episodes),
            "attribution_scope": "observed symptoms; causes require a controlled intervention",
        }
        return publish(
            spec, artifacts, {"failure_report": report}, metrics={"failures": float(len(failures))}
        )


class PolicyInterventionEvidence(DomainModel):
    schema_id: Literal["physrealx.policy-intervention-evidence/1"] = Field(
        default="physrealx.policy-intervention-evidence/1", alias="schema"
    )
    failure_report: ArtifactRef
    recovery_datasets: tuple[VersionedRef, ...] = Field(min_length=1)


class PolicyGateConfig(DomainModel):
    max_target_error_regression: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    max_tracking_rmse_regression: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    max_contact_variation_regression_n: float = Field(default=5.0, ge=0, allow_inf_nan=False)


def analyze_policy_intervention(artifacts: ArtifactStore, evidence_ref: ArtifactRef) -> ArtifactRef:
    """Validate policy causality using same-condition failed/recovery episode pairs."""
    request = PolicyInterventionEvidence.model_validate(
        read_json(
            artifacts,
            evidence_ref,
            "physrealx.policy-intervention-evidence/1",
        )
    )
    report = read_json(artifacts, request.failure_report, "physrealx.failure-report/2")
    baseline_dataset = load_resource(
        artifacts, VersionedRef.model_validate(report["dataset"]), DatasetManifest
    )
    baseline = {
        episode.episode_id: episode
        for episode in (load_episode(artifacts, row) for row in baseline_dataset.episodes)
    }
    recoveries: list[Episode] = []
    for reference in request.recovery_datasets:
        dataset = load_resource(artifacts, reference, DatasetManifest)
        validate_dataset(artifacts, dataset)
        recoveries.extend(load_episode(artifacts, row) for row in dataset.episodes)
    paired = []
    for failure in report["failures"]:
        original = baseline[failure["episode_id"]]
        matches = [
            episode
            for episode in recoveries
            if (
                episode.seed,
                episode.scene,
                episode.task,
                episode.backend,
                episode.source,
            )
            == (
                original.seed,
                original.scene,
                original.task,
                original.backend,
                original.source,
            )
        ]
        if len(matches) != 1 or matches[0].policy == original.policy:
            raise DomainValidationError(
                "each failure requires one same-condition rerun that changes only the policy"
            )
        recovered = matches[0]
        task = load_resource(artifacts, recovered.task, TaskSpec)
        after = episode_metrics(artifacts, recovered, task)
        validated = bool(
            recovered.success
            and not original.success
            and all(goal["satisfied"] for goal in after["task_goals"])
        )
        paired.append(
            {
                **failure,
                "causal_category": "planning" if validated else "unresolved",
                "causal_validation": validated,
                "intervention": "policy",
                "recovery_episode_id": recovered.episode_id,
                "outcome": {
                    "success_before": original.success,
                    "success_after": recovered.success,
                    "target_error_before": failure["control_target_error"],
                    "target_error_after": after["control_target_error"],
                    "tracking_rmse_before": failure["tracking_rmse"],
                    "tracking_rmse_after": after["tracking_rmse"],
                },
            }
        )
    causal = {
        "schema": "physrealx.causal-failure-report/1",
        "failure_report": request.failure_report.model_dump(mode="json"),
        "changed_factor": "policy",
        "control": "same simulator backend, scene, task, source, and seed",
        "paired_failures": paired,
        "validated_count": sum(row["causal_validation"] for row in paired),
        "attribution_scope": "paired policy intervention; other causal factors remain unresolved",
    }
    return artifacts.put_json(causal)


def assess_policy_update(
    artifacts: ArtifactStore,
    initial: EvaluationReport,
    updated: EvaluationReport,
    config: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    """Gate a policy update with paired closed-loop episodes from identical conditions."""
    import numpy as np

    gate = PolicyGateConfig.model_validate(config or {})
    if initial.model == updated.model:
        raise DomainValidationError("policy update assessment requires two distinct models")
    if initial.dataset is None or updated.dataset is None:
        raise DomainValidationError("policy update assessment requires evaluation datasets")

    def episodes(report: EvaluationReport) -> dict[int, Episode]:
        dataset_ref = report.dataset
        if dataset_ref is None:
            raise DomainValidationError("policy update assessment requires an evaluation dataset")
        dataset = load_resource(artifacts, dataset_ref, DatasetManifest)
        rows = [load_episode(artifacts, summary) for summary in dataset.episodes]
        if len({row.seed for row in rows}) != len(rows):
            raise DomainValidationError("policy evaluation seeds must be unique")
        return {row.seed: row for row in rows}

    before, after = episodes(initial), episodes(updated)
    if set(before) != set(after):
        raise DomainValidationError("policy evaluations must use identical seeds")
    paired: list[dict[str, Any]] = []
    for seed in sorted(before):
        baseline, candidate = before[seed], after[seed]
        if (
            baseline.scene != candidate.scene
            or baseline.task != candidate.task
            or baseline.backend != candidate.backend
            or baseline.source != candidate.source
        ):
            raise DomainValidationError(
                "policy evaluations must share scene, task, backend, and deployment source"
            )
        if baseline.policy == candidate.policy:
            raise DomainValidationError("paired policy evaluations did not change the policy")
        task = load_resource(artifacts, baseline.task, TaskSpec)
        baseline_metrics = episode_metrics(artifacts, baseline, task)
        candidate_metrics = episode_metrics(artifacts, candidate, task)
        paired.append(
            {
                "seed": seed,
                "initial_episode_id": baseline.episode_id,
                "updated_episode_id": candidate.episode_id,
                "initial_success": baseline.success,
                "updated_success": candidate.success,
                "initial_target_error": baseline_metrics["control_target_error"],
                "updated_target_error": candidate_metrics["control_target_error"],
                "initial_tracking_rmse": baseline_metrics["tracking_rmse"],
                "updated_tracking_rmse": candidate_metrics["tracking_rmse"],
                "initial_contact_variation_n": baseline_metrics["contact_force_variation_rms_n"],
                "updated_contact_variation_n": candidate_metrics["contact_force_variation_rms_n"],
            }
        )
    initial_success = float(np.mean([row["initial_success"] for row in paired]))
    updated_success = float(np.mean([row["updated_success"] for row in paired]))
    success_delta = updated_success - initial_success

    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in paired]))

    metrics = {
        "initial_success_rate": initial_success,
        "updated_success_rate": updated_success,
        "success_rate_delta": success_delta,
        "target_error_delta": mean("updated_target_error") - mean("initial_target_error"),
        "tracking_rmse_delta": mean("updated_tracking_rmse") - mean("initial_tracking_rmse"),
        "contact_variation_delta_n": mean("updated_contact_variation_n")
        - mean("initial_contact_variation_n"),
    }
    violations = [
        name
        for name, threshold in (
            ("target_error_delta", gate.max_target_error_regression),
            ("tracking_rmse_delta", gate.max_tracking_rmse_regression),
            ("contact_variation_delta_n", gate.max_contact_variation_regression_n),
        )
        if metrics[name] > threshold
    ]
    if success_delta < 0:
        status = "REJECT"
        reason = "paired closed-loop success rate regressed"
    elif success_delta == 0:
        status = "COLLECT_MORE_DATA"
        reason = "paired closed-loop success rate did not improve"
    elif violations:
        status = "COLLECT_MORE_DATA"
        reason = "success improved but trajectory or contact guardrails regressed"
    else:
        status = "PROMOTE"
        reason = "success improved without violating trajectory or contact guardrails"

    return artifacts.put_json(
        {
            "schema": "physrealx.policy-update-assessment/1",
            "status": status,
            "reason": reason,
            "control": "same scene, task, backend, deployment source, and seed",
            "initial_evaluation": initial.reference().model_dump(mode="json"),
            "updated_evaluation": updated.reference().model_dump(mode="json"),
            "metrics": metrics,
            "guardrails": gate.model_dump(mode="json"),
            "guardrail_violations": violations,
            "paired_episodes": paired,
        }
    )


class FailureQuery(DomainModel):
    schema_id: Literal["physrealx.failure-query/2"] = Field(
        default="physrealx.failure-query/2", alias="schema"
    )
    report: ArtifactRef
    categories: tuple[str, ...] = ()
    resource_id: str
    version: str = "1.0.0"


class FailureDatasetAdapter:
    descriptor = builtin_method(
        MethodCapability.FEEDBACK_BUILD_DATASET,
        "failure_episodes",
        version="2.0.0",
        input_contract="FailureQuery/2",
        output_contract="DatasetBundle/1+SelectionReport/2",
    )

    def probe(self) -> MethodProbe:
        return TelemetryArchiveAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.config:
            raise DomainValidationError("failure selection is fully specified by its query")
        query = FailureQuery.model_validate(
            read_json(
                artifacts, input_artifact(spec, "query", artifacts), "physrealx.failure-query/2"
            )
        )
        report = read_json(artifacts, query.report, "physrealx.failure-report/2")
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
                    "schema": "physrealx.selection-report/2",
                    "selected_episode_ids": sorted(selected),
                    "regeneration_seeds": seeds,
                    "failure_report": query.report.model_dump(mode="json"),
                    "relabelled_success": False,
                },
            },
            metrics={"selected_episodes": float(len(summaries))},
        )


def build_physics_calibration_plan(
    artifacts: ArtifactStore, telemetry_ref: ArtifactRef, failure_report_ref: ArtifactRef
) -> ArtifactRef:
    """Create an honest calibration decision from deployment source and diagnosed symptoms."""
    telemetry = read_json(artifacts, telemetry_ref, "physrealx.telemetry-bundle/2")
    report = read_json(artifacts, failure_report_ref, "physrealx.failure-report/2")
    if report["telemetry"] != telemetry_ref.model_dump(mode="json"):
        raise DomainValidationError("calibration plan telemetry differs from the failure report")
    real = set(telemetry["source_counts"]) == {EpisodeSource.REAL.value}
    affected = [
        row["episode_id"] for row in report["failures"] if row["category"] in {"control", "contact"}
    ]
    if not real:
        status, reason = (
            "AWAITING_REAL_EVIDENCE",
            "physical parameter identification requires attested real deployment trajectories",
        )
    elif not affected:
        status, reason = "NOT_REQUIRED", "no control or contact symptom requires calibration"
    else:
        status, reason = "READY", "real train/holdout trajectories are available for identification"
    return artifacts.put_json(
        {
            "schema": "physrealx.physics-calibration-plan/1",
            "status": status,
            "reason": reason,
            "telemetry": telemetry_ref.model_dump(mode="json"),
            "failure_report": failure_report_ref.model_dump(mode="json"),
            "affected_episodes": affected,
            "method": "bounded_inverse_dynamics",
            "required_signals": ["joint.position", "joint.velocity", "joint.effort"],
            "identified_parameters": [
                "effective_inertia",
                "viscous_damping",
                "coulomb_friction",
            ],
            "validation": "independent real holdout trajectory",
        }
    )


def prepare_joint_identification_evidence(
    artifacts: ArtifactStore,
    telemetry_ref: ArtifactRef,
    *,
    joint_name: str,
    joint_type: Literal["slide", "hinge"],
    joint_index: int,
    prior_parameters: tuple[float, float, float],
) -> ArtifactRef:
    """Convert two attested real deployment episodes to the existing physics identifier."""
    import numpy as np

    from physrealx.integrations.physics import JointIdentificationEvidence

    telemetry = read_json(artifacts, telemetry_ref, "physrealx.telemetry-bundle/2")
    if set(telemetry["source_counts"]) != {EpisodeSource.REAL.value}:
        raise DomainValidationError("physics calibration evidence must come from real deployment")
    dataset = load_resource(
        artifacts, VersionedRef.model_validate(telemetry["dataset"]), DatasetManifest
    )
    episodes = [load_episode(artifacts, row) for row in dataset.episodes]
    if len(episodes) < 2:
        raise DomainValidationError(
            "physics calibration requires independent train and holdout episodes"
        )
    episodes.sort(
        key=lambda episode: float(
            np.ptp(
                np.asarray(
                    [step.observations["joint.position"].inline for step in episode.steps],
                    dtype=float,
                )[:, joint_index]
            )
        ),
        reverse=True,
    )
    captures: list[str] = []
    arrays: list[ArtifactRef] = []
    for episode in episodes:
        capture = episode.metadata.get("capture_id")
        if not isinstance(capture, str) or not capture or capture in captures:
            continue
        time = np.asarray([step.timestamp_seconds for step in episode.steps], dtype=float)
        position = np.asarray(
            [step.observations["joint.position"].inline for step in episode.steps], dtype=float
        )[:, joint_index]
        velocity = np.asarray(
            [step.observations["joint.velocity"].inline for step in episode.steps], dtype=float
        )[:, joint_index]
        effort = np.asarray(
            [step.observations["joint.effort"].inline for step in episode.steps], dtype=float
        )[:, joint_index]
        acceleration = np.gradient(velocity, time, edge_order=2)
        values = np.column_stack((time, position, velocity, acceleration, effort))
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        arrays.append(artifacts.put_bytes(stream.getvalue(), media_type="application/x-npy"))
        captures.append(capture)
        if len(arrays) == 2:
            break
    if len(arrays) != 2:
        raise DomainValidationError(
            "physics calibration requires two distinct real capture_id groups"
        )
    return artifacts.put_json(
        JointIdentificationEvidence(
            joint_name=joint_name,
            joint_type=joint_type,
            train=arrays[0],
            holdout=arrays[1],
            train_capture_id=captures[0],
            holdout_capture_id=captures[1],
            position_unit="m" if joint_type == "slide" else "rad",
            force_unit="N" if joint_type == "slide" else "N*m",
            gravity_compensated=True,
            prior_parameters=prior_parameters,
        )
    )


__all__ = [
    "DeploymentEvidence",
    "FailureQuery",
    "PolicyGateConfig",
    "PolicyInterventionEvidence",
    "analyze_policy_intervention",
    "assess_policy_update",
    "build_physics_calibration_plan",
    "episode_metrics",
    "prepare_joint_identification_evidence",
]
