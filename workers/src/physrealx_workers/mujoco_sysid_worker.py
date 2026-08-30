"""Evidence-driven single-joint system-identification worker.

This module is a PhysReal-X adapter.  It fits a small dynamics model that can be
written back to a MuJoCo joint, but it is not part of the upstream MuJoCo
project and does not claim that MuJoCo performed the optimisation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physrealx.external_workers import (
    WorkerProducedFile,
    WorkerRequest,
    WorkerResponse,
    run_worker,
)
from physrealx_workers.common import config, dependency_path, is_probe, report_probe, resource_json

_ADAPTER = "physrealx_mujoco_sysid_adapter"
_EVIDENCE_SCHEMA = "physrealx.joint-system-identification-evidence/1"
_JOINT_PHYSICS_SCHEMA = "physrealx.identified-joint-physics/1"
_REPORT_SCHEMA = "physrealx.physics-identification-report/1"
_STATUS_IDENTIFIED = "IDENTIFIED"
_STATUS_NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
_PARAMETER_NAMES = ("effective_inertia", "viscous_damping", "coulomb_friction")
_REQUIRED_SIGNAL_NAMES = ("timestamp", "position", "velocity", "applied_effort", "split")
_OPTIONAL_SIGNAL_NAMES = ("external_effort", "contact_effort", "known_bias_effort")
_ALLOWED_CONFIG = {
    "condition_number_limit",
    "confidence_level",
    "friction_velocity_scale",
    "inertia_bounds",
    "damping_bounds",
    "friction_bounds",
    "minimum_acceleration_span",
    "minimum_effort_span",
    "minimum_holdout_samples",
    "minimum_prior_improvement",
    "minimum_train_samples",
    "minimum_velocity_span",
    "maximum_kinematic_relative_rmse",
    "multi_start_count",
}


@dataclass(frozen=True)
class _Options:
    minimum_train_samples: int
    minimum_holdout_samples: int
    minimum_velocity_span: float
    minimum_acceleration_span: float
    minimum_effort_span: float
    condition_number_limit: float
    minimum_prior_improvement: float
    maximum_kinematic_relative_rmse: float
    friction_velocity_scale: float
    confidence_level: float
    multi_start_count: int
    lower_bounds: tuple[float, float, float]
    upper_bounds: tuple[float, float, float]


@dataclass(frozen=True)
class _Identification:
    status: str
    identified_joint_physics: dict[str, Any]
    report: dict[str, Any]
    metrics: dict[str, float]


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be greater than zero")
    return result


def _positive_integer(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _bounds(
    options: dict[str, Any],
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    value = options.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly [lower, upper]")
    lower = _finite_float(value[0], f"{name}[0]")
    upper = _finite_float(value[1], f"{name}[1]")
    if lower < 0.0 or lower >= upper:
        raise ValueError(f"{name} must satisfy 0 <= lower < upper")
    return lower, upper


def _options(raw: dict[str, Any]) -> _Options:
    unknown = sorted(set(raw) - _ALLOWED_CONFIG)
    if unknown:
        raise ValueError(f"unknown MuJoCo SysID adapter options: {unknown}")
    inertia = _bounds(raw, "inertia_bounds", (1.0e-6, 1.0e4))
    damping = _bounds(raw, "damping_bounds", (0.0, 1.0e4))
    friction = _bounds(raw, "friction_bounds", (0.0, 1.0e4))
    confidence = _finite_float(raw.get("confidence_level", 0.95), "confidence_level")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1.0")
    improvement = _finite_float(
        raw.get("minimum_prior_improvement", 0.02),
        "minimum_prior_improvement",
    )
    if not 0.0 <= improvement < 1.0:
        raise ValueError("minimum_prior_improvement must be in [0, 1)")
    return _Options(
        minimum_train_samples=_positive_integer(
            raw.get("minimum_train_samples", 40), "minimum_train_samples", 8
        ),
        minimum_holdout_samples=_positive_integer(
            raw.get("minimum_holdout_samples", 20), "minimum_holdout_samples", 4
        ),
        minimum_velocity_span=_positive_float(
            raw.get("minimum_velocity_span", 0.1), "minimum_velocity_span"
        ),
        minimum_acceleration_span=_positive_float(
            raw.get("minimum_acceleration_span", 0.1), "minimum_acceleration_span"
        ),
        minimum_effort_span=_positive_float(
            raw.get("minimum_effort_span", 0.1), "minimum_effort_span"
        ),
        condition_number_limit=_positive_float(
            raw.get("condition_number_limit", 1.0e6), "condition_number_limit"
        ),
        minimum_prior_improvement=improvement,
        maximum_kinematic_relative_rmse=_positive_float(
            raw.get("maximum_kinematic_relative_rmse", 0.15),
            "maximum_kinematic_relative_rmse",
        ),
        friction_velocity_scale=_positive_float(
            raw.get("friction_velocity_scale", 0.02), "friction_velocity_scale"
        ),
        confidence_level=confidence,
        multi_start_count=_positive_integer(
            raw.get("multi_start_count", 9), "multi_start_count", 3
        ),
        lower_bounds=(inertia[0], damping[0], friction[0]),
        upper_bounds=(inertia[1], damping[1], friction[1]),
    )


def _validate_manifest(manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    if manifest.get("schema") != _EVIDENCE_SCHEMA:
        raise ValueError(f"evidence schema must be {_EVIDENCE_SCHEMA!r}")
    joint = manifest.get("joint")
    if not isinstance(joint, dict):
        raise ValueError("evidence must declare a joint object")
    if joint.get("type") not in {"hinge", "slide"}:
        raise ValueError("joint.type must be 'hinge' or 'slide'")
    if not isinstance(joint.get("name"), str) or not joint["name"].strip():
        raise ValueError("joint.name must be a non-empty string")
    signals = manifest.get("signals")
    if not isinstance(signals, dict):
        raise ValueError("evidence must declare a signals object")
    unknown_signals = sorted(set(signals) - set(_REQUIRED_SIGNAL_NAMES + _OPTIONAL_SIGNAL_NAMES))
    if unknown_signals:
        raise ValueError(f"evidence declares unsupported signals: {unknown_signals}")
    missing = [name for name in _REQUIRED_SIGNAL_NAMES if name not in signals]
    if missing:
        raise ValueError(f"evidence is missing required signals: {missing}")
    if any(not isinstance(signals[name], str) or not signals[name] for name in signals):
        raise ValueError("signal mappings must name non-empty NPZ arrays")
    if len(set(signals.values())) != len(signals):
        raise ValueError("signal mappings must reference distinct NPZ arrays")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("digest"), str):
        raise ValueError("evidence must reference one immutable NPZ artifact")
    if manifest.get("format") != "npz":
        raise ValueError("evidence format must be 'npz'")
    effort = manifest.get("effort_evidence")
    if not isinstance(effort, dict):
        raise ValueError("effort_evidence is required; controller commands alone are insufficient")
    if effort.get("kind") not in {
        "measured_actuator_effort",
        "calibrated_force_sensor",
        "measured_contact_wrench",
    }:
        raise ValueError("effort_evidence.kind must identify measured or calibrated effort")
    if not isinstance(effort.get("source"), str) or not effort["source"].strip():
        raise ValueError("effort_evidence.source must identify the measurement source")
    if manifest.get("effort_sign_convention") != "positive_assists_joint_motion":
        raise ValueError("effort_sign_convention must be 'positive_assists_joint_motion'")
    expected_units = {
        "timestamp": "s",
        "position": "rad" if joint["type"] == "hinge" else "m",
        "velocity": "rad/s" if joint["type"] == "hinge" else "m/s",
        "applied_effort": "N*m" if joint["type"] == "hinge" else "N",
    }
    for optional_name in _OPTIONAL_SIGNAL_NAMES:
        if optional_name in signals:
            expected_units[optional_name] = expected_units["applied_effort"]
    units = manifest.get("units")
    if not isinstance(units, dict) or units != expected_units:
        raise ValueError(f"units must exactly match the declared joint signals: {expected_units}")
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError("evidence must define explicit train and holdout split values")
    if set(split) != {"train", "holdout"}:
        raise ValueError("split must contain exactly train and holdout values")
    if split["train"] == split["holdout"]:
        raise ValueError("train and holdout split values must differ")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("measurement provenance is required")
    capture_ids = provenance.get("capture_ids")
    if (
        not isinstance(capture_ids, list)
        or not capture_ids
        or any(not isinstance(item, str) or not item.strip() for item in capture_ids)
    ):
        raise ValueError("provenance.capture_ids must contain measured run identifiers")
    if len(capture_ids) != len(set(capture_ids)):
        raise ValueError("provenance.capture_ids must be unique")
    split_provenance = manifest.get("split_provenance")
    if not isinstance(split_provenance, dict) or set(split_provenance) != {
        "train_capture_ids",
        "holdout_capture_ids",
    }:
        raise ValueError("split_provenance must map train and holdout to measured capture IDs")
    train_capture_ids = split_provenance["train_capture_ids"]
    holdout_capture_ids = split_provenance["holdout_capture_ids"]
    for name, identifiers in (
        ("train_capture_ids", train_capture_ids),
        ("holdout_capture_ids", holdout_capture_ids),
    ):
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or any(not isinstance(item, str) or not item.strip() for item in identifiers)
        ):
            raise ValueError(f"split_provenance.{name} must be a non-empty string array")
    if set(train_capture_ids) & set(holdout_capture_ids):
        raise ValueError("train and holdout capture IDs must be disjoint")
    if not (set(train_capture_ids) | set(holdout_capture_ids)) <= set(capture_ids):
        raise ValueError("split provenance references an undeclared capture ID")
    prior = manifest.get("prior")
    if not isinstance(prior, dict):
        raise ValueError("a declared engineering prior is required for holdout comparison")
    if prior.get("source_type") != "engineering_baseline":
        raise ValueError(
            "prior.source_type must be 'engineering_baseline', not measured material data"
        )
    values = prior.get("parameters")
    if not isinstance(values, dict) or set(values) != set(_PARAMETER_NAMES):
        raise ValueError(f"prior.parameters must contain exactly {_PARAMETER_NAMES}")
    for name in _PARAMETER_NAMES:
        if _finite_float(values[name], f"prior.parameters.{name}") < 0.0:
            raise ValueError(f"prior.parameters.{name} cannot be negative")
    return {str(key): str(value) for key, value in signals.items()}, prior


def _load_npz(path: Path, signal_names: dict[str, str]) -> dict[str, Any]:
    import numpy as np

    if path.suffix.lower() != ".npz":
        raise ValueError("system-identification evidence artifact must be an .npz file")
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = [member for member in signal_names.values() if member not in archive.files]
        if missing:
            raise ValueError(f"NPZ evidence is missing declared arrays: {missing}")
        for semantic_name, member in signal_names.items():
            array = np.asarray(archive[member])
            if array.ndim != 1:
                raise ValueError(f"signal {semantic_name!r} must be one-dimensional")
            arrays[semantic_name] = array
    lengths = {int(array.shape[0]) for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all evidence signals must have the same sample count")
    return arrays


def _rmse(residual: Any) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(residual))))


def _not_identifiable(
    blockers: list[str],
    *,
    evidence_digest: str,
    joint: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> _Identification:
    physics: dict[str, Any] = {
        "schema": _JOINT_PHYSICS_SCHEMA,
        "status": _STATUS_NOT_IDENTIFIABLE,
        "adapter": _ADAPTER,
        "evidence_digest": evidence_digest,
        "blockers": blockers,
    }
    if joint is not None:
        physics["joint"] = joint
    report = {
        "schema": _REPORT_SCHEMA,
        "status": _STATUS_NOT_IDENTIFIABLE,
        "adapter": _ADAPTER,
        "algorithm": "bounded_multi_start_least_squares",
        "blockers": blockers,
        "diagnostics": diagnostics or {},
    }
    diagnostics_value = diagnostics or {}
    train_count = diagnostics_value.get("train_samples", 0.0)
    holdout_count = diagnostics_value.get("holdout_samples", 0.0)
    metrics = {
        "identifiable": 0.0,
        "train_samples": float(train_count),
        "holdout_samples": float(holdout_count),
    }
    return _Identification(_STATUS_NOT_IDENTIFIABLE, physics, report, metrics)


def _identify(
    manifest: dict[str, Any],
    arrays: dict[str, Any],
    options: _Options,
    *,
    evidence_digest: str,
) -> _Identification:
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.stats import norm

    signals, prior = _validate_manifest(manifest)
    del signals
    joint_manifest = manifest["joint"]
    joint = {"name": str(joint_manifest["name"]), "type": str(joint_manifest["type"])}
    values: dict[str, Any] = {}
    for name, array in arrays.items():
        if name == "split":
            values[name] = np.asarray(array)
            continue
        converted = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(converted)):
            return _not_identifiable(
                [f"signal {name!r} contains NaN or infinity"],
                evidence_digest=evidence_digest,
                joint=joint,
            )
        values[name] = converted
    sample_count = len(values["timestamp"])
    if sample_count < 5:
        return _not_identifiable(
            ["at least five ordered samples are required"],
            evidence_digest=evidence_digest,
            joint=joint,
        )
    timestamps = values["timestamp"]
    delta = np.diff(timestamps)
    if not np.all(delta > 0.0):
        return _not_identifiable(
            ["timestamps must be finite and strictly increasing"],
            evidence_digest=evidence_digest,
            joint=joint,
        )
    split_manifest = manifest["split"]
    train_mask = values["split"] == split_manifest["train"]
    holdout_mask = values["split"] == split_manifest["holdout"]
    known_mask = train_mask | holdout_mask
    train_count = int(np.count_nonzero(train_mask))
    holdout_count = int(np.count_nonzero(holdout_mask))
    diagnostics: dict[str, Any] = {
        "sample_count": sample_count,
        "train_samples": train_count,
        "holdout_samples": holdout_count,
        "unknown_split_samples": int(np.count_nonzero(~known_mask)),
    }
    blockers: list[str] = []
    if np.any(~known_mask):
        blockers.append("every sample must belong to the declared train or holdout split")
    if train_count < options.minimum_train_samples:
        blockers.append(
            f"train split has {train_count} samples; requires {options.minimum_train_samples}"
        )
    if holdout_count < options.minimum_holdout_samples:
        blockers.append(
            f"holdout split has {holdout_count} samples; requires {options.minimum_holdout_samples}"
        )
    velocity = values["velocity"]
    acceleration = np.gradient(velocity, timestamps, edge_order=2)
    applied = values["applied_effort"].copy()
    for name in ("external_effort", "contact_effort"):
        if name in values:
            applied = applied + values[name]
    if "known_bias_effort" in values:
        applied = applied - values["known_bias_effort"]
    if not np.all(np.isfinite(acceleration)) or not np.all(np.isfinite(applied)):
        blockers.append("derived acceleration or net effort is non-finite")
    velocity_span = float(np.ptp(velocity[train_mask])) if train_count else 0.0
    acceleration_span = float(np.ptp(acceleration[train_mask])) if train_count else 0.0
    effort_span = float(np.ptp(applied[train_mask])) if train_count else 0.0
    reconstructed_velocity = np.gradient(values["position"], timestamps, edge_order=2)
    velocity_rmse = _rmse(reconstructed_velocity - velocity)
    velocity_scale = max(_rmse(velocity), np.finfo(np.float64).eps)
    kinematic_relative_rmse = float(velocity_rmse / velocity_scale)
    diagnostics.update(
        {
            "velocity_span": velocity_span,
            "acceleration_span": acceleration_span,
            "effort_span": effort_span,
            "position_velocity_relative_rmse": kinematic_relative_rmse,
        }
    )
    if velocity_span < options.minimum_velocity_span:
        blockers.append("training motion does not span enough joint velocity")
    if acceleration_span < options.minimum_acceleration_span:
        blockers.append("training motion does not span enough joint acceleration")
    if effort_span < options.minimum_effort_span:
        blockers.append("training motion does not span enough measured effort")
    if kinematic_relative_rmse > options.maximum_kinematic_relative_rmse:
        blockers.append("measured position and velocity are kinematically inconsistent")
    if train_count and not (
        np.any(velocity[train_mask] > options.friction_velocity_scale)
        and np.any(velocity[train_mask] < -options.friction_velocity_scale)
    ):
        blockers.append("training motion must excite both joint-motion directions")
    design = np.column_stack(
        (
            acceleration,
            velocity,
            np.tanh(velocity / options.friction_velocity_scale),
        )
    )
    if train_count:
        train_design = design[train_mask]
        rank = int(np.linalg.matrix_rank(train_design))
        condition_number = float(np.linalg.cond(train_design))
    else:
        rank = 0
        condition_number = options.condition_number_limit + 1.0
    diagnostics["design_rank"] = rank
    if math.isfinite(condition_number):
        diagnostics["condition_number"] = condition_number
    else:
        diagnostics["condition_number_status"] = "non_finite"
    if rank < len(_PARAMETER_NAMES):
        blockers.append("training dynamics matrix is rank deficient")
    if not math.isfinite(condition_number) or condition_number > options.condition_number_limit:
        blockers.append("training dynamics matrix is too ill-conditioned")
    if blockers:
        return _not_identifiable(
            blockers,
            evidence_digest=evidence_digest,
            joint=joint,
            diagnostics=diagnostics,
        )

    lower = np.asarray(options.lower_bounds, dtype=np.float64)
    upper = np.asarray(options.upper_bounds, dtype=np.float64)
    prior_parameters = np.asarray(
        [prior["parameters"][name] for name in _PARAMETER_NAMES],
        dtype=np.float64,
    )
    clipped_prior = np.clip(prior_parameters, lower, upper)
    starts = [clipped_prior, (lower + upper) * 0.5]
    fractions = np.linspace(0.1, 0.9, options.multi_start_count)
    for index, _fraction in enumerate(fractions):
        rolled = np.roll(fractions, index)
        starts.append(lower + (upper - lower) * rolled[: len(_PARAMETER_NAMES)])
    fitted: list[Any] = []
    train_target = applied[train_mask]
    for start in starts[: options.multi_start_count]:
        result = least_squares(
            lambda candidate: train_design @ candidate - train_target,
            start,
            bounds=(lower, upper),
            method="trf",
            loss="linear",
            max_nfev=5_000,
        )
        if result.success and np.all(np.isfinite(result.x)):
            fitted.append(result)
    if not fitted:
        return _not_identifiable(
            ["bounded optimizer did not produce a finite converged solution"],
            evidence_digest=evidence_digest,
            joint=joint,
            diagnostics=diagnostics,
        )
    best = min(fitted, key=lambda item: float(np.dot(item.fun, item.fun)))
    parameters = np.asarray(best.x, dtype=np.float64)
    holdout_design = design[holdout_mask]
    holdout_target = applied[holdout_mask]
    train_residual = train_design @ parameters - train_target
    holdout_residual = holdout_design @ parameters - holdout_target
    prior_holdout_residual = holdout_design @ prior_parameters - holdout_target
    train_rmse = _rmse(train_residual)
    holdout_rmse = _rmse(holdout_residual)
    prior_holdout_rmse = _rmse(prior_holdout_residual)
    denominator = max(prior_holdout_rmse, np.finfo(np.float64).eps)
    improvement = float((prior_holdout_rmse - holdout_rmse) / denominator)
    diagnostics.update(
        {
            "converged_starts": len(fitted),
            "train_rmse": train_rmse,
            "holdout_rmse": holdout_rmse,
            "prior_holdout_rmse": prior_holdout_rmse,
            "prior_improvement_fraction": improvement,
        }
    )
    if holdout_rmse >= prior_holdout_rmse or improvement < options.minimum_prior_improvement:
        return _not_identifiable(
            ["identified parameters do not improve the declared prior on the holdout split"],
            evidence_digest=evidence_digest,
            joint=joint,
            diagnostics=diagnostics,
        )

    degrees_of_freedom = max(1, train_count - len(_PARAMETER_NAMES))
    variance = max(
        float(np.dot(train_residual, train_residual)) / degrees_of_freedom,
        np.finfo(np.float64).eps,
    )
    covariance = variance * np.linalg.pinv(train_design.T @ train_design)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_score = float(norm.ppf(0.5 + options.confidence_level / 2.0))
    all_solutions = np.stack([np.asarray(item.x, dtype=np.float64) for item in fitted])
    multi_start_spread = np.ptp(all_solutions, axis=0)
    parameter_output: dict[str, Any] = {}
    for index, name in enumerate(_PARAMETER_NAMES):
        estimate = float(parameters[index])
        low = float(max(lower[index], estimate - z_score * standard_error[index]))
        high = float(min(upper[index], estimate + z_score * standard_error[index]))
        unit = "kg*m^2" if name == "effective_inertia" else "N*m*s/rad"
        if name == "coulomb_friction":
            unit = "N*m"
        if joint["type"] == "slide":
            unit = "kg" if name == "effective_inertia" else "N*s/m"
            if name == "coulomb_friction":
                unit = "N"
        parameter_output[name] = {
            "value": estimate,
            "unit": unit,
            "confidence_interval": {
                "level": options.confidence_level,
                "lower": low,
                "upper": high,
            },
            "multi_start_spread": float(multi_start_spread[index]),
        }
    identified = {
        "schema": _JOINT_PHYSICS_SCHEMA,
        "status": _STATUS_IDENTIFIED,
        "adapter": _ADAPTER,
        "upstream_attribution": "not_an_upstream_mujoco_component",
        "joint": joint,
        "parameters": parameter_output,
        "evidence_digest": evidence_digest,
        "evidence_state": "identified",
    }
    report = {
        "schema": _REPORT_SCHEMA,
        "status": _STATUS_IDENTIFIED,
        "adapter": _ADAPTER,
        "algorithm": "bounded_multi_start_least_squares",
        "dynamics_equation": (
            "net_effort = effective_inertia*acceleration + viscous_damping*velocity + "
            "coulomb_friction*tanh(velocity/friction_velocity_scale)"
        ),
        "friction_velocity_scale": options.friction_velocity_scale,
        "diagnostics": diagnostics,
        "parameter_bounds": {
            name: {"lower": float(lower[index]), "upper": float(upper[index])}
            for index, name in enumerate(_PARAMETER_NAMES)
        },
        "provenance": {
            "capture_ids": manifest["provenance"]["capture_ids"],
            "split_provenance": manifest["split_provenance"],
            "effort_evidence": manifest["effort_evidence"],
            "prior_source": prior.get("source", "unspecified_engineering_baseline"),
        },
    }
    metrics = {
        "identifiable": 1.0,
        "train_samples": float(train_count),
        "holdout_samples": float(holdout_count),
        "train_rmse": train_rmse,
        "holdout_rmse": holdout_rmse,
        "prior_holdout_rmse": prior_holdout_rmse,
        "prior_improvement_fraction": improvement,
        "condition_number": condition_number,
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("identification produced a non-finite metric")
    return _Identification(_STATUS_IDENTIFIED, identified, report, metrics)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    selected_options = _options(config(request))
    evidence_digest = next(
        (item.digest for item in request.inputs if item.name == "evidence"),
        f"sha256:{'0' * 64}",
    )
    try:
        manifest = resource_json(request, "evidence")
        signal_names, _ = _validate_manifest(manifest)
        artifact = manifest["artifact"]
        evidence_path = dependency_path(request, "evidence", str(artifact["digest"]))
        arrays = _load_npz(evidence_path, signal_names)
        result = _identify(
            manifest,
            arrays,
            selected_options,
            evidence_digest=evidence_digest,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        result = _not_identifiable(
            [str(error) or type(error).__name__],
            evidence_digest=evidence_digest,
        )
    physics_path = output_directory / "identified_joint_physics.json"
    report_path = output_directory / "identification_report.json"
    _write_json(physics_path, result.identified_joint_physics)
    _write_json(report_path, result.report)
    produced = {
        "identified_joint_physics": WorkerProducedFile(
            name="identified_joint_physics",
            contract="IdentifiedJointPhysics/1",
            path=physics_path.name,
            media_type="application/json",
        ),
        "identification_report": WorkerProducedFile(
            name="identification_report",
            contract="PhysicsIdentificationReport/1",
            path=report_path.name,
            media_type="application/json",
        ),
    }
    try:
        outputs = tuple(produced[item.name] for item in request.operation.outputs)
    except KeyError as error:
        raise ValueError(f"unsupported MuJoCo SysID output: {error.args[0]}") from error
    return WorkerResponse(outputs=outputs, metrics=result.metrics)


def _synthetic_self_test() -> dict[str, str]:
    """Recover known parameters so readiness proves more than importability."""

    import numpy as np

    timestamp = np.linspace(0.0, 12.0, 601, dtype=np.float64)
    velocity = 0.7 * np.sin(0.9 * timestamp) + 0.3 * np.sin(2.1 * timestamp + 0.2)
    acceleration = 0.7 * 0.9 * np.cos(0.9 * timestamp) + 0.3 * 2.1 * np.cos(2.1 * timestamp + 0.2)
    position = np.cumsum(velocity) * float(timestamp[1] - timestamp[0])
    effort = 2.3 * acceleration + 0.7 * velocity + 0.24 * np.tanh(velocity / 0.02)
    split = np.zeros(timestamp.shape, dtype=np.int8)
    split[451:] = 1
    manifest = {
        "schema": _EVIDENCE_SCHEMA,
        "joint": {"name": "probe_slide", "type": "slide"},
        "format": "npz",
        "artifact": {
            "digest": f"sha256:{'a' * 64}",
            "size_bytes": 1,
            "media_type": "application/x-npz",
        },
        "signals": {
            "timestamp": "timestamp",
            "position": "position",
            "velocity": "velocity",
            "applied_effort": "applied_effort",
            "split": "split",
        },
        "units": {
            "timestamp": "s",
            "position": "m",
            "velocity": "m/s",
            "applied_effort": "N",
        },
        "effort_evidence": {
            "kind": "measured_actuator_effort",
            "source": "deterministic-installation-fixture",
        },
        "effort_sign_convention": "positive_assists_joint_motion",
        "split": {"train": 0, "holdout": 1},
        "prior": {
            "source_type": "engineering_baseline",
            "source": "deterministic-installation-fixture",
            "parameters": {
                "effective_inertia": 0.8,
                "viscous_damping": 0.1,
                "coulomb_friction": 0.05,
            },
        },
        "provenance": {"capture_ids": ["probe-train", "probe-holdout"]},
        "split_provenance": {
            "train_capture_ids": ["probe-train"],
            "holdout_capture_ids": ["probe-holdout"],
        },
    }
    result = _identify(
        manifest,
        {
            "timestamp": timestamp,
            "position": position,
            "velocity": velocity,
            "applied_effort": effort,
            "split": split,
        },
        _options(
            {
                "inertia_bounds": [0.1, 5.0],
                "damping_bounds": [0.0, 3.0],
                "friction_bounds": [0.0, 1.0],
                "multi_start_count": 9,
            }
        ),
        evidence_digest=f"sha256:{'b' * 64}",
    )
    if result.status != _STATUS_IDENTIFIED:
        raise RuntimeError(f"SysID synthetic self-test was not identifiable: {result.report}")
    expected = {
        "effective_inertia": 2.3,
        "viscous_damping": 0.7,
        "coulomb_friction": 0.24,
    }
    maximum_relative_error = max(
        abs(result.identified_joint_physics["parameters"][name]["value"] - value) / value
        for name, value in expected.items()
    )
    if maximum_relative_error > 0.02:
        raise RuntimeError("SysID synthetic self-test exceeded 2% parameter error")
    return {
        "self_test": "passed",
        "self_test_max_relative_error": f"{maximum_relative_error:.8g}",
        "self_test_holdout_rmse": f"{result.metrics['holdout_rmse']:.8g}",
        "self_test_prior_improvement": (f"{result.metrics['prior_improvement_fraction']:.8g}"),
    }


def main() -> int:
    if is_probe():
        import numpy
        import scipy

        major = int(numpy.__version__.split(".", 1)[0])
        if major != 1:
            raise RuntimeError("MuJoCo SysID adapter requires NumPy 1.x")
        self_test = _synthetic_self_test()
        return report_probe(
            capability="physics.calibrate",
            name=_ADAPTER,
            version="1.0.0",
            source_revision="adapter-1.0.0",
            dependencies={
                "adapter": "1.0.0",
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
            },
            self_test_checks=self_test,
        )
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
