"""Resolve COLMAP's similarity gauge from independent metric 3D evidence."""

from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from physrealx.external_workers import (
    WorkerProducedFile,
    WorkerRequest,
    WorkerResponse,
    run_worker,
)
from physrealx_workers.common import (
    config,
    declared_input_path,
    input_file,
    is_probe,
    report_probe,
)

_ADAPTER = "metric_alignment"
_VERSION = "1.0.0"
_SOURCE_REVISION = "adapter-1.0.0"
_RIG_SCHEMA = "physrealx.camera-rig-calibration/1"
_EVIDENCE_SCHEMA = "physrealx.metric-alignment-evidence/1"
_REPORT_SCHEMA = "physrealx.metric-alignment-report/1"
_OUTPUT_FILES = {
    "camera_rig_calibration": (
        "camera-rig-calibration.json",
        "CameraRigCalibration/1",
    ),
    "metric_alignment_report": (
        "metric-alignment-report.json",
        "MetricAlignmentReport/1",
    ),
}
_ALLOWED_OPTIONS = {
    "inlier_threshold_m",
    "max_geometry_condition",
    "max_holdout_relative_rmse",
    "max_holdout_rmse_m",
    "max_scale_cv",
    "maximum_scale_m_per_source_unit",
    "minimum_inlier_ratio",
    "minimum_inliers",
    "minimum_scale_m_per_source_unit",
    "random_seed",
    "ransac_iterations",
}
_EVIDENCE_FIELDS = {
    "schema",
    "resource_id",
    "version",
    "unit",
    "camera_rig_digest",
    "component_id",
    "source_frame",
    "target_frame",
    "metric_reference",
    "provenance",
    "train_correspondences",
    "holdout_correspondences",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"input JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} must be unambiguous finite UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{label} must be a finite {qualifier}number")
    return result


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _options(request: WorkerRequest) -> dict[str, float | int]:
    raw = config(request)
    unknown = sorted(set(raw) - _ALLOWED_OPTIONS)
    if unknown:
        raise ValueError(f"unknown metric alignment options: {unknown}")
    options: dict[str, float | int] = {
        "inlier_threshold_m": 0.01,
        "max_geometry_condition": 10_000.0,
        "max_holdout_relative_rmse": 0.02,
        "max_holdout_rmse_m": 0.01,
        "max_scale_cv": 0.02,
        "maximum_scale_m_per_source_unit": 1_000_000.0,
        "minimum_inlier_ratio": 0.75,
        "minimum_inliers": 4,
        "minimum_scale_m_per_source_unit": 1e-9,
        "random_seed": 0,
        "ransac_iterations": 512,
    }
    options.update(raw)
    for name in (
        "inlier_threshold_m",
        "max_geometry_condition",
        "max_holdout_relative_rmse",
        "max_holdout_rmse_m",
        "max_scale_cv",
        "maximum_scale_m_per_source_unit",
        "minimum_scale_m_per_source_unit",
    ):
        options[name] = _number(options[name], name, positive=True)
    ratio = _number(options["minimum_inlier_ratio"], "minimum_inlier_ratio", positive=True)
    if ratio > 1.0:
        raise ValueError("minimum_inlier_ratio must be <= 1")
    options["minimum_inlier_ratio"] = ratio
    options["minimum_inliers"] = _integer(options["minimum_inliers"], "minimum_inliers", minimum=4)
    options["ransac_iterations"] = _integer(
        options["ransac_iterations"], "ransac_iterations", minimum=1
    )
    options["random_seed"] = _integer(options["random_seed"], "random_seed", minimum=0)
    if float(options["minimum_scale_m_per_source_unit"]) >= float(
        options["maximum_scale_m_per_source_unit"]
    ):
        raise ValueError("minimum scale must be smaller than maximum scale")
    return options


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _point(value: Any, label: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates")
    return np.asarray([_number(item, label) for item in value], dtype=np.float64)


def _correspondences(
    value: Any,
    label: str,
    *,
    minimum: int,
) -> tuple[
    tuple[str, ...], np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} requires at least {minimum} correspondences")
    identifiers: list[str] = []
    source: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    target: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        unknown = sorted(set(item) - {"id", "source_xyz", "target_xyz_m", "uncertainty_m"})
        if unknown:
            raise ValueError(f"{label}[{index}] contains unknown fields: {unknown}")
        identifier = _string(item.get("id"), f"{label}[{index}].id")
        if "uncertainty_m" in item:
            _number(item["uncertainty_m"], f"{label}[{index}].uncertainty_m", positive=True)
        identifiers.append(identifier)
        source.append(_point(item.get("source_xyz"), f"{label}[{index}].source_xyz"))
        target.append(_point(item.get("target_xyz_m"), f"{label}[{index}].target_xyz_m"))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} correspondence ids must be unique")
    if len({tuple(point) for point in source}) != len(source):
        raise ValueError(f"{label} source coordinates must be unique")
    if len({tuple(point) for point in target}) != len(target):
        raise ValueError(f"{label} target coordinates must be unique")
    return identifiers, np.stack(source), np.stack(target)


def _geometry_condition(points: np.ndarray[Any, np.dtype[np.float64]], label: str) -> float:
    centered = points - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = max(centered.shape) * np.finfo(np.float64).eps * max(float(singular[0]), 1.0)
    rank = int(np.count_nonzero(singular > tolerance))
    if rank < 2:
        raise ValueError(f"{label} geometry is degenerate or collinear")
    return float(singular[0] / singular[1])


def _umeyama(
    source: np.ndarray[Any, np.dtype[np.float64]],
    target: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[float, np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Sim(3) inputs must be matching N x 3 arrays")
    _geometry_condition(source, "source correspondence")
    _geometry_condition(target, "target correspondence")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = target_zero.T @ source_zero / source.shape[0]
    left, singular, right_transpose = np.linalg.svd(covariance)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0.0:
        correction[-1] = -1.0
    rotation = left @ np.diag(correction) @ right_transpose
    source_variance = float(np.sum(source_zero * source_zero) / source.shape[0])
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("source correspondence geometry has zero scale")
    scale = float(np.sum(singular * correction) / source_variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("metric alignment produced a non-positive scale")
    translation = target_center - scale * (rotation @ source_center)
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-8:
        raise ValueError("metric alignment did not produce a proper rotation")
    return scale, rotation, translation


def _apply(
    points: np.ndarray[Any, np.dtype[np.float64]],
    transform: tuple[
        float,
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    scale, rotation, translation = transform
    return scale * (points @ rotation.T) + translation


def _candidate_subsets(count: int, iterations: int, seed: int) -> tuple[tuple[int, int, int], ...]:
    total = math.comb(count, 3)
    if total <= iterations:
        return tuple(itertools.combinations(range(count), 3))
    generator = np.random.default_rng(seed)
    subsets: set[tuple[int, int, int]] = set()
    while len(subsets) < iterations:
        subsets.add(tuple(sorted(int(item) for item in generator.choice(count, 3, replace=False))))
    return tuple(sorted(subsets))


def _fit(
    train_source: np.ndarray[Any, np.dtype[np.float64]],
    train_target: np.ndarray[Any, np.dtype[np.float64]],
    holdout_source: np.ndarray[Any, np.dtype[np.float64]],
    holdout_target: np.ndarray[Any, np.dtype[np.float64]],
    options: dict[str, float | int],
) -> tuple[
    tuple[
        float,
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ],
    dict[str, float],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    source_condition = _geometry_condition(train_source, "training source")
    target_condition = _geometry_condition(train_target, "training target")
    holdout_source_condition = _geometry_condition(holdout_source, "holdout source")
    holdout_target_condition = _geometry_condition(holdout_target, "holdout target")
    maximum_condition = float(options["max_geometry_condition"])
    if (
        max(
            source_condition,
            target_condition,
            holdout_source_condition,
            holdout_target_condition,
        )
        > maximum_condition
    ):
        raise ValueError("metric alignment correspondence geometry is ill-conditioned")

    threshold = float(options["inlier_threshold_m"])
    best: tuple[int, float, tuple[float, np.ndarray[Any, Any], np.ndarray[Any, Any]]] | None = None
    for subset in _candidate_subsets(
        train_source.shape[0],
        int(options["ransac_iterations"]),
        int(options["random_seed"]),
    ):
        indices = np.asarray(subset, dtype=np.int64)
        try:
            candidate = _umeyama(train_source[indices], train_target[indices])
        except ValueError:
            continue
        residuals = np.linalg.norm(_apply(train_source, candidate) - train_target, axis=1)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        score = float(np.sum(np.minimum(residuals, threshold) ** 2))
        ordering = (count, -score)
        if best is None or ordering > (best[0], -best[1]):
            best = (count, score, candidate)
    if best is None:
        raise ValueError("RANSAC could not estimate a non-degenerate Sim(3) transform")
    residuals = np.linalg.norm(_apply(train_source, best[2]) - train_target, axis=1)
    inliers = residuals <= threshold
    minimum_inliers = int(options["minimum_inliers"])
    ratio = float(np.mean(inliers))
    if int(np.count_nonzero(inliers)) < minimum_inliers or ratio < float(
        options["minimum_inlier_ratio"]
    ):
        raise ValueError("metric alignment has insufficient RANSAC inlier support")
    transform = _umeyama(train_source[inliers], train_target[inliers])
    train_residuals = np.linalg.norm(_apply(train_source, transform) - train_target, axis=1)
    refined_inliers = train_residuals <= threshold
    if int(np.count_nonzero(refined_inliers)) < minimum_inliers or float(
        np.mean(refined_inliers)
    ) < float(options["minimum_inlier_ratio"]):
        raise ValueError("metric alignment lost inlier support after refinement")
    transform = _umeyama(train_source[refined_inliers], train_target[refined_inliers])
    train_residuals = np.linalg.norm(_apply(train_source, transform) - train_target, axis=1)
    holdout_residuals = np.linalg.norm(_apply(holdout_source, transform) - holdout_target, axis=1)

    jackknife_scales = []
    inlier_indices = np.flatnonzero(refined_inliers)
    for omitted in inlier_indices:
        selected = inlier_indices[inlier_indices != omitted]
        if selected.size < 3:
            continue
        jackknife_scales.append(_umeyama(train_source[selected], train_target[selected])[0])
    if len(jackknife_scales) < 4:
        raise ValueError("metric alignment requires at least four stable jackknife fits")
    scale_mean = float(np.mean(jackknife_scales))
    scale_cv = float(np.std(jackknife_scales, ddof=1) / scale_mean)
    scale = transform[0]
    if not (
        float(options["minimum_scale_m_per_source_unit"])
        <= scale
        <= float(options["maximum_scale_m_per_source_unit"])
    ):
        raise ValueError("metric alignment scale lies outside the declared physical bounds")
    if scale_cv > float(options["max_scale_cv"]):
        raise ValueError("metric alignment scale is unstable under jackknife validation")
    holdout_rmse = float(np.sqrt(np.mean(holdout_residuals**2)))
    target_spread = float(
        np.sqrt(np.mean(np.sum((holdout_target - np.mean(holdout_target, axis=0)) ** 2, axis=1)))
    )
    if target_spread <= np.finfo(np.float64).eps:
        raise ValueError("holdout metric geometry has zero physical extent")
    holdout_relative_rmse = holdout_rmse / target_spread
    if holdout_rmse > float(options["max_holdout_rmse_m"]):
        raise ValueError("metric alignment exceeds the absolute holdout RMSE gate")
    if holdout_relative_rmse > float(options["max_holdout_relative_rmse"]):
        raise ValueError("metric alignment exceeds the relative holdout RMSE gate")

    metrics = {
        "scale_m_per_source_unit": scale,
        "scale_cv": scale_cv,
        "training_inlier_ratio": float(np.mean(refined_inliers)),
        "training_inlier_rmse_m": float(np.sqrt(np.mean(train_residuals[refined_inliers] ** 2))),
        "training_all_rmse_m": float(np.sqrt(np.mean(train_residuals**2))),
        "holdout_rmse_m": holdout_rmse,
        "holdout_max_error_m": float(np.max(holdout_residuals)),
        "holdout_relative_rmse": holdout_relative_rmse,
        "source_geometry_condition": source_condition,
        "target_geometry_condition": target_condition,
        "rotation_determinant": float(np.linalg.det(transform[1])),
    }
    return transform, metrics, refined_inliers, holdout_residuals


def _quaternion_to_rotation(value: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("camera quaternion_wxyz must contain four finite values")
    quaternion = np.asarray([_number(item, "camera quaternion") for item in value])
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("camera quaternion must have non-zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_to_quaternion(rotation: np.ndarray[Any, np.dtype[np.float64]]) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        values = np.asarray(
            [
                0.25 * root,
                (rotation[2, 1] - rotation[1, 2]) / root,
                (rotation[0, 2] - rotation[2, 0]) / root,
                (rotation[1, 0] - rotation[0, 1]) / root,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            root = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            values = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / root,
                    0.25 * root,
                    (rotation[0, 1] + rotation[1, 0]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root,
                ]
            )
        elif index == 1:
            root = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            values = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / root,
                    (rotation[0, 1] + rotation[1, 0]) / root,
                    0.25 * root,
                    (rotation[1, 2] + rotation[2, 1]) / root,
                ]
            )
        else:
            root = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            values = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / root,
                    (rotation[0, 2] + rotation[2, 0]) / root,
                    (rotation[1, 2] + rotation[2, 1]) / root,
                    0.25 * root,
                ]
            )
    values /= np.linalg.norm(values)
    if values[0] < 0.0:
        values *= -1.0
    return [float(item) for item in values]


def _transform_rig(
    rig: dict[str, Any],
    evidence: dict[str, Any],
    transform: tuple[
        float,
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ],
    report_digest: str,
    camera_rig_reference: dict[str, Any],
    evidence_reference: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(rig)
    component = result["components"][0]
    scale, target_from_source, target_translation = transform
    for view in component["registered_views"]:
        pose = view.get("world_to_camera")
        if not isinstance(pose, dict):
            raise ValueError("registered camera view is missing world_to_camera")
        camera_from_source = _quaternion_to_rotation(pose.get("quaternion_wxyz"))
        camera_translation = _point(pose.get("translation"), "camera translation")
        camera_center_source = -(camera_from_source.T @ camera_translation)
        camera_center_target = (
            scale * (target_from_source @ camera_center_source) + target_translation
        )
        camera_from_target = camera_from_source @ target_from_source.T
        camera_translation_target = -(camera_from_target @ camera_center_target)
        pose["quaternion_wxyz"] = _rotation_to_quaternion(camera_from_target)
        pose["translation"] = [float(item) for item in camera_translation_target]
    component["coordinate_frame"] = evidence["target_frame"]
    result["method"] = {"name": _ADAPTER, "version": _VERSION}
    result["source_camera_rig"] = camera_rig_reference
    result["metric_alignment_evidence"] = evidence_reference
    result["coordinate_convention"] = {
        "frame": evidence["target_frame"],
        "pose": "world_to_camera",
        "quaternion_order": "wxyz",
        "translation_unit": "m",
    }
    result["scale_status"] = "resolved"
    result.pop("scale_reason", None)
    transform_payload = {
        "scale_m_per_source_unit": scale,
        "rotation_target_from_source": [
            [float(item) for item in row] for row in target_from_source
        ],
        "translation_target_m": [float(item) for item in target_translation],
        "source_frame": evidence["source_frame"],
        "target_frame": evidence["target_frame"],
    }
    result["metric_alignment"] = {
        "report_digest": report_digest,
        "transform": transform_payload,
    }
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validated_inputs(
    request: WorkerRequest,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[tuple[str, ...], np.ndarray[Any, Any], np.ndarray[Any, Any]],
    tuple[tuple[str, ...], np.ndarray[Any, Any], np.ndarray[Any, Any]],
]:
    rig_input = input_file(request, "camera_rig")
    evidence_input = input_file(request, "evidence")
    if rig_input.kind != "artifact" or evidence_input.kind != "artifact":
        raise ValueError("metric alignment requires immutable camera-rig and evidence artifacts")
    if (
        rig_input.media_type != "application/json"
        or evidence_input.media_type != "application/json"
    ):
        raise ValueError("metric alignment inputs must use application/json")
    rig = _load_json(declared_input_path(request, "camera_rig"), "camera rig")
    evidence = _load_json(declared_input_path(request, "evidence"), "metric evidence")
    if rig.get("schema") != _RIG_SCHEMA or rig.get("scale_status") != "unresolved":
        raise ValueError("metric alignment requires an unresolved CameraRigCalibration/1")
    components = rig.get("components")
    if (
        not isinstance(components, list)
        or len(components) != 1
        or not isinstance(components[0], dict)
    ):
        raise ValueError("each independent COLMAP component must be aligned separately")
    unknown = sorted(set(evidence) - _EVIDENCE_FIELDS)
    if unknown:
        raise ValueError(f"metric alignment evidence contains unknown fields: {unknown}")
    if evidence.get("schema") != _EVIDENCE_SCHEMA:
        raise ValueError(f"metric alignment evidence schema must be {_EVIDENCE_SCHEMA!r}")
    if evidence.get("unit") != "m":
        raise ValueError("metric alignment evidence unit must explicitly be 'm'")
    if evidence.get("camera_rig_digest") != rig_input.digest:
        raise ValueError("metric evidence is not bound to the supplied camera rig digest")
    component_id = _string(evidence.get("component_id"), "component_id")
    if component_id != _string(components[0].get("component_id"), "camera component_id"):
        raise ValueError("metric evidence component_id does not match the camera rig")
    source_frame = _string(evidence.get("source_frame"), "source_frame")
    if source_frame != _string(components[0].get("coordinate_frame"), "camera coordinate_frame"):
        raise ValueError("metric evidence source_frame does not match the camera component")
    target_frame = _string(evidence.get("target_frame"), "target_frame")
    if target_frame == source_frame:
        raise ValueError("metric target_frame must differ from the arbitrary-scale source frame")
    reference = evidence.get("metric_reference")
    if not isinstance(reference, dict):
        raise ValueError("metric evidence requires a traceable metric_reference object")
    for field in ("kind", "identifier", "source"):
        _string(reference.get(field), f"metric_reference.{field}")
    reference_value = _number(reference.get("value_m"), "metric_reference.value_m", positive=True)
    provenance = evidence.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("metric evidence requires provenance")
    for field in ("capture_id", "measurement_method", "source_uri"):
        _string(provenance.get(field), f"provenance.{field}")
    if "://" not in str(provenance["source_uri"]):
        raise ValueError("provenance.source_uri must be an absolute URI")
    marker_evidence = "marker" in str(reference["kind"]).lower() or any(
        token in str(provenance["measurement_method"]).lower() for token in ("apriltag", "aruco")
    )
    if marker_evidence:
        for field in ("marker_family", "layout", "layout_source_uri"):
            _string(reference.get(field), f"metric_reference.{field}")
        marker_size = _number(
            reference.get("marker_size_m"),
            "metric_reference.marker_size_m",
            positive=True,
        )
        if not math.isclose(marker_size, reference_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "metric_reference.value_m must equal marker_size_m for marker evidence"
            )
        if "://" not in str(reference["layout_source_uri"]):
            raise ValueError("metric_reference.layout_source_uri must be an absolute URI")
    train = _correspondences(evidence.get("train_correspondences"), "training", minimum=4)
    holdout = _correspondences(evidence.get("holdout_correspondences"), "holdout", minimum=3)
    overlap = set(train[0]) & set(holdout[0])
    if overlap:
        raise ValueError(f"training and holdout correspondence ids overlap: {sorted(overlap)}")
    train_source = {tuple(point) for point in train[1]}
    holdout_source = {tuple(point) for point in holdout[1]}
    train_target = {tuple(point) for point in train[2]}
    holdout_target = {tuple(point) for point in holdout[2]}
    if train_source & holdout_source or train_target & holdout_target:
        raise ValueError("training and holdout correspondence coordinates overlap")
    return rig, evidence, train, holdout


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    if request.operation.method.capability.value != "camera.align_metric":
        raise ValueError("metric alignment worker requires camera.align_metric capability")
    expected_outputs = {name: contract for name, (_, contract) in _OUTPUT_FILES.items()}
    actual_outputs = {item.name: item.contract for item in request.operation.outputs}
    if actual_outputs != expected_outputs:
        raise ValueError("metric alignment worker requires its exact versioned output contracts")
    options = _options(request)
    rig, evidence, train, holdout = _validated_inputs(request)
    transform, metrics, inliers, holdout_residuals = _fit(
        train[1], train[2], holdout[1], holdout[2], options
    )
    camera_input = input_file(request, "camera_rig")
    evidence_input = input_file(request, "evidence")
    transform_payload = {
        "scale_m_per_source_unit": transform[0],
        "rotation_target_from_source": [[float(item) for item in row] for row in transform[1]],
        "translation_target_m": [float(item) for item in transform[2]],
        "source_frame": evidence["source_frame"],
        "target_frame": evidence["target_frame"],
    }
    report = {
        "schema": _REPORT_SCHEMA,
        "status": "PASSED",
        "method": {"name": _ADAPTER, "version": _VERSION},
        "input": {
            "camera_rig_digest": camera_input.digest,
            "evidence_digest": evidence_input.digest,
            "component_id": evidence["component_id"],
        },
        "provenance": evidence["provenance"],
        "metric_reference": evidence["metric_reference"],
        "transform": transform_payload,
        "metrics": {
            **metrics,
            "training_correspondences": len(train[0]),
            "training_inliers": int(np.count_nonzero(inliers)),
            "holdout_correspondences": len(holdout[0]),
        },
        "gates": {
            "independent_holdout": True,
            "metric_unit_declared": True,
            "metric_reference_traceable": True,
            "nondegenerate_geometry": True,
            "ransac_inlier_support": True,
            "scale_bounds": True,
            "scale_stability": True,
            "holdout_absolute_rmse": True,
            "holdout_relative_rmse": True,
        },
        "training": {
            "ids": list(train[0]),
            "inliers": [bool(item) for item in inliers],
        },
        "holdout": {
            "ids": list(holdout[0]),
            "errors_m": [float(item) for item in holdout_residuals],
        },
    }
    report_path = output_directory / _OUTPUT_FILES["metric_alignment_report"][0]
    _write_json(report_path, report)
    camera_reference = {
        "digest": camera_input.digest,
        "size_bytes": camera_input.size_bytes,
        "media_type": camera_input.media_type,
    }
    evidence_reference = {
        "digest": evidence_input.digest,
        "size_bytes": evidence_input.size_bytes,
        "media_type": evidence_input.media_type,
    }
    resolved = _transform_rig(
        rig,
        evidence,
        transform,
        _sha256(report_path),
        camera_reference,
        evidence_reference,
    )
    rig_path = output_directory / _OUTPUT_FILES["camera_rig_calibration"][0]
    _write_json(rig_path, resolved)
    produced = {
        name: WorkerProducedFile(
            name=name,
            contract=contract,
            path=filename,
            media_type="application/json",
        )
        for name, (filename, contract) in _OUTPUT_FILES.items()
    }
    return WorkerResponse(
        outputs=tuple(produced[item.name] for item in request.operation.outputs),
        metrics={
            "scale_m_per_source_unit": metrics["scale_m_per_source_unit"],
            "scale_cv": metrics["scale_cv"],
            "training_inlier_ratio": metrics["training_inlier_ratio"],
            "holdout_rmse_m": metrics["holdout_rmse_m"],
            "holdout_relative_rmse": metrics["holdout_relative_rmse"],
        },
    )


def _self_test() -> dict[str, Any]:
    source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.2, 0.3, 0.7],
            [0.8, 0.1, 0.5],
            [0.1, 0.8, 0.4],
            [0.7, 0.7, 0.9],
        ],
        dtype=np.float64,
    )
    angle = 0.37
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = 0.42 * (source @ rotation.T) + np.asarray([0.3, -0.2, 1.1])
    target[-1] += np.asarray([0.1, -0.08, 0.12])
    holdout_source = np.asarray(
        [[0.15, 0.25, 0.2], [0.75, 0.25, 0.3], [0.2, 0.75, 0.6], [0.7, 0.65, 0.8]]
    )
    holdout_target = 0.42 * (holdout_source @ rotation.T) + np.asarray([0.3, -0.2, 1.1])
    options: dict[str, float | int] = {
        "inlier_threshold_m": 0.002,
        "max_geometry_condition": 10_000.0,
        "max_holdout_relative_rmse": 0.001,
        "max_holdout_rmse_m": 0.001,
        "max_scale_cv": 0.001,
        "maximum_scale_m_per_source_unit": 10.0,
        "minimum_inlier_ratio": 0.75,
        "minimum_inliers": 4,
        "minimum_scale_m_per_source_unit": 1e-6,
        "random_seed": 0,
        "ransac_iterations": 256,
    }
    transform, metrics, inliers, _ = _fit(source, target, holdout_source, holdout_target, options)
    if abs(transform[0] - 0.42) > 1e-10 or int(np.count_nonzero(inliers)) != 7:
        raise RuntimeError("metric alignment self-test recovered the wrong Sim(3)")
    return {
        "sim3_recovery": True,
        "outlier_rejection": True,
        "holdout_gate": True,
        "scale_error": abs(transform[0] - 0.42),
        "holdout_rmse_m": metrics["holdout_rmse_m"],
    }


def main() -> int:
    if is_probe():
        major = int(np.__version__.split(".", 1)[0])
        if major != 1:
            raise RuntimeError("metric alignment worker requires NumPy 1.x")
        return report_probe(
            capability="camera.align_metric",
            name=_ADAPTER,
            version=_VERSION,
            source_revision=_SOURCE_REVISION,
            dependencies={"adapter": _VERSION, "numpy": np.__version__},
            self_test_checks=_self_test(),
        )
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
