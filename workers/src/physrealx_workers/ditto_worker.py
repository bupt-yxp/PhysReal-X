"""Strict adapter for the official UT Austin Ditto articulation model.

The worker accepts measured, metric before/after interaction evidence only.  It
does not read URDF, simulator kinematics, asset annotations, or guessed joint
limits.  A network prediction is published only after an independent held-out
set of tracked points verifies the predicted screw transform.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physrealx.external_workers import (
    WorkerProducedFile,
    WorkerRequest,
    WorkerResponse,
    run_worker,
)
from physrealx_workers.common import (
    config,
    dependency_path,
    is_probe,
    report_probe,
    resource_json,
)

_WORKER = "ditto"
_UPSTREAM_REPOSITORY = "https://github.com/UT-Austin-RPL/Ditto"
_UPSTREAM_REVISION = "f69579f49bbfaadc665fee511ad960f5243e1c1a"
_BUNDLE_SCHEMA = "physrealx.paired-interaction-point-cloud-bundle/1"
_CHECKPOINT_SCHEMA = "physrealx.ditto-checkpoint-manifest/1"
_MODEL_SCHEMA = "physrealx.articulation-model/1"
_REPORT_SCHEMA = "physrealx.articulation-recovery-report/1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_EXPERIMENTS = {"Ditto_s2m", "Ditto_syn"}
_ALLOWED_CONFIG = {
    "maximum_axis_dispersion_deg",
    "maximum_holdout_rmse_m",
    "maximum_static_alignment_rmse_m",
    "minimum_holdout_improvement",
    "minimum_holdout_points",
    "minimum_joint_type_margin",
    "minimum_mask_points",
    "minimum_mesh_faces",
    "minimum_mesh_vertices",
    "minimum_motion_m",
    "minimum_part_extent_m",
    "minimum_points_per_state",
    "resolution",
    "seed",
}


@dataclass(frozen=True)
class _Options:
    minimum_points_per_state: int
    minimum_mask_points: int
    minimum_holdout_points: int
    minimum_motion_m: float
    maximum_static_alignment_rmse_m: float
    maximum_holdout_rmse_m: float
    minimum_holdout_improvement: float
    minimum_joint_type_margin: float
    maximum_axis_dispersion_deg: float
    minimum_part_extent_m: float
    minimum_mesh_vertices: int
    minimum_mesh_faces: int
    resolution: int
    seed: int


@dataclass(frozen=True)
class _Evidence:
    manifest: dict[str, Any]
    before_points: Any
    after_points: Any
    before_mask: Any
    after_mask: Any
    holdout_before: Any
    holdout_after: Any
    holdout_moving_mask: Any
    checkpoint_path: Path
    checkpoint_digest: str


@dataclass(frozen=True)
class _Prediction:
    joint_type: str
    prismatic_probability: float
    axis: Any
    pivot_m: Any
    motion: float
    axis_candidates: Any
    static_vertices_m: Any
    static_faces: Any
    moving_vertices_m: Any
    moving_faces: Any
    normalization_center_m: Any
    normalization_scale_m: float


@dataclass(frozen=True)
class _Recovery:
    model: dict[str, Any]
    report: dict[str, Any]
    metrics: dict[str, float]


def _keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _integer(value: Any, label: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return int(value)


def _options(raw: dict[str, Any]) -> _Options:
    unknown = sorted(set(raw) - _ALLOWED_CONFIG)
    if unknown:
        raise ValueError(f"unknown Ditto adapter options: {unknown}")
    return _Options(
        minimum_points_per_state=_integer(
            raw.get("minimum_points_per_state", 1_024),
            "minimum_points_per_state",
            minimum=512,
        ),
        minimum_mask_points=_integer(
            raw.get("minimum_mask_points", 1_024), "minimum_mask_points", minimum=64
        ),
        minimum_holdout_points=_integer(
            raw.get("minimum_holdout_points", 64), "minimum_holdout_points", minimum=16
        ),
        minimum_motion_m=_number(
            raw.get("minimum_motion_m", 0.005), "minimum_motion_m", minimum=1.0e-5
        ),
        maximum_static_alignment_rmse_m=_number(
            raw.get("maximum_static_alignment_rmse_m", 0.01),
            "maximum_static_alignment_rmse_m",
            minimum=1.0e-6,
        ),
        maximum_holdout_rmse_m=_number(
            raw.get("maximum_holdout_rmse_m", 0.02),
            "maximum_holdout_rmse_m",
            minimum=1.0e-6,
        ),
        minimum_holdout_improvement=_number(
            raw.get("minimum_holdout_improvement", 0.35),
            "minimum_holdout_improvement",
            minimum=0.0,
            maximum=0.99,
        ),
        minimum_joint_type_margin=_number(
            raw.get("minimum_joint_type_margin", 0.10),
            "minimum_joint_type_margin",
            minimum=0.0,
            maximum=0.49,
        ),
        maximum_axis_dispersion_deg=_number(
            raw.get("maximum_axis_dispersion_deg", 20.0),
            "maximum_axis_dispersion_deg",
            minimum=0.1,
            maximum=90.0,
        ),
        minimum_part_extent_m=_number(
            raw.get("minimum_part_extent_m", 0.01),
            "minimum_part_extent_m",
            minimum=1.0e-5,
        ),
        minimum_mesh_vertices=_integer(
            raw.get("minimum_mesh_vertices", 4), "minimum_mesh_vertices", minimum=4
        ),
        minimum_mesh_faces=_integer(
            raw.get("minimum_mesh_faces", 4), "minimum_mesh_faces", minimum=4
        ),
        resolution=_integer(raw.get("resolution", 32), "resolution", minimum=16, maximum=128),
        seed=_integer(raw.get("seed", 0), "seed", minimum=0, maximum=2**31 - 1),
    )


def _artifact(value: Any, label: str, *, media_types: set[str]) -> dict[str, Any]:
    artifact = _object(value, label)
    _keys(artifact, required={"digest", "size_bytes", "media_type"}, label=label)
    digest = artifact["digest"]
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError(f"{label}.digest must be a SHA-256 digest")
    _integer(artifact["size_bytes"], f"{label}.size_bytes", minimum=1)
    if artifact["media_type"] not in media_types:
        raise ValueError(
            f"{label}.media_type must be one of {sorted(media_types)}, "
            f"not {artifact['media_type']!r}"
        )
    return artifact


def _matrix(value: Any, label: str, rows: int, columns: int) -> list[list[float]]:
    raw_rows = _array(value, label)
    if len(raw_rows) != rows:
        raise ValueError(f"{label} must have {rows} rows")
    result: list[list[float]] = []
    for row_index, raw_row in enumerate(raw_rows):
        row = _array(raw_row, f"{label}[{row_index}]")
        if len(row) != columns:
            raise ValueError(f"{label}[{row_index}] must have {columns} values")
        result.append(
            [_number(item, f"{label}[{row_index}][{index}]") for index, item in enumerate(row)]
        )
    return result


def _validate_transform(value: Any, label: str) -> None:
    import numpy as np

    transform = np.asarray(_matrix(value, label, 4, 4), dtype=np.float64)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError(f"{label} must be a rigid homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-5):
        raise ValueError(f"{label} rotation must be right-handed")


def _load_npy(
    request: WorkerRequest,
    owner: str,
    artifact: dict[str, Any],
    label: str,
) -> Any:
    import numpy as np

    path = dependency_path(request, owner, str(artifact["digest"]))
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a valid non-pickle NPY array: {error}") from error
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{label} must contain one NPY array")
    return value


def _point_cloud(value: Any, label: str, minimum: int) -> Any:
    import numpy as np

    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < minimum:
        raise ValueError(f"{label} must have shape [N,3] with N >= {minimum}")
    if points.dtype.kind not in {"f"}:
        raise ValueError(f"{label} must use a floating-point dtype")
    points = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{label} contains NaN or infinity")
    if float(np.max(np.ptp(points, axis=0))) <= 1.0e-6:
        raise ValueError(f"{label} is geometrically degenerate")
    return points


def _mask(
    value: Any,
    label: str,
    count: int,
    minimum_true: int,
    *,
    require_false: bool,
) -> Any:
    import numpy as np

    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape[0] != count:
        raise ValueError(f"{label} must have shape [{count}]")
    if raw.dtype.kind not in {"b", "u", "i"}:
        raise ValueError(f"{label} must be a boolean or integer array")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{label} must contain only 0 and 1")
    result = raw.astype(bool, copy=False)
    moving = int(result.sum())
    static = int((~result).sum())
    if moving < minimum_true or (require_false and static < minimum_true):
        required_populations = "true and false" if require_false else "true"
        raise ValueError(
            f"{label} must retain at least {minimum_true} {required_populations} points"
        )
    return result


def _validate_checkpoint(value: Any) -> dict[str, Any]:
    checkpoint = _object(value, "checkpoint")
    _keys(
        checkpoint,
        required={
            "schema",
            "artifact",
            "upstream_repository",
            "upstream_revision",
            "experiment",
            "license_spdx",
            "architecture",
        },
        label="checkpoint",
    )
    if checkpoint["schema"] != _CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint.schema must be {_CHECKPOINT_SCHEMA!r}")
    upstream_repository = str(checkpoint["upstream_repository"])
    if upstream_repository.endswith(".git"):
        upstream_repository = upstream_repository[:-4]
    upstream_repository = upstream_repository.rstrip("/")
    if upstream_repository != _UPSTREAM_REPOSITORY:
        raise ValueError("checkpoint must name the official UT-Austin-RPL/Ditto repository")
    revision = checkpoint["upstream_revision"]
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("checkpoint.upstream_revision must be a full Git commit SHA")
    if revision != _UPSTREAM_REVISION:
        raise ValueError(f"this worker is pinned to official Ditto revision {_UPSTREAM_REVISION}")
    if checkpoint["experiment"] not in _ALLOWED_EXPERIMENTS:
        raise ValueError(f"checkpoint.experiment must be one of {sorted(_ALLOWED_EXPERIMENTS)}")
    if checkpoint["license_spdx"] != "MIT":
        raise ValueError("checkpoint.license_spdx must record the upstream MIT license")
    architecture = _object(checkpoint["architecture"], "checkpoint.architecture")
    _keys(
        architecture,
        required={"encoder", "geometry_backbone", "joint_decoder"},
        label="checkpoint.architecture",
    )
    if architecture != {
        "encoder": "PointNet++",
        "geometry_backbone": "ConvONets",
        "joint_decoder": "dense_screw_parameter_voting",
    }:
        raise ValueError("checkpoint architecture does not describe official Ditto")
    _artifact(
        checkpoint["artifact"],
        "checkpoint.artifact",
        media_types={"application/x-pytorch", "application/octet-stream"},
    )
    return checkpoint


def _validate_manifest(
    request: WorkerRequest,
    options: _Options,
) -> _Evidence:
    import numpy as np

    document = resource_json(request, "pair")
    _keys(
        document,
        required={
            "schema",
            "resource_id",
            "version",
            "source_kind",
            "coordinate_frame",
            "alignment",
            "before",
            "after",
            "cameras",
            "interaction",
            "checkpoint",
        },
        label="PairedInteractionPointCloudBundle",
    )
    if document["schema"] != _BUNDLE_SCHEMA:
        raise ValueError(f"bundle.schema must be {_BUNDLE_SCHEMA!r}")
    _string(document["resource_id"], "bundle.resource_id")
    _string(document["version"], "bundle.version")
    if document["source_kind"] != "measured_interaction":
        raise ValueError("bundle.source_kind must be 'measured_interaction', never simulation")

    coordinate_frame = _object(document["coordinate_frame"], "coordinate_frame")
    _keys(
        coordinate_frame,
        required={"frame_id", "units", "handedness", "up_axis"},
        label="coordinate_frame",
    )
    _string(coordinate_frame["frame_id"], "coordinate_frame.frame_id")
    if coordinate_frame["units"] != "m":
        raise ValueError("Ditto evidence must already be expressed in meters")
    if coordinate_frame["handedness"] != "right":
        raise ValueError("Ditto evidence must use a right-handed frame")
    if coordinate_frame["up_axis"] not in {"Y", "Z"}:
        raise ValueError("coordinate_frame.up_axis must be Y or Z")

    alignment = _object(document["alignment"], "alignment")
    _keys(
        alignment,
        required={"status", "convention", "method", "evidence"},
        label="alignment",
    )
    if alignment["status"] != "verified":
        raise ValueError("before/after object-frame alignment must be independently verified")
    if alignment["convention"] != "points_expressed_in_same_object_frame":
        raise ValueError("alignment convention must state one shared object frame")
    _string(alignment["method"], "alignment.method")
    alignment_artifact = _artifact(
        alignment["evidence"], "alignment.evidence", media_types={"application/json"}
    )
    dependency_path(request, "pair", str(alignment_artifact["digest"]))

    states: dict[str, tuple[Any, Any]] = {}
    for state_name in ("before", "after"):
        state = _object(document[state_name], state_name)
        _keys(state, required={"points", "object_mask", "capture_id"}, label=state_name)
        _string(state["capture_id"], f"{state_name}.capture_id")
        points_ref = _artifact(
            state["points"], f"{state_name}.points", media_types={"application/x-npy"}
        )
        mask_ref = _artifact(
            state["object_mask"],
            f"{state_name}.object_mask",
            media_types={"application/x-npy"},
        )
        points = _point_cloud(
            _load_npy(request, "pair", points_ref, f"{state_name}.points"),
            f"{state_name}.points",
            options.minimum_points_per_state,
        )
        mask = _mask(
            _load_npy(request, "pair", mask_ref, f"{state_name}.object_mask"),
            f"{state_name}.object_mask",
            len(points),
            max(options.minimum_mask_points, options.minimum_points_per_state),
            require_false=False,
        )
        states[state_name] = (points, mask)
    if document["before"]["capture_id"] == document["after"]["capture_id"]:
        raise ValueError("before and after must originate from distinct measured captures")

    cameras = _array(document["cameras"], "cameras")
    if not cameras:
        raise ValueError("at least one calibrated camera is required")
    camera_ids: set[str] = set()
    for index, raw_camera in enumerate(cameras):
        label = f"cameras[{index}]"
        camera = _object(raw_camera, label)
        _keys(
            camera,
            required={
                "camera_id",
                "intrinsics",
                "object_from_camera",
                "calibration_evidence",
            },
            label=label,
        )
        camera_id = _string(camera["camera_id"], f"{label}.camera_id")
        if camera_id in camera_ids:
            raise ValueError("camera IDs must be unique")
        camera_ids.add(camera_id)
        intrinsics = _matrix(camera["intrinsics"], f"{label}.intrinsics", 3, 3)
        if intrinsics[0][0] <= 0.0 or intrinsics[1][1] <= 0.0:
            raise ValueError(f"{label} focal lengths must be positive")
        if not math.isclose(intrinsics[2][2], 1.0, abs_tol=1.0e-9):
            raise ValueError(f"{label}.intrinsics[2][2] must be 1")
        _validate_transform(camera["object_from_camera"], f"{label}.object_from_camera")
        calibration = _artifact(
            camera["calibration_evidence"],
            f"{label}.calibration_evidence",
            media_types={"application/json"},
        )
        dependency_path(request, "pair", str(calibration["digest"]))

    interaction = _object(document["interaction"], "interaction")
    _keys(
        interaction,
        required={
            "kind",
            "timestamp_order",
            "evidence",
            "holdout_correspondences",
        },
        label="interaction",
    )
    if interaction["kind"] not in {"robot_manipulation", "instrumented_manual_motion"}:
        raise ValueError("interaction.kind must identify a measured physical intervention")
    if interaction["timestamp_order"] != "before_then_after":
        raise ValueError("interaction.timestamp_order must be 'before_then_after'")
    interaction_evidence = _artifact(
        interaction["evidence"], "interaction.evidence", media_types={"application/json"}
    )
    dependency_path(request, "pair", str(interaction_evidence["digest"]))
    holdout = _object(interaction["holdout_correspondences"], "holdout_correspondences")
    _keys(
        holdout,
        required={"before_points", "after_points", "moving_mask", "source"},
        label="holdout_correspondences",
    )
    if holdout["source"] not in {"tracked_rgbd_features", "instrumented_fiducials"}:
        raise ValueError("holdout correspondences require tracked RGB-D features or fiducials")
    holdout_refs = {
        name: _artifact(
            holdout[name], f"holdout_correspondences.{name}", media_types={"application/x-npy"}
        )
        for name in ("before_points", "after_points", "moving_mask")
    }
    holdout_before = _point_cloud(
        _load_npy(request, "pair", holdout_refs["before_points"], "holdout.before_points"),
        "holdout.before_points",
        options.minimum_holdout_points * 2,
    )
    holdout_after = _point_cloud(
        _load_npy(request, "pair", holdout_refs["after_points"], "holdout.after_points"),
        "holdout.after_points",
        options.minimum_holdout_points * 2,
    )
    if holdout_before.shape != holdout_after.shape:
        raise ValueError("holdout before/after point arrays must be one-to-one correspondences")
    holdout_mask = _mask(
        _load_npy(request, "pair", holdout_refs["moving_mask"], "holdout.moving_mask"),
        "holdout.moving_mask",
        len(holdout_before),
        options.minimum_holdout_points,
        require_false=True,
    )
    static_residual = holdout_after[~holdout_mask] - holdout_before[~holdout_mask]
    static_alignment_rmse = float(np.sqrt(np.mean(np.sum(static_residual**2, axis=1))))
    if static_alignment_rmse > options.maximum_static_alignment_rmse_m:
        raise ValueError(
            "held-out static correspondences contradict the declared object-frame alignment: "
            f"RMSE={static_alignment_rmse:.6g} m"
        )
    motion = holdout_after[holdout_mask] - holdout_before[holdout_mask]
    motion_rms = float(np.sqrt(np.mean(np.sum(motion**2, axis=1))))
    if motion_rms < options.minimum_motion_m:
        raise ValueError(
            f"measured moving-part motion {motion_rms:.6g} m is below the excitation gate"
        )

    checkpoint = _validate_checkpoint(document["checkpoint"])
    checkpoint_artifact = checkpoint["artifact"]
    checkpoint_path = dependency_path(request, "pair", str(checkpoint_artifact["digest"]))
    return _Evidence(
        manifest=document,
        before_points=states["before"][0],
        after_points=states["after"][0],
        before_mask=states["before"][1],
        after_mask=states["after"][1],
        holdout_before=holdout_before,
        holdout_after=holdout_after,
        holdout_moving_mask=holdout_mask,
        checkpoint_path=checkpoint_path,
        checkpoint_digest=str(checkpoint_artifact["digest"]),
    )


def _normalize(vector: Any, label: str) -> Any:
    import numpy as np

    value = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} contains NaN or infinity")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-9:
        raise ValueError(f"{label} has zero length")
    return value / norm


def _bridge_environment(python: Path) -> dict[str, str]:
    prefix = python.parent.parent
    environment = {
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "LD_LIBRARY_PATH": str(prefix / "lib"),
    }
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _run_official_ditto(evidence: _Evidence, options: _Options) -> _Prediction:
    import numpy as np

    python_value = os.environ.get("PHYSREALX_DITTO_PYTHON")
    repository_value = os.environ.get("PHYSREALX_DITTO_REPOSITORY")
    installed_checkpoint_value = os.environ.get("PHYSREALX_DITTO_CHECKPOINT")
    installed_checkpoint_digest = os.environ.get("PHYSREALX_DITTO_CHECKPOINT_DIGEST")
    installed_experiment = os.environ.get("PHYSREALX_DITTO_EXPERIMENT")
    if not all(
        (
            python_value,
            repository_value,
            installed_checkpoint_value,
            installed_checkpoint_digest,
            installed_experiment,
        )
    ):
        raise RuntimeError(
            "Ditto deployment must provide its attested Python, repository, checkpoint, "
            "checkpoint digest and experiment binding"
        )
    checkpoint_manifest = evidence.manifest["checkpoint"]
    if installed_checkpoint_digest != evidence.checkpoint_digest:
        raise RuntimeError(
            "request checkpoint digest does not match the checkpoint bound at installation"
        )
    if installed_experiment != checkpoint_manifest["experiment"]:
        raise RuntimeError(
            "request checkpoint experiment does not match the experiment bound at installation"
        )
    python = Path(python_value).resolve(strict=True)
    repository = Path(repository_value).resolve(strict=True)
    installed_checkpoint = Path(installed_checkpoint_value).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("PHYSREALX_DITTO_PYTHON is not an executable file")
    if repository.is_symlink() or not repository.is_dir():
        raise RuntimeError("PHYSREALX_DITTO_REPOSITORY is not a real directory")
    if not installed_checkpoint.is_file():
        raise RuntimeError("PHYSREALX_DITTO_CHECKPOINT is not a regular file")
    bridge = Path(__file__).with_name("ditto_inference_bridge.py").resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="ditto-bridge-", dir=Path.cwd()) as temporary:
        directory = Path(temporary)
        input_path = directory / "input.npz"
        output_path = directory / "output.npz"
        request_path = directory / "request.json"
        np.savez_compressed(
            input_path,
            before=np.asarray(evidence.before_points[evidence.before_mask], dtype=np.float64),
            after=np.asarray(evidence.after_points[evidence.after_mask], dtype=np.float64),
        )
        request_document = {
            "repository": str(repository),
            "revision": checkpoint_manifest["upstream_revision"],
            "checkpoint": str(evidence.checkpoint_path),
            "checkpoint_digest": evidence.checkpoint_digest,
            "experiment": checkpoint_manifest["experiment"],
            "input_npz": str(input_path),
            "output_npz": str(output_path),
            "resolution": options.resolution,
            "seed": options.seed,
        }
        request_path.write_text(
            json.dumps(request_document, allow_nan=False, sort_keys=True), encoding="utf-8"
        )
        result = subprocess.run(
            [str(python), str(bridge), "--request", str(request_path)],
            cwd=repository,
            env=_bridge_environment(python),
            check=False,
            capture_output=True,
            text=True,
            timeout=3_600,
        )
        if result.returncode != 0:
            message = result.stdout.strip() or result.stderr.strip() or "unknown bridge failure"
            raise RuntimeError(f"official Ditto inference bridge failed: {message[:4096]}")
        if not output_path.is_file() or output_path.is_symlink():
            raise RuntimeError("official Ditto inference bridge did not produce its NPZ result")
        with np.load(output_path, allow_pickle=False) as archive:
            expected = {
                "joint_type",
                "prismatic_probability",
                "axis",
                "pivot_m",
                "motion",
                "axis_candidates",
                "static_vertices_m",
                "static_faces",
                "moving_vertices_m",
                "moving_faces",
                "normalization_center_m",
                "normalization_scale_m",
            }
            if set(archive.files) != expected:
                raise RuntimeError("official Ditto bridge result uses an unexpected schema")
            joint_code = int(np.asarray(archive["joint_type"]).reshape(-1)[0])
            if joint_code not in {0, 1}:
                raise RuntimeError("official Ditto bridge returned an invalid joint type")
            return _Prediction(
                joint_type="prismatic" if joint_code == 1 else "revolute",
                prismatic_probability=float(
                    np.asarray(archive["prismatic_probability"]).reshape(-1)[0]
                ),
                axis=np.asarray(archive["axis"], dtype=np.float64),
                pivot_m=np.asarray(archive["pivot_m"], dtype=np.float64),
                motion=float(np.asarray(archive["motion"]).reshape(-1)[0]),
                axis_candidates=np.asarray(archive["axis_candidates"], dtype=np.float64),
                static_vertices_m=np.asarray(archive["static_vertices_m"], dtype=np.float64),
                static_faces=np.asarray(archive["static_faces"], dtype=np.int64),
                moving_vertices_m=np.asarray(archive["moving_vertices_m"], dtype=np.float64),
                moving_faces=np.asarray(archive["moving_faces"], dtype=np.int64),
                normalization_center_m=np.asarray(
                    archive["normalization_center_m"], dtype=np.float64
                ),
                normalization_scale_m=float(
                    np.asarray(archive["normalization_scale_m"]).reshape(-1)[0]
                ),
            )


def _rotation(axis: Any, angle: float) -> Any:
    import numpy as np

    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _transform(points: Any, prediction: _Prediction) -> Any:
    import numpy as np

    if prediction.joint_type == "prismatic":
        return points + prediction.axis[None, :] * prediction.motion
    rotation = _rotation(prediction.axis, prediction.motion)
    centered = points - prediction.pivot_m[None, :]
    return centered @ np.asarray(rotation).T + prediction.pivot_m[None, :]


def _canonical_screw(axis: Any, motion: float) -> tuple[Any, float]:
    value = _normalize(axis, "joint axis")
    first = next((component for component in value if abs(float(component)) > 1.0e-12), 1.0)
    if first < 0.0:
        return -value, -motion
    return value, motion


def _mesh_gate(vertices: Any, faces: Any, label: str, options: _Options) -> dict[str, Any]:
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Ditto {label} vertices are malformed")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Ditto {label} faces are malformed")
    if len(vertices) < options.minimum_mesh_vertices or len(faces) < options.minimum_mesh_faces:
        raise ValueError(f"Ditto {label} mesh is too small to represent a physical part")
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"Ditto {label} mesh contains NaN or infinity")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError(f"Ditto {label} mesh contains invalid face indices")
    extent = np.ptp(vertices, axis=0)
    if float(np.max(extent)) < options.minimum_part_extent_m:
        raise ValueError(f"Ditto {label} part is geometrically degenerate")
    centered = vertices - np.mean(vertices, axis=0)
    if int(np.linalg.matrix_rank(centered, tol=1.0e-8)) < 2:
        raise ValueError(f"Ditto {label} part is rank deficient")
    triangles = vertices[faces]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    nondegenerate_faces = int(np.count_nonzero(double_areas > 1.0e-12))
    if nondegenerate_faces < options.minimum_mesh_faces:
        raise ValueError(f"Ditto {label} mesh has too few non-degenerate triangles")
    return {
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "nondegenerate_face_count": nondegenerate_faces,
        "extent_m": [float(item) for item in extent],
    }


def _recover(evidence: _Evidence, prediction: _Prediction, options: _Options) -> _Recovery:
    import numpy as np

    if prediction.joint_type not in {"revolute", "prismatic"}:
        raise ValueError("Ditto returned an unsupported joint type")
    probability = _number(
        prediction.prismatic_probability,
        "prismatic_probability",
        minimum=0.0,
        maximum=1.0,
    )
    type_margin = abs(probability - 0.5)
    if type_margin < options.minimum_joint_type_margin:
        raise ValueError(f"Ditto joint-type prediction is ambiguous: margin={type_margin:.6g}")
    expected_type = "prismatic" if probability >= 0.5 else "revolute"
    if prediction.joint_type != expected_type:
        raise ValueError("Ditto joint type and probability are inconsistent")
    axis, motion = _canonical_screw(prediction.axis, prediction.motion)
    normalized_candidates = np.asarray(
        [_normalize(item, "axis candidate") for item in prediction.axis_candidates],
        dtype=np.float64,
    )
    if len(normalized_candidates) < 16:
        raise ValueError("Ditto emitted too few dense joint votes")
    angular_errors = np.degrees(np.arccos(np.clip(np.abs(normalized_candidates @ axis), 0.0, 1.0)))
    axis_dispersion = float(np.percentile(angular_errors, 90.0))
    if axis_dispersion > options.maximum_axis_dispersion_deg:
        raise ValueError(f"Ditto axis votes are ambiguous: p90={axis_dispersion:.6g} degrees")
    pivot = np.asarray(prediction.pivot_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pivot)) or not math.isfinite(motion):
        raise ValueError("Ditto screw parameters contain NaN or infinity")
    if prediction.joint_type == "revolute" and abs(motion) < math.radians(1.0):
        raise ValueError("Ditto predicted less than one degree of revolute excitation")
    if prediction.joint_type == "prismatic" and abs(motion) < options.minimum_motion_m:
        raise ValueError("Ditto predicted prismatic motion below the excitation gate")

    canonical = _Prediction(
        joint_type=prediction.joint_type,
        prismatic_probability=probability,
        axis=axis,
        pivot_m=pivot,
        motion=motion,
        axis_candidates=normalized_candidates,
        static_vertices_m=prediction.static_vertices_m,
        static_faces=prediction.static_faces,
        moving_vertices_m=prediction.moving_vertices_m,
        moving_faces=prediction.moving_faces,
        normalization_center_m=prediction.normalization_center_m,
        normalization_scale_m=prediction.normalization_scale_m,
    )
    moving_mask = evidence.holdout_moving_mask
    predicted_after = _transform(evidence.holdout_before[moving_mask], canonical)
    observed_after = evidence.holdout_after[moving_mask]
    residual = predicted_after - observed_after
    holdout_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    identity_residual = evidence.holdout_before[moving_mask] - observed_after
    identity_rmse = float(np.sqrt(np.mean(np.sum(identity_residual**2, axis=1))))
    holdout_improvement = (identity_rmse - holdout_rmse) / max(identity_rmse, 1.0e-12)
    if holdout_rmse > options.maximum_holdout_rmse_m:
        raise ValueError(f"Ditto screw transform fails held-out points: RMSE={holdout_rmse:.6g} m")
    if holdout_improvement < options.minimum_holdout_improvement:
        raise ValueError(
            "Ditto screw transform does not improve identity on held-out moving points: "
            f"improvement={holdout_improvement:.6g}"
        )
    static_summary = _mesh_gate(
        prediction.static_vertices_m, prediction.static_faces, "static", options
    )
    moving_summary = _mesh_gate(
        prediction.moving_vertices_m, prediction.moving_faces, "moving", options
    )
    checkpoint = evidence.manifest["checkpoint"]
    alignment_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (evidence.holdout_after[~moving_mask] - evidence.holdout_before[~moving_mask])
                    ** 2,
                    axis=1,
                )
            )
        )
    )
    model = {
        "schema": _MODEL_SCHEMA,
        "status": "IDENTIFIED",
        "method": "ditto",
        "upstream": {
            "repository": _UPSTREAM_REPOSITORY,
            "revision": checkpoint["upstream_revision"],
            "checkpoint_digest": evidence.checkpoint_digest,
            "experiment": checkpoint["experiment"],
        },
        "coordinate_frame": evidence.manifest["coordinate_frame"],
        "links": [
            {"name": "base", "role": "static", "geometry_summary": static_summary},
            {"name": "moving", "role": "moving", "geometry_summary": moving_summary},
        ],
        "joints": [
            {
                "name": "joint_0",
                "parent": "base",
                "child": "moving",
                "type": prediction.joint_type,
                "axis": [float(item) for item in axis],
                "pivot_m": [float(item) for item in pivot],
                "observed_displacement": {
                    "value": float(motion),
                    "unit": "m" if prediction.joint_type == "prismatic" else "rad",
                },
                "limits": {
                    "status": "unresolved",
                    "reason": "one interaction pair cannot establish mechanical end stops",
                },
            }
        ],
        "evidence_state": "identified_from_measured_interaction",
        "source_bundle": {
            "resource_id": evidence.manifest["resource_id"],
            "version": evidence.manifest["version"],
        },
    }
    report = {
        "schema": _REPORT_SCHEMA,
        "status": "PASSED",
        "method": "ditto",
        "gates": {
            "shared_metric_frame": "passed",
            "measured_interaction": "passed",
            "part_non_degeneracy": "passed",
            "joint_type_margin": "passed",
            "axis_vote_dispersion": "passed",
            "held_out_screw_transform": "passed",
            "joint_limits": "unresolved",
        },
        "diagnostics": {
            "before_points": len(evidence.before_points),
            "after_points": len(evidence.after_points),
            "holdout_moving_points": int(moving_mask.sum()),
            "holdout_static_points": int((~moving_mask).sum()),
            "static_alignment_rmse_m": alignment_rmse,
            "identity_holdout_rmse_m": identity_rmse,
            "screw_holdout_rmse_m": holdout_rmse,
            "holdout_improvement_fraction": holdout_improvement,
            "joint_type_prismatic_probability": probability,
            "joint_type_margin": type_margin,
            "axis_vote_p90_deg": axis_dispersion,
            "normalization_center_m": [float(item) for item in prediction.normalization_center_m],
            "normalization_scale_m": float(prediction.normalization_scale_m),
        },
        "provenance": {
            "alignment_evidence_digest": evidence.manifest["alignment"]["evidence"]["digest"],
            "interaction_evidence_digest": evidence.manifest["interaction"]["evidence"]["digest"],
            "checkpoint_digest": evidence.checkpoint_digest,
            "capture_ids": [
                evidence.manifest["before"]["capture_id"],
                evidence.manifest["after"]["capture_id"],
            ],
        },
    }
    metrics = {
        "identified": 1.0,
        "before_points": float(len(evidence.before_points)),
        "after_points": float(len(evidence.after_points)),
        "holdout_points": float(moving_mask.sum()),
        "static_alignment_rmse_m": alignment_rmse,
        "holdout_rmse_m": holdout_rmse,
        "holdout_identity_rmse_m": identity_rmse,
        "holdout_improvement_fraction": float(holdout_improvement),
        "joint_type_margin": type_margin,
        "axis_vote_p90_deg": axis_dispersion,
    }
    if any(not math.isfinite(item) for item in metrics.values()):
        raise ValueError("Ditto validation produced a non-finite metric")
    return _Recovery(model=model, report=report, metrics=metrics)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    options = _options(config(request))
    evidence = _validate_manifest(request, options)
    prediction = _run_official_ditto(evidence, options)
    result = _recover(evidence, prediction, options)
    model_path = output_directory / "articulation_model.json"
    report_path = output_directory / "articulation_recovery_report.json"
    _write_json(model_path, result.model)
    _write_json(report_path, result.report)
    available = {
        "articulation_model": WorkerProducedFile(
            name="articulation_model",
            contract="ArticulationModel/1",
            path=model_path.name,
            media_type="application/json",
        ),
        "articulation_recovery_report": WorkerProducedFile(
            name="articulation_recovery_report",
            contract="ArticulationRecoveryReport/1",
            path=report_path.name,
            media_type="application/json",
        ),
    }
    try:
        outputs = tuple(available[item.name] for item in request.operation.outputs)
    except KeyError as error:
        raise ValueError(f"unsupported Ditto output: {error.args[0]}") from error
    return WorkerResponse(outputs=outputs, metrics=result.metrics)


def _probe() -> int:
    checkpoint_digest: str | None = None
    try:
        python = Path(os.environ["PHYSREALX_DITTO_PYTHON"]).resolve(strict=True)
        repository = Path(os.environ["PHYSREALX_DITTO_REPOSITORY"]).resolve(strict=True)
        checkpoint = Path(os.environ["PHYSREALX_DITTO_CHECKPOINT"]).resolve(strict=True)
        digest = os.environ["PHYSREALX_DITTO_CHECKPOINT_DIGEST"]
        if _DIGEST.fullmatch(digest) is None:
            raise RuntimeError("PHYSREALX_DITTO_CHECKPOINT_DIGEST must be sha256:<hex>")
        checkpoint_digest = digest
        experiment = os.environ.get("PHYSREALX_DITTO_EXPERIMENT", "Ditto_s2m")
        if experiment not in _ALLOWED_EXPERIMENTS:
            raise RuntimeError(f"unsupported Ditto experiment: {experiment!r}")
        bridge = Path(__file__).with_name("ditto_inference_bridge.py").resolve(strict=True)
        result = subprocess.run(
            [
                str(python),
                str(bridge),
                "--probe",
                "--repository",
                str(repository),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-digest",
                digest,
                "--experiment",
                experiment,
            ],
            cwd=repository,
            env=_bridge_environment(python),
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            message = result.stdout.strip() or result.stderr.strip() or "unknown bridge failure"
            raise RuntimeError(f"Ditto readiness bridge failed: {message[:4096]}")
        bridge_result = json.loads(result.stdout)
        if not isinstance(bridge_result, dict) or bridge_result.get("ready") is not True:
            raise RuntimeError("Ditto readiness bridge did not attest readiness")
        versions = bridge_result.get("versions")
        if not isinstance(versions, dict):
            raise RuntimeError("Ditto readiness bridge omitted version evidence")
        dependencies = {
            key: str(versions[key]) for key in ("python", "torch", "cuda") if key in versions
        }
        return report_probe(
            capability="articulation.recover",
            name=_WORKER,
            version="1.0.0",
            source_revision=_UPSTREAM_REVISION,
            dependencies=dependencies,
            checkpoint_digest=digest,
            checkpoint_source_revision=_UPSTREAM_REVISION,
            self_test_checks={key: str(value) for key, value in versions.items()},
        )
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        return report_probe(
            capability="articulation.recover",
            name=_WORKER,
            version="1.0.0",
            source_revision=_UPSTREAM_REVISION,
            dependencies={},
            checkpoint_digest=checkpoint_digest,
            checkpoint_source_revision=(
                _UPSTREAM_REVISION if checkpoint_digest is not None else None
            ),
            self_test_checks={"runtime_attestation": False},
            ready=False,
            reason=str(error) or type(error).__name__,
        )


def main() -> int:
    if is_probe():
        return _probe()
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
