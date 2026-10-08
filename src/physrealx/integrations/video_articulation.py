"""Video-constrained articulation and local surface refinement.

The module-one reconstruction is immutable evidence.  This service derives a
new physical-asset candidate from calibrated RGB-D video, rather than editing
the reconstruction in place or asking collision proxies to hide a visual
geometry defect.
"""

from __future__ import annotations

import io
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_array, read_json
from physrealx.integrations._surface_query import SurfaceQuery, create_surface_query
from physrealx.integrations.collision import load_meshes
from physrealx.reconstruction import joint_transform


class VideoArticulationFrame(DomainModel):
    """One calibrated video observation in the module-one reference frame."""

    frame_id: str = Field(min_length=1)
    partition: Literal["fit", "check"]
    rgb: ArtifactRef
    depth: ArtifactRef
    # Integer PNG: zero is background/hand/unknown, positive IDs map exactly to links.
    link_mask: ArtifactRef | None = None
    object_mask: ArtifactRef | None = None
    depth_scale_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy; aligned pixels
    camera_from_reference: tuple[tuple[float, ...], ...]  # OpenCV world-to-camera SE(3)
    positions: dict[str, float]

    @model_validator(mode="after")
    def one_mask_kind(self) -> VideoArticulationFrame:
        if (self.link_mask is None) == (self.object_mask is None):
            raise ValueError("frame requires exactly one link mask or object mask")
        return self


class VideoArticulationEvidence(DomainModel):
    """Explicit evidence contract; raw adjacent files never become calibration truth."""

    schema_id: Literal["physrealx.video-articulation-evidence/1"] = Field(
        default="physrealx.video-articulation-evidence/1", alias="schema"
    )
    reference: ArtifactRef
    registration: ArtifactRef
    source_geometry: ArtifactRef
    capture_id: str = Field(min_length=1)
    depth_scale_m: float = Field(gt=0, allow_inf_nan=False)
    link_ids: dict[str, int]
    frames: tuple[VideoArticulationFrame, ...] = Field(min_length=4)


class VideoArticulationRefinementConfig(DomainModel):
    """Bounded evidence, joint and shape trust regions."""

    resume_checkpoint: ArtifactRef | None = None
    initialize_from_checkpoint: ArtifactRef | None = None
    max_surface_displacement_m: float = Field(default=0.01, gt=0, le=0.01)
    surface_correspondence_distance_m: float = Field(default=0.02, gt=0, le=0.1)
    joint_correspondence_distance_m: float = Field(default=0.03, gt=0, le=0.1)
    points_per_frame: int = Field(default=2000, ge=100, le=20000)
    render_max_dimension: int = Field(default=640, ge=160, le=1920)
    surface_smoothness: float = Field(default=0.2, gt=0)
    state_reconstruction_steps: int = Field(default=3, ge=1, le=20)
    motion_consistency_weight: float = Field(default=0.5, gt=0)
    max_check_degradation_m: float = Field(default=0.00025, ge=0, le=0.003)
    max_mask_iou_degradation: float = Field(default=0.01, ge=0, le=0.05)
    max_rgb_mae_degradation: float = Field(default=0.01, ge=0, le=0.05)
    max_axis_change_deg: float = Field(default=8.0, gt=0, le=30)
    max_origin_displacement_m: float = Field(default=0.01, gt=0, le=0.05)
    hinge_band_m: float = Field(default=0.04, gt=0, le=0.15)
    minimum_clearance_m: float = Field(default=0.0002, gt=0, le=0.003)
    clearance_samples: int = Field(default=61, ge=3, le=361)
    surface_query_backend: Literal["auto", "cpu", "warp_cuda"] = "auto"
    gpu_devices: tuple[int, ...] = ()
    require_accelerator: bool = False
    cpu_query_workers: int = Field(default=0, ge=0, le=64)
    surface_query_batch_size: int = Field(default=4096, ge=512, le=1_048_576)
    remesh_edge_m: float = Field(default=0.003, gt=0, le=0.02)
    remesh_rounds: int = Field(default=2, ge=0, le=5)
    surface_support_m: float = Field(default=0.005, gt=0, le=0.02)
    solver_iterations: int = Field(default=120, ge=5, le=2000)
    correspondence_steps: int = Field(default=5, ge=1, le=20)
    solver_learning_rate: float = Field(default=0.00015, gt=0, le=0.001)
    contact_reserve_m: float = Field(default=0.0005, ge=0, le=0.003)
    active_constraints_per_pose: int = Field(default=400, ge=100, le=50000)
    temporal_subdivisions: int = Field(default=3, ge=0, le=8)
    verification_cell_budget: int = Field(default=200000, ge=1000, le=2000000)
    seed: int = Field(default=2026, ge=1, le=2_147_483_647)

    @model_validator(mode="after")
    def ordered_distances(self) -> VideoArticulationRefinementConfig:
        if self.resume_checkpoint is not None and self.initialize_from_checkpoint is not None:
            raise ValueError("choose exact resume or geometry initialization, not both")
        if self.solver_iterations % self.correspondence_steps:
            raise ValueError("solver iterations must be divisible by correspondence steps")
        if self.minimum_clearance_m >= self.max_surface_displacement_m:
            raise ValueError("clearance must be smaller than the surface trust region")
        if self.hinge_band_m <= self.max_surface_displacement_m:
            raise ValueError("hinge band must exceed the surface trust region")
        if len(set(self.gpu_devices)) != len(self.gpu_devices) or any(
            device < 0 for device in self.gpu_devices
        ):
            raise ValueError("GPU device indices must be unique and nonnegative")
        if self.require_accelerator and self.surface_query_backend == "cpu":
            raise ValueError("accelerator-required search cannot select the CPU backend")
        return self


class VideoArticulationRefinement(DomainModel):
    """Published result; only a passed candidate exposes derived geometries."""

    schema_id: Literal["physrealx.video-articulation-refinement/2"] = Field(
        default="physrealx.video-articulation-refinement/2", alias="schema"
    )
    status: Literal["PASSED", "FAILED"]
    evidence_scope: Literal["calibrated_video_constrained_articulation"]
    source_reference: ArtifactRef
    source_geometry: ArtifactRef
    registration: ArtifactRef
    source_evidence: ArtifactRef
    config: VideoArticulationRefinementConfig
    source_motion: dict[str, Any]
    optimized_motion: dict[str, Any]
    range_preserved: bool
    joint: dict[str, Any]
    links: dict[str, dict[str, Any]]
    heldout: dict[str, dict[str, Any]]
    heldout_images: dict[str, Any] = Field(default_factory=dict)
    swept_surface_clearance: dict[str, Any]
    surface_query: dict[str, Any]
    optimizer: dict[str, Any]
    diagnostic_geometries: dict[str, ArtifactRef]
    checkpoints: tuple[ArtifactRef, ...] = ()
    geometries: dict[str, ArtifactRef] | None
    visual_geometry_changed: bool
    joint_model_changed: bool
    joint_range_changed: bool
    collision_proxy_status: Literal["REQUIRES_REBUILD", "BLOCKED"]
    independent_observation_validation: bool
    physical_validation: bool

    @model_validator(mode="after")
    def accepted_candidate_is_complete(self) -> VideoArticulationRefinement:
        if not self.range_preserved or self.joint_range_changed:
            raise ValueError("video refinement may not change the observed joint range")
        if any(
            self.source_motion.get(k) != self.optimized_motion.get(k)
            for k in ("parent", "child", "kind", "joint", "range", "positions")
        ):
            raise ValueError("video refinement must preserve source articulation and positions")
        if self.status == "PASSED":
            if self.geometries is None or self.collision_proxy_status != "REQUIRES_REBUILD":
                raise ValueError("passed video refinement requires derived geometry rebuild")
            if (
                self.geometries != self.diagnostic_geometries
                or self.optimizer.get("geometry_valid") is not True
                or self.swept_surface_clearance.get("continuous_surface_clearance_certified")
                is not True
                or self.swept_surface_clearance.get("minimum_clearance_lower_bound_m", -math.inf)
                < self.config.minimum_clearance_m
            ):
                raise ValueError(
                    "passed refinement requires valid geometry and certified clearance"
                )
            if not _heldout_surface_passed(
                self.heldout, set(self.geometries), self.config.max_check_degradation_m
            ):
                raise ValueError("passed refinement requires independent surface validation")
            image_frames = self.heldout_images.get("frames", [])
            if (
                self.heldout_images.get("passed") is not True
                or not image_frames
                or not all(
                    _image_pair_passed(row.get("before", {}), row.get("candidate", {}), self.config)
                    for row in image_frames
                )
            ):
                raise ValueError("passed refinement requires independent image validation")
        elif self.geometries is not None or self.collision_proxy_status != "BLOCKED":
            raise ValueError("failed video refinement cannot publish derived geometries")
        if self.independent_observation_validation or self.physical_validation:
            raise ValueError("video refinement is not independent physical validation")
        return self


def _heldout_surface_passed(
    measurements: dict[str, Any], required_links: set[str], tolerance: float
) -> bool:
    if not required_links or not required_links.issubset(measurements):
        return False
    try:
        for value in measurements.values():
            before, after = value["before"], value["candidate"]
            if (
                any(
                    r["samples"] <= 0 or not math.isfinite(r["p90_m"]) or r["p90_m"] < 0
                    for r in (before, after)
                )
                or after["p90_m"] > before["p90_m"] + tolerance
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _image_pair_passed(
    before: dict[str, Any], after: dict[str, Any], config: VideoArticulationRefinementConfig
) -> bool:
    try:
        return (
            before["frame_id"] == after["frame_id"]
            and all(
                r[k] is not None and math.isfinite(r[k]) and r[k] >= 0
                for r in (before, after)
                for k in ("mask_iou", "rgb_mae", "depth_p90_m")
            )
            and after["mask_iou"] >= before["mask_iou"] - config.max_mask_iou_degradation
            and after["rgb_mae"] <= before["rgb_mae"] + config.max_rgb_mae_degradation
            and after["depth_p90_m"] <= before["depth_p90_m"] + config.max_check_degradation_m
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class FramePoints:
    frame_id: str
    position: float
    points_reference: Any
    image: Any = None


def _closest(query: SurfaceQuery | None, mesh: Any, points: Any) -> tuple[Any, Any, Any]:
    """Recover exact closest points from the query backend's triangle ownership."""
    import numpy as np
    import trimesh

    values = np.asarray(points, dtype=float)
    if query is None:
        return cast(
            tuple[Any, Any, Any],
            trimesh.proximity.closest_point(mesh, values),  # type: ignore[no-untyped-call]
        )
    distances, triangles = query.nearest(mesh, values)
    if np.any(triangles < 0):
        raise DomainValidationError("surface query could not bind every observation")
    closest = trimesh.triangles.closest_point(  # type: ignore[no-untyped-call]
        mesh.triangles[triangles], values
    )
    return closest, distances, triangles


def _validate_camera(matrix: Any) -> bool:
    import numpy as np

    return bool(
        matrix.shape == (4, 4)
        and np.isfinite(matrix).all()
        and np.allclose(matrix[3], [0, 0, 0, 1])
        and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        and np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
    )


def _resize_metric_depth(depth: Any, size: tuple[int, int]) -> Any:
    """Pool finite positive depth without dropping sparse sensor samples.

    Nearest-neighbour decimation can pick only invalid pixels on a sparse grid.
    Normalize the area-weighted depth by its valid support; empty footprints
    remain unknown (zero), rather than fabricating an interpolated measurement.
    """
    import numpy as np
    from PIL import Image

    valid = np.isfinite(depth) & (depth > 0)
    numerator = np.asarray(
        Image.fromarray(np.where(valid, depth, 0).astype(np.float32)).resize(
            size, Image.Resampling.BOX
        )
    )
    support = np.asarray(
        Image.fromarray(valid.astype(np.float32)).resize(size, Image.Resampling.BOX)
    )
    return np.divide(numerator, support, out=np.zeros_like(numerator), where=support > 0)


def observations(
    store: ArtifactStore,
    evidence: VideoArticulationEvidence,
    model: dict[str, Any],
    motion: dict[str, Any],
    config: VideoArticulationRefinementConfig,
) -> dict[str, dict[str, list[FramePoints]]]:
    """Decode metric link point clouds without applying the current joint model."""
    import numpy as np
    import trimesh
    from PIL import Image

    names = {link["name"] for link in model["links"]}
    if (
        set(evidence.link_ids) != names
        or len(set(evidence.link_ids.values())) != len(names)
        or any(value <= 0 for value in evidence.link_ids.values())
    ):
        raise DomainValidationError("positive unique mask IDs must cover every link")
    if len({frame.frame_id for frame in evidence.frames}) != len(evidence.frames):
        raise DomainValidationError("video frame IDs must be unique")
    stream_artifacts = [
        ref.artifact_id
        for frame in evidence.frames
        for ref in (frame.rgb, frame.depth, frame.link_mask or frame.object_mask)
        if ref is not None
    ]
    if len(stream_artifacts) != len(set(stream_artifacts)):
        raise DomainValidationError("every video frame stream must be a unique artifact")
    partitions = {frame.partition for frame in evidence.frames}
    if partitions != {"fit", "check"}:
        raise DomainValidationError("video evidence requires fit and held-out check frames")
    fit_artifacts = {
        ref.artifact_id
        for frame in evidence.frames
        if frame.partition == "fit"
        for ref in (frame.rgb, frame.depth, frame.link_mask or frame.object_mask)
        if ref is not None
    }
    check_artifacts = {
        ref.artifact_id
        for frame in evidence.frames
        if frame.partition == "check"
        for ref in (frame.rgb, frame.depth, frame.link_mask or frame.object_mask)
        if ref is not None
    }
    if fit_artifacts & check_artifacts:
        raise DomainValidationError("video fit/check partitions must use different artifacts")

    result: dict[str, dict[str, list[FramePoints]]] = {
        name: {"fit": [], "check": []} for name in sorted(names)
    }
    if any(f.object_mask is not None for f in evidence.frames):
        result["__object__"] = {"fit": [], "check": []}
    random = np.random.default_rng(config.seed)
    for frame in evidence.frames:
        depth = read_array(store, frame.depth).astype(float) * (
            frame.depth_scale_m or evidence.depth_scale_m
        )
        mask_ref = frame.link_mask or frame.object_mask
        assert mask_ref is not None  # EvidenceFrame validates exactly one mask kind.
        mask = np.asarray(Image.open(io.BytesIO(store.get_bytes(mask_ref))))
        rgb = np.asarray(Image.open(io.BytesIO(store.get_bytes(frame.rgb))).convert("RGB"))
        matrix = np.asarray(frame.camera_from_reference, dtype=float)
        fx, fy, cx, cy = frame.intrinsics
        if (
            depth.ndim != 2
            or mask.shape != depth.shape
            or rgb.shape[:2] != depth.shape
            or mask.dtype.kind not in "ui"
            or not _validate_camera(matrix)
            or not np.isfinite(frame.intrinsics).all()
            or min(fx, fy) <= 0
            or (
                frame.link_mask is not None
                and not set(np.unique(mask)) <= {0, *evidence.link_ids.values()}
            )
        ):
            raise DomainValidationError(
                "video requires calibrated RGB-D, integer link masks and SE(3) camera poses"
            )
        if set(frame.positions) != {motion["joint"]}:
            raise DomainValidationError(
                "video positions must cover exactly the reconstructed joint"
            )
        position = frame.positions[motion["joint"]]
        if not motion["range"][0] <= position <= motion["range"][1]:
            raise DomainValidationError("video position exceeds the observed start/end interval")
        ratio = min(1.0, config.render_max_dimension / max(depth.shape))
        render_size = (max(8, round(depth.shape[1] * ratio)), max(8, round(depth.shape[0] * ratio)))
        render_rgb = np.asarray(Image.fromarray(rgb).resize(render_size, Image.Resampling.BILINEAR))
        render_depth = _resize_metric_depth(depth, render_size)
        render_mask = np.asarray(
            Image.fromarray(mask).resize(render_size, Image.Resampling.NEAREST)
        )
        sx, sy = render_size[0] / depth.shape[1], render_size[1] / depth.shape[0]
        render_intrinsics = np.array(
            [fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5]
        )
        identifiers = evidence.link_ids if frame.link_mask is not None else {"__object__": 1}
        if frame.object_mask is not None:
            mask = (mask > 0).astype(np.uint8)
            render_mask = (render_mask > 0).astype(np.uint8)
        for name, identifier in identifiers.items():
            y, x = np.where((mask == identifier) & np.isfinite(depth) & (depth > 0))
            if len(x) < 100:
                continue
            z = depth[y, x]
            camera_points = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
            points = trimesh.transform_points(camera_points, np.linalg.inv(matrix))
            count = min(len(points), config.points_per_frame)
            if frame.partition == "fit":
                # Preserve full-frame coverage while explicitly retaining small
                # observed hinge features which uniform random pixels miss.
                axis = np.asarray(motion["axis"], dtype=float)
                axis /= np.linalg.norm(axis)
                delta = points - np.asarray(motion["origin_m"])
                distance = np.linalg.norm(delta - np.outer(delta @ axis, axis), axis=1)
                local = np.flatnonzero(distance < config.hinge_band_m)
                selected = random.choice(local, min(len(local), count // 2), replace=False)
                remaining = np.setdiff1d(np.arange(len(points)), selected, assume_unique=True)
                indexes = np.r_[
                    selected, random.choice(remaining, count - len(selected), replace=False)
                ]
            else:
                # Held-out sampling remains uniform, independent of the optimizer.
                indexes = random.choice(len(points), count, replace=False)
            points = points[indexes]
            result[name][frame.partition].append(
                FramePoints(
                    frame.frame_id,
                    float(position),
                    points,
                    (
                        render_rgb,
                        render_depth,
                        render_mask == identifier,
                        matrix,
                        render_intrinsics,
                    ),
                )
            )
    return result


def _axis_angle_deg(left: Any, right: Any) -> float:
    import numpy as np

    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _local_points(frames: list[FramePoints], motion: dict[str, Any], link: str) -> Any:
    import numpy as np
    import trimesh

    arrays = []
    for frame in frames:
        matrix = (
            joint_transform(motion["kind"], motion["axis"], motion["origin_m"], frame.position)
            if link == motion["child"]
            else np.eye(4)
        )
        arrays.append(trimesh.transform_points(frame.points_reference, np.linalg.inv(matrix)))
    return np.concatenate(arrays) if arrays else np.empty((0, 3), dtype=float)


def _surface_metrics(
    mesh: Any,
    frames: list[FramePoints],
    motion: dict[str, Any],
    link: str,
    query: SurfaceQuery | None = None,
) -> dict[str, float]:
    import numpy as np
    import trimesh

    values = []
    for frame in frames:
        matrix = (
            joint_transform(motion["kind"], motion["axis"], motion["origin_m"], frame.position)
            if link == motion["child"]
            else np.eye(4)
        )
        local = trimesh.transform_points(frame.points_reference, np.linalg.inv(matrix))
        values.append(_closest(query, mesh, local)[1])
    distances = np.concatenate(values) if values else np.asarray([math.inf])
    return {
        "median_m": float(np.median(distances)),
        "p90_m": float(np.quantile(distances, 0.9)),
        "samples": len(distances),
    }


def _object_metrics(
    meshes: dict[str, Any], frames: list[FramePoints], motion: dict[str, Any], query: SurfaceQuery
) -> dict[str, Any]:
    import numpy as np
    import trimesh

    values = []
    for frame in frames:
        child = meshes[motion["child"]].copy()
        child.apply_transform(
            joint_transform("hinge", motion["axis"], motion["origin_m"], frame.position)
        )
        combined = trimesh.util.concatenate([meshes[motion["parent"]], child])
        values.append(_closest(query, combined, frame.points_reference)[1])
    d = np.concatenate(values) if values else np.array([math.inf])
    return {"median_m": float(np.median(d)), "p90_m": float(np.quantile(d, 0.9)), "samples": len(d)}


def _refine_video_articulation(
    store: ArtifactStore,
    evidence_ref: ArtifactRef,
    *,
    config: VideoArticulationRefinementConfig,
    query: SurfaceQuery,
    candidate: ArtifactRef | None = None,
) -> ArtifactRef:
    """Derive a video-supported geometry/joint candidate with an unchanged range."""
    import numpy as np

    evidence = VideoArticulationEvidence.model_validate_json(store.get_bytes(evidence_ref))
    model = read_json(store, evidence.reference, "physrealx.usd-reference/1")
    registration = read_json(store, evidence.registration, "physrealx.metric-alignment-report/1")
    source_geometry = read_json(
        store, evidence.source_geometry, "physrealx.registered-reconstruction-geometry/4"
    )
    if registration["source_reference"] != evidence.reference.model_dump(mode="json"):
        raise DomainValidationError("video registration differs from source reference")
    if (
        source_geometry.get("source_reference") != evidence.reference.model_dump(mode="json")
        or source_geometry.get("source_registration")
        != evidence.registration.model_dump(mode="json")
        or set(source_geometry.get("geometries", {})) != {link["name"] for link in model["links"]}
    ):
        raise DomainValidationError("video source geometry differs from reconstruction")
    source_motion = source_geometry["reconstructed_motion"]
    if source_motion["kind"] != "hinge":
        raise DomainValidationError("video articulation refinement currently requires one hinge")
    decoded = observations(store, evidence, model, source_motion, config)
    meshes: dict[str, Any] = {}
    for link in model["links"]:
        name = link["name"]
        values = load_meshes(
            store, ArtifactRef.model_validate(source_geometry["geometries"][name]), process=False
        )
        if len(values) != 1:
            raise DomainValidationError("video articulation requires one visual mesh per link")
        meshes[name] = values[0]
    if set(meshes) != {source_motion["parent"], source_motion["child"]}:
        raise DomainValidationError("joint surface refinement requires exactly two hinged parts")

    from ._joint_surface_solver import solve_joint_surfaces
    from ._surface_clearance import swept_clearance

    checkpoints = []

    def checkpoint(
        metrics: dict[str, Any],
        state: tuple[dict[str, Any], dict[str, Any]],
        optimization_state: dict[str, Any],
    ) -> None:
        logging.getLogger(__name__).info("Joint refinement: %s", metrics)
        candidates, motion = state
        geometries = {
            name: store.put_bytes(
                mesh.export(file_type="glb"), media_type="model/gltf-binary"
            ).model_dump(mode="json")
            for name, mesh in candidates.items()
        }
        checkpoints.append(
            store.put_json(
                {
                    "schema": "physrealx.joint-surface-checkpoint/2",
                    "optimization_state": optimization_state,
                    "source_evidence": evidence_ref.model_dump(mode="json"),
                    "config": config.model_dump(mode="json"),
                    "metrics": metrics,
                    "motion": motion,
                    "diagnostic_geometries": geometries,
                    "validated": False,
                }
            )
        )

        logging.getLogger(__name__).info("Joint checkpoint: %s", checkpoints[-1].model_dump_json())

    resume = None
    checkpoint_ref = config.resume_checkpoint or config.initialize_from_checkpoint
    if checkpoint_ref is not None:
        saved = read_json(store, checkpoint_ref, "physrealx.joint-surface-checkpoint/2")
        saved_evidence = ArtifactRef.model_validate(saved["source_evidence"])
        ignored = {
            "resume_checkpoint",
            "initialize_from_checkpoint",
            "solver_iterations",
            "surface_query_backend",
            "gpu_devices",
            "cpu_query_workers",
            "require_accelerator",
            "surface_query_batch_size",
        }
        if config.initialize_from_checkpoint is not None:
            ignored.update(
                {"solver_learning_rate", "clearance_samples", "active_constraints_per_pose"}
            )
        current_config = config.model_dump(mode="json")
        if store.get_bytes(saved_evidence) != store.get_bytes(evidence_ref) or {
            k: v for k, v in saved["config"].items() if k not in ignored
        } != {k: v for k, v in current_config.items() if k not in ignored}:
            raise DomainValidationError("checkpoint input evidence or shape constraints differ")
        resume = saved["optimization_state"]
        if str(resume.get("solver_identity", "")).startswith("video-feasible/"):
            raise DomainValidationError("deprecated pre-unfolded checkpoint is not video-first")
        if config.initialize_from_checkpoint is not None:
            # Reuse geometry only after rebuilding the original problem. Old
            # optimizer/contact states and acceptance claims are not imported.
            saved_meshes = {
                name: load_meshes(store, ArtifactRef.model_validate(ref), process=False)[0]
                for name, ref in saved["diagnostic_geometries"].items()
            }
            resume = {
                **resume,
                "initialize_only": True,
                "mesh_faces": {name: mesh.faces for name, mesh in saved_meshes.items()},
                "mesh_vertices": {name: mesh.vertices for name, mesh in saved_meshes.items()},
            }
        elif resume["iteration"] >= config.solver_iterations:
            raise DomainValidationError("resume requires a larger total iteration budget")
    if candidate is None:
        candidates, optimized_motion, optimizer = solve_joint_surfaces(
            meshes, decoded, source_motion, config, query, checkpoint, resume=resume
        )
    else:
        from ._native_untangling import load_unfolded_candidate

        candidates, optimized_motion, optimizer = load_unfolded_candidate(
            store, candidate, evidence.source_geometry, meshes, source_motion, config
        )
    geometries = {
        name: store.put_bytes(
            mesh.export(file_type="glb"), media_type="model/gltf-binary"
        ).model_dump(mode="json")
        for name, mesh in candidates.items()
    }
    # All independent gates examine the exact exported float32 mesh.
    candidates = {
        name: load_meshes(store, ArtifactRef.model_validate(ref), process=False)[0]
        for name, ref in geometries.items()
    }
    from ._mesh_validity import self_intersections, surface_topology

    for name, mesh in candidates.items():
        topology = surface_topology(mesh)
        optimizer["links"][name]["exported_topology"] = topology
        intersects = self_intersections(mesh)
        optimizer["links"][name]["self_intersections"] = intersects
        optimizer["geometry_valid"] &= topology["valid"] and not intersects
    clearance = swept_clearance(
        candidates[source_motion["parent"]],
        candidates[source_motion["child"]],
        optimized_motion,
        query,
        clearance=config.minimum_clearance_m,
        samples=max(361, config.clearance_samples),
        temporal_levels=config.temporal_subdivisions,
        max_cells=config.verification_cell_budget,
    )
    joint_report = {
        "method": optimizer["method"],
        "axis_change_deg": _axis_angle_deg(source_motion["axis"], optimized_motion["axis"]),
        "origin_displacement_m": float(
            np.linalg.norm(np.array(source_motion["origin_m"]) - optimized_motion["origin_m"])
        ),
        "range_preserved": True,
    }
    link_reports = optimizer["links"]
    heldout = {}
    for name in candidates:
        before = _surface_metrics(meshes[name], decoded[name]["check"], source_motion, name, query)
        after = _surface_metrics(
            candidates[name], decoded[name]["check"], optimized_motion, name, query
        )
        heldout[name] = {"before": before, "candidate": after}
    if "__object__" in decoded:
        heldout["start_end"] = {
            "before": _object_metrics(meshes, decoded["__object__"]["check"], source_motion, query),
            "candidate": _object_metrics(
                candidates, decoded["__object__"]["check"], optimized_motion, query
            ),
        }
    check_passed = _heldout_surface_passed(heldout, set(candidates), config.max_check_degradation_m)
    from ._video_render_loss import check_images

    device = query.describe()["devices"][0] if query.describe()["accelerated"] else "cpu"
    image_before = check_images(meshes, decoded, source_motion, device)
    image_after = check_images(candidates, decoded, optimized_motion, device)
    image_rows = []
    for before, after in zip(image_before, image_after, strict=True):
        passed = _image_pair_passed(before, after, config)
        image_rows.append({"before": before, "candidate": after, "passed": passed})
    image_passed = bool(image_rows) and all(r["passed"] for r in image_rows)
    enough_views = all(
        min(len(value["fit"]), len(value["check"])) >= 2 for value in decoded.values()
    )
    status = (
        "PASSED"
        if enough_views
        and optimizer["geometry_valid"]
        and check_passed
        and image_passed
        and clearance["status"] in {"PASSED", "CLEAR"}
        else "FAILED"
    )
    result = VideoArticulationRefinement(
        **{
            "schema": "physrealx.video-articulation-refinement/2",
            "status": status,
            "evidence_scope": "calibrated_video_constrained_articulation",
            "source_reference": evidence.reference.model_dump(mode="json"),
            "source_geometry": evidence.source_geometry.model_dump(mode="json"),
            "registration": evidence.registration.model_dump(mode="json"),
            "source_evidence": evidence_ref.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
            "source_motion": source_motion,
            "optimized_motion": optimized_motion,
            "range_preserved": optimized_motion["range"] == source_motion["range"],
            "joint": joint_report,
            "links": link_reports,
            "heldout": heldout,
            "heldout_images": {"passed": image_passed, "frames": image_rows},
            "swept_surface_clearance": clearance,
            "surface_query": query.describe(),
            "optimizer": optimizer,
            "diagnostic_geometries": geometries,
            "checkpoints": checkpoints,
            "geometries": geometries if status == "PASSED" else None,
            "visual_geometry_changed": status == "PASSED"
            and any(geometries[name] != source_geometry["geometries"][name] for name in geometries),
            "joint_model_changed": status == "PASSED"
            and (
                optimized_motion["axis"] != source_motion["axis"]
                or optimized_motion["origin_m"] != source_motion["origin_m"]
            ),
            "joint_range_changed": False,
            "collision_proxy_status": "REQUIRES_REBUILD" if status == "PASSED" else "BLOCKED",
            "independent_observation_validation": False,
            "physical_validation": False,
        }
    )
    return store.put_json(result.model_dump(mode="json", by_alias=True))


def refine_video_articulation(
    store: ArtifactStore,
    evidence_ref: ArtifactRef,
    *,
    config: VideoArticulationRefinementConfig,
    cached_result: ArtifactRef | None = None,
    candidate: ArtifactRef | None = None,
) -> ArtifactRef:
    """Derive a video-supported candidate using the selected bounded query backend."""
    if candidate is not None and (
        cached_result is not None
        or config.resume_checkpoint is not None
        or config.initialize_from_checkpoint is not None
    ):
        raise DomainValidationError("external candidate validation cannot reuse solver checkpoints")
    if cached_result is not None:
        from ._joint_surface_solver import SOLVER_IDENTITY

        document = json.loads(store.get_bytes(cached_result)) if store.verify(cached_result) else {}
        if document.get("optimizer", {}).get("solver_identity") == SOLVER_IDENTITY:
            cached = VideoArticulationRefinement.model_validate(document)
            # A completed run retains its restart provenance. A normal request
            # may reuse that completed result without repeating the restart;
            # an explicitly requested different checkpoint must still miss.
            cached_config = cached.config
            if config.resume_checkpoint is None and config.initialize_from_checkpoint is None:
                cached_config = cached_config.model_copy(
                    update={
                        "resume_checkpoint": None,
                        "initialize_from_checkpoint": None,
                    }
                )
            if (
                cached_config == config
                and cached.optimizer.get("solver_identity") == SOLVER_IDENTITY
                and store.get_bytes(cached.source_evidence) == store.get_bytes(evidence_ref)
                and all(store.verify(ref) for ref in cached.diagnostic_geometries.values())
            ):
                return cached_result
    query = create_surface_query(
        backend=config.surface_query_backend,
        gpu_devices=config.gpu_devices,
        require_accelerator=config.require_accelerator,
        cpu_workers=config.cpu_query_workers,
        batch_size=config.surface_query_batch_size,
    )
    try:
        return _refine_video_articulation(
            store, evidence_ref, config=config, query=query, candidate=candidate
        )
    finally:
        query.close()
