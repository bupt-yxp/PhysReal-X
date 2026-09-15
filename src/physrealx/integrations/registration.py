"""Register a two-state observed horizontal-axis joint to its supplied articulated USD reference.

The support plane initializes tilt; a full yaw search uses observed surfaces. Robust surface
registration fits pose and the two joint positions on training views only.
This is model-conditioned registration, never independent physical metrology.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import ArtifactRef, DomainModel, MethodCapability, OperationSpec
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import input_artifact, publish, read_json
from physrealx.integrations.capture_rgbd import capture_support_region
from physrealx.integrations.collision import load_meshes
from physrealx.integrations.rgbd import MetricRGBDInput, RGBDView, load_rgbd
from physrealx.methods import builtin_method
from physrealx.reconstruction import ReconstructionPrediction, joint_transform


class ReferenceRegistrationInput(DomainModel):
    schema_id: Literal["physrealx.reference-registration-input/1"] = Field(
        default="physrealx.reference-registration-input/1", alias="schema"
    )
    rgbd: MetricRGBDInput
    reconstruction_report: ArtifactRef


class RegistrationConfig(DomainModel):
    joint: str = Field(min_length=1)
    seed: int = Field(default=2026, ge=0)
    points_per_view: int = Field(default=1200, ge=300, le=5000)
    mesh_samples: int = Field(default=20000, ge=5000, le=100000)
    fit_points_per_state: int = Field(default=2500, ge=1000, le=10000)
    component_voxel_m: float = Field(default=0.012, gt=0, le=0.03, allow_inf_nan=False)
    max_holdout_median_m: float = Field(default=0.015, gt=0, le=0.03, allow_inf_nan=False)
    max_holdout_p90_m: float = Field(default=0.03, gt=0, le=0.05, allow_inf_nan=False)


class AlignmentStateEvidence(DomainModel):
    median_m: float = Field(ge=0, allow_inf_nan=False)
    p90_m: float = Field(ge=0, allow_inf_nan=False)
    points: int = Field(ge=100)
    joint_position: float = Field(allow_inf_nan=False)


class ReferenceDepthAlignment(DomainModel):
    """One train-only real-depth to supplied-USD alignment with held-out checks."""

    schema_id: Literal["physrealx.reference-depth-alignment/1"] = Field(
        default="physrealx.reference-depth-alignment/1", alias="schema"
    )
    source_camera: ArtifactRef
    source_depth: ArtifactRef
    source_reference: ArtifactRef
    source_frame: str
    target_frame: str
    observed_joint: str
    joint_type: Literal["hinge", "slide"]
    joint_axis: tuple[float, float, float]
    joint_origin_m: tuple[float, float, float]
    joint_limits: tuple[float, float]
    state_order: tuple[str, str]
    rotation: tuple[tuple[float, float, float], ...]
    translation_m: tuple[float, float, float]
    evaluations: dict[str, AlignmentStateEvidence]
    foreground_fractions: dict[str, float]
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    metrics: dict[str, float]

    @model_validator(mode="after")
    def validate_geometry_and_split(self) -> Self:
        import numpy as np

        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation_m, dtype=float)
        axis = np.asarray(self.joint_axis, dtype=float)
        origin = np.asarray(self.joint_origin_m, dtype=float)
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
            or not np.isfinite(axis).all()
            or not np.isfinite(origin).all()
            or not np.isclose(np.linalg.norm(axis), 1, atol=1e-6)
            or not np.isfinite(self.joint_limits).all()
            or not self.joint_limits[0] < self.joint_limits[1]
        ):
            raise ValueError("reference depth alignment requires a finite proper rigid transform")
        if (
            self.state_order[0] == self.state_order[1]
            or set(self.evaluations) != set(self.state_order)
            or set(self.train_frame_ids) & set(self.holdout_frame_ids)
            or not self.train_frame_ids
            or not self.holdout_frame_ids
            or len(self.train_frame_ids) != len(set(self.train_frame_ids))
            or len(self.holdout_frame_ids) != len(set(self.holdout_frame_ids))
            or not self.source_frame
            or not self.target_frame
            or any(
                not 0 <= value <= 1 or not np.isfinite(value)
                for value in self.foreground_fractions.values()
            )
            or any(not np.isfinite(value) for value in self.metrics.values())
            or any(
                not self.joint_limits[0] <= item.joint_position <= self.joint_limits[1]
                for item in self.evaluations.values()
            )
        ):
            raise ValueError("reference depth alignment requires two states and a disjoint split")
        return self


def _select_registration_views(
    views: Sequence[RGBDView], training: object, holdout: object
) -> tuple[list[RGBDView], set[str], set[str]]:
    """Resolve the reconstruction split against RGB-D without consuming unused source views."""
    if (
        not isinstance(training, (list, tuple))
        or not isinstance(holdout, (list, tuple))
        or not training
        or not holdout
        or any(not isinstance(item, str) or not item for item in (*training, *holdout))
        or len(training) != len(set(training))
        or len(holdout) != len(set(holdout))
    ):
        raise DomainValidationError(
            "registration requires unique nonempty train and held-out views"
        )
    train_ids, test_ids = set(training), set(holdout)
    if train_ids & test_ids:
        raise DomainValidationError("registration train and held-out views overlap")
    available_ids = [view.frame_id for view in views]
    if len(available_ids) != len(set(available_ids)):
        raise DomainValidationError("registration RGB-D view identities must be unique")
    unknown = (train_ids | test_ids) - set(available_ids)
    if unknown:
        raise DomainValidationError(
            f"registration split references unknown capture frames: {sorted(unknown)!r}"
        )
    selected = train_ids | test_ids
    return [view for view in views if view.frame_id in selected], train_ids, test_ids


def connected_foreground(
    points: Any,
    plane_z: float,
    voxel: float,
    plane_clearance: float,
    reference_dimensions: Any,
) -> Any:
    """Isolate the model-sized object above the support plane."""
    import numpy as np
    from scipy.ndimage import label

    dimensions = np.asarray(reference_dimensions, dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise DomainValidationError("registration requires finite positive reference dimensions")
    points = points[points[:, 2] - plane_z > plane_clearance]
    if len(points) < 100:
        raise MethodExecutionError("registration has insufficient above-plane object points")
    seed = points[points[:, 2] - plane_z > max(2 * plane_clearance, 0.15 * dimensions[2])]
    if len(seed) < 100:
        raise MethodExecutionError("registration has insufficient elevated object support")
    center = np.median(seed[:, :2], axis=0)
    radius = 0.65 * float(np.linalg.norm(dimensions))
    points = points[np.linalg.norm(points[:, :2] - center, axis=1) <= radius]
    if len(points) < 100:
        raise MethodExecutionError("registration model-sized foreground is insufficient")
    grid = np.floor((points - points.min(0)) / voxel)
    shape = grid.max(0) + 1
    if not np.isfinite(shape).all() or np.prod(shape) > 10_000_000:
        raise DomainValidationError("registration foreground exceeds its bounded spatial grid")
    cells = grid.astype(int)
    occupancy = np.zeros(tuple(shape.astype(int)), bool)
    occupancy[tuple(cells.T)] = True
    components, _ = label(occupancy, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    # Articulated surfaces can be disconnected when the guide/hinge is occluded.
    # Retain substantial components; tiny detached markers remain excluded.
    retained = sizes >= max(1, sizes.max() * 0.1)
    retained[0] = False
    return points[retained[components[tuple(cells.T)]]]


def reference_footprint_foreground(
    points: Any,
    target: Any,
    rotation: Any,
    translation: Any,
    margin: float,
) -> Any:
    """Keep observed points inside the fitted reference footprint and height range."""
    import numpy as np
    from scipy.spatial import ConvexHull

    transformed = points @ rotation.T + translation
    hull = ConvexHull(target[:, :2])
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]
    inside = (
        transformed[:, :2] @ normals.T + offsets <= margin * np.linalg.norm(normals, axis=1)
    ).all(axis=1)
    lower, upper = target[:, 2].min() - margin, target[:, 2].max() + margin
    inside &= (transformed[:, 2] >= lower) & (transformed[:, 2] <= upper)
    selected = points[inside]
    if len(selected) < 100:
        raise MethodExecutionError("registration reference footprint has insufficient points")
    return selected


def fit_articulated_surfaces(
    points: list[Any],
    static: Any,
    moving: Any,
    initial_translation: Any,
    limits: tuple[float, float],
    *,
    seed: int,
    max_points: int,
    joint_type: str = "slide",
    axis: Any = (1, 0, 0),
    origin: Any = (0, 0, 0),
    initial_solution: tuple[Any, Any, Any] | None = None,
) -> tuple[Any, Any, Any, float]:
    """Fit one rigid transform and observed positions; evaluate no held-out data here."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    generator = np.random.default_rng(seed)
    samples = [
        row[generator.choice(len(row), min(len(row), max_points), replace=False)] for row in points
    ]
    static_tree, moving_tree = cKDTree(static), cKDTree(moving)

    def residual(parameters: Any) -> Any:
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        errors = []
        for index, sample in enumerate(samples):
            transformed = sample @ rotation.T + parameters[3:6]
            motion = joint_transform(joint_type, axis, origin, parameters[6 + index])
            errors.append(
                np.minimum(
                    static_tree.query(transformed)[0],
                    moving_tree.query((transformed - motion[:3, 3]) @ motion[:3, :3])[0],
                )
            )
        # The observed support plane is an orientation/location constraint, not a
        # camera-independent surface truth. Small model/depth mismatch is allowed.
        errors.append(
            2 * np.array([parameters[0], parameters[1], parameters[5] - initial_translation[2]])
        )
        return np.concatenate(errors)

    center = np.median(np.concatenate(samples), axis=0)

    def solve(initial: Any, lower: Any, upper: Any) -> Any:
        return least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.006,
            max_nfev=100,
            diff_step=1e-4,
        )

    def fit_candidate(candidate: tuple[float, float, float]) -> Any:
        yaw, start, end = candidate
        yaw_center = yaw
        shifted = (
            initial_translation + center - Rotation.from_rotvec([0, 0, yaw_center]).apply(center)
        )
        initial = np.r_[
            0.0,
            0.0,
            yaw,
            shifted,
            limits[0] + start * (limits[1] - limits[0]),
            limits[0] + end * (limits[1] - limits[0]),
        ]
        lower = np.r_[
            -0.08, -0.08, yaw_center - 0.8, shifted - [0.08, 0.08, 0.02], limits[0], limits[0]
        ]
        upper = np.r_[
            0.08,
            0.08,
            yaw_center + 0.8,
            shifted + np.array([0.08, 0.08, 0.02]),
            limits[1],
            limits[1],
        ]
        return solve(initial, lower, upper)

    if initial_solution is None:
        candidates = [
            (yaw, start, end)
            for yaw in np.linspace(-np.pi, np.pi, 8, endpoint=False)
            for start, end in ((0.15, 0.85), (0.85, 0.15))
        ]
        # Seeds are independent; ordered map preserves deterministic tie-breaking.
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(fit_candidate, candidates))
    else:
        prior_rotation, prior_translation, prior_positions = initial_solution
        rotvec = Rotation.from_matrix(prior_rotation).as_rotvec()
        initial = np.r_[rotvec, prior_translation, prior_positions]
        lower = np.r_[
            rotvec - 0.1,
            np.asarray(prior_translation) - 0.03,
            limits[0],
            limits[0],
        ]
        upper = np.r_[
            rotvec + 0.1,
            np.asarray(prior_translation) + 0.03,
            limits[1],
            limits[1],
        ]
        results = [solve(initial, lower, upper)]
    finite = [result for result in results if np.isfinite(result.cost)]
    best = min(finite, key=lambda result: result.cost) if finite else None
    if best is None or not np.isfinite(best.x).all():
        raise MethodExecutionError("reference registration did not find a finite pose")
    if abs(best.x[7] - best.x[6]) < 0.01:
        raise MethodExecutionError("reference registration lacks a resolved observed joint span")
    return (Rotation.from_rotvec(best.x[:3]).as_matrix(), best.x[3:6], best.x[6:], float(best.cost))


def fit_reference_depth_alignment(
    source: MetricRGBDInput,
    artifacts: ArtifactStore,
    training_frame_ids: object,
    holdout_frame_ids: object,
    config: RegistrationConfig,
) -> ReferenceDepthAlignment:
    """Load and fit supplied articulated geometry to masked real metric depth."""
    camera, _, views = load_rgbd(source, artifacts)
    reference = read_json(artifacts, source.reference_model, "physrealx.usd-reference/1")
    return fit_loaded_reference_depth_alignment(
        source,
        artifacts,
        camera,
        views,
        reference,
        training_frame_ids,
        holdout_frame_ids,
        config,
    )


def fit_loaded_reference_depth_alignment(
    source: MetricRGBDInput,
    artifacts: ArtifactStore,
    camera: dict[str, Any],
    views: Sequence[RGBDView],
    reference: dict[str, Any],
    training_frame_ids: object,
    holdout_frame_ids: object,
    config: RegistrationConfig,
) -> ReferenceDepthAlignment:
    """Fit supplied articulated geometry directly to masked real metric depth.

    Only ``training_frame_ids`` influence the fitted rigid pose and state positions.
    ``holdout_frame_ids`` are evaluated after fitting and can only reject the result.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    trimesh = import_module("trimesh")
    selected_views, train_ids, test_ids = _select_registration_views(
        views, training_frame_ids, holdout_frame_ids
    )
    joint = next((item for item in reference["joints"] if item["name"] == config.joint), None)
    if (
        joint is None
        or joint["joint_type"] not in ("hinge", "slide")
        or abs(joint["axis"][2]) > 1e-5
    ):
        raise DomainValidationError(
            "surface registration requires one selected horizontal reference joint"
        )
    support = capture_support_region(camera)
    normal = support["normal"]
    direction = np.eye(3)[np.argmin(np.abs(normal))]
    horizontal = direction - normal * (direction @ normal)
    horizontal /= np.linalg.norm(horizontal)
    basis = np.array([horizontal, np.cross(normal, horizontal), normal])
    generator = np.random.default_rng(config.seed)
    streams_by_id = {view.frame_id: view.stream for view in views}
    assert isinstance(training_frame_ids, (list, tuple))
    streams = tuple(dict.fromkeys(streams_by_id[frame] for frame in training_frame_ids))
    if len(streams) != 2:
        raise DomainValidationError("reference depth alignment requires two observed states")
    for stream in streams:
        if sum(streams_by_id[key] == stream for key in train_ids) < 2 or not any(
            streams_by_id[key] == stream for key in test_ids
        ):
            raise DomainValidationError(
                "each reference depth alignment state needs train and held-out views"
            )
    groups: dict[str, dict[str, Any]] = {name: {"train": [], "test": []} for name in streams}
    for view in selected_views:
        xyz = (view.rays * view.depth[..., None] - view.pose[:, 3]) @ view.pose[:, :3]
        points = xyz[view.mask & (view.depth > 0)]
        if len(points) < 100:
            raise MethodExecutionError(
                f"reference depth alignment view {view.frame_id} has insufficient object depth"
            )
        indices = generator.choice(
            len(points), min(len(points), config.points_per_view), replace=False
        )
        split = "train" if view.frame_id in train_ids else "test"
        groups[view.stream][split].append(points[indices] @ basis.T)
    fractions: dict[str, float] = {}
    plane_z = float(support["origin"] @ normal)
    plane_clearance = max(6 * float(support["plane_rmse_m"]), 0.03 * float(support["height_m"]))
    for stream, partitions in groups.items():
        for split, rows in partitions.items():
            cloud = np.concatenate(rows)
            foreground = connected_foreground(
                cloud,
                plane_z,
                config.component_voxel_m,
                plane_clearance,
                reference["dimensions_m"],
            )
            fractions[f"{stream}/{split}"] = len(foreground) / len(cloud)
            partitions[split] = foreground
    meshes = {}
    for link in reference["links"]:
        mesh = trimesh.util.concatenate(
            load_meshes(artifacts, ArtifactRef.model_validate(link["geometry"]))
        )
        meshes[link["name"]] = trimesh.sample.sample_surface(
            mesh, config.mesh_samples, seed=config.seed
        )[0]
    body = meshes[reference["root_link"]]
    z_min, z_max = body[:, 2].min(), body[:, 2].max()
    top = body[body[:, 2] > z_min + 0.68 * (z_max - z_min)]
    fit = np.concatenate([part["train"] for part in groups.values()])
    upper = fit[fit[:, 2] - plane_z > 0.7 * (z_max - z_min)]
    if len(upper) < 100:
        raise MethodExecutionError("reference depth alignment lacks an upper static surface")
    translation = np.median(top, axis=0) - np.median(upper, axis=0)
    translation[2] = z_min - plane_z
    tree = cKDTree(top)
    for _ in range(25):
        distances, ids = tree.query(upper + translation)
        keep = distances < np.quantile(distances, 0.8)
        delta = np.median(top[ids[keep]] - upper[keep] - translation, axis=0)
        translation += delta
        if np.linalg.norm(delta) < 1e-6:
            break
    moving = meshes[joint["child"]]
    static = np.concatenate([points for name, points in meshes.items() if name != joint["child"]])
    initial_translation = translation.copy()
    rotation, translation, positions, cost = fit_articulated_surfaces(
        [groups[name]["train"] for name in streams],
        static,
        moving,
        translation,
        (joint["lower_limit"], joint["upper_limit"]),
        seed=config.seed,
        max_points=config.fit_points_per_state,
        joint_type=joint["joint_type"],
        axis=joint["axis"],
        origin=joint["origin_m"],
    )
    footprint_margin = max(
        6 * float(support["plane_rmse_m"]),
        0.02 * float(np.linalg.norm(reference["dimensions_m"])),
    )
    refined_train = []
    for index, stream in enumerate(streams):
        motion = joint_transform(
            joint["joint_type"], joint["axis"], joint["origin_m"], positions[index]
        )
        target = np.concatenate([static, moving @ motion[:3, :3].T + motion[:3, 3]])
        points = groups[stream]["train"]
        refined = reference_footprint_foreground(
            points, target, rotation, translation, footprint_margin
        )
        fractions[f"{stream}/train_reference_footprint"] = len(refined) / len(points)
        refined_train.append(refined)
    rotation, translation, positions, cost = fit_articulated_surfaces(
        refined_train,
        static,
        moving,
        initial_translation,
        (joint["lower_limit"], joint["upper_limit"]),
        seed=config.seed,
        max_points=config.fit_points_per_state,
        joint_type=joint["joint_type"],
        axis=joint["axis"],
        origin=joint["origin_m"],
        initial_solution=(rotation, translation, positions),
    )
    evaluations = {}
    for index, stream in enumerate(streams):
        motion = joint_transform(
            joint["joint_type"], joint["axis"], joint["origin_m"], positions[index]
        )
        target_points = np.concatenate([static, moving @ motion[:3, :3].T + motion[:3, 3]])
        raw_points = groups[stream]["test"]
        foreground = reference_footprint_foreground(
            raw_points, target_points, rotation, translation, footprint_margin
        )
        fractions[f"{stream}/test_reference_footprint"] = len(foreground) / len(raw_points)
        points = foreground @ rotation.T + translation
        distances = cKDTree(target_points).query(points)[0]
        median, p90 = np.quantile(distances, [0.5, 0.9])
        if median > config.max_holdout_median_m or p90 > config.max_holdout_p90_m:
            raise MethodExecutionError(
                f"reference depth alignment failed held-out surface gates for {stream}: "
                f"median={median:.6f} m, p90={p90:.6f} m, "
                f"joint_position={positions[index]:.6f}"
            )
        evaluations[stream] = AlignmentStateEvidence(
            median_m=float(median),
            p90_m=float(p90),
            points=len(points),
            joint_position=float(positions[index]),
        )
    rotation = rotation @ basis
    limit_range = joint["upper_limit"] - joint["lower_limit"]
    limit_margin = max(1e-6, 0.005 * limit_range)
    saturated = any(
        min(abs(value - joint["lower_limit"]), abs(value - joint["upper_limit"])) <= limit_margin
        for value in positions
    )
    metrics = {
        "max_holdout_median_m": max(row.median_m for row in evaluations.values()),
        "max_holdout_p90_m": max(row.p90_m for row in evaluations.values()),
        "fit_cost": cost,
        "observed_joint_span": float(positions[1] - positions[0]),
        "joint_limit_saturated": float(saturated),
    }
    return ReferenceDepthAlignment(
        source_camera=source.camera_rig,
        source_depth=source.enhanced_depth,
        source_reference=source.reference_model,
        source_frame=camera["coordinate_frame"],
        target_frame=reference["coordinate_frame"],
        observed_joint=joint["name"],
        joint_type=joint["joint_type"],
        joint_axis=tuple(float(value) for value in joint["axis"]),
        joint_origin_m=tuple(float(value) for value in joint["origin_m"]),
        joint_limits=(float(joint["lower_limit"]), float(joint["upper_limit"])),
        state_order=streams,
        rotation=tuple(tuple(float(value) for value in row) for row in rotation),
        translation_m=tuple(float(value) for value in translation),
        evaluations=evaluations,
        foreground_fractions=fractions,
        train_frame_ids=tuple(sorted(train_ids)),
        holdout_frame_ids=tuple(sorted(test_ids)),
        metrics=metrics,
    )


def registration_documents(
    alignment: ReferenceDepthAlignment,
    artifacts: ArtifactStore,
    evidence: ArtifactRef,
    visual: ArtifactRef,
    prediction: ReconstructionPrediction,
    method: Mapping[str, Any],
    *,
    fit_stage: Literal["pre_reconstruction", "post_reconstruction"],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind one depth alignment to a reconstructed appearance without refitting it."""
    import numpy as np

    if not artifacts.verify(visual):
        raise DomainValidationError("registration visual bundle is unavailable")
    reference = read_json(artifacts, alignment.source_reference, "physrealx.usd-reference/1")
    joint = next(
        (item for item in reference["joints"] if item["name"] == alignment.observed_joint),
        None,
    )
    moving_groups = [name for name, part in prediction.parts.items() if part.kind != "fixed"]
    roots = [name for name, part in prediction.parts.items() if part.parent is None]
    if (
        joint is None
        or not prediction.segmented
        or len(prediction.parts) != 2
        or len(moving_groups) != 1
        or len(roots) != 1
    ):
        raise DomainValidationError(
            "appearance registration requires one reconstructed articulated motion group"
        )
    camera = read_json(artifacts, alignment.source_camera, "physrealx.camera-rig-calibration/1")
    if (
        camera.get("coordinate_frame") != alignment.source_frame
        or reference.get("coordinate_frame") != alignment.target_frame
    ):
        raise DomainValidationError("reference alignment coordinate frames differ from its sources")
    rotation = np.asarray(alignment.rotation, dtype=float)
    translation = np.asarray(alignment.translation_m, dtype=float)
    for stream in camera["streams"]:
        for component in stream["components"]:
            for image in component["images"]:
                pose = np.asarray(image["cam_from_world"], dtype=float)
                rotated = pose[:, :3] @ rotation.T
                image["cam_from_world"] = np.column_stack(
                    [rotated, pose[:, 3] - rotated @ translation]
                ).tolist()
            for track in component.get("fiducial_tracks", []):
                track["xyz"] = (np.asarray(track["xyz"]) @ rotation.T + translation).tolist()
    camera.update(
        coordinate_frame=alignment.target_frame,
        source_camera_artifact_id=alignment.source_camera.artifact_id,
        metric_alignment_evidence_artifact_id=evidence.artifact_id,
    )
    camera["coordinate_convention"]["frame"] = alignment.target_frame
    report = {
        "schema": "physrealx.metric-alignment-report/1",
        "status": "PASSED",
        "method": dict(method),
        "evidence_artifact_id": evidence.artifact_id,
        "evidence_scope": "model_conditioned_full_pose_registration",
        "fit_stage": fit_stage,
        "depth_correspondence": "masked_real_depth_to_supplied_usd_surfaces",
        "orientation_evidence": "capture_support_plane_and_training_surfaces",
        "source_frame": alignment.source_frame,
        "target_frame": alignment.target_frame,
        "source_camera": alignment.source_camera.model_dump(mode="json"),
        "source_depth": alignment.source_depth.model_dump(mode="json"),
        "source_reference": alignment.source_reference.model_dump(mode="json"),
        "source_visual": visual.model_dump(mode="json"),
        "rotation": alignment.rotation,
        "translation_m": alignment.translation_m,
        "observed_joint": alignment.observed_joint,
        "joint_type": alignment.joint_type,
        "joint_axis": alignment.joint_axis,
        "joint_origin_m": alignment.joint_origin_m,
        "joint_limits": alignment.joint_limits,
        "state_order": alignment.state_order,
        "evaluations": {
            name: value.model_dump(mode="json") for name, value in alignment.evaluations.items()
        },
        "foreground_fractions": alignment.foreground_fractions,
        "train_frame_ids": alignment.train_frame_ids,
        "holdout_frame_ids": alignment.holdout_frame_ids,
        "metrics": alignment.metrics,
        "motion_groups": {
            moving_groups[0]: [joint["child"]],
            roots[0]: [
                link["name"] for link in reference["links"] if link["name"] != joint["child"]
            ],
        },
        "unobserved_joint_motion": [
            item["name"] for item in reference["joints"] if item["name"] != joint["name"]
        ],
        "gates": {
            "held_out_views": True,
            "independent_metric_truth": False,
            "reference_guided_training": fit_stage == "pre_reconstruction",
        },
    }
    return camera, report


class ReferenceRegistrationAdapter:
    descriptor = builtin_method(
        MethodCapability.CAMERA_ALIGN_METRIC,
        "articulated_surface_icp",
        input_contract="CameraRigCalibration/1+MetricAlignmentEvidence/1",
        output_contract="CameraRigCalibration/1+MetricAlignmentReport/1",
        python=">=3.10,<3.13",
        version="1.6.0",
    )

    def probe(self) -> MethodProbe:
        from physrealx.integrations.rgbd_fusion import RGBDFusionAdapter

        return RGBDFusionAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        config = RegistrationConfig.model_validate(spec.method.config)
        evidence = input_artifact(spec, "evidence", artifacts)
        request = ReferenceRegistrationInput.model_validate(
            read_json(artifacts, evidence, "physrealx.reference-registration-input/1")
        )
        camera_ref = input_artifact(spec, "camera_rig", artifacts)
        if request.rgbd.camera_rig != camera_ref:
            raise DomainValidationError("reference registration is bound to another camera")
        reconstruction = read_json(
            artifacts, request.reconstruction_report, "physrealx.geometry-reconstruction-report/1"
        )
        if (
            reconstruction.get("source_camera_artifact_id") != camera_ref.artifact_id
            or reconstruction.get("source_depth_artifact_id")
            != request.rgbd.enhanced_depth.artifact_id
            or reconstruction.get("evaluation_scope") != "held_out_views"
        ):
            raise DomainValidationError(
                "registration requires the matching held-out reconstruction"
            )
        visual = ArtifactRef.model_validate(reconstruction["bundle"])
        prediction = ReconstructionPrediction.model_validate(reconstruction.get("prediction"))
        stored_alignment = reconstruction.get("reference_depth_alignment")
        if stored_alignment is None:
            alignment = fit_reference_depth_alignment(
                request.rgbd,
                artifacts,
                reconstruction.get("training_frame_ids"),
                reconstruction.get("validation_frame_ids"),
                config,
            )
        else:
            alignment = ReferenceDepthAlignment.model_validate(stored_alignment)
            if (
                alignment.source_camera != camera_ref
                or alignment.source_depth != request.rgbd.enhanced_depth
                or alignment.source_reference != request.rgbd.reference_model
                or alignment.observed_joint != config.joint
                or tuple(alignment.train_frame_ids)
                != tuple(sorted(reconstruction.get("training_frame_ids", ())))
                or tuple(alignment.holdout_frame_ids)
                != tuple(sorted(reconstruction.get("validation_frame_ids", ())))
            ):
                raise DomainValidationError(
                    "stored reference depth alignment differs from reconstruction inputs"
                )
        camera, report = registration_documents(
            alignment,
            artifacts,
            evidence,
            visual,
            prediction,
            self.descriptor.model_dump(mode="json"),
            fit_stage=("post_reconstruction" if stored_alignment is None else "pre_reconstruction"),
        )
        return publish(
            spec,
            artifacts,
            {"camera_rig_calibration": camera, "metric_alignment_report": report},
            metrics=dict(alignment.metrics),
        )


def load_registration(store: ArtifactStore, reference: ArtifactRef) -> dict[str, Any]:
    """Validate the portable appearance binding before assembly or rendering."""
    import numpy as np

    report = read_json(store, reference, "physrealx.metric-alignment-report/1")
    rotation, translation = (
        np.asarray(report.get("rotation")),
        np.asarray(report.get("translation_m")),
    )
    if (
        report.get("status") != "PASSED"
        or report.get("evidence_scope") != "model_conditioned_full_pose_registration"
        or report.get("target_frame") != "reference_z_up"
        or rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise DomainValidationError(
            "registered appearance requires a finite proper rigid transform"
        )
    for key in ("source_camera", "source_reference", "source_visual"):
        if not store.verify(ArtifactRef.model_validate(report[key])):
            raise DomainValidationError("registered appearance source is unavailable or modified")
    model = read_json(
        store, ArtifactRef.model_validate(report["source_reference"]), "physrealx.usd-reference/1"
    )
    covered = [link for values in report["motion_groups"].values() for link in values]
    if len(set(covered)) != len(covered) or set(covered) != {
        link["name"] for link in model["links"]
    }:
        raise DomainValidationError("appearance motion groups must cover each reference link once")
    if set(report["train_frame_ids"]) & set(report["holdout_frame_ids"]):
        raise DomainValidationError("appearance registration has overlapping evaluation views")
    if len(report["state_order"]) != 2 or set(report["state_order"]) != set(report["evaluations"]):
        raise DomainValidationError("appearance registration needs two ordered observed states")
    return report
