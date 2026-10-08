"""Coupled bounded surface/hinge fit with hard multi-pose contact projection.

Closest triangle correspondences are refreshed outside autograd. Every inner
step optimizes both links and the same hinge against video and swept contact
constraints. Check frames never enter this objective. No closing faces are invented.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from physrealx.reconstruction import joint_transform

from ._part_remeshing import (
    refine_observed_crossings,
    restricted_delaunay,
    retriangulate_observed_states,
)
from ._surface_clearance import conforming_refine

SOLVER_IDENTITY = hashlib.sha256(
    b"".join(
        Path(__file__).with_name(name).read_bytes()
        for name in (
            "_joint_surface_solver.py",
            "_triangle_contacts.py",
            "_surface_untangling.py",
            "_warp_untangling.py",
            "_surface_clearance.py",
            "_video_render_loss.py",
            "_part_remeshing.py",
            "_restricted_delaunay.cpp",
            "_state_surfaces.py",
            "_contact_projection.py",
            "_mesh_validity.py",
            "_warp_surface_query.py",
            "_surface_query.py",
            "video_articulation.py",
        )
    )
).hexdigest()


def _candidate_rank(
    intersects: bool,
    gaps: Any,
    weights: Any,
    fit: float,
    minimum: float,
    *,
    intersection_count: int = 1,
) -> tuple[bool, float, float, float]:
    """Rank infeasible candidates by depth before tessellation-dependent pair count."""
    deficits = np.maximum(0.0, minimum - gaps)
    if intersects:
        return True, float(deficits.max(initial=0)), float(intersection_count), fit
    return intersects, float(deficits.max(initial=0)), float(deficits @ weights), fit


def _observed_separation_gradients(separation: Any, observation: Any, params: Any) -> None:
    """Use video motion along directions that do not oppose unfolding.

    The separation energy must decrease while crossing triangles are opened.
    Orthogonalizing each parameter group's conflicting observation gradient
    keeps the fit evidence active during that phase without cancelling the
    nonlocal unfolding direction. Nonlinear acceptance still checks both.
    """
    import torch

    separating = torch.autograd.grad(
        separation, params, retain_graph=True, allow_unused=True, materialize_grads=True
    )
    observed = torch.autograd.grad(observation, params, allow_unused=True, materialize_grads=True)
    for parameter, separating_gradient, observed_gradient in zip(
        params, separating, observed, strict=True
    ):
        magnitude = separating_gradient.norm()
        if float(magnitude) > 1e-20:
            opposing = torch.minimum(
                torch.dot(separating_gradient.flatten(), observed_gradient.flatten()),
                separating_gradient.new_zeros(()),
            )
            observed_gradient = observed_gradient - opposing * separating_gradient / (
                magnitude.square() + 1e-30
            )
            observed_gradient = observed_gradient * torch.clamp(
                magnitude / observed_gradient.norm().clamp_min(1e-30), max=1
            )
        parameter.grad = separating_gradient + observed_gradient


def solve_joint_surfaces(
    meshes: dict[str, Any],
    observations: dict[str, Any],
    motion: dict[str, Any],
    config: Any,
    query: Any,
    checkpoint: Any = None,
    resume: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Own workers for the whole solve; spawn never inherits a CUDA context."""
    with ExitStack() as resources:
        workers = min(config.cpu_query_workers or 4, config.clearance_samples)
        pool = resources.enter_context(
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_contact_worker_init,
            )
        )
        return _solve_joint_surfaces(
            meshes, observations, motion, config, query, checkpoint, resume, pool, resources
        )


def _contact_worker_init() -> None:
    # Native NumPy/BLAS work is already distributed by pose. Nested thread teams
    # would oversubscribe the host and starve both CUDA submission threads.
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    threadpool_limits(limits=1)


def _restoration_contacts(constraints: dict[str, Any], per_pose: int) -> dict[str, Any]:
    """Bound the LP active set; retain all contacts for energy and final sweep.

    Select the deepest crossing on distinct source/target faces first. Every
    previously separated near pair stays in the hard projection. Re-querying
    after each update adds new witnesses without hiding any from validation.
    """
    if per_pose < 1:
        raise ValueError("restoration contact budget must be positive")
    poses, gaps = constraints["positions"], constraints["gaps"]
    chosen: list[int] = []
    for q in np.unique(poses):
        ids = np.flatnonzero(poses == q)
        bad = ids[gaps[ids] < 0]
        ordered = bad[np.argsort(gaps[bad], kind="stable")]
        if len(ordered) > per_pose:
            left, right = constraints["parent_faces"], constraints["child_faces"]
            priority: list[int] = []
            selected_ids: set[int] = set()
            seen_left: set[tuple[int, ...]] = set()
            seen_right: set[tuple[int, ...]] = set()
            for coverage in (2, 1, 0):
                for index in ordered:
                    a, b = tuple(left[index]), tuple(right[index])
                    novelty = int(a not in seen_left) + int(b not in seen_right)
                    if novelty >= coverage and int(index) not in selected_ids:
                        priority.append(int(index))
                        selected_ids.add(int(index))
                        seen_left.add(a)
                        seen_right.add(b)
                    if len(priority) == per_pose:
                        break
                if len(priority) == per_pose:
                    break
            ordered = np.asarray(priority[:per_pose], dtype=int)
        chosen.extend(map(int, ordered))
        chosen.extend(map(int, ids[gaps[ids] >= 0]))
    selection = np.asarray(chosen, dtype=int)
    return {
        key: np.asarray(constraints[key])[selection]
        for key in ("parent_faces", "child_faces", "normals", "positions", "gaps")
    }


def _solve_joint_surfaces(
    meshes: Any,
    observations: Any,
    motion: Any,
    config: Any,
    query: Any,
    checkpoint: Any,
    resume: Any,
    pool: Any,
    resources: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import ipctk
    import torch

    started = monotonic()

    from ._mesh_validity import (
        collision_mesh,
        deformation_step,
        limit_collision_nodes,
        limit_triangle_area_nodes,
        self_intersections,
        surface_topology,
    )

    ipctk.set_num_threads(config.cpu_query_workers or 4)

    device = query.describe()["devices"][0] if query.describe()["accelerated"] else "cpu"
    if device == "cpu":
        original_threads = torch.get_num_threads()
        torch.set_num_threads(min(config.cpu_query_workers or 4, 8))
        resources.callback(torch.set_num_threads, original_threads)
    dtype = torch.float64

    def tensor(x: Any, *, integer: bool = False) -> Any:
        return torch.as_tensor(np.array(x), dtype=torch.long if integer else dtype, device=device)

    axis0, origin0 = np.array(motion["axis"]), np.array(motion["origin_m"])
    axis0 = axis0 / np.linalg.norm(axis0)
    da: Any = torch.nn.Parameter(tensor(np.zeros(3)))
    do: Any = torch.nn.Parameter(tensor(np.zeros(3)))
    params = [da, do]
    ipc_meshes = {}
    graphs, refined, remesh = {}, {}, {}
    from ._triangle_contacts import swept_constraints as query_constraints

    swept_constraints = partial(query_constraints, executor=pool)

    affected = swept_constraints(
        meshes[motion["parent"]],
        meshes[motion["child"]],
        motion,
        np.linspace(motion["range"][0], motion["range"][1], min(17, config.clearance_samples)),
        margin=2 * config.max_surface_displacement_m + config.minimum_clearance_m,
        budget=config.active_constraints_per_pose,
        workers=config.cpu_query_workers or 4,
    )
    for name, original in meshes.items():
        logging.getLogger(__name__).info("Restricted Delaunay: %s", name)
        rebuilt, topology = restricted_delaunay(original, query)
        logging.getLogger(__name__).info("Restricted Delaunay result: %s", topology)
        mesh, report = conforming_refine(
            rebuilt,
            axis0,
            origin0,
            config.hinge_band_m,
            config.remesh_edge_m,
            config.remesh_rounds,
        )
        report["restricted_delaunay"] = topology
        evidence = []
        for frame in observations[name]["fit"]:
            M = (
                joint_transform("hinge", axis0, origin0, frame.position)
                if name == motion["child"]
                else np.eye(4)
            )
            evidence.append(trimesh.transform_points(frame.points_reference, np.linalg.inv(M)))
        face_key = "child_faces" if name == motion["child"] else "parent_faces"
        crossing_faces = affected[face_key][affected["gaps"] <= 0]
        crossing_points = (
            np.asarray(original.vertices)[np.unique(crossing_faces)]
            if len(crossing_faces)
            else np.empty((0, 3))
        )
        sampling_edge = max(4 * config.minimum_clearance_m, config.remesh_edge_m / 3)
        mesh, video_sampling = refine_observed_crossings(
            mesh,
            evidence,
            crossing_points,
            query,
            edge_length_m=sampling_edge,
            support_radius_m=min(config.surface_support_m, 3 * sampling_edge),
        )
        report["video_sampling"] = video_sampling
        refined[name], remesh[name] = mesh, report
        if not surface_topology(mesh)["valid"]:
            raise ValueError(f"link {name} has invalid topology after remeshing")
        if self_intersections(mesh):
            raise ValueError(f"source link {name} contains self intersections after remeshing")
        ipc_meshes[name] = collision_mesh(mesh)
        v = np.array(mesh.vertices)
        points = np.concatenate(evidence) if evidence else np.empty((0, 3))
        delta = v - origin0
        radial = np.linalg.norm(delta - np.outer(delta @ axis0, axis0), axis=1)
        support_distance, support_ids = (
            cKDTree(points).query(v)
            if len(points)
            else (np.full(len(v), np.inf), np.zeros(len(v), dtype=int))
        )
        measured = (
            np.sum((points[support_ids] - v) * mesh.vertex_normals, axis=1)
            if len(points)
            else np.zeros(len(v))
        )
        observation_interval = np.column_stack(
            [
                measured - abs(measured) - config.max_check_degradation_m,
                measured + abs(measured) + config.max_check_degradation_m,
            ]
        )
        support_taper = np.clip(
            (config.surface_correspondence_distance_m - support_distance)
            / config.surface_support_m,
            0,
            1,
        )
        taper = (
            np.clip((config.hinge_band_m - radial) / config.surface_support_m, 0, 1) * support_taper
        )
        key = "affected_child_vertices" if name == motion["child"] else "affected_parent_vertices"
        conflict_vertices = np.asarray(original.vertices)[affected[key]]
        if len(conflict_vertices):
            distance = cKDTree(conflict_vertices).query(v)[0]
            conflict_taper = np.clip(2 - distance / config.surface_support_m, 0, 1)
            # Every swept conflict has actual degrees of freedom, including
            # occluded source faces. Evidence gates still bound the final change.
            taper = np.maximum(taper, conflict_taper)
        editable = taper > 0
        # Topological vertex degrees of freedom keep nearby folded sheets
        # independent. Euclidean lattice nodes incorrectly weld those sheets.
        displacement = torch.nn.Parameter(tensor(np.zeros((len(v), 3))))
        params.append(displacement)
        graphs[name] = dict(
            base=tensor(v),
            faces=tensor(mesh.faces, integer=True),
            idx=tensor(np.arange(len(v))[:, None], integer=True),
            weights=tensor(editable[:, None].astype(float)),
            delta=displacement,
            edges=tensor(mesh.edges_unique, integer=True),
            editable=editable,
            taper=taper,
            observed=support_distance < 0.003,
            observation_normal=np.asarray(mesh.vertex_normals).copy(),
            observation_interval=observation_interval,
            support_points=points[support_ids] if len(points) else v.copy(),
        )

    def state() -> tuple[dict[str, Any], Any, Any]:
        a = tensor(axis0) + da
        a = a / torch.linalg.vector_norm(a)
        o = tensor(origin0) + do
        verts = {
            name: g["base"] + (g["delta"][g["idx"]] * g["weights"][..., None]).sum(dim=1)
            for name, g in graphs.items()
        }
        return verts, a, o

    def observation_violation(vertices: dict[str, Any]) -> float:
        maximum = 0.0
        for name, graph in graphs.items():
            seen = tensor(graph["observed"], integer=True).bool()
            if not bool(seen.any()):
                continue
            distance = (
                (vertices[name][seen] - graph["base"][seen])
                * tensor(graph["observation_normal"])[seen]
            ).sum(1)
            interval = tensor(graph["observation_interval"])[seen]
            maximum = max(
                maximum,
                float((interval[:, 0] - distance).max()),
                float((distance - interval[:, 1]).max()),
            )
        return maximum

    def transform(points: Any, q: Any, axis: Any, origin: Any) -> Any:
        q = torch.as_tensor(q, dtype=dtype, device=points.device)
        if q.ndim:
            q = q[:, None]
        d = points - origin
        return (
            origin
            + d * torch.cos(q)
            + torch.linalg.cross(axis.expand_as(d), d) * torch.sin(q)
            + (d * axis).sum(dim=-1, keepdim=True) * axis * (1 - torch.cos(q))
        )

    def snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
        with torch.no_grad():
            verts, a, o = state()
            output = {name: refined[name].copy() for name in refined}
            for name, m in output.items():
                m.vertices = verts[name].cpu().numpy()
            return output, {**motion, "axis": a.cpu().tolist(), "origin_m": o.cpu().tolist()}

    def bind(target: Any, points: Any) -> tuple[Any, Any, Any]:
        dist, ids = query.nearest(target, points)
        if not np.isfinite(dist).all() or np.any(ids < 0):
            raise ValueError("invalid surface correspondence")
        triangles = target.triangles[ids]
        closest = trimesh.triangles.closest_point(triangles, points)  # type: ignore[no-untyped-call]
        bary = trimesh.triangles.points_to_barycentric(triangles, closest)  # type: ignore[no-untyped-call]
        return dist, target.faces[ids], bary

    def barypoint(v: Any, faces: Any, weights: Any) -> Any:
        return (v[faces] * weights[..., None]).sum(dim=1)

    lever_arm = max(
        float(np.linalg.norm(m.vertices - origin0, axis=1).max()) for m in refined.values()
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [da], "lr": config.solver_learning_rate / max(lever_arm, 0.05)},
            {"params": params[1:], "lr": config.solver_learning_rate},
        ]
    )
    from ._contact_projection import ContactProjection
    from ._state_surfaces import MotionConsistency, reconstruct_states
    from ._surface_untangling import SurfaceUntangling
    from ._video_render_loss import VideoRenderLoss

    logging.getLogger(__name__).info("Independent state reconstruction")
    state_meshes, reconstruction_report = reconstruct_states(
        refined, observations, motion, config, query, executor=pool
    )
    if affected["intersecting_pairs"]:
        # The first rDel pass regularizes the module-one source. A second,
        # strictly guarded pass lets fitted VIDEO states propose connectivity
        # where their shared surface supports a different local triangulation.
        # All poses keep one vertex identity and one topology.
        for name, graph in graphs.items():
            canonical = []
            for q, fitted_state in state_meshes.items():
                part = fitted_state[name].copy()
                if name == motion["child"]:
                    part.apply_transform(joint_transform("hinge", axis0, origin0, -q))
                canonical.append(part)
            face_key = "child_faces" if name == motion["child"] else "parent_faces"
            crossing_faces = affected[face_key][affected["gaps"] <= 0]
            focus_points = (
                np.asarray(meshes[name].vertices)[np.unique(crossing_faces)]
                if len(crossing_faces)
                else np.empty((0, 3))
            )
            changed, _states, observed_topology = retriangulate_observed_states(
                refined[name],
                canonical,
                query,
                focus_points=focus_points,
                local_tolerance=config.remesh_edge_m,
            )
            remesh[name]["video_topology"] = observed_topology
            logging.getLogger(__name__).info(
                "Video-supported topology %s: %s (%s)",
                name,
                observed_topology["status"],
                observed_topology.get("reason", observed_topology.get("region", "accepted")),
            )
            if observed_topology["status"] != "APPLIED":
                continue
            refined[name] = changed
            remesh[name]["faces"] = len(changed.faces)
            for fitted_state in state_meshes.values():
                fitted_state[name].faces = np.asarray(changed.faces).copy()
            ipc_meshes[name] = collision_mesh(changed)
            graph["faces"] = tensor(changed.faces, integer=True)
            graph["edges"] = tensor(changed.edges_unique, integer=True)
            normals = np.asarray(changed.vertex_normals).copy()
            graph["observation_normal"] = normals
            measured = np.sum((graph["support_points"] - changed.vertices) * normals, axis=1)
            graph["observation_interval"] = np.column_stack(
                [
                    measured - abs(measured) - config.max_check_degradation_m,
                    measured + abs(measured) + config.max_check_degradation_m,
                ]
            )
    for name, graph in graphs.items():
        canonical = []
        for q, observed_state in state_meshes.items():
            matrix = (
                joint_transform("hinge", axis0, origin0, -q)
                if name == motion["child"]
                else np.eye(4)
            )
            canonical.append(trimesh.transform_points(observed_state[name].vertices, matrix))
        graph["observed_reference"] = (
            tensor(np.mean(canonical, axis=0)) - graph["base"]
            if affected["intersecting_pairs"]
            else torch.zeros_like(graph["base"])
        )
        graph["observed_reference"] *= tensor(graph["editable"][:, None])
    motion_consistency = MotionConsistency(state_meshes, motion, config, query, str(device))
    contact_projection = ContactProjection(motion, refined, graphs, config)
    untangling = SurfaceUntangling(refined, motion, config, query.describe()["devices"])
    resources.callback(untangling.close)
    rendering = VideoRenderLoss(
        refined,
        observations,
        str(device),
        motion,
        state_meshes[max(state_meshes)],
        max(state_meshes),
    )
    history = []
    best = None
    best_score = (np.inf, np.inf, np.inf, np.inf)
    positions = set(np.linspace(motion["range"][0], motion["range"][1], config.clearance_samples))
    start = 0

    def training_error(candidate: Any, fitted: Any) -> float:
        """Fresh fit-only correspondences; never rank a post-step mesh by stale loss."""
        losses = []
        for name, partitions in observations.items():
            for frame in partitions["fit"]:
                matrix = joint_transform(
                    "hinge", fitted["axis"], fitted["origin_m"], frame.position
                )
                if name == "__object__":
                    child_mesh = candidate[motion["child"]].copy()
                    child_mesh.apply_transform(matrix)
                    target = trimesh.util.concatenate([candidate[motion["parent"]], child_mesh])
                    points = frame.points_reference
                else:
                    target = candidate[name]
                    points = frame.points_reference
                    if name == motion["child"]:
                        points = trimesh.transform_points(points, np.linalg.inv(matrix))
                distance, _ = query.nearest(target, points)
                residual = distance / 0.001
                # No dropping distant points: losing observation coverage must
                # never improve the selection score.
                losses.append(
                    np.where(residual < 3, 0.5 * residual**2, 3 * (residual - 1.5)).mean()
                )
        return float(np.mean(losses))

    def candidate_score(candidate: Any, fitted: Any, checked: Any = None) -> tuple[Any, ...]:
        if checked is None:
            checked = swept_constraints(
                candidate[motion["parent"]],
                candidate[motion["child"]],
                fitted,
                sorted(positions),
                margin=config.minimum_clearance_m + config.contact_reserve_m,
                budget=config.active_constraints_per_pose,
                workers=config.cpu_query_workers or 4,
            )
        intersects = bool(checked["intersecting_pairs"])
        gaps = checked["gaps"]
        weights = np.ones(len(gaps))
        # An old Gaussian field can decrease while new faces start crossing.
        # Rank from freshly queried geometry instead of the old energy.
        return _candidate_rank(
            intersects,
            gaps,
            weights,
            training_error(candidate, fitted),
            config.minimum_clearance_m,
            intersection_count=checked["intersecting_pairs"],
        )

    if resume is not None:
        initialize_only = resume.get("initialize_only", False)
        if not initialize_only and resume["solver_identity"] != SOLVER_IDENTITY:
            raise ValueError("checkpoint solver implementation differs")
        if initialize_only:
            if set(resume.get("mesh_faces", {})) != set(refined) or any(
                not np.array_equal(resume["mesh_faces"][name], mesh.faces)
                for name, mesh in refined.items()
            ):
                raise ValueError("checkpoint surface topology differs")
            best = [p.detach().clone() for p in params]
            best_score = candidate_score(*snapshot())
        with torch.no_grad():
            for p, value in zip(params, resume["parameters"], strict=True):
                v = tensor(value)
                if v.shape != p.shape or not torch.isfinite(v).all():
                    raise ValueError("checkpoint deformation graph differs")
                p.copy_(v)
        if not initialize_only:
            for p, saved in zip(params, resume["adam"], strict=True):
                if saved:
                    optimizer.state[p] = {
                        k: torch.tensor(v) if k == "step" else tensor(v) for k, v in saved.items()
                    }
            for af, bf, normal, q in resume["contact_planes"]:
                contact_projection.rows[(q, *af, *bf)] = (
                    np.array(af),
                    np.array(bf),
                    np.array(normal),
                    q,
                )
            for key, direction in zip(
                contact_projection.rows, resume["contact_directions"], strict=True
            ):
                contact_projection.branch_directions[key] = np.asarray(direction)
            contact_projection.coherent_keys = {
                tuple(key) for key in resume["coherent_contact_keys"]
            }
            contact_projection.hard_safe_keys = {
                tuple(key) for key in resume["hard_safe_contact_keys"]
            }
            if not contact_projection.hard_safe_keys <= contact_projection.rows.keys():
                raise ValueError("checkpoint protected contacts differ from stored planes")
            for name, a, b, normal, gap in resume["self_contact_planes"]:
                contact_projection.add_self_contacts(
                    name, [(np.array(a), np.array(b), np.array(normal), gap)]
                )
            for name, values in resume["local_trust_radii_m"].items():
                radii = np.asarray(values, dtype=float)
                if (
                    radii.shape != contact_projection.local_radii[name].shape
                    or not np.isfinite(radii).all()
                    or np.any((radii <= 0) | (radii > 0.0005))
                ):
                    raise ValueError("checkpoint local trust radii are invalid")
                contact_projection.local_radii[name] = radii
            history = resume["history"]
            start = int(resume["iteration"]) // config.correspondence_steps
        positions.update(resume["positions"])
        if not 0 <= start < config.solver_iterations // config.correspondence_steps:
            raise ValueError("checkpoint iteration is outside the requested solve")
        if any(not motion["range"][0] <= q <= motion["range"][1] for q in positions):
            raise ValueError("checkpoint contains a pose outside the source range")
        limits = [np.tan(np.radians(config.max_axis_change_deg)), config.max_origin_displacement_m]
        limits.extend([config.max_surface_displacement_m] * len(graphs))
        saved_best = None if initialize_only else resume.get("best_parameters")
        states = [resume["parameters"]] + ([saved_best] if saved_best is not None else [])
        for saved_state in states:
            for p, value, limit in zip(params, saved_state, limits, strict=True):
                array = np.asarray(value)
                if (
                    array.shape != tuple(p.shape)
                    or not np.isfinite(array).all()
                    or np.linalg.norm(array, axis=-1).max() > limit + 1e-9
                ):
                    raise ValueError("checkpoint violates the source trust region")
        if any(self_intersections(m) for m in snapshot()[0].values()):
            raise ValueError("checkpoint contains an intersecting link")
        if initialize_only and (
            set(resume.get("mesh_vertices", {})) != set(refined)
            or any(
                np.asarray(resume["mesh_vertices"][name]).shape != mesh.vertices.shape
                or not np.allclose(resume["mesh_vertices"][name], mesh.vertices, rtol=0, atol=1e-7)
                for name, mesh in snapshot()[0].items()
            )
        ):
            raise ValueError("checkpoint geometry and parameters differ")
        if initialize_only and resume.get("contact_planes"):
            from ._triangle_contacts import separating_axes

            candidate, fitted = snapshot()
            af = np.array([r[0] for r in resume["contact_planes"]], dtype=int)
            bf = np.array([r[1] for r in resume["contact_planes"]], dtype=int)
            qs = np.array([r[3] for r in resume["contact_planes"]], dtype=float)
            parent, child = motion["parent"], motion["child"]
            if (
                af.shape != bf.shape
                or af.shape[1:] != (3,)
                or np.any(af < 0)
                or np.any(af >= len(candidate[parent].vertices))
                or np.any(bf < 0)
                or np.any(bf >= len(candidate[child].vertices))
                or not np.isfinite(qs).all()
                or np.any(qs < motion["range"][0])
                or np.any(qs > motion["range"][1])
            ):
                raise ValueError(
                    "checkpoint contact witnesses differ from source topology or range"
                )
            gaps, normals = np.empty(len(qs)), np.empty((len(qs), 3))
            for q in np.unique(qs):
                ids = qs == q
                M = joint_transform("hinge", fitted["axis"], fitted["origin_m"], q)
                left = candidate[parent].vertices[af[ids]]
                right = candidate[child].vertices[bf[ids]] @ M[:3, :3].T + M[:3, 3]
                gaps[ids], normals[ids] = separating_axes(left, right)
            # Reuse discovered feature pairs as geometric witnesses, but rebuild
            # their directions and bounds from the verified current surface.
            contact_projection.add(
                dict(parent_faces=af, child_faces=bf, positions=qs, gaps=gaps, normals=normals),
                candidate,
                fitted,
            )
        if saved_best is not None:
            best = [tensor(value) for value in saved_best]
            best_score = tuple(resume["best_score"])
    initialization_report: dict[str, Any] = {"applied": False}
    if resume is None and affected["intersecting_pairs"]:
        # Independently reconstructed states must actually initialize a shared
        # observed surface, not merely appear as another weak loss beside the
        # intersecting source. Transport every fit state to the SAME canonical
        # frame; do not create independent per-pose geometry.
        original_error = training_error(*snapshot())
        with torch.no_grad():
            for name, graph in graphs.items():
                proposal = graph["observed_reference"].clone()
                seen = tensor(graph["observed"], integer=True).bool()
                normals = tensor(graph["observation_normal"])
                bounds = tensor(graph["observation_interval"])
                distance = (proposal * normals).sum(1)
                clipped = torch.maximum(bounds[:, 0], torch.minimum(bounds[:, 1], distance))
                proposal[seen] += ((clipped - distance)[:, None] * normals)[seen]
                proposal *= torch.clamp(
                    config.max_surface_displacement_m
                    / torch.linalg.vector_norm(proposal, dim=1, keepdim=True).clamp_min(1e-12),
                    max=1,
                )
                proposal[~tensor(graph["editable"], integer=True).bool()] = 0
                old = graph["delta"].detach().clone()
                graph["delta"].copy_(proposal)
                limit_triangle_area_nodes(graph, old)
                limit_collision_nodes(ipc_meshes[name], graph, old, executor=pool)
                limit_triangle_area_nodes(graph, old)
                before = graph["base"].cpu().numpy()
                after = (graph["base"] + graph["delta"]).cpu().numpy()
                graph["delta"] *= deformation_step(ipc_meshes[name], before, after, executor=pool)
            initialized_error = training_error(*snapshot())
            accepted = (
                initialized_error <= original_error and observation_violation(state()[0]) <= 2e-9
            )
            if not accepted:
                for graph in graphs.values():
                    graph["delta"].zero_()
            initialization_report = {
                "applied": accepted,
                "source_training_error": original_error,
                "reconstructed_training_error": initialized_error,
                "source": "fit_states_transported_to_shared_canonical_surface",
            }
    if best is None:
        best = [p.detach().clone() for p in params]
        best_score = candidate_score(*snapshot())
    stop_reason = "iteration_limit"
    preparation_seconds = monotonic() - started
    logging.getLogger(__name__).info("Joint solve preparation: %.3f s", preparation_seconds)
    for outer in range(start, config.solver_iterations // config.correspondence_steps):
        outer_started = monotonic()
        # Penalty continuation makes local separation dominate once the coherent
        # unfolding force has opened a route. A fixed finite soft penalty can
        # otherwise settle at a crossing equilibrium. Final clearance stays hard.
        untangling.local_weight = 2.0 ** min(10, outer // 4)
        # A stalled restoration is diagnostic, never a feasibility certificate.
        # Only stop after repeated unchanged active sets and sub-micron updates;
        # the independent full-range and observation gates still run afterwards.
        recent = history[-3:]
        if (
            len(recent) == 3
            and all(r["phase"] == "fit_with_separation" for r in recent)
            and len({(r["active_poses"], r["contact_pairs"]) for r in recent}) == 1
            and all(r["maximum_motion_step_m"] < 1e-7 for r in recent)
        ):
            stop_reason = "stationary_updates_require_independent_validation"
            break
        current, fitted_motion = snapshot()
        motion_consistency.bind(current, fitted_motion)
        a = np.array(fitted_motion["axis"])
        o = np.array(fitted_motion["origin_m"])
        data = []
        for name, mesh in current.items():
            for frame in observations[name]["fit"]:
                M = (
                    joint_transform("hinge", a, o, frame.position)
                    if name == motion["child"]
                    else np.eye(4)
                )
                local = trimesh.transform_points(frame.points_reference, np.linalg.inv(M))
                distances, faces, bary = bind(mesh, local)
                keep = distances < config.joint_correspondence_distance_m
                if np.any(keep):
                    data.append(
                        (
                            name,
                            tensor(faces[keep], integer=True),
                            tensor(bary[keep]),
                            tensor(frame.points_reference[keep]),
                            frame.position,
                        )
                    )
                # Reverse term only on observed faces, not on unseen backsides.
                unique = np.unique(faces[keep].reshape(-1))
                if len(unique):
                    d, ids = cKDTree(local).query(mesh.vertices[unique])
                    valid = d < min(config.surface_correspondence_distance_m, 0.002)
                    ids, unique = ids[valid], unique[valid]
                    if len(unique):
                        data.append(
                            (
                                name,
                                tensor(np.repeat(unique[:, None], 3, axis=1), integer=True),
                                tensor(np.tile([1.0, 0, 0], (len(unique), 1))),
                                tensor(frame.points_reference[ids]),
                                frame.position,
                            )
                        )
        for frame in observations.get("__object__", {}).get("fit", []):
            posed = current[motion["child"]].copy()
            posed.apply_transform(joint_transform("hinge", a, o, frame.position))
            combined = trimesh.util.concatenate([current[motion["parent"]], posed])
            distances, faces, bary = bind(combined, frame.points_reference)
            keep = distances < config.joint_correspondence_distance_m
            if np.any(keep):
                data.append(
                    (
                        "__object__",
                        tensor(faces[keep], integer=True),
                        tensor(bary[keep]),
                        tensor(frame.points_reference[keep]),
                        frame.position,
                    )
                )
        parent, child = motion["parent"], motion["child"]
        constraints = swept_constraints(
            current[parent],
            current[child],
            fitted_motion,
            sorted(positions),
            margin=config.minimum_clearance_m + config.contact_reserve_m + 0.001,
            budget=config.active_constraints_per_pose,
            workers=config.cpu_query_workers or 4,
        )
        restoring = bool(constraints["intersecting_pairs"]) or bool(
            len(constraints["gaps"]) and constraints["gaps"].min() < config.minimum_clearance_m
        )
        if restoring:
            untangling.bind(current, fitted_motion, constraints, sorted(positions))
            contact_projection.add(
                _restoration_contacts(constraints, config.active_constraints_per_pose),
                current,
                fitted_motion,
            )
        else:
            if history and history[-1]["phase"] == "restore_feasibility":
                optimizer.state.clear()
            contact_projection.add(constraints, current, fitted_motion)
            untangling.jobs = []
        # Evaluate nonlinear progress against the SAME coherent branch used by
        # the feasibility projection. Raw SAT normals may point through a
        # crossed sheet and falsely reject a valid unfolding direction. The
        # complete raw crossing set still drives the nonlocal objective and
        # independent full-range geometry check.
        fa, fb, normals, angles = contact_projection.arrays()
        contact_a, contact_b = tensor(fa, integer=True), tensor(fb, integer=True)
        contact_n, contact_q = tensor(normals), tensor(angles)
        contact_buffers = {"active": (contact_a, contact_b, contact_n, contact_q)}
        starting_intersections = constraints["intersecting_pairs"]
        required = config.minimum_clearance_m + config.contact_reserve_m
        hard_required = config.minimum_clearance_m

        def contact_gaps(
            v: Any,
            axis: Any,
            origin: Any,
            contacts: Any = contact_buffers,
            links: Any = (parent, child),
        ) -> Any:
            faces_a, faces_b, normals, angles = contacts["active"]
            left = v[links[0]][faces_a]
            right = transform(
                v[links[1]][faces_b].reshape(-1, 3),
                angles.repeat_interleave(3),
                axis,
                origin,
            ).reshape(-1, 3, 3)
            return (left * normals[:, None]).sum(2).amin(1) - (right * normals[:, None]).sum(
                2
            ).amax(1)

        def video_terms(
            v: Any,
            axis: Any,
            origin: Any,
            iteration: int,
            fit_data: Any = data,
            fixed_parent: str = parent,
            fixed_child: str = child,
            recovery_phase: bool = restoring,
        ) -> tuple[Any, ...]:
            losses = []
            for name, faces, bary, points, q in fit_data:
                values = (
                    torch.cat([v[fixed_parent], transform(v[fixed_child], q, axis, origin)])
                    if name == "__object__"
                    else v[name]
                )
                predicted = barypoint(values, faces, bary)
                if name == fixed_child:
                    predicted = transform(predicted, q, axis, origin)
                residual = torch.linalg.vector_norm(predicted - points, dim=1) / 0.001
                losses.append(
                    torch.nn.functional.huber_loss(residual, torch.zeros_like(residual), delta=3.0)
                )
            fit = torch.stack(losses).mean() if losses else axis.sum() * 0
            if recovery_phase:
                image = rendering.observed_loss(
                    v,
                    transform,
                    axis,
                    origin,
                    fixed_child,
                    config.max_rgb_mae_degradation,
                    config.max_mask_iou_degradation,
                )
            else:
                image = rendering(v, transform, axis, origin, fixed_child, iteration)
            return fit, image, motion_consistency(v, transform, axis, origin)

        area_steps = {name: {"backtracked": 0, "restored": 0} for name in graphs}
        collision_steps = {
            name: {"initial_global_step": 1.0, "fallback_scale": 1.0, "local_nodes": 0}
            for name in graphs
        }
        for inner in range(config.correspondence_steps):
            if restoring and inner:
                # An accepted deformation can introduce previously unseen
                # triangle pairs. Refresh the complete current contact set and
                # nonlocal energy before computing another update.
                current, fitted_motion = snapshot()
                constraints = swept_constraints(
                    current[parent],
                    current[child],
                    fitted_motion,
                    sorted(positions),
                    margin=config.minimum_clearance_m + config.contact_reserve_m + 0.001,
                    budget=config.active_constraints_per_pose,
                    workers=config.cpu_query_workers or 4,
                )
                untangling.bind(current, fitted_motion, constraints, sorted(positions))
                contact_projection.add(
                    _restoration_contacts(constraints, config.active_constraints_per_pose),
                    current,
                    fitted_motion,
                )
                fa, fb, normals, angles = contact_projection.arrays()
                contact_a, contact_b = tensor(fa, integer=True), tensor(fb, integer=True)
                contact_n, contact_q = tensor(normals), tensor(angles)
                contact_buffers["active"] = (contact_a, contact_b, contact_n, contact_q)
            elif not restoring:
                contact_projection.refit_supports(*snapshot())
                contact_n.copy_(tensor(contact_projection.arrays()[2]))
            optimizer.zero_grad()
            verts, axis, origin = state()
            fit, image_loss, motion_loss = video_terms(
                verts, axis, origin, outer * config.correspondence_steps + inner
            )
            gaps = contact_gaps(verts, axis, origin)
            # Crossed pairs enter an elastic feasibility LP. Slacks shrink
            # during recovery; already separated pairs keep zero slack. The
            # complete refreshed set still drives the nonlocal objective.
            residual = torch.relu((required - gaps) / 0.001)
            if len(residual):
                contact = residual.square().mean() + residual.square().max()
            else:
                contact = da.sum() * 0
            smooth = da.sum() * 0
            shape = da.sum() * 0
            orientation = da.sum() * 0
            for name, g in graphs.items():
                # Regularize residual deformation around the observed surface.
                # Anchoring to the crossed source would undo its reconstruction.
                # The separate source trust region remains unchanged.
                d = (g["delta"] - g["observed_reference"]) / 0.001
                edges = g["edges"]
                shape = shape + d.square().mean()
                if len(edges):
                    smooth = smooth + (d[edges[:, 0]] - d[edges[:, 1]]).square().mean()
                faces = g["faces"]
                base = g["base"][faces]
                v = verts[name][faces]
                n0 = torch.linalg.cross(base[:, 1] - base[:, 0], base[:, 2] - base[:, 0])
                n = torch.linalg.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
                ratio = torch.linalg.vector_norm(n, dim=1) / torch.linalg.vector_norm(
                    n0, dim=1
                ).clamp_min(1e-20)
                orientation = (
                    orientation
                    + torch.relu(0.2 - ratio).square().mean()
                    + torch.relu(0.1 - ratio).square().max()
                )
            prior = (da / max(np.radians(config.max_axis_change_deg), 1e-6)).square().sum() + (
                do / config.max_origin_displacement_m
            ).square().sum()
            gaussian, minkowski = (
                untangling(verts, axis, origin, transform) if restoring else (fit * 0, fit * 0)
            )
            evidence_loss = (
                fit
                + config.motion_consistency_weight * motion_loss
                + (image_loss if restoring else 0.1 * image_loss)
            )
            fitting_loss = (
                evidence_loss
                + config.surface_smoothness * smooth
                + 0.05 * shape
                + 0.1 * prior
                + 0.1 * contact
                + 100 * orientation
            )
            # Crossing recovery uses the observed direction tangent to the
            # unfolding descent. The video is active from the source state;
            # neither a crossed source nor an independently unfolded seed is
            # the fitting target.
            loss = untangling.local_weight * minkowski + gaussian if restoring else fitting_loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite coupled objective")
            if not torch.isfinite(evidence_loss):
                raise ValueError("nonfinite video objective")
            if restoring:
                _observed_separation_gradients(loss, evidence_loss, params)
            else:
                loss.backward()
            for graph in graphs.values():
                graph["delta"].grad[~tensor(graph["editable"], integer=True).bool()] = 0
            torch.nn.utils.clip_grad_norm_(params, 1000)
            previous = [p.detach().clone() for p in params]
            optimizer.step()
            with torch.no_grad():
                desired = [(p - old).cpu().numpy() for p, old in zip(params, previous, strict=True)]
                projection_metric = None
                if restoring:
                    # A single worst-contact gradient must not freeze thousands
                    # of independent surface nodes. Use current gradients with
                    # a bounded RMS metric, not Adam's possibly stale momentum.
                    # Project in that SAME metric to retain a descent direction.
                    desired, projection_metric = [], []
                    for index, p in enumerate(params):
                        saved = optimizer.state[p]
                        rms = torch.sqrt(saved["exp_avg_sq"] / (1 - 0.999 ** float(saved["step"])))
                        rms = rms.clamp_min(max(float(rms.max()) * 1e-4, 1e-8))
                        rate = config.solver_learning_rate / (
                            max(lever_arm, 0.05) if index == 0 else 1
                        )
                        diagonal = (rms / rate).cpu().numpy()
                        projection_metric.append(diagonal)
                        desired.append(-p.grad.detach().cpu().numpy() / diagonal)
                    magnitude = max(
                        max(lever_arm, 0.05) * np.linalg.norm(desired[0]),
                        *(np.linalg.norm(d, axis=-1).max() for d in desired[1:]),
                    )
                    scale = min(1.0, config.solver_learning_rate / max(magnitude, 1e-20))
                    desired = [d * scale for d in desired]
                for p, old in zip(params, previous, strict=True):
                    p.copy_(old)
                restricted = {
                    name: np.zeros(len(g["delta"]), dtype=bool) for name, g in graphs.items()
                }
                local_reports = []
                for _local_attempt in range(3):
                    attempt_restricted = False
                    for p, old in zip(params, previous, strict=True):
                        p.copy_(old)
                    if restoring:
                        # Project the nonlinear unfolding direction into the
                        # observation and geometric feasible tangent region.
                        if contact_projection.rows or contact_projection.self_planes:
                            step, projection_report = contact_projection.project(
                                *snapshot(),
                                desired,
                                quadratic=True,
                                quadratic_metric=projection_metric,
                                # Clarabel can time out on the dense hinge
                                # neighbourhood. Its finite iterate is only
                                # a proposal: the nonlinear video, geometry,
                                # observation-slab and protected-gap guards
                                # below still decide whether it may move.
                                allow_inexact_proposal=True,
                            )
                            projection_report["objective"] = "project_nonlinear_untangling"
                        else:
                            step, projection_report = desired, {"status": "nonlinear_untangling"}
                    elif len(contact_a):
                        step, projection_report = contact_projection.project(*snapshot(), desired)
                    else:
                        step, projection_report = desired, {"status": "no_active_contacts"}
                    if step is not None:
                        for p, old, d in zip(params, previous, step, strict=True):
                            p.copy_(old + tensor(d))
                    # Project all updates into independent physical trust regions.
                    da -= torch.dot(da, tensor(axis0)) * tensor(axis0)
                    da *= torch.clamp(
                        torch.tensor(np.tan(np.radians(config.max_axis_change_deg)), device=device)
                        / torch.linalg.vector_norm(da).clamp_min(1e-12),
                        max=1,
                    )
                    do -= torch.dot(do, tensor(axis0)) * tensor(
                        axis0
                    )  # remove hinge-origin gauge freedom
                    do *= torch.clamp(
                        config.max_origin_displacement_m
                        / torch.linalg.vector_norm(do).clamp_min(1e-12),
                        max=1,
                    )
                    for g in graphs.values():
                        if restoring:
                            observed = tensor(g["observed"], integer=True).bool()
                            normals = tensor(g["observation_normal"])
                            bounds = tensor(g["observation_interval"])
                            distance = (g["delta"] * normals).sum(1)
                            target = torch.maximum(
                                bounds[:, 0], torch.minimum(bounds[:, 1], distance)
                            )
                            g["delta"][observed] += ((target - distance)[:, None] * normals)[
                                observed
                            ]
                        g["delta"] *= torch.clamp(
                            config.max_surface_displacement_m
                            / torch.linalg.vector_norm(g["delta"], dim=1, keepdim=True).clamp_min(
                                1e-12
                            ),
                            max=1,
                        )
                    validity_proposals = {
                        name: (graph["delta"] - old).detach().cpu().numpy().copy()
                        for (name, graph), old in zip(graphs.items(), previous[2:], strict=True)
                    }
                    for (name, graph), old in zip(graphs.items(), previous[2:], strict=True):
                        step_report = limit_triangle_area_nodes(graph, old)
                        for key, count in step_report.items():
                            area_steps[name][key] = max(area_steps[name][key], count)
                    for (name, graph), old in zip(graphs.items(), previous[2:], strict=True):
                        step_report = limit_collision_nodes(
                            ipc_meshes[name],
                            graph,
                            old,
                            executor=pool,
                            on_blockers=lambda planes, link=name: (
                                contact_projection.add_self_contacts(link, planes)
                            ),
                        )
                        # Mixed local backtracking can affect a shared face. Repeat
                        # orientation projection and certify that combined proposal.
                        limit_triangle_area_nodes(graph, old)
                        previous_vertices = graph["base"] + (
                            old[graph["idx"]] * graph["weights"][..., None]
                        ).sum(1)
                        trial_vertices = graph["base"] + (
                            graph["delta"][graph["idx"]] * graph["weights"][..., None]
                        ).sum(1)
                        final_alpha = deformation_step(
                            ipc_meshes[name],
                            previous_vertices.cpu().numpy(),
                            trial_vertices.cpu().numpy(),
                            executor=pool,
                        )
                        graph["delta"].copy_(old + final_alpha * (graph["delta"] - old))
                        final_step = {
                            "initial_global_step": final_alpha,
                            "fallback_scale": final_alpha,
                            "local_nodes": 0,
                        }
                        for key in ("initial_global_step", "fallback_scale"):
                            collision_steps[name][key] = min(
                                collision_steps[name][key], step_report[key], final_step[key]
                            )
                        collision_steps[name]["local_nodes"] = max(
                            collision_steps[name]["local_nodes"],
                            step_report["local_nodes"],
                            final_step["local_nodes"],
                        )
                        # Learn only area/CCD restrictions. Observation-slab and
                        # global trust projections above are already enforced
                        # by their own constraints; treating them as collisions
                        # also freezes otherwise admissible tangential motion.
                        requested = validity_proposals[name]
                        limited = contact_projection.learn_local_limits(
                            name, requested, (graph["delta"] - old).cpu().numpy()
                        )
                        attempt_restricted |= bool(limited.any())
                        restricted[name] |= limited
                        limited = tensor(limited, integer=True).bool()
                        # Adam must not repeatedly accumulate a blocked direction.
                        for key in ("exp_avg", "exp_avg_sq"):
                            optimizer.state[graph["delta"]][key][limited] = 0
                    local_reports.append(dict(projection_report))
                    local_vertices, local_axis, local_origin = state()
                    local_gaps = contact_gaps(local_vertices, local_axis, local_origin)
                    protected_local = bool(
                        (local_gaps[gaps >= hard_required] >= hard_required - 2e-9).all()
                    )
                    deficit_local = (
                        float(torch.relu(hard_required - local_gaps).max())
                        if len(local_gaps)
                        else 0.0
                    )
                    deficit_before = (
                        float(torch.relu(hard_required - gaps).max()) if len(gaps) else 0.0
                    )
                    if step is None or (
                        protected_local
                        and (not restoring or deficit_local <= deficit_before + 2e-9)
                    ):
                        break
                    if not attempt_restricted:
                        break
                projection_report["local_reprojections"] = len(local_reports) - 1
                projection_report["local_projection_attempts"] = local_reports
                projection_report["locally_restricted_nodes"] = {
                    name: int(mask.sum()) for name, mask in restricted.items()
                }
                # Guard the *actual* nonlinear proposal after every local
                # validity correction. A restoration label alone is not progress.
                guarded = [p.detach().clone() for p in params]
                initial_deficit = (
                    float(torch.relu(hard_required - gaps).max()) if len(gaps) else 0.0
                )
                weights = tensor(
                    np.ones(len(gaps))
                    if restoring
                    else contact_projection.area_weights(snapshot()[0])
                )
                initial_sum = float((torch.relu(hard_required - gaps) * weights).sum())
                initial_observation_violation = observation_violation(verts)
                # The optional optimization reserve is not the physical safety
                # threshold. Local validity corrections may consume that reserve
                # while every already-safe pair still retains the required gap.
                protected = gaps >= hard_required
                accepted = False
                initial_untangling = float(untangling.local_weight * minkowski + gaussian)
                attempts = (
                    18
                    if step is not None
                    and any(
                        bool((trial != old).any())
                        for trial, old in zip(guarded, previous, strict=True)
                    )
                    else 1
                )
                for backtrack in range(attempts):
                    scale = 0.5**backtrack
                    for p, old, trial in zip(params, previous, guarded, strict=True):
                        p.copy_(old + scale * (trial - old))
                    trial_vertices_by_link, ax, orig = state()
                    trial_gaps = contact_gaps(trial_vertices_by_link, ax, orig)
                    deficit = (
                        float(torch.relu(hard_required - trial_gaps).max()) if len(gaps) else 0.0
                    )
                    protected_valid = bool((trial_gaps[protected] >= hard_required - 2e-9).all())
                    total = float((torch.relu(hard_required - trial_gaps) * weights).sum())
                    observation_error = observation_violation(trial_vertices_by_link)
                    observation_valid = observation_error <= max(
                        2e-9, initial_observation_violation - 1e-12
                    )
                    if backtrack == 0:
                        projection_report["unscaled_trial"] = {
                            "maximum_deficit_m": deficit,
                            "protected_valid": protected_valid,
                            "weighted_deficit_sum_m": total,
                            "observation_violation_m": observation_error,
                        }
                    improves = deficit <= initial_deficit + 2e-9 and total < initial_sum - min(
                        1e-9, initial_sum * 1e-6
                    )
                    if restoring and step is not None:
                        trial_global, trial_local = untangling(
                            trial_vertices_by_link, ax, orig, transform
                        )
                        trial_energy = float(untangling.local_weight * trial_local + trial_global)
                        # Feasibility restoration is lexicographic: reduce the
                        # worst hard-gap deficit first, then total deficit, and
                        # use the smooth untangling energy only as a tie-break.
                        # Requiring energy descent while a hard gap improves
                        # can stall a necessary local unfolding step.
                        tolerance = min(1e-9, initial_deficit * 1e-6)
                        improves = deficit < initial_deficit - tolerance or (
                            deficit <= initial_deficit + 2e-9
                            and (
                                total < initial_sum - min(1e-9, initial_sum * 1e-6)
                                or (
                                    total <= initial_sum + 2e-9
                                    and trial_energy < initial_untangling - 1e-12
                                )
                            )
                        )
                    video_valid = True
                    if restoring and step is not None and (improves or deficit <= 2e-9):
                        trial_fit, trial_image, trial_motion = video_terms(
                            trial_vertices_by_link,
                            ax,
                            orig,
                            outer * config.correspondence_steps + inner,
                        )
                        trial_evidence = float(
                            trial_fit
                            + config.motion_consistency_weight * trial_motion
                            + trial_image
                        )
                        # This local tolerance is expressed in the same
                        # millimetre-normalized units as the observed fit. It
                        # permits sensor-scale motion while rejecting a step
                        # that opens the mesh by losing video agreement.
                        video_valid = trial_evidence <= float(evidence_loss) + (
                            config.max_check_degradation_m / 0.001
                        )
                        projection_report["observed_objective_before"] = float(evidence_loss)
                        projection_report["observed_objective_after"] = trial_evidence
                    if (
                        step is not None
                        and protected_valid
                        and observation_valid
                        and video_valid
                        and (improves or deficit <= 2e-9)
                    ):
                        accepted = True
                        break
                if not accepted:
                    for p, old in zip(params, previous, strict=True):
                        p.copy_(old)
                        for key in ("exp_avg", "exp_avg_sq"):
                            optimizer.state[p][key].zero_()
                if accepted:
                    contact_projection.release_local_limits(restricted)
                projection_report["nonlinear_update_accepted"] = accepted
                if restoring:
                    projection_report["untangling_energy_before"] = initial_untangling
                    projection_report["untangling_energy_after"] = (
                        trial_energy if accepted else initial_untangling
                    )
                projection_report["observation_violation_m"] = (
                    observation_error if accepted else initial_observation_violation
                )
                projection_report["nonlinear_backtracks"] = backtrack if accepted else attempts
                projection_report["weighted_deficit_sum_before_m"] = initial_sum
                projection_report["weighted_deficit_sum_after_m"] = (
                    total if accepted else initial_sum
                )
                projection_report["nonlinear_deficit_before_m"] = initial_deficit
                projection_report["nonlinear_deficit_after_m"] = (
                    deficit if accepted else initial_deficit
                )
                logging.getLogger(__name__).info(
                    "Contact projection %s: %s",
                    outer * config.correspondence_steps + inner + 1,
                    projection_report,
                )
        safe_step = min(r["fallback_scale"] for r in collision_steps.values())
        with torch.no_grad():
            end_v, end_a, end_o = state()
            end_gaps = contact_gaps(end_v, end_a, end_o)
            max_violation = max(0.0, required - float(end_gaps.min())) if len(end_gaps) else 0.0
        row = {
            "iteration": (outer + 1) * config.correspondence_steps,
            "phase": "restore_feasibility" if restoring else "fit_with_separation",
            "objective": float(loss.detach()),
            "video": float(fit.detach()),
            "render": float(image_loss.detach()),
            "bidirectional_motion": float(motion_loss.detach()),
            "gaussian_shape_repulsion": float(gaussian.detach()),
            "triangle_minkowski_penalty": float(minkowski.detach()),
            "untangling_penalty_weight": untangling.local_weight,
            "hard_contact_projection": projection_report,
            "contact": float(contact.detach()),
            "area_collapse": float(orientation.detach()),
            "contact_pairs": len(contact_a),
            "intersecting_pairs_before_step": starting_intersections,
            "maximum_constraint_violation_m": max_violation,
            "deformation_ccd_step": safe_step,
            "active_poses": len(positions),
            "area_backtracking": area_steps,
            "self_collision_backtracking": collision_steps,
            "surface_step_max_m": {
                name: float(
                    np.linalg.norm(end_v[name].cpu().numpy() - current[name].vertices, axis=1).max()
                )
                for name in current
            },
        }
        row["maximum_motion_step_m"] = max(
            max(row["surface_step_max_m"].values()),
            lever_arm * float(np.linalg.norm(end_a.cpu().numpy() - a))
            + float(np.linalg.norm(end_o.cpu().numpy() - o)),
        )
        history.append(row)
        if (outer + 1) % 4 == 0 and max_violation <= config.contact_reserve_m + 2e-9:
            from ._surface_clearance import swept_clearance

            candidate, candidate_motion = snapshot()
            audit = swept_clearance(
                candidate[parent],
                candidate[child],
                candidate_motion,
                query,
                clearance=config.minimum_clearance_m,
                samples=9,
                temporal_levels=1,
                max_cells=min(20000, config.verification_cell_budget),
            )
            bad = [r for r in audit["intervals"] if r["lower_bound_m"] < config.minimum_clearance_m]
            positions.update(r["q"] for r in sorted(bad, key=lambda r: r["lower_bound_m"])[:8])
        candidate, candidate_motion = snapshot()
        contact_projection.refit_supports(candidate, candidate_motion)
        checked = swept_constraints(
            candidate[parent],
            candidate[child],
            candidate_motion,
            sorted(positions),
            margin=required,
            budget=config.active_constraints_per_pose,
            workers=config.cpu_query_workers or 4,
        )
        row["intersecting_pairs_after_step"] = checked["intersecting_pairs"]
        row["active_poses"] = len(positions)
        current_parameters = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, value in zip(params, best, strict=True):
                p.copy_(value)
        best_score = candidate_score(*snapshot())
        with torch.no_grad():
            for p, value in zip(params, current_parameters, strict=True):
                p.copy_(value)
        # Feasibility precedes fit: a smaller image loss cannot buy intersections.
        score = candidate_score(candidate, candidate_motion, checked)
        row["candidate_training_error"] = score[-1]
        if score < best_score:
            best_score = score
            best = [p.detach().clone() for p in params]
        logging.getLogger(__name__).info(
            "Untangling batch %d: %.3f s, crossings %d -> %d, devices=%s",
            row["iteration"],
            monotonic() - outer_started,
            row["intersecting_pairs_before_step"],
            row["intersecting_pairs_after_step"],
            untangling.devices,
        )
        if checkpoint is not None:
            checkpoint(
                row,
                snapshot(),
                {
                    "solver_identity": SOLVER_IDENTITY,
                    "iteration": row["iteration"],
                    "parameters": [p.detach().cpu().tolist() for p in params],
                    "adam": [
                        {k: v.detach().cpu().tolist() for k, v in optimizer.state[p].items()}
                        for p in params
                    ],
                    "positions": sorted(positions),
                    "history": history,
                    "best_parameters": [p.detach().cpu().tolist() for p in best]
                    if best is not None
                    else None,
                    "best_score": list(best_score),
                    "contact_planes": [
                        [a.tolist(), b.tolist(), n.tolist(), q]
                        for a, b, n, q in contact_projection.rows.values()
                    ],
                    "coherent_contact_keys": [
                        list(key) for key in contact_projection.coherent_keys
                    ],
                    "hard_safe_contact_keys": [
                        list(key) for key in contact_projection.hard_safe_keys
                    ],
                    "contact_directions": [
                        contact_projection.branch_directions[key].tolist()
                        for key in contact_projection.rows
                    ],
                    "local_trust_radii_m": {
                        name: radii.tolist()
                        for name, radii in contact_projection.local_radii.items()
                    },
                    "self_contact_planes": [
                        [name, a.tolist(), b.tolist(), n.tolist(), gap]
                        for name, a, b, n, gap in contact_projection.self_planes.values()
                    ],
                },
            )
    if best is not None:
        with torch.no_grad():
            for p, value in zip(params, best, strict=True):
                p.copy_(value)
    output, fitted = snapshot()
    final_observation_violation = observation_violation(state()[0])
    valid = final_observation_violation <= 2e-9
    links = {}
    for name, mesh in output.items():
        initial = refined[name]
        displacement = np.linalg.norm(mesh.vertices - initial.vertices, axis=1)
        cross0 = np.cross(
            initial.triangles[:, 1] - initial.triangles[:, 0],
            initial.triangles[:, 2] - initial.triangles[:, 0],
        )
        cross = np.cross(
            mesh.triangles[:, 1] - mesh.triangles[:, 0], mesh.triangles[:, 2] - mesh.triangles[:, 0]
        )
        collapsed = int(
            np.count_nonzero(np.linalg.norm(cross, axis=1) <= 0.1 * np.linalg.norm(cross0, axis=1))
        )
        intersects = self_intersections(mesh)
        topology = surface_topology(mesh)
        valid &= (
            remesh[name]["restricted_delaunay"]["coverage_certified"]
            and topology["valid"]
            and not intersects
            and collapsed == 0
            and bool(displacement.max() <= config.max_surface_displacement_m + 1e-9)
        )
        links[name] = {
            "remesh": remesh[name],
            "editable_vertices": int(graphs[name]["editable"].sum()),
            "deformation_nodes": len(graphs[name]["delta"]),
            "max_displacement_m": float(displacement.max()),
            "collapsed_faces": collapsed,
            "self_intersections": intersects,
            "topology": topology,
            "geometry_changed": bool(displacement.max() > 1e-9),
        }
    return (
        output,
        fitted,
        {
            "method": "observed_surface_constrained_joint_separation",
            "solver_identity": SOLVER_IDENTITY,
            "device": str(device),
            "history": history,
            "links": links,
            "geometry_valid": bool(valid),
            "maximum_observation_constraint_violation_m": final_observation_violation,
            "training_partitions": ["fit"],
            "heldout_used_for_optimization": False,
            "collision_representation": "triangle_minkowski_and_gaussian_surface_repulsion",
            "contact_constraints": "nine_vertex_inequalities_per_triangle_pair",
            "restoration": "observed_nonlinear_untangling_then_hard_contact_projection",
            "parallelism": {
                "cpu_pose_processes": pool._max_workers,
                "energy_devices": untangling.devices,
            },
            "timing_seconds": {"preparation": preparation_seconds, "total": monotonic() - started},
            "termination": stop_reason,
            "state_reconstruction": reconstruction_report,
            "observed_initialization": initialization_report,
            "motion_terms": ["bidirectional_same_part_vertices", "bidirectional_rgbd_render"],
            "source_artifacts_mutated": False,
            "initialized_from_solver_identity": (
                resume["solver_identity"] if resume and resume.get("initialize_only") else None
            ),
            "candidate_selection": (
                "independent_intersection_then_common_contact_field_then_training_fit"
            ),
            "self_collision_backend": "ipc_candidates_bounded_plane_intervals",
            "continuous_witness_feedback": True,
            "image_terms": ["rgb", "metric_depth", "projected_silhouette"],
        },
    )
