"""Per-part restricted Delaunay: retain duals intersecting the observed surface.

This is a reconstruction of the rDel operation in ArtMesh Eq. 8, using the
CGAL exact constructions and the existing Warp distance queries. It changes
connectivity, never vertex positions or attributes. Finite Voronoi edges AND unbounded hull rays are
restricted to the input surface; no convex-hull closing faces are invented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from ._mesh_validity import surface_topology


def refine_observed_crossings(
    mesh: Any,
    frame_points: list[NDArray[Any]],
    crossing_points: NDArray[Any],
    query: Any,
    *,
    edge_length_m: float,
    support_radius_m: float,
) -> tuple[Any, dict[str, Any]]:
    """Split source edges seen in at least two independent fit RGB-D frames.

    Each added vertex is an edge interpolation on the immutable source surface.
    Both incident faces are split together; no gap is capped or new surface is
    inferred from a single noisy pixel. Later state fitting gives these shared
    vertices video-dependent positions before the coupled collision solve.
    """
    from scipy.spatial import cKDTree

    from ._surface_clearance import conforming_refine

    if edge_length_m <= 0 or support_radius_m <= 0:
        raise ValueError("video refinement scales must be positive")
    crossings = np.asarray(crossing_points, dtype=float).reshape(-1, 3)
    edges = np.asarray(mesh.edges_unique)
    vertices = np.asarray(mesh.vertices)
    midpoints = vertices[edges].mean(axis=1)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    eligible = lengths > edge_length_m
    if len(crossings):
        eligible &= cKDTree(crossings).query(midpoints)[0] <= support_radius_m
    else:
        eligible[:] = False
    support = np.zeros(len(edges), dtype=np.int32)
    candidates = np.flatnonzero(eligible)
    for points in frame_points:
        measured = np.asarray(points, dtype=float).reshape(-1, 3)
        if not len(candidates) or not len(measured):
            continue
        if not np.isfinite(measured).all():
            raise ValueError("video fit points must be finite")
        # Require the pixel's nearest source face to touch the edge's one-ring.
        # The exact incident face would under-sample tiny triangles at the
        # video's pixel pitch; an unrelated folded sheet shares no vertex.
        distance, owners = query.nearest(mesh, measured)
        valid = (distance <= support_radius_m) & (owners >= 0)
        if not np.any(valid):
            continue
        measured = measured[valid]
        owner_faces = np.asarray(mesh.faces)[owners[valid]]
        separation, nearby = cKDTree(measured).query(midpoints[candidates], k=min(4, len(measured)))
        separation = np.atleast_2d(separation).reshape(len(candidates), -1)
        nearby = np.atleast_2d(nearby).reshape(len(candidates), -1)
        observed_faces = owner_faces[nearby]
        ends = edges[candidates]
        same_sheet = (observed_faces == ends[:, None, 0, None]).any(axis=2) | (
            observed_faces == ends[:, None, 1, None]
        ).any(axis=2)
        support[candidates] += ((separation <= support_radius_m) & same_sheet).any(axis=1)
    selected = eligible & (support >= 2)
    if not selected.any():
        return mesh.copy(), {
            "method": "video_supported_conforming_edge_split",
            "status": "NO_SUPPORTED_EDGES",
            "candidate_edges": int(eligible.sum()),
            "supported_edges": 0,
            "added_vertices": 0,
            "heldout_used": False,
        }
    refined, report = conforming_refine(
        mesh,
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3),
        np.inf,
        edge_length_m,
        rounds=1,
        edge_mask=selected,
    )
    return refined, {
        **report,
        "method": "video_supported_conforming_edge_split",
        "status": "APPLIED",
        "candidate_edges": int(eligible.sum()),
        "supported_edges": int(selected.sum()),
        "support_frames_required": 2,
        "heldout_used": False,
    }


def manifold_patches(
    source: Any,
    candidate: Any,
    *,
    focus_vertices: NDArray[Any] | None = None,
    validation_meshes: list[Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Accept rDel replacements only within the same observed surface domain.

    Common edges bound independent retriangulation cavities. A replacement must
    have exactly the same boundary and Euler characteristic as its source cavity.
    Ambiguous rDel facets cannot create a third sheet or remove source coverage:
    their original cavity is retained, with this choice reported explicitly.
    No boundary is capped, no vertex is moved, and no source region is dropped.
    """
    common = set(map(tuple, source.edges_unique)) & set(map(tuple, candidate.edges_unique))

    def patches(mesh: Any) -> Any:
        edges = mesh.face_adjacency_edges
        keep = np.array([tuple(e) not in common for e in edges], dtype=bool)
        groups = trimesh.graph.connected_components(  # type: ignore[no-untyped-call]
            mesh.face_adjacency[keep], nodes=np.arange(len(mesh.faces)), engine="scipy"
        )
        result: dict[Any, list[Any]] = {}
        for group in groups:
            faces = np.asarray(mesh.faces)[group]
            edge = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
            unique, count = np.unique(edge, axis=0, return_counts=True)
            boundary = tuple(map(tuple, unique[count == 1]))
            euler = len(np.unique(faces)) - len(unique) + len(faces)
            result.setdefault(boundary, []).append((faces, euler, bool(np.all(count <= 2))))
        return result

    proposed = patches(candidate)
    original = patches(source)
    accepted, retained, output, originals = 0, 0, [], []
    focus = None if focus_vertices is None else np.asarray(focus_vertices, dtype=bool)
    if focus is not None and len(focus) != len(source.vertices):
        raise ValueError("focus mask must match the source vertices")
    for boundary, cavities in original.items():
        options = proposed.get(boundary, [])
        for faces, euler, valid in cavities:
            originals.append(faces)
            if (
                len(cavities) == len(options) == 1
                and valid
                and options[0][2]
                and options[0][1] == euler
                and (focus is None or bool(focus[np.unique(faces)].any()))
            ):
                output.append(options[0][0])
                accepted += 1
            else:
                output.append(faces)
                retained += 1
    result = source.copy()
    result.faces = np.concatenate(output)
    trimesh.repair.fix_winding(result)  # type: ignore[no-untyped-call]
    before, after = surface_topology(source), surface_topology(result)
    invariants = ("boundary_loops", "components", "euler_characteristic")
    # Individually valid cavities can share a shortcut edge through a fold.
    # Resolve conflicts jointly by retaining the affected source cavities.
    while not after["valid"]:
        bad = np.asarray(after["nonmanifold_vertex_ids"], dtype=int)
        changed = False
        for i, faces in enumerate(output):
            if faces is originals[i]:
                continue
            if np.isin(faces, bad).any():
                output[i] = originals[i]
                accepted -= 1
                retained += 1
                changed = True
        if not changed:
            break
        result.faces = np.concatenate(output)
        trimesh.repair.fix_winding(result)  # type: ignore[no-untyped-call]
        after = surface_topology(result)
    if not after["valid"] or any(before[k] != after[k] for k in invariants):
        # Fail closed, preserving the complete source surface. This is not a
        # successful remesh and must never be labelled as one.
        raise ValueError(f"rDel cavity assembly violates source topology: {after}")
    geometry_checks = 0
    rejected_cavities = 0
    if validation_meshes and accepted:
        from ._mesh_validity import self_intersections

        baselines = [surface_topology(mesh) for mesh in validation_meshes]
        if any(not row["valid"] for row in baselines):
            raise ValueError("observed cavity validation requires valid input surfaces")
        cavity_ids = [i for i, faces in enumerate(output) if faces is not originals[i]]
        selected: set[int] = set()

        def valid_with(batch: list[int]) -> bool:
            nonlocal geometry_checks
            geometry_checks += 1
            active = selected | set(batch)
            faces = np.concatenate(
                [output[i] if i in active else originals[i] for i in range(len(output))]
            )
            for mesh, baseline in zip(validation_meshes, baselines, strict=True):
                tested = mesh.copy()
                tested.faces = faces.copy()
                topology = surface_topology(tested)
                if (
                    not topology["valid"]
                    or any(topology[k] != baseline[k] for k in invariants)
                    or self_intersections(tested)
                ):
                    return False
            return True

        # A single folded cavity must not veto all other observed cavities.
        # Bisection keeps whole patches and their shared boundary intact. A
        # finite validation budget fails closed by retaining unchecked source
        # cavities; downstream bidirectional coverage still certifies the result.
        if valid_with(cavity_ids):
            selected.update(cavity_ids)
        else:

            def admit(batch: list[int]) -> None:
                if not batch or geometry_checks >= 32:
                    return
                if valid_with(batch):
                    selected.update(batch)
                elif len(batch) > 1:
                    middle = len(batch) // 2
                    admit(batch[:middle])
                    admit(batch[middle:])

            admit(cavity_ids)
            rejected_cavities = len(cavity_ids) - len(selected)
            result.faces = np.concatenate(
                [output[i] if i in selected else originals[i] for i in range(len(output))]
            )
            trimesh.repair.fix_winding(result)  # type: ignore[no-untyped-call]
            accepted = len(selected)
            retained = len(output) - accepted
            after = surface_topology(result)
    return result, {
        "accepted_cavities": accepted,
        "retained_source_cavities": retained,
        "unmatched_candidate_cavities": sum(
            len(v) for k, v in proposed.items() if k not in original
        ),
        "source_domain_preserved": True,
        "geometry_validation_checks": geometry_checks,
        "geometry_rejected_cavities": rejected_cavities,
        "topology": after,
    }


def restricted_delaunay(
    mesh: Any, query: Any, tolerance: float = 0.001
) -> tuple[Any, dict[str, Any]]:
    """Refine observed triangles until rDel satisfies bidirectional coverage."""
    from ._surface_clearance import conforming_refine

    source = mesh.copy()
    topology = surface_topology(source)
    if not topology["valid"]:
        raise ValueError(f"restricted Delaunay requires a valid source two-manifold: {topology}")
    attempts: list[dict[str, Any]] = []
    for level in range(8):
        result, report = _restricted_delaunay_once(source, query, tolerance)
        attempts.append({k: v for k, v in report.items() if k not in {"method", "attempts"}})
        if report["coverage_certified"]:
            break
        witnesses = []
        for key in ("source_to_remesh", "remesh_to_source"):
            if "witness_point_m" in report[key]:
                witnesses.append(np.asarray([report[key]["witness_point_m"]]))
        for left, right in ((source, result), (result, source)):
            triangles = np.asarray(left.triangles)
            samples = np.concatenate(
                [
                    triangles.mean(1),
                    triangles.reshape(-1, 3),
                    ((triangles + np.roll(triangles, 1, axis=1)) / 2).reshape(-1, 3),
                ]
            )
            distances, _ = query.nearest(right, samples)
            witnesses.append(samples[distances > tolerance * 0.5])
        points = np.concatenate(witnesses)
        if not len(points):
            break
        _, ids = query.nearest(source, points)
        points = source.vertices[np.unique(source.faces[ids])]
        source, _ = conforming_refine(
            source,
            np.array([0.0, 0.0, 1.0]),
            source.centroid,
            tolerance * 2,
            tolerance * 2 / (2**level),
            rounds=2,
            near=points,
        )
    report["attempts"] = attempts
    report["observed_surface_steiner_vertices"] = len(result.vertices) - len(mesh.vertices)
    return result, report


def retriangulate_observed_states(
    source: Any,
    states: list[Any],
    query: Any,
    tolerance: float = 0.001,
    *,
    focus_points: NDArray[Any] | None = None,
    local_tolerance: float | None = None,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Accept one video-derived connectivity only if every state supports it.

    State vertices have already been transported into the same part frame. The
    observed mean proposes connectivity, but neither its vertices nor its faces
    are trusted until the immutable source and *each* fitted state certify
    topology, self-intersection freedom, and bidirectional surface coverage.
    Rejection leaves the existing connectivity untouched; it never fills a gap.
    """
    from scipy.spatial import cKDTree

    from ._collision_shell import approximation_bound
    from ._mesh_validity import self_intersections

    if not states or any(len(state.vertices) != len(source.vertices) for state in states):
        raise ValueError("observed remeshing requires shared part vertices")
    if tolerance <= 0 or (local_tolerance is not None and local_tolerance <= 0):
        raise ValueError("surface coverage tolerances must be positive")
    observed = source.copy()
    observed.vertices = np.mean([np.asarray(state.vertices) for state in states], axis=0)
    baseline = surface_topology(source)
    attempts: list[dict[str, Any]] = []
    unchanged = False
    # A mean of two valid deformations can itself fold. Each state is an
    # independent video-supported proposal; all other states still veto it.
    regions: list[tuple[NDArray[Any] | None, float]] = [(None, tolerance)]
    if focus_points is not None and len(focus_points):
        distance = cKDTree(np.asarray(focus_points)).query(np.asarray(source.vertices))[0]
        # Region radius and allowable surface deviation are independent. An
        # expanded budget must first retry the SAME crossing patch; widening
        # the patch at the same time can admit an unrelated folded cavity.
        local_trials = [(tolerance, 2 * tolerance)]
        if local_tolerance is not None and local_tolerance > tolerance:
            local_trials.extend(
                [(local_tolerance, 2 * tolerance), (local_tolerance, 2 * local_tolerance)]
            )
        for budget, radius in local_trials:
            mask = distance <= radius
            if mask.any() and not any(
                previous is not None and np.array_equal(previous, mask) and limit == budget
                for previous, limit in regions
            ):
                regions.append((mask, budget))
    for basis, geometry in [("mean", observed), *[(f"state_{i}", s) for i, s in enumerate(states)]]:
        if not surface_topology(geometry)["valid"] or self_intersections(geometry):
            attempts.append({"basis": basis, "reason": "invalid_proposal_surface"})
            continue
        # First try the whole observed surface; if an unsupported remote cavity
        # vetoes it, restrict the SAME video proposal to the actual crossings.
        proposal_faces = None
        for mask, budget in regions:
            proposed, method = _restricted_delaunay_once(
                geometry,
                query,
                budget,
                focus_vertices=mask,
                candidate_faces=proposal_faces,
                return_proposal=mask is None and len(regions) > 1,
                validation_meshes=(
                    [source, *states] if mask is not None and budget > tolerance else None
                ),
            )
            proposal_faces = method.pop("proposal_faces", proposal_faces)
            region = "whole" if mask is None else f"local_{int(mask.sum())}_vertices"
            if not method["coverage_certified"]:
                attempts.append(
                    {
                        "basis": basis,
                        "region": region,
                        "coverage_budget_m": budget,
                        "reason": "proposal_coverage_unresolved",
                        "forward": method["source_to_remesh"],
                        "backward": method["remesh_to_source"],
                    }
                )
                continue
            if np.array_equal(proposed.faces, source.faces):
                unchanged = True
                attempts.append(
                    {
                        "basis": basis,
                        "region": region,
                        "coverage_budget_m": budget,
                        "reason": "no_supported_cavity",
                    }
                )
                continue
            checks = {}
            updated = []
            rejection = None
            for label, mesh in [
                ("source", source),
                *[(f"state_{i}", s) for i, s in enumerate(states)],
            ]:
                candidate = mesh.copy()
                candidate.faces = np.asarray(proposed.faces).copy()
                topology = surface_topology(candidate)
                same_domain = all(
                    topology.get(key) == baseline[key]
                    for key in ("boundary_loops", "components", "euler_characteristic")
                )
                if not topology["valid"] or not same_domain or self_intersections(candidate):
                    rejection = f"{label}_topology"
                    break
                forward = approximation_bound(mesh, candidate, query, budget, max_cells=2000000)
                backward = approximation_bound(candidate, mesh, query, budget, max_cells=2000000)
                checks[label] = {"source_to_candidate": forward, "candidate_to_source": backward}
                if forward["status"] != "CERTIFIED" or backward["status"] != "CERTIFIED":
                    rejection = f"{label}_surface_coverage"
                    break
                updated.append(candidate)
            if rejection is not None:
                attempts.append(
                    {
                        "basis": basis,
                        "region": region,
                        "coverage_budget_m": budget,
                        "reason": rejection,
                        "coverage": checks,
                    }
                )
                continue
            old_faces = set(map(tuple, np.sort(source.faces, axis=1)))
            new_faces = set(map(tuple, np.sort(proposed.faces, axis=1)))
            return (
                updated[0],
                updated[1:],
                {
                    "status": "APPLIED",
                    "method": "shared_connectivity_from_fit_states",
                    "proposal_basis": basis,
                    "region": region,
                    "coverage_budget_m": budget,
                    "faces_removed": len(old_faces - new_faces),
                    "faces_added": len(new_faces - old_faces),
                    "source_vertices_preserved": True,
                    "heldout_used": False,
                    "coverage": checks,
                    "attempts": attempts,
                },
            )
    return (
        source,
        states,
        {
            "status": "UNCHANGED" if unchanged else "REJECTED",
            "reason": "no_supported_cavity" if unchanged else attempts[-1]["reason"],
            "attempts": attempts,
        },
    )


def _restricted_delaunay_once(
    mesh: Any,
    query: Any,
    tolerance: float = 0.001,
    *,
    focus_vertices: NDArray[Any] | None = None,
    candidate_faces: NDArray[Any] | None = None,
    return_proposal: bool = False,
    validation_meshes: list[Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if not np.isfinite(vertices).all() or not len(mesh.faces):
        raise ValueError("restricted Delaunay requires a finite nonempty surface")
    center = vertices.mean(0)
    scale = float(np.ptp(vertices, axis=0).max())
    if scale <= 0:
        raise ValueError("restricted Delaunay requires a nondegenerate surface")
    if np.linalg.matrix_rank(vertices - center, tol=scale * 1e-10) < 3:
        return mesh.copy(), {
            "method": "planar_surface_retained",
            "faces": len(mesh.faces),
            "coverage_certified": True,
            "topology": surface_topology(mesh),
        }
    faces = _exact_faces(mesh) if candidate_faces is None else np.asarray(candidate_faces).copy()
    if not len(faces):
        raise ValueError("restricted Delaunay returned no observed surface")
    candidate = trimesh.Trimesh(
        vertices=vertices.copy(),
        faces=faces,
        vertex_colors=mesh.visual.vertex_colors.copy(),
        process=False,
    )
    # Orient against the observed part, then propagate consistent winding.
    _, ids = query.nearest(mesh, candidate.triangles_center)
    reverse = (candidate.face_normals * mesh.face_normals[ids]).sum(1) < 0
    faces[reverse] = faces[reverse, ::-1]
    candidate.faces = faces
    trimesh.repair.fix_winding(candidate)  # type: ignore[no-untyped-call]
    candidate, patch_report = manifold_patches(
        mesh,
        candidate,
        focus_vertices=focus_vertices,
        validation_meshes=validation_meshes,
    )
    from ._collision_shell import approximation_bound

    coverage = approximation_bound(mesh, candidate, query, tolerance, max_cells=2000000)
    fidelity = approximation_bound(candidate, mesh, query, tolerance, max_cells=2000000)
    return candidate, {
        "method": "per_part_restricted_delaunay_voronoi_surface_intersection",
        "source_faces": len(mesh.faces),
        "faces": len(candidate.faces),
        "vertices_preserved": True,
        "attributes_preserved": True,
        "cross_part_triangles_created": False,
        "predicates": "CGAL_exact_predicates_exact_constructions",
        "hull_rays_included": True,
        "coverage_certified": coverage["status"] == fidelity["status"] == "CERTIFIED",
        "source_to_remesh": coverage,
        "remesh_to_source": fidelity,
        "surface_budget_m": tolerance,
        **({"proposal_faces": faces.copy()} if return_proposal else {}),
        **patch_report,
    }


def _exact_faces(mesh: Any) -> Any:
    """Build the tiny CGAL worker once per source/compiler version, atomically."""
    import fcntl
    import hashlib
    import io
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    source = Path(__file__).with_name("_restricted_delaunay.cpp")
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("restricted Delaunay requires a C++17 compiler and libcgal-dev")
    version = subprocess.run([compiler, "--version"], check=True, capture_output=True).stdout
    key = hashlib.sha256(source.read_bytes() + version).hexdigest()
    cache = Path(tempfile.gettempdir()) / f"physrealx-rdel-{os.getuid()}"
    cache.mkdir(mode=0o700, exist_ok=True)
    executable = cache / key
    with (cache / (key + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not executable.is_file():
            build = cache / (key + f".{os.getpid()}.tmp")
            result = subprocess.run(
                [compiler, "-std=c++17", "-O2", str(source), "-o", str(build), "-lgmp", "-lmpfr"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode:
                build.unlink(missing_ok=True)
                raise RuntimeError(
                    "CGAL rDel build failed; install libcgal-dev, libgmp-dev and libmpfr-dev: "
                    + result.stderr[-3000:]
                )
            os.replace(build, executable)
    stream = io.StringIO()
    stream.write(f"{len(mesh.vertices)} {len(mesh.faces)}\n")
    np.savetxt(stream, mesh.vertices, fmt="%.17g")
    np.savetxt(stream, mesh.faces, fmt="%d")
    result = subprocess.run(
        [str(executable)],
        input=stream.getvalue(),
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    values = np.fromstring(result.stdout, sep=" ", dtype=np.int64)
    if not len(values) or len(values) != 1 + 3 * values[0]:
        raise ValueError("invalid restricted Delaunay worker output")
    faces = values[1:].reshape(-1, 3)
    if np.any(faces < 0) or np.any(faces >= len(mesh.vertices)):
        raise ValueError("restricted Delaunay returned an invalid vertex index")
    return faces
