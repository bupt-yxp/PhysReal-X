"""Collision-only refinement over an immutable articulated source asset."""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel, GeometryRepresentation, GeometryRole
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations._collision_refit import (
    SurfaceCoverage,
    SurfacePatch,
    _halfspace_polyhedron,
    bind_surface,
    clip_patch_support,
    fit_one_sided_patch,
    fit_patches,
    nearest,
)
from physrealx.integrations._refinement_checkpoint import RefinementCheckpointStore
from physrealx.integrations._surface_query import SurfaceQuery, create_surface_query
from physrealx.integrations._visual_motion_audit import audit_visual_motion
from physrealx.integrations.assembly import assembled_asset
from physrealx.integrations.asset_prior import ReferenceAssemblyInput, reference_prior_asset
from physrealx.integrations.collision import load_meshes, mesh_reference
from physrealx.integrations.physical_feasibility import (
    FeasibilityConfig,
    evaluate_asset_feasibility,
)

# MuJoCo's convex contact mode and float32 GLB vertices quantize sub-micrometre
# support changes.  Keep comparison slack below one percent of the default 50 um
# safety reserve; the independent physics gate still enforces its exact 1 mm limit.
_PENETRATION_NUMERIC_ATOL_M = 5e-7
_SURFACE_NUMERIC_ATOL_M = 1e-7
_COVERAGE_NUMERIC_ATOL = 1e-6
_REFINEMENT_CHECKPOINT_CONTRACT = "source-surface-support/3"
_INDEPENDENT_SWEEP_SAMPLES = 361
_INDEPENDENT_ADAPTIVE_SWEEP_LEVELS = 3


def _meets_penetration_target(value_m: float, target_m: float) -> bool:
    """Compare solver penetration with sub-micrometre numerical tolerance."""
    return value_m <= target_m + _PENETRATION_NUMERIC_ATOL_M


def _definitely_fails_fidelity(
    value: dict[str, Any], config: ArticulatedCollisionRefinementConfig
) -> bool:
    """Reject failures too large for float32 GLB round-trip to change."""
    return not config.fidelity_metrics_within_limits(value)


class ArticulatedCollisionRefinementConfig(DomainModel):
    """Finite refinement budget and local surface/contact fidelity gates."""

    max_local_error_m: float = Field(default=0.003, gt=0, le=0.01)
    surface_fidelity_reserve_m: float = Field(default=0.00025, ge=0, le=0.001)
    coverage_distance_m: float = Field(default=0.002, gt=0, le=0.01)
    min_patch_coverage: float = Field(default=0.99, gt=0, le=1)
    patch_size_m: float = Field(default=0.01, gt=0, le=0.1)
    proxy_skin_m: float = Field(default=0.00025, gt=0, le=0.002)
    shape_trust_region_m: float = Field(default=0.0018, gt=0, le=0.005)
    min_proxy_volume_ratio: float = Field(default=0.1, gt=0, le=1)
    min_proxy_thickness_ratio: float = Field(default=0.25, gt=0, le=1)
    penetration_safety_margin_m: float = Field(default=5e-5, ge=0, le=0.0001)
    max_rounds: int = Field(default=6, ge=1, le=128)
    max_parts_per_link: int = Field(default=768, ge=2, le=2048)
    max_patch_updates_per_link: int = Field(default=128, ge=1, le=512)
    max_collision_patch_updates_per_link: int = Field(default=8, ge=1, le=64)
    max_fidelity_rounds: int = Field(default=32, ge=1, le=64)
    max_materialized_candidates_per_round: int = Field(default=3, ge=1, le=16)
    max_exact_candidates_per_round: int = Field(default=2, ge=1, le=8)
    surface_query_backend: Literal["auto", "cpu", "warp_cuda"] = "auto"
    gpu_devices: tuple[int, ...] = ()
    require_accelerator: bool = False
    cpu_query_workers: int = Field(default=0, ge=0, le=64)
    surface_query_batch_size: int = Field(default=4096, ge=512, le=1_048_576)
    visual_audit_samples: int = Field(default=61, ge=3, le=361)
    visual_audit_probes_per_link: int = Field(default=2048, ge=128, le=16384)
    checkpoint_mode: Literal["auto", "disabled", "required"] = "auto"
    resume_checkpoint: ArtifactRef | None = None
    seed: int = Field(default=1, ge=1, le=2_147_483_647)

    @model_validator(mode="after")
    def ordered_tolerances(self) -> ArticulatedCollisionRefinementConfig:
        if not (
            self.proxy_skin_m
            < self.shape_trust_region_m
            <= self.coverage_distance_m
            <= self.max_local_error_m
        ):
            raise ValueError(
                "require proxy skin < shape trust region <= coverage distance <= local error limit"
            )
        if self.max_local_error_m - self.surface_fidelity_reserve_m < self.coverage_distance_m:
            raise ValueError(
                "surface fidelity reserve must leave at least the configured coverage distance"
            )
        if len(set(self.gpu_devices)) != len(self.gpu_devices) or any(
            device < 0 for device in self.gpu_devices
        ):
            raise ValueError("GPU device indices must be unique and nonnegative")
        if self.require_accelerator and self.surface_query_backend == "cpu":
            raise ValueError("accelerator-required search cannot select the CPU backend")
        return self

    def fidelity_metrics_within_limits(self, value: dict[str, Any]) -> bool:
        """Evaluate serialized hard metrics with only round-trip numerical slack."""
        try:
            maximum = float(value["bidirectional_sample_max_m"])
            worst_coverage = float(value["worst_cell_coverage"])
            contact_coverage = float(value["contact_coverage"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            all(math.isfinite(item) for item in (maximum, worst_coverage, contact_coverage))
            and maximum <= self.max_local_error_m + _SURFACE_NUMERIC_ATOL_M
            and worst_coverage >= self.min_patch_coverage - _COVERAGE_NUMERIC_ATOL
            and contact_coverage >= self.min_patch_coverage - _COVERAGE_NUMERIC_ATOL
        )


def _conflicting_parts(
    evaluation: dict[str, Any], prefixes: dict[str, str], tolerance: float
) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = {link: set() for link in prefixes}
    for pair in evaluation["collision_pairs"]:
        if pair["max_penetration_m"] <= tolerance:
            continue
        for side in ("a", "b"):
            link, name = str(pair[f"link_{side}"]), str(pair[f"geom_{side}"])
            if link not in prefixes or not name.startswith(prefixes[link]):
                raise DomainValidationError("failed collision pair has no canonical source")
            index = int(name.removeprefix(prefixes[link]))
            if index < 0:
                raise DomainValidationError("collision part index must be nonnegative")
            selected[link].add(index)
    return selected


def _stabilize_contact_normals(
    contacts: list[dict[str, Any]], *, cluster_cosine: float = 0.82
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve sign-ambiguous penetration normals without weakening validation.

    Deeply overlapping convex shapes can make EPA select alternating faces at
    adjacent joint samples.  The reported axes then differ only by sign, and a
    support-plane optimizer incorrectly trims both sides of the same proxy.  We
    cluster unoriented axes per geometry pair, anchor each cluster at its deepest
    contact, and align shallower samples to that anchor.  Exact MuJoCo sweeps still
    accept or reject every resulting candidate.
    """
    import numpy as np

    if not 0 < cluster_cosine < 1:
        raise ValueError("contact-normal cluster cosine must be between zero and one")
    adjusted = [dict(contact) for contact in contacts]
    clusters: dict[tuple[tuple[str, str], tuple[str, str]], list[dict[str, Any]]] = {}
    flipped = 0
    order = sorted(
        range(len(contacts)),
        key=lambda index: (
            -float(contacts[index].get("penetration_m", 0.0)),
            float(contacts[index].get("q", 0.0)),
            index,
        ),
    )
    for index in order:
        contact = contacts[index]
        endpoint_a = (str(contact["link_a"]), str(contact["geom_a"]))
        endpoint_b = (str(contact["link_b"]), str(contact["geom_b"]))
        canonical_sign = 1.0 if endpoint_a <= endpoint_b else -1.0
        pair = (endpoint_a, endpoint_b) if endpoint_a <= endpoint_b else (endpoint_b, endpoint_a)
        normal = canonical_sign * np.asarray(contact["normal_a_to_b"], dtype=float)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal).all() or norm <= 1e-12:
            raise DomainValidationError("collision evidence contains an invalid contact normal")
        normal /= norm
        pair_clusters = clusters.setdefault(pair, [])
        similarities = [abs(float(normal @ cluster["axis"])) for cluster in pair_clusters]
        if similarities and max(similarities) >= cluster_cosine:
            cluster = pair_clusters[int(np.argmax(similarities))]
            was_flipped = float(normal @ cluster["axis"]) < 0
            aligned = -normal if was_flipped else normal
            weight = max(float(contact.get("penetration_m", 0.0)), 1e-12)
            direction_sum = cluster["direction_sum"] + weight * aligned
            cluster["direction_sum"] = direction_sum
            cluster["axis"] = direction_sum / np.linalg.norm(direction_sum)
            cluster["samples"] += 1
        else:
            was_flipped = False
            aligned = normal
            weight = max(float(contact.get("penetration_m", 0.0)), 1e-12)
            pair_clusters.append(
                {
                    "axis": aligned.copy(),
                    "direction_sum": weight * aligned,
                    "samples": 1,
                }
            )
        if was_flipped:
            flipped += 1
        adjusted[index]["normal_a_to_b"] = (canonical_sign * aligned).tolist()
    return adjusted, {
        "contacts": len(contacts),
        "geometry_pairs": len(clusters),
        "direction_clusters": sum(len(values) for values in clusters.values()),
        "sign_flips": flipped,
        "cluster_cosine": cluster_cosine,
    }


def _split_with_current_envelope(
    visual: Any,
    patch: SurfacePatch,
    config: ArticulatedCollisionRefinementConfig,
    query: Any = nearest,
) -> list[SurfacePatch] | None:
    """Split source faces while preserving every accepted parent support plane."""
    from scipy.spatial import ConvexHull, QhullError

    children = fit_patches(visual, patch, config.proxy_skin_m)
    if not children:
        return children
    try:
        equations = ConvexHull(patch.mesh.vertices).equations
    except QhullError:
        return None
    envelope = [(row[:3], -float(row[3])) for row in equations]
    clipped_children: list[SurfacePatch] = []
    for child in children:
        violations = (equations[:, :3] @ child.mesh.vertices.T + equations[:, 3, None]).max(axis=0)
        outside = child.mesh.vertices[violations > 1e-9]
        inherited_displacement_m = 0.0
        if len(outside):
            inherited_displacement_m = float(query(patch.mesh, outside)[0].max())
        clipped, _ = clip_patch_support(
            child,
            visual,
            envelope,
            # This is not a new shape update: the extra allowance is exactly the
            # distance needed to inherit the already accepted parent envelope.
            trust_region_m=max(
                config.shape_trust_region_m,
                inherited_displacement_m + config.proxy_skin_m,
            ),
            min_thickness_ratio=config.min_proxy_thickness_ratio,
            min_volume_ratio=config.min_proxy_volume_ratio,
            query=query,
        )
        if clipped is None:
            return _partition_current_envelope(visual, patch, config, query=query)
        # Envelope inheritance establishes a new topology; future shape updates
        # must preserve this accepted child's volume/thickness, not the expanded
        # source-shell proposal that existed only before intersection.
        clipped_children.append(SurfacePatch(clipped.mesh, child.faces, child.depth))
    return clipped_children


def _partition_current_envelope(
    visual: Any,
    patch: SurfacePatch,
    config: ArticulatedCollisionRefinementConfig,
    *,
    query: Any = nearest,
) -> list[SurfacePatch] | None:
    """Bisect an accepted convex proxy without adding or removing occupied volume.

    A heavily refitted parent can be too tight for source-shell children: rebuilding
    each child from source faces first expands outside the parent, and clipping it
    back may become degenerate.  This fallback partitions the current convex volume
    at the median source-face position. The two children conserve the parent union,
    every source face retains exactly one owner, and later source coverage plus the
    exact sweep remain authoritative.
    """
    import numpy as np

    if len(patch.faces) < 2:
        return None
    centers = np.asarray(visual.triangles_center[patch.faces], dtype=float)
    centered = centers - centers.mean(axis=0)
    try:
        _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not len(singular) or float(singular[0]) <= 1e-12:
        return None
    normal = np.asarray(basis[0], dtype=float)
    normal /= np.linalg.norm(normal)
    source_projection = centers @ normal
    split = float(np.median(source_projection))
    parent_projection = np.asarray(patch.mesh.vertices, dtype=float) @ normal
    lower, upper = float(parent_projection.min()), float(parent_projection.max())
    margin = max(1e-9, 1e-6 * (upper - lower))
    if upper - lower <= 2 * margin:
        return None
    split = float(np.clip(split, lower + margin, upper - margin))
    left = np.asarray(patch.faces)[source_projection <= split]
    right = np.asarray(patch.faces)[source_projection > split]
    if not len(left) or not len(right):
        order = np.argsort(source_projection, kind="stable")
        boundary = len(order) // 2
        left = np.asarray(patch.faces)[order[:boundary]]
        right = np.asarray(patch.faces)[order[boundary:]]
        split = float(
            0.5 * (source_projection[order[boundary - 1]] + source_projection[order[boundary]])
        )
        split = float(np.clip(split, lower + margin, upper - margin))
    children: list[SurfacePatch] = []
    for faces, direction, support in (
        (left, normal, split),
        (right, -normal, -split),
    ):
        seed = SurfacePatch(patch.mesh, faces, patch.depth + 1)
        clipped, _ = clip_patch_support(
            seed,
            visual,
            [(direction, support)],
            trust_region_m=config.shape_trust_region_m,
            min_thickness_ratio=1e-6,
            min_volume_ratio=1e-6,
            query=query,
        )
        if clipped is None:
            return None
        children.append(SurfacePatch(clipped.mesh, faces, patch.depth + 1))
    child_volume = sum(float(child.mesh.volume) for child in children)
    if not np.isclose(child_volume, float(patch.mesh.volume), rtol=5e-5, atol=1e-12):
        return None
    ids = np.concatenate([child.faces for child in children])
    if not np.array_equal(np.sort(ids), np.sort(np.asarray(patch.faces))):
        raise MethodExecutionError("convex envelope partition changed source-face ownership")
    return children


def _glb_stable_convex_proxy(mesh: Any) -> Any | None:
    """Retriangulate one float32 vertex hull and prove its GLB topology is valid."""
    import io
    from contextlib import suppress

    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull, QhullError

    source_points = np.unique(
        np.asarray(mesh.vertices, dtype=np.float32).astype(np.float64), axis=0
    )
    if len(source_points) < 4:
        return None

    def qhull_mesh(points: Any, options: str) -> Any:
        center = points.mean(axis=0)
        scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
        hull = ConvexHull((points - center) / scale, qhull_options=options)
        faces = hull.simplices.copy()
        for index, (face, equation) in enumerate(zip(faces, hull.equations, strict=True)):
            triangle = points[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if float(normal @ equation[:3]) < 0:
                faces[index, 1], faces[index, 2] = faces[index, 2], faces[index, 1]
        return trimesh.Trimesh(vertices=points, faces=faces, process=True)

    pending = [source_points]
    seen: set[bytes] = set()
    while pending:
        points = pending.pop(0)
        key = np.asarray(points, dtype=np.float32).tobytes()
        if key in seen or len(points) < 4:
            continue
        seen.add(key)
        candidates = []
        with suppress(QhullError, ValueError, RuntimeError):
            candidates.append(_halfspace_polyhedron(points))
        for options in ("Qt Qx", "QJ Qt Qx"):
            try:
                candidates.append(qhull_mesh(points, options))
            except (QhullError, ValueError, RuntimeError):
                continue
        for candidate in candidates:
            if not candidate.is_volume or not candidate.is_convex:
                continue
            payload = candidate.export(file_type="glb")
            decoded_scene: Any = trimesh.load(
                io.BytesIO(payload),
                file_type="glb",
                force="scene",
                resolver={},
                process=True,
            )
            values = list(decoded_scene.geometry.values())
            if len(values) != 1:
                continue
            decoded = values[0]
            retained = np.unique(
                np.asarray(decoded.vertices, dtype=np.float32).astype(np.float64),
                axis=0,
            )
            if not decoded.is_volume or not decoded.is_convex:
                if len(retained) < len(points):
                    pending.append(retained)
                continue
            forward = trimesh.proximity.closest_point_naive(  # type: ignore[no-untyped-call]
                decoded, source_points
            )[1]
            reverse = trimesh.proximity.closest_point_naive(  # type: ignore[no-untyped-call]
                mesh, decoded.vertices
            )[1]
            if max(float(forward.max()), float(reverse.max())) <= _PENETRATION_NUMERIC_ATOL_M:
                return candidate
    return None


def _publish_parts(
    store: ArtifactStore, link: str, parts: list[Any]
) -> tuple[ArtifactRef, list[int]]:
    import hashlib
    import io
    from collections import defaultdict

    import numpy as np
    import trimesh

    def signature(mesh: Any) -> str:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        vertices = vertices[np.lexsort(vertices.T)]
        return hashlib.sha256(vertices.tobytes()).hexdigest()

    published_parts = list(parts)
    payload = b""
    for attempt in range(2):
        scene: Any = trimesh.Scene()
        for index, part in enumerate(published_parts):
            scene.add_geometry(
                part,
                node_name=f"{link}_{index:05d}",
                geom_name=f"convex_{index:05d}",
            )
        payload = scene.export(file_type="glb")
        decoded_scene: Any = trimesh.load(
            io.BytesIO(payload), file_type="glb", force="scene", resolver={}, process=True
        )
        invalid_indices = [
            int(name.removeprefix("convex_"))
            for name, part in decoded_scene.geometry.items()
            if not part.is_volume or not part.is_convex
        ]
        if len(decoded_scene.geometry) != len(published_parts):
            raise MethodExecutionError(
                "refitted collision proxy GLB changed the number of convex parts"
            )
        if not invalid_indices:
            break
        if attempt:
            raise MethodExecutionError(
                f"collision proxies remain invalid at GLB precision: {invalid_indices[:3]}"
            )
        for index in invalid_indices:
            # Rebuild only the proxy whose topology actually degenerates after
            # float32 serialization. Stable patches retain their accepted faces.
            canonical = _glb_stable_convex_proxy(published_parts[index])
            if canonical is None:
                raise MethodExecutionError(
                    f"collision proxy {link}/{index} is invalid at GLB precision"
                )
            published_parts[index] = canonical
    order: dict[str, list[int]] = defaultdict(list)
    for index, part in enumerate(published_parts):
        order[signature(part)].append(index)
    ref = store.put_bytes(payload, media_type="model/gltf-binary")
    decoded = load_meshes(store, ref)
    invalid = [
        {"index": index, "is_volume": bool(part.is_volume), "is_convex": bool(part.is_convex)}
        for index, part in enumerate(decoded)
        if not part.is_volume or not part.is_convex
    ]
    if len(decoded) != len(published_parts) or invalid:
        raise MethodExecutionError(
            "refitted collision proxy failed convex GLB roundtrip: "
            f"decoded={len(decoded)} expected={len(published_parts)} invalid={invalid[:3]}"
        )
    permutation = []
    for part in decoded:
        matches = order.get(signature(part), [])
        if not matches:
            raise MethodExecutionError("collision roundtrip changed a source patch")
        permutation.append(matches.pop())
    return ref, permutation


def _maximum_partition_seam_error(
    visual: Any,
    patches: list[SurfacePatch],
    changed: set[int],
    query: Any = nearest,
) -> float:
    """Measure changed source seams against the collision union actually consumed."""
    import numpy as np

    face_to_patch = {
        int(face): index for index, patch in enumerate(patches) for face in patch.faces
    }
    probes = []
    for left, right in visual.face_adjacency:
        a, b = face_to_patch[int(left)], face_to_patch[int(right)]
        if a == b or not changed.intersection((a, b)):
            continue
        common = np.intersect1d(visual.faces[int(left)], visual.faces[int(right)])
        edge = visual.vertices[common]
        if len(edge) == 2:
            edge = np.vstack([edge, edge.mean(axis=0)])
        probes.extend(edge)
    if not probes:
        return 0.0
    points = np.unique(np.asarray(probes), axis=0)
    distances = np.full(len(points), np.inf)
    for patch in patches:
        bounds = np.asarray(patch.mesh.bounds)
        lower_bound = np.linalg.norm(
            np.maximum(np.maximum(bounds[0] - points, points - bounds[1]), 0.0),
            axis=1,
        )
        candidates = np.flatnonzero(lower_bound <= distances + 1e-12)
        if len(candidates):
            measured, _ = query(patch.mesh, points[candidates])
            distances[candidates] = np.minimum(distances[candidates], measured)
    return float(distances.max())


def _materialize_candidate(
    store: ArtifactStore,
    asset: Any,
    patches: dict[str, list[SurfacePatch]],
) -> tuple[dict[str, list[SurfacePatch]], dict[str, ArtifactRef], Any]:
    """Round-trip the exact proxies that simulation and evidence will consume."""
    decoded_patches: dict[str, list[SurfacePatch]] = {}
    references: dict[str, ArtifactRef] = {}
    for link, values in patches.items():
        references[link], order = _publish_parts(store, link, [patch.mesh for patch in values])
        decoded_patches[link] = [values[index] for index in order]
        decoded = load_meshes(store, references[link])
        for patch, mesh in zip(decoded_patches[link], decoded, strict=True):
            patch.mesh = mesh
    candidate = asset.model_copy(
        update={
            "geometries": tuple(
                geometry.model_copy(
                    update={"uri": f"artifact://{references[geometry.link].artifact_id}"}
                )
                if geometry.role is GeometryRole.COLLISION
                else geometry
                for geometry in asset.geometries
            )
        }
    )
    return decoded_patches, references, candidate


def _support_line_search_fractions(
    remaining_penetration_m: float, fraction_hint: float | None
) -> list[float]:
    """Schedule bounded under- and over-relaxed sequential-convex steps.

    Contact normals linearize a nonlinear convex-pair penetration metric.  Near
    the target, a nominal 1.0 step can remove only part of the measured excess and
    converge asymptotically.  Probe up to 2x the linear request only in the final
    0.2 mm band; exact thickness, volume, surface, and MuJoCo gates still decide
    whether that over-relaxed proposal is admissible.
    """
    maximum_fraction = 2.0 if remaining_penetration_m <= 0.0002 else 1.0
    base = list(
        (2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
        if maximum_fraction > 1.0
        else (0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.5, 0.75, 1.0)
    )
    fractions: list[float] = []
    if fraction_hint is not None and 0 < fraction_hint <= maximum_fraction:
        expansion = 2.0 if maximum_fraction > 1.0 else 1.5
        fractions.extend(
            [
                min(maximum_fraction, expansion * fraction_hint),
                min(maximum_fraction, 1.5 * fraction_hint),
                fraction_hint,
                0.75 * fraction_hint,
                0.5 * fraction_hint,
            ]
        )
        base = [value for value in base if value <= fraction_hint]
    fractions.extend(base)
    return list(dict.fromkeys(value for value in fractions if value >= 1e-4))


def _shape_refit_candidates(
    asset: Any,
    patches: dict[str, list[SurfacePatch]],
    visuals: dict[str, Any],
    coverage: dict[str, SurfaceCoverage],
    evaluation: dict[str, Any],
    tolerance: float,
    config: ArticulatedCollisionRefinementConfig,
    query: Any = nearest,
    fraction_hint: float | None = None,
) -> list[dict[str, Any]]:
    """Clip local convex supports using every failed pose in the joint sweep."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    from physrealx.integrations._asset_preview import link_transforms, pose_matrix

    joint = asset.joints[0]
    geometries = {g.link: g for g in asset.geometries if g.role is GeometryRole.COLLISION}
    contacts = list(evaluation.get("contact_constraints", ()))
    if not contacts:
        contacts = [
            {
                **pair,
                "penetration_m": pair["max_penetration_m"],
                "q": pair["peak_q"],
            }
            for pair in evaluation["collision_pairs"]
            if pair["max_penetration_m"] > tolerance
        ]
    if not contacts:
        return []
    contacts, normal_stabilization = _stabilize_contact_normals(contacts)

    def contact_regions() -> dict[str, dict[str, int]]:
        regions: dict[str, dict[str, int]] = {}
        for link, visual in visuals.items():
            active_faces = {
                int(face)
                for contact in contacts
                for side in ("a", "b")
                if contact[f"link_{side}"] == link
                for face in patches[link][
                    int(str(contact[f"geom_{side}"]).rsplit("/", 1)[-1])
                ].faces
            }
            neighbors: dict[int, set[int]] = {face: set() for face in active_faces}
            for left, right in visual.face_adjacency:
                a, b = int(left), int(right)
                if a in active_faces and b in active_faces:
                    neighbors[a].add(b)
                    neighbors[b].add(a)
            pending = set(active_faces)
            components: list[list[int]] = []
            while pending:
                stack = [pending.pop()]
                component: list[int] = []
                while stack:
                    face = stack.pop()
                    component.append(face)
                    adjacent = neighbors[face].intersection(pending)
                    pending.difference_update(adjacent)
                    stack.extend(adjacent)
                components.append(sorted(component))
            regions[link] = {
                "components": len(components),
                "source_faces": sum(len(component) for component in components),
            }
        return regions

    regions = contact_regions()
    support_variables: dict[tuple[str, int, float, float, float], tuple[str, int, Any, float]] = {}
    linear_constraints: list[tuple[float, list[tuple[Any, float]]]] = []
    for contact in contacts:
        depth = float(contact["penetration_m"])
        if depth <= tolerance:
            continue
        transforms = link_transforms(asset, {joint.name: float(contact["q"])})
        normal = np.asarray(contact["normal_a_to_b"], dtype=float)
        terms: list[tuple[Any, float]] = []
        for side, direction in (("a", 1.0), ("b", -1.0)):
            link = str(contact[f"link_{side}"])
            index = int(str(contact[f"geom_{side}"]).rsplit("/", 1)[-1])
            geometry = geometries[link]
            transform = transforms[link] @ pose_matrix(geometry.local_pose)
            linear = transform[:3, :3] * np.asarray(geometry.scale)[None, :]
            local_normal = direction * linear.T @ normal
            norm = float(np.linalg.norm(local_normal))
            if norm <= 1e-12:
                continue
            unit = local_normal / norm
            key = (link, index, *tuple(np.round(unit, 4).tolist()))
            if key not in support_variables:
                support = float(np.max(np.asarray(patches[link][index].mesh.vertices) @ unit))
                support_variables[key] = (link, index, unit, support)
            terms.append((key, norm))
        if len(terms) == 2:
            linear_constraints.append((depth - tolerance, terms))
    if not support_variables or not linear_constraints:
        return []
    variable_order = list(support_variables)
    variable_index = {key: index for index, key in enumerate(variable_order)}
    count = len(variable_order)
    constraint_count = len(linear_constraints)
    width = count + 1 + constraint_count
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    for constraint_index, (_, terms) in enumerate(linear_constraints):
        for key, coefficient in terms:
            matrix_rows.append(constraint_index)
            matrix_columns.append(variable_index[key])
            matrix_values.append(-coefficient)
        matrix_rows.append(constraint_index)
        matrix_columns.append(count + 1 + constraint_index)
        matrix_values.append(-1.0)
    for variable in range(count):
        row = constraint_count + variable
        matrix_rows.extend((row, row))
        matrix_columns.extend((variable, count))
        matrix_values.extend((1.0, -1.0))
    constraint_matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(constraint_count + count, width),
    ).tocsr()

    candidates: list[dict[str, Any]] = []
    maximum_penetration = float(
        evaluation.get(
            "max_penetration_m",
            max(float(contact["penetration_m"]) for contact in contacts),
        )
    )
    remaining_penetration = maximum_penetration - tolerance
    # Sequential convex optimization needs a real line search: a valid LP step
    # can still violate the nonlinear thickness or volume gates after the exact
    # half-space intersection.  Backtrack before trying a more aggressive step;
    # never weaken a geometry gate merely to keep making progress.
    fractions = _support_line_search_fractions(remaining_penetration, fraction_hint)
    scheduled_fractions = set(fractions)
    bisections_scheduled = 0
    passing_fractions: list[float] = []
    failing_fractions: list[float] = []
    fraction_index = 0
    while fraction_index < len(fractions):
        fraction = fractions[fraction_index]
        fraction_index += 1
        requested_clearance = np.asarray(
            [fraction * target for target, _ in linear_constraints], dtype=float
        )
        objective = np.zeros(width)
        objective[:count] = 1e-6 / count
        objective[count] = 1.0 / config.shape_trust_region_m
        # Normalized L1 slacks form a robust joint objective.  A full violation
        # of every contact costs 100x the largest trust-region step, while an
        # individually impossible contact remains bounded and diagnosable.
        objective[count + 1 :] = 100.0 / (constraint_count * np.maximum(requested_clearance, 1e-12))
        blocked_patches: set[tuple[str, int]] = set()
        rejected_by_patch: dict[tuple[str, int], dict[str, Any]] = {}
        fallback: tuple[Any, ...] | None = None
        accepted: tuple[Any, ...] | None = None
        active_set_rounds = 0
        while active_set_rounds <= len(support_variables):
            active_set_rounds += 1
            variable_bounds = [
                (0.0, 0.0)
                if (support_variables[key][0], support_variables[key][1]) in blocked_patches
                else (0.0, config.shape_trust_region_m)
                for key in variable_order
            ]
            solution = linprog(
                objective,
                A_ub=constraint_matrix,
                b_ub=np.r_[-requested_clearance, np.zeros(count)],
                bounds=[
                    *variable_bounds,
                    (0.0, config.shape_trust_region_m),
                    *((0.0, float(requested)) for requested in requested_clearance),
                ],
                method="highs",
            )
            if not solution.success:
                if fallback is not None:
                    accepted = fallback
                else:
                    logging.getLogger(__name__).info(
                        "support optimization rejected fraction=%s reason=%s",
                        fraction,
                        solution.message,
                    )
                break

            trims = solution.x[:count]
            slacks = solution.x[count + 1 :]
            cuts: dict[str, dict[int, list[tuple[Any, float]]]] = {link: {} for link in patches}
            cut_variables: dict[str, dict[int, list[Any]]] = {link: {} for link in patches}
            for key, trim in zip(variable_order, trims, strict=True):
                if trim <= 1e-9:
                    continue
                link, index, unit, support = support_variables[key]
                cuts[link].setdefault(index, []).append((unit, support - float(trim)))
                cut_variables[link].setdefault(index, []).append(key)

            proposal = {link: list(values) for link, values in patches.items()}
            shape_metrics: dict[str, dict[int, dict[str, Any]]] = {link: {} for link in patches}
            applied_variables: set[Any] = set()
            newly_blocked: set[tuple[str, int]] = set()
            for link, indexed in cuts.items():
                for index, halfspaces in indexed.items():
                    clipped, clip_metrics = clip_patch_support(
                        patches[link][index],
                        visuals[link],
                        halfspaces,
                        trust_region_m=config.shape_trust_region_m,
                        min_thickness_ratio=config.min_proxy_thickness_ratio,
                        min_volume_ratio=config.min_proxy_volume_ratio,
                        query=query,
                    )
                    if clipped is None:
                        patch_key = (link, index)
                        newly_blocked.add(patch_key)
                        rejected_by_patch[patch_key] = {
                            "link": link,
                            "patch": index,
                            **clip_metrics,
                        }
                        continue
                    proposal[link][index], shape_metrics[link][index] = (
                        clipped,
                        clip_metrics,
                    )
                    applied_variables.update(cut_variables[link][index])
            state = (
                proposal,
                shape_metrics,
                applied_variables,
                trims,
                slacks,
            )
            if any(shape_metrics.values()):
                fallback = state
            if not newly_blocked and any(shape_metrics.values()):
                # A locally valid convex cut can still consume the global surface
                # error budget. Freeze only changed patches identified by the
                # bidirectional coverage gate, then re-solve so the opposing link
                # can absorb the requested clearance when it has fidelity margin.
                for link in patches:
                    link_fidelity, violations, _ = coverage[link].evaluate_with_priorities(
                        proposal[link],
                        max_error_m=config.max_local_error_m,
                        coverage_distance_m=config.coverage_distance_m,
                        min_coverage=config.min_patch_coverage,
                    )
                    if link_fidelity["status"] == "PASSED":
                        continue
                    for index in violations.intersection(cuts[link]):
                        patch_key = (link, index)
                        newly_blocked.add(patch_key)
                        rejected_by_patch[patch_key] = {
                            "link": link,
                            "patch": index,
                            "status": "REJECTED",
                            "reason": "surface_fidelity_gate",
                            "bidirectional_sample_max_m": link_fidelity[
                                "bidirectional_sample_max_m"
                            ],
                            "worst_cell_coverage": link_fidelity["worst_cell_coverage"],
                            "contact_coverage": link_fidelity["contact_coverage"],
                        }
            if not newly_blocked:
                accepted = state if any(shape_metrics.values()) else fallback
                break
            previous_count = len(blocked_patches)
            blocked_patches.update(newly_blocked)
            if len(blocked_patches) == previous_count:
                accepted = fallback
                break

        if rejected_by_patch:
            reasons: dict[str, int] = {}
            for value in rejected_by_patch.values():
                reason = str(value.get("reason", "unknown"))
                reasons[reason] = reasons.get(reason, 0) + 1
            logging.getLogger(__name__).info(
                "support active-set fraction=%s rounds=%s blocked=%s reasons=%s",
                fraction,
                active_set_rounds,
                len(rejected_by_patch),
                reasons,
            )
        if accepted is None:
            continue
        proposal, shape_metrics, applied_variables, trims, slacks = accepted
        rejected_patches = list(rejected_by_patch.values())

        max_adjacency_gap = 0.0
        for link, visual in visuals.items():
            max_adjacency_gap = max(
                max_adjacency_gap,
                _maximum_partition_seam_error(
                    visual, proposal[link], set(shape_metrics[link]), query=query
                ),
            )
        if max_adjacency_gap > config.max_local_error_m:
            logging.getLogger(__name__).info(
                "support candidate rejected fraction=%s adjacency_gap_m=%s",
                fraction,
                max_adjacency_gap,
            )
            continue

        fidelity = {
            link: coverage[link].evaluate(
                proposal[link],
                max_error_m=config.max_local_error_m,
                coverage_distance_m=config.coverage_distance_m,
                min_coverage=config.min_patch_coverage,
            )[0]
            for link in patches
        }
        if not all(value["status"] == "PASSED" for value in fidelity.values()):
            logging.getLogger(__name__).info(
                "support candidate remains outside fidelity gate fraction=%s fidelity=%s",
                fraction,
                {
                    link: {
                        key: value[key]
                        for key in (
                            "status",
                            "bidirectional_sample_max_m",
                            "worst_cell_coverage",
                            "contact_coverage",
                        )
                    }
                    for link, value in fidelity.items()
                },
            )
        accepted_metrics = [value for values in shape_metrics.values() for value in values.values()]
        linear_residuals = np.asarray(
            [
                requested
                - sum(
                    coefficient * trims[variable_index[key]]
                    for key, coefficient in terms
                    if key in applied_variables
                )
                for requested, (_, terms) in zip(
                    requested_clearance, linear_constraints, strict=True
                )
            ],
            dtype=float,
        )
        positive_residuals = np.maximum(linear_residuals, 0.0)
        linear_residual = float(positive_residuals.max(initial=0.0))
        residual_ratios = positive_residuals / np.maximum(requested_clearance, 1e-12)
        fidelity_passed = all(value["status"] == "PASSED" for value in fidelity.values())
        candidates.append(
            {
                "policy": "support_plane_trust_region",
                "patches": proposal,
                "fraction": fraction,
                "step_m": max(value["max_vertex_displacement_m"] for value in accepted_metrics),
                "max_adjacency_gap_m": max_adjacency_gap,
                "min_volume_ratio": min(value["volume_ratio"] for value in accepted_metrics),
                "minimum_thickness_m": min(
                    value["minimum_thickness_m"] for value in accepted_metrics
                ),
                "constraint_count": len(linear_constraints),
                "support_plane_count": len(applied_variables),
                "linear_constraint_residual_m": max(0.0, float(linear_residual)),
                "mean_constraint_residual_ratio": float(residual_ratios.mean()),
                "satisfied_constraint_fraction": float(np.mean(positive_residuals <= 1e-7)),
                "lp_slack_max_m": float(slacks.max(initial=0.0)),
                "active_set_rounds": active_set_rounds,
                "rejected_patch_count": len(rejected_patches),
                "rejected_patches": rejected_patches,
                "contact_regions": regions,
                "contact_normal_stabilization": normal_stabilization,
                "fidelity": fidelity,
            }
        )
        if fidelity_passed:
            passing_fractions.append(fraction)
        else:
            failing_fractions.append(fraction)

        # Fixed powers of two can leave a large unused fidelity budget. Once one
        # valid and one invalid materialized shape bracket the boundary, bisect it
        # before trying more distant steps. Exact MuJoCo scoring still decides
        # which of these geometry-valid candidates is accepted.
        lower = max((value for value in passing_fractions if value < fraction), default=None)
        upper = min((value for value in failing_fractions if value > fraction), default=None)
        if fidelity_passed:
            upper = min((value for value in failing_fractions if value > fraction), default=None)
            lower = fraction
        elif lower is not None:
            upper = fraction
        if (
            lower is not None
            and upper is not None
            and upper - lower > 1e-4
            and bisections_scheduled < 5
        ):
            midpoint = (lower + upper) / 2
            if midpoint not in scheduled_fractions:
                fractions.insert(fraction_index, midpoint)
                scheduled_fractions.add(midpoint)
                bisections_scheduled += 1
        if (
            sum(
                all(value["status"] == "PASSED" for value in candidate["fidelity"].values())
                for candidate in candidates
            )
            >= config.max_materialized_candidates_per_round
        ):
            break
        # Keep ordered backtracking candidates until GLB round-trip and the exact
        # sweep have run.  A geometrically valid in-memory hull can still become
        # non-volumetric at float32 serialization and must not end the line search.
    return candidates


def _contact_sided_refit_proposal(
    asset: Any,
    patches: dict[str, list[SurfacePatch]],
    visuals: dict[str, Any],
    evaluation: dict[str, Any],
    tolerance: float,
    config: ArticulatedCollisionRefinementConfig,
) -> tuple[dict[str, list[SurfacePatch]], dict[str, Any]] | None:
    """Move open-surface proxy thickness away from every observed contact.

    The source surface remains a boundary of each rebuilt proxy.  A patch is
    changed only when one normal polarity reduces its support along *all* sampled
    outward contact directions and retains a bounded fraction of the current
    volume.  Exact surface and MuJoCo gates remain authoritative afterwards.
    """
    import numpy as np

    from physrealx.integrations._asset_preview import link_transforms, pose_matrix

    joint = asset.joints[0]
    geometries = {g.link: g for g in asset.geometries if g.role is GeometryRole.COLLISION}
    contacts = [
        contact
        for contact in evaluation.get("contact_constraints", ())
        if float(contact["penetration_m"]) > tolerance
    ]
    if not contacts:
        contacts = [
            {
                **pair,
                "penetration_m": pair["max_penetration_m"],
                "q": pair["peak_q"],
            }
            for pair in evaluation["collision_pairs"]
            if float(pair["max_penetration_m"]) > tolerance
        ]
    contacts, normal_stabilization = _stabilize_contact_normals(contacts)
    directions: dict[tuple[str, int], list[Any]] = {}
    severities: dict[tuple[str, int], float] = {}
    for contact in contacts:
        transforms = link_transforms(asset, {joint.name: float(contact["q"])})
        normal = np.asarray(contact["normal_a_to_b"], dtype=float)
        for side, direction in (("a", 1.0), ("b", -1.0)):
            link = str(contact[f"link_{side}"])
            if link not in patches:
                continue
            index = int(str(contact[f"geom_{side}"]).rsplit("/", 1)[-1])
            if not 0 <= index < len(patches[link]):
                raise DomainValidationError("contact references an unavailable collision patch")
            geometry = geometries[link]
            transform = transforms[link] @ pose_matrix(geometry.local_pose)
            linear = transform[:3, :3] * np.asarray(geometry.scale)[None, :]
            local = direction * linear.T @ normal
            norm = float(np.linalg.norm(local))
            if norm > 1e-12:
                patch_key = (link, index)
                directions.setdefault(patch_key, []).append(local / norm)
                severities[patch_key] = max(
                    severities.get(patch_key, 0.0),
                    float(contact["penetration_m"]) - tolerance,
                )
    maximum_severity = max(severities.values(), default=0.0)
    ranked: list[tuple[float, float, float, str, int, SurfacePatch, int, float]] = []
    for (link, index), values in directions.items():
        # Changing a shallower pair cannot lower the current minimax objective.
        # Restrict this reparameterization to patches participating in the deepest
        # contact set; the exact sweep will expose the next active set afterwards.
        if severities[(link, index)] < maximum_severity - _PENETRATION_NUMERIC_ATOL_M:
            continue
        local_directions = np.asarray(values, dtype=float)
        current = patches[link][index]
        current_support = np.max(
            np.asarray(current.mesh.vertices, dtype=float) @ local_directions.T,
            axis=0,
        )
        alternatives: list[tuple[float, float, SurfacePatch, int, float]] = []
        for polarity in (-1, 1):
            fitted = fit_one_sided_patch(
                visuals[link],
                current,
                skin_m=config.proxy_skin_m,
                polarity=polarity,
            )
            if fitted is None:
                continue
            volume_ratio = float(fitted.mesh.volume / current.mesh.volume)
            if volume_ratio < config.min_proxy_volume_ratio:
                continue
            support_delta = (
                np.max(np.asarray(fitted.mesh.vertices) @ local_directions.T, axis=0)
                - current_support
            )
            alternatives.append(
                (
                    float(np.max(support_delta)),
                    float(np.mean(support_delta)),
                    fitted,
                    polarity,
                    volume_ratio,
                )
            )
        if not alternatives:
            continue
        worst, mean, fitted, polarity, volume_ratio = min(
            alternatives, key=lambda value: (value[0], value[1])
        )
        if worst < -_SURFACE_NUMERIC_ATOL_M:
            ranked.append(
                (
                    severities[(link, index)],
                    worst,
                    mean,
                    link,
                    index,
                    fitted,
                    polarity,
                    volume_ratio,
                )
            )
    if not ranked:
        return None
    # Bound each round independently of contact count.  Strongest uniform support
    # reductions are applied first; remaining contacts are reconsidered after the
    # exact sweep has changed the active set.
    selected: list[tuple[float, float, float, str, int, SurfacePatch, int, float]] = []
    per_link = {link: 0 for link in patches}
    for candidate in sorted(ranked, key=lambda value: (-value[0], value[1], value[2])):
        link = candidate[3]
        if per_link[link] >= config.max_collision_patch_updates_per_link:
            continue
        selected.append(candidate)
        per_link[link] += 1
    proposal = {link: list(values) for link, values in patches.items()}
    for _, _, _, link, index, fitted, _, _ in selected:
        proposal[link][index] = fitted
    return proposal, {
        "changed_patches": len(selected),
        "changed_by_link": per_link,
        "maximum_support_reduction_m": -min(value[1] for value in selected),
        "minimum_support_reduction_m": -max(value[1] for value in selected),
        "maximum_penetration_excess_m": max(value[0] for value in selected),
        "minimum_penetration_excess_m": min(value[0] for value in selected),
        "polarities": {
            "negative": sum(value[6] == -1 for value in selected),
            "positive": sum(value[6] == 1 for value in selected),
        },
        "minimum_volume_ratio": min(value[7] for value in selected),
        "active_penetration_excess_m": maximum_severity,
        "contact_normal_stabilization": normal_stabilization,
    }


def _assert_source_asset_lock(source: Any, candidate: Any) -> None:
    """Collision fitting may change only collision URIs of its source asset."""
    source_visuals = tuple(
        geometry for geometry in source.geometries if geometry.role is GeometryRole.VISUAL
    )
    candidate_visuals = tuple(
        geometry for geometry in candidate.geometries if geometry.role is GeometryRole.VISUAL
    )
    if (
        source_visuals != candidate_visuals
        or source.joints != candidate.joints
        or source.root_pose != candidate.root_pose
        or source.physics != candidate.physics
        or source.contact_exclusions != candidate.contact_exclusions
        or source.metadata.get("reconstructed_motion")
        != candidate.metadata.get("reconstructed_motion")
    ):
        raise MethodExecutionError(
            "collision candidate changed immutable source geometry, motion, or physics"
        )


def _fidelity_penalty(
    fidelity: dict[str, Any], config: ArticulatedCollisionRefinementConfig
) -> float:
    """Normalize all hard geometry gates into one deterministic search merit."""
    penalty = 0.0
    for value in fidelity.values():
        penalty += max(
            0.0,
            float(value["bidirectional_sample_max_m"]) / config.max_local_error_m - 1.0,
        )
        penalty += 4.0 * max(0.0, config.min_patch_coverage - float(value["worst_cell_coverage"]))
        penalty += 4.0 * max(0.0, config.min_patch_coverage - float(value["contact_coverage"]))
    return penalty


def _candidate_score(
    evaluation: dict[str, Any],
    fidelity: dict[str, Any],
    patches: dict[str, list[SurfacePatch]],
    optimization_tolerance: float,
    config: ArticulatedCollisionRefinementConfig,
) -> float:
    """Balance clearance and source-surface fidelity without relaxing either gate."""
    penetration = float(evaluation["max_penetration_m"]) / optimization_tolerance
    complexity = 1e-4 * sum(len(values) for values in patches.values())
    return penetration + 2.0 * _fidelity_penalty(fidelity, config) + complexity


def _shape_candidate_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave topology families before spending the exact-sweep budget.

    The smallest LP residual and largest admissible support step answer different
    questions.  The current and subdivided proxy topologies answer a third one.
    Ranking both current-topology representatives first used the entire bounded
    MuJoCo budget before a split-and-refit candidate could be evaluated.  Keep the
    largest admissible step from every available topology first, then its minimum
    residual representative.  With only one topology this still preserves both
    trust-region extremes.
    """
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    policies = ("current", "topology_supported_split")
    representatives: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for policy in policies:
        group = [value for value in candidates if value.get("base_policy") == policy]
        if not group:
            continue
        representatives[policy] = (
            max(
                group,
                key=lambda value: (
                    float(value["fraction"]),
                    -float(value["linear_constraint_residual_m"]),
                ),
            ),
            min(
                group,
                key=lambda value: (
                    float(value["linear_constraint_residual_m"]),
                    -float(value["fraction"]),
                ),
            ),
        )
    for representative_index in range(2):
        for policy in policies:
            if policy not in representatives:
                continue
            value = representatives[policy][representative_index]
            if id(value) not in seen:
                ordered.append(value)
                seen.add(id(value))
    for value in sorted(
        candidates,
        key=lambda candidate: (
            candidate.get("base_policy") != "current",
            float(candidate["linear_constraint_residual_m"]),
            -float(candidate["fraction"]),
        ),
    ):
        if id(value) not in seen:
            ordered.append(value)
            seen.add(id(value))
    return ordered


def _shape_progress_requires_topology(
    policy: str,
    previous_penetration_m: float,
    current_penetration_m: float,
    target_m: float,
) -> tuple[bool, float, float]:
    """Escalate a stalled trust-region step to source-driven subdivision."""
    improvement = previous_penetration_m - current_penetration_m
    required = 0.05 * max(0.0, previous_penetration_m - target_m)
    stalled = (
        policy in {"joint_support_plane_refit", "topology_support_plane_refit"}
        and improvement < required
    )
    return stalled, improvement, required


def _topology_refit_exploration_candidates(
    admissible: list[dict[str, Any]],
    split_state: dict[str, Any],
    *,
    current_penetration_m: float,
    optimization_tolerance_m: float,
) -> list[dict[str, Any]]:
    """Keep a useful split long enough to optimize its discontinuous metric.

    MuJoCo reports penetration per convex pair. Partitioning an identical occupied
    volume can therefore increase the reported maximum even though the union and
    source support are unchanged. Comparing a split-and-refit step only with the
    previous convex decomposition rejects the new degrees of freedom before they
    can be used. Permit a bounded transition only when the refit makes material
    progress relative to the raw split, remains within one acceptance tolerance of
    the current state, and all hard fidelity gates have already passed upstream.
    The globally best state is retained separately and remains the publication
    fallback.
    """
    split_penetration = float(split_state["evaluation"]["max_penetration_m"])
    required_progress = 0.05 * max(0.0, split_penetration - optimization_tolerance_m)
    required_progress = max(required_progress, 1e-6)
    exploration_limit = current_penetration_m + optimization_tolerance_m
    return [
        proposal
        for proposal in admissible
        if proposal.get("policy") == "topology_support_plane_refit"
        and float(proposal["evaluation"]["max_penetration_m"])
        <= split_penetration - required_progress
        and float(proposal["evaluation"]["max_penetration_m"])
        <= exploration_limit + _PENETRATION_NUMERIC_ATOL_M
    ]


def _materialize_fidelity_candidate(
    store: ArtifactStore,
    source_asset: Any,
    proposal: dict[str, list[SurfacePatch]],
    coverage: dict[str, SurfaceCoverage],
    links: tuple[str, str],
    config: ArticulatedCollisionRefinementConfig,
) -> dict[str, Any]:
    """Round-trip and measure the exact collision GLB before costly simulation."""
    patches, references, asset = _materialize_candidate(store, source_asset, proposal)
    _assert_source_asset_lock(source_asset, asset)
    fidelity = {
        link: coverage[link].evaluate(
            patches[link],
            max_error_m=config.max_local_error_m,
            coverage_distance_m=config.coverage_distance_m,
            min_coverage=config.min_patch_coverage,
        )[0]
        for link in links
    }
    return {
        "patches": patches,
        "references": references,
        "asset": asset,
        "fidelity": fidelity,
    }


def _score_materialized_candidate(
    store: ArtifactStore,
    state: dict[str, Any],
    evaluation_config: FeasibilityConfig,
    optimization_tolerance: float,
    config: ArticulatedCollisionRefinementConfig,
    *,
    run_rollouts: bool = False,
) -> dict[str, Any]:
    """Score one candidate; defer dynamic rollouts until final validation."""
    if not all(value["status"] == "PASSED" for value in state["fidelity"].values()):
        return {**state, "evaluation": None, "score": float("inf")}
    evaluation = evaluate_asset_feasibility(
        store,
        state["asset"],
        config=evaluation_config,
        retain_contact_constraints=True,
        contact_constraint_tolerance_m=optimization_tolerance,
        run_rollouts=run_rollouts,
    )
    return {
        **state,
        "evaluation": evaluation,
        "score": _candidate_score(
            evaluation,
            state["fidelity"],
            state["patches"],
            optimization_tolerance,
            config,
        ),
    }


def _prepare_surface_fidelity(
    visuals: dict[str, Any],
    patches: dict[str, list[SurfacePatch]],
    coverage: dict[str, SurfaceCoverage],
    links: tuple[str, str],
    config: ArticulatedCollisionRefinementConfig,
    *,
    target_error_m: float | None = None,
) -> tuple[dict[str, list[SurfacePatch]], list[dict[str, Any]]]:
    """Fit source-supported patches before contact optimization changes their shape."""
    import numpy as np

    target_error_m = config.max_local_error_m if target_error_m is None else target_error_m
    if not config.coverage_distance_m <= target_error_m <= config.max_local_error_m:
        raise ValueError("surface preparation target must lie between coverage and hard error")
    history: list[dict[str, Any]] = []
    current = patches
    for iteration in range(config.max_fidelity_rounds + 1):
        measurements: dict[str, Any] = {}
        priorities: dict[str, dict[int, float]] = {}
        selected: dict[str, set[int]] = {}
        for link in links:
            measurements[link], selected[link], priorities[link] = coverage[
                link
            ].evaluate_with_priorities(
                current[link],
                max_error_m=target_error_m,
                coverage_distance_m=config.coverage_distance_m,
                min_coverage=config.min_patch_coverage,
            )
        history.append(
            {
                "round": iteration,
                "target_error_m": target_error_m,
                "fidelity": measurements,
                "parts": {link: len(current[link]) for link in links},
                "selected_patches": {link: len(selected[link]) for link in links},
            }
        )
        if all(value["status"] == "PASSED" for value in measurements.values()):
            return current, history
        if iteration == config.max_fidelity_rounds:
            break
        proposal: dict[str, list[SurfacePatch]] = {}
        progress = False
        for link in links:
            available = config.max_parts_per_link - len(current[link])
            ordered = sorted(
                selected[link],
                key=lambda index: (
                    priorities[link].get(index, 0.0),
                    len(current[link][index].faces),
                    -index,
                ),
                reverse=True,
            )
            replacements: dict[int, list[SurfacePatch]] = {}
            for index in ordered[: config.max_patch_updates_per_link]:
                children = fit_patches(visuals[link], current[link][index], config.proxy_skin_m)
                if children is None:
                    continue
                added = len(children) - 1
                if added > available:
                    continue
                replacements[index] = children
                available -= max(0, added)
            updated: list[SurfacePatch] = []
            for index, patch in enumerate(current[link]):
                if index in replacements:
                    updated.extend(replacements[index])
                    progress = True
                else:
                    updated.append(patch)
            ids = np.concatenate([patch.faces for patch in updated])
            if not np.array_equal(np.sort(ids), np.arange(len(visuals[link].faces))):
                raise MethodExecutionError(
                    "surface fidelity fitting dropped or duplicated source face ownership"
                )
            proposal[link] = updated
        if not progress:
            break
        current = proposal
    return current, history


def _meets_surface_target(
    value: dict[str, Any],
    *,
    target_error_m: float,
    config: ArticulatedCollisionRefinementConfig,
) -> bool:
    """Apply a stricter optimization target without weakening final hard gates."""
    return (
        float(value["bidirectional_sample_max_m"]) <= target_error_m + _SURFACE_NUMERIC_ATOL_M
        and float(value["worst_cell_coverage"])
        >= config.min_patch_coverage - _COVERAGE_NUMERIC_ATOL
        and float(value["contact_coverage"]) >= config.min_patch_coverage - _COVERAGE_NUMERIC_ATOL
    )


def _surface_reserve_exhausted(
    fidelity: dict[str, Any], config: ArticulatedCollisionRefinementConfig
) -> list[str]:
    """Identify links that still pass publication gates but need fitting freedom."""
    target = config.max_local_error_m - config.surface_fidelity_reserve_m
    return sorted(
        link
        for link, value in fidelity.items()
        if float(value["bidirectional_sample_max_m"]) > target + _SURFACE_NUMERIC_ATOL_M
    )


def _close_candidate_surface_fidelity(
    store: ArtifactStore,
    source_asset: Any,
    proposal: dict[str, list[SurfacePatch]],
    visuals: dict[str, Any],
    coverage: dict[str, SurfaceCoverage],
    links: tuple[str, str],
    evaluation_config: FeasibilityConfig,
    optimization_tolerance: float,
    config: ArticulatedCollisionRefinementConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Repair and remeasure one proposal until its consumed GLB is faithful."""
    history: list[dict[str, Any]] = []
    current = proposal
    stalled_roundtrip = False
    state: dict[str, Any] | None = None
    for serialization_pass in range(config.max_fidelity_rounds):
        current, preparation = _prepare_surface_fidelity(visuals, current, coverage, links, config)
        history.extend({**row, "serialization_pass": serialization_pass} for row in preparation)
        state = _materialize_fidelity_candidate(
            store, source_asset, current, coverage, links, config
        )
        current = state["patches"]
        if all(value["status"] == "PASSED" for value in state["fidelity"].values()):
            break
        if len(preparation) == 1:
            if stalled_roundtrip:
                break
            stalled_roundtrip = True
        else:
            stalled_roundtrip = False
    if state is None:
        raise MethodExecutionError("surface fidelity closure produced no materialized candidate")
    return (
        _score_materialized_candidate(
            store, state, evaluation_config, optimization_tolerance, config
        ),
        history,
    )


def refine_articulated_collision(
    store: ArtifactStore,
    assembly_ref: ArtifactRef,
    feasibility_ref: ArtifactRef,
    *,
    config: ArticulatedCollisionRefinementConfig,
) -> ArtifactRef:
    """Run collision refinement with one explicitly selected query backend."""
    surface_query = create_surface_query(
        backend=config.surface_query_backend,
        gpu_devices=config.gpu_devices,
        require_accelerator=config.require_accelerator,
        cpu_workers=config.cpu_query_workers,
        batch_size=config.surface_query_batch_size,
    )
    logging.getLogger(__name__).info("collision surface query backend=%s", surface_query.describe())
    try:
        return _refine_articulated_collision(
            store,
            assembly_ref,
            feasibility_ref,
            config=config,
            surface_query=surface_query,
        )
    finally:
        surface_query.close()


def _refine_articulated_collision(
    store: ArtifactStore,
    assembly_ref: ArtifactRef,
    feasibility_ref: ArtifactRef,
    *,
    config: ArticulatedCollisionRefinementConfig,
    surface_query: SurfaceQuery,
) -> ArtifactRef:
    """Refit conflicting surface patches, accepting fidelity AND collision feasibility.

    Source triangles are assigned once and conserved through binary subdivision.
    Fixed local/contact probes prevent passing by deleting supported collision faces.
    Failed or budget-exhausted candidates are reported without publication as assets.
    """
    import numpy as np

    assembly = read_json(store, assembly_ref, "physrealx.asset-assembly-bundle/1")
    feasibility = read_json(store, feasibility_ref, "physrealx.physical-feasibility-report/1")
    if feasibility.get("source_assembly") != assembly_ref.model_dump(
        mode="json"
    ) or feasibility.get("status") not in {"PASSED", "FAILED"}:
        raise DomainValidationError(
            "articulated collision refinement requires its current feasibility report"
        )
    asset = assembled_asset(assembly, store)
    if len(asset.joints) != 1:
        raise DomainValidationError("articulated collision refinement currently requires one joint")
    joint = asset.joints[0]
    links = (joint.parent, joint.child)
    reconstructed_motion = asset.metadata.get("reconstructed_motion")
    expected_motion = {
        "joint": joint.name,
        "parent": joint.parent,
        "child": joint.child,
        "axis": list(joint.axis),
        "origin_m": list(joint.parent_to_joint.translation_m),
        "range": [joint.lower_limit, joint.upper_limit],
    }
    if not isinstance(reconstructed_motion, dict) or any(
        reconstructed_motion.get(key) != value for key, value in expected_motion.items()
    ):
        raise DomainValidationError(
            "asset joint must exactly reproduce its source reconstructed motion"
        )
    if asset.contact_exclusions:
        raise DomainValidationError("collision refinement requires enabled parent/child collision")
    source_evaluation_config = FeasibilityConfig.model_validate(feasibility["config"])
    if config.penetration_safety_margin_m >= source_evaluation_config.penetration_tolerance_m:
        raise DomainValidationError(
            "collision refinement safety margin must be below the feasibility tolerance"
        )
    optimization_tolerance = (
        source_evaluation_config.penetration_tolerance_m - config.penetration_safety_margin_m
    )
    # Candidate selection and publication must observe the same joint samples.
    # A coarse search followed by a denser final sweep aliases narrow collision
    # peaks and stores incomparable scores in a resumable checkpoint.
    search_evaluation_config = source_evaluation_config.model_copy(
        update={
            "sweep_samples": max(
                _INDEPENDENT_SWEEP_SAMPLES, source_evaluation_config.sweep_samples
            ),
            "adaptive_sweep_levels": max(
                _INDEPENDENT_ADAPTIVE_SWEEP_LEVELS,
                source_evaluation_config.adaptive_sweep_levels,
            ),
        }
    )
    before = evaluate_asset_feasibility(
        store,
        asset,
        config=source_evaluation_config,
        retain_contact_constraints=True,
        contact_constraint_tolerance_m=optimization_tolerance,
        run_rollouts=False,
    )
    if (
        before["status"] != feasibility["status"]
        or before["locked_range"] != feasibility.get("locked_range")
        or not np.isclose(before["max_penetration_m"], feasibility["max_penetration_m"])
    ):
        raise DomainValidationError("collision evidence is stale or non-reproducible")

    source_input = ArtifactRef.model_validate(assembly["source_input"])
    request = ReferenceAssemblyInput.model_validate(
        read_json(store, source_input, "physrealx.reference-assembly-input/2")
    )
    if request.articulated_collision_refinement is not None:
        raise DomainValidationError("assembly already contains an articulated collision refinement")
    report_sources: dict[str, tuple[ArtifactRef, dict[str, Any]]] = {}
    for ref in request.collision_reports:
        document = read_json(store, ref, "physrealx.collision-build-report/1")
        report_sources[str(document["link"])] = (ref, document)
    if set(report_sources) != set(asset.links):
        raise DomainValidationError("collision reports do not cover exactly the articulated asset")

    collision_geometries: dict[str, tuple[int, Any]] = {}
    visual_geometries: dict[str, Any] = {}
    for index, geometry in enumerate(asset.geometries):
        if geometry.link not in links:
            continue
        if geometry.role is GeometryRole.COLLISION:
            if geometry.link in collision_geometries:
                raise DomainValidationError("each articulated link requires one collision mesh")
            collision_geometries[geometry.link] = (index, geometry)
        elif geometry.role is GeometryRole.VISUAL:
            if geometry.link in visual_geometries:
                raise DomainValidationError("each articulated link requires one visual mesh")
            visual_geometries[geometry.link] = geometry
    if set(collision_geometries) != set(links) or set(visual_geometries) != set(links):
        raise DomainValidationError("articulated links require visual and collision geometry")
    if any(
        geometry.representation is not GeometryRepresentation.MESH
        for _, geometry in collision_geometries.values()
    ):
        raise DomainValidationError("articulated collision refinement requires mesh proxies")

    parts: dict[str, list[Any]] = {}
    visuals: dict[str, Any] = {}
    prefixes: dict[str, str] = {}
    for link in links:
        report_ref, report = report_sources[link]
        collision_ref = ArtifactRef.model_validate(report["collision_geometry"])
        if collision_ref.artifact_id != collision_geometries[link][1].uri.removeprefix(
            "artifact://"
        ):
            raise DomainValidationError("collision geometry differs from its build report")
        parts[link] = load_meshes(store, collision_ref, process=False)
        if any(not part.is_volume or not part.is_convex for part in parts[link]):
            raise DomainValidationError("collision refinement requires convex positive volumes")
        visual_ref = mesh_reference(visual_geometries[link].uri, store)
        decoded_visual = load_meshes(store, visual_ref, process=False)
        if len(decoded_visual) != 1 or report["source_geometry"] != visual_ref.model_dump(
            mode="json"
        ):
            raise DomainValidationError("collision report differs from immutable visual geometry")
        visuals[link] = decoded_visual[0]
        prefixes[link] = f"asset/geometry/{collision_geometries[link][0]}/"
        if not store.verify(report_ref):
            raise DomainValidationError("source collision report is unavailable")
    visual_refs = {
        link: mesh_reference(visual_geometries[link].uri, store).model_dump(mode="json")
        for link in links
    }

    # Round and part budgets control continuation length, not candidate semantics.
    # Keep them outside checkpoint identity so a bounded run can be extended from
    # the exact same accepted state instead of repeating source-surface binding.
    checkpoint_config = config.model_dump(
        mode="json",
        exclude={"resume_checkpoint", "max_rounds", "max_parts_per_link"},
    )
    checkpoint_config.update(
        {
            "optimizer_contract": _REFINEMENT_CHECKPOINT_CONTRACT,
            "search_evaluation": {
                "sweep_samples": search_evaluation_config.sweep_samples,
                "adaptive_sweep_levels": search_evaluation_config.adaptive_sweep_levels,
            },
        }
    )
    checkpoint_store = RefinementCheckpointStore(
        store,
        source_assembly=assembly_ref,
        source_feasibility=feasibility_ref,
        config=checkpoint_config,
        mode=config.checkpoint_mode,
        resume_from=config.resume_checkpoint,
        lock_writer=True,
    )
    resumed = checkpoint_store.load(visuals)
    reason = "refinement budget exhausted"
    best_state: dict[str, Any] | None = None

    def candidate_with_collision_references(references: dict[str, ArtifactRef]) -> Any:
        candidate = asset.model_copy(
            update={
                "geometries": tuple(
                    geometry.model_copy(
                        update={"uri": f"artifact://{references[geometry.link].artifact_id}"}
                    )
                    if geometry.role is GeometryRole.COLLISION
                    else geometry
                    for geometry in asset.geometries
                )
            }
        )
        _assert_source_asset_lock(asset, candidate)
        return candidate

    if resumed is not None:
        patches = resumed["patches"]
        refined_collision_refs = resumed["references"]
        candidate_asset = candidate_with_collision_references(refined_collision_refs)
        coverage = {
            link: SurfaceCoverage(
                visuals[link],
                patches[link],
                set(),
                cell_size_m=config.patch_size_m,
                seed=config.seed,
                query=surface_query.nearest,
                contact_faces=resumed["contact_faces"][link],
            )
            for link in links
        }
        iterations = list(resumed.get("iterations", ()))
        fidelity = dict(resumed["fidelity"])
        after = dict(resumed["evaluation"])
        surface_fidelity_preparation = list(resumed.get("surface_fidelity_preparation", ()))
        surface_prepared = bool(resumed.get("surface_prepared", False))
        shape_refit_rounds = int(resumed.get("shape_refit_rounds", 0))
        shape_refit_history = list(resumed.get("shape_refit_history", ()))
        shape_refit_trials = list(resumed.get("shape_refit_trials", ()))
        contact_sided_refit_history = list(resumed.get("contact_sided_refit_history", ()))
        topology_required = bool(resumed.get("topology_required", False))
        visual_motion_audit = dict(resumed["visual_motion_audit"])
        start_iteration = int(resumed["next_round"])
        restored_best = resumed.get("best")
        if restored_best is not None:
            best_state = {
                **restored_best,
                "asset": candidate_with_collision_references(restored_best["references"]),
            }
        logging.getLogger(__name__).info(
            "resumed articulated collision refinement checkpoint=%s next_round=%s",
            resumed["checkpoint"].artifact_id,
            start_iteration,
        )
    else:
        conflicts = _conflicting_parts(
            before, prefixes, source_evaluation_config.penetration_tolerance_m
        )
        if any(i >= len(parts[link]) for link in links for i in conflicts[link]):
            raise DomainValidationError("failed sweep references an unavailable collision part")
        patches = {
            link: bind_surface(visuals[link], parts[link], query=surface_query.nearest)
            for link in links
        }
        coverage = {
            link: SurfaceCoverage(
                visuals[link],
                patches[link],
                conflicts[link],
                cell_size_m=config.patch_size_m,
                seed=config.seed,
                query=surface_query.nearest,
            )
            for link in links
        }
        visual_motion_audit = audit_visual_motion(
            asset,
            visuals,
            links,
            surface_query,
            samples=config.visual_audit_samples,
            probes_per_link=config.visual_audit_probes_per_link,
        )
        iterations = []
        fidelity = {}
        refined_collision_refs = {
            link: mesh_reference(collision_geometries[link][1].uri, store) for link in links
        }
        candidate_asset = asset
        after = before
        surface_fidelity_preparation = []
        surface_prepared = False
        prepared_surface_state: dict[str, Any] | None = None
        # Fidelity is measured on the exact float32 GLB later consumed by MuJoCo.
        # Re-enter local fitting when serialization exposes a new boundary extreme.
        for serialization_pass in range(config.max_fidelity_rounds):
            preparation_target = config.max_local_error_m - config.surface_fidelity_reserve_m
            patches, preparation = _prepare_surface_fidelity(
                visuals,
                patches,
                coverage,
                links,
                config,
                target_error_m=preparation_target,
            )
            for row in preparation:
                surface_fidelity_preparation.append(
                    {**row, "serialization_pass": serialization_pass}
                )
            changed = len(preparation) > 1
            surface_prepared = surface_prepared or changed
            if not changed:
                break
            prepared_surface_state = _materialize_fidelity_candidate(
                store, asset, patches, coverage, links, config
            )
            patches = prepared_surface_state["patches"]
            refined_collision_refs = prepared_surface_state["references"]
            candidate_asset = prepared_surface_state["asset"]
            fidelity = prepared_surface_state["fidelity"]
            if all(
                _meets_surface_target(
                    value,
                    target_error_m=preparation_target,
                    config=config,
                )
                for value in fidelity.values()
            ):
                break
        if prepared_surface_state is not None:
            prepared = _score_materialized_candidate(
                store,
                prepared_surface_state,
                search_evaluation_config,
                optimization_tolerance,
                config,
            )
            after = prepared["evaluation"]
        else:
            after = evaluate_asset_feasibility(
                store,
                candidate_asset,
                config=search_evaluation_config,
                retain_contact_constraints=True,
                contact_constraint_tolerance_m=optimization_tolerance,
                run_rollouts=False,
            )
        shape_refit_rounds = 0
        shape_refit_history = []
        shape_refit_trials = []
        contact_sided_refit_history = []
        topology_required = False
        start_iteration = 0

    def save_checkpoint(next_round: int) -> None:
        current_score = _candidate_score(after, fidelity, patches, optimization_tolerance, config)
        checkpoint_best = best_state
        if checkpoint_best is None or current_score < checkpoint_best["score"]:
            checkpoint_best = {
                "patches": patches,
                "references": refined_collision_refs,
                "asset": candidate_asset,
                "evaluation": after,
                "fidelity": fidelity,
                "score": current_score,
            }
        checkpoint_store.save(
            patches=patches,
            contact_faces={link: coverage[link].contact_faces for link in links},
            references=refined_collision_refs,
            state={
                "next_round": next_round,
                "evaluation": after,
                "fidelity": fidelity,
                "iterations": iterations,
                "surface_fidelity_preparation": surface_fidelity_preparation,
                "surface_prepared": surface_prepared,
                "shape_refit_rounds": shape_refit_rounds,
                "shape_refit_history": shape_refit_history,
                "shape_refit_trials": shape_refit_trials,
                "contact_sided_refit_history": contact_sided_refit_history,
                "topology_required": topology_required,
                "visual_motion_audit": visual_motion_audit,
            },
            best={
                "patches": checkpoint_best["patches"],
                "references": checkpoint_best["references"],
                "state": {
                    "evaluation": checkpoint_best["evaluation"],
                    "fidelity": checkpoint_best["fidelity"],
                    "score": checkpoint_best["score"],
                },
            },
        )

    if resumed is None:
        save_checkpoint(0)
    for iteration in range(start_iteration, config.max_rounds + 1):
        selected = _conflicting_parts(after, prefixes, optimization_tolerance)
        unsupported_parts: dict[str, set[int]] = {}
        fidelity_priority: dict[str, dict[int, float]] = {}
        for link in links:
            fidelity[link], unsupported, priorities = coverage[link].evaluate_with_priorities(
                patches[link],
                max_error_m=config.max_local_error_m,
                coverage_distance_m=config.coverage_distance_m,
                min_coverage=config.min_patch_coverage,
            )
            unsupported_parts[link] = unsupported
            fidelity_priority[link] = priorities
            selected[link].update(unsupported)
        current_score = _candidate_score(after, fidelity, patches, optimization_tolerance, config)
        exhausted_reserve = _surface_reserve_exhausted(fidelity, config)
        if exhausted_reserve:
            topology_required = True
        if best_state is None or current_score < best_state["score"]:
            best_state = {
                "patches": patches,
                "references": refined_collision_refs,
                "asset": candidate_asset,
                "evaluation": after,
                "fidelity": fidelity,
                "score": current_score,
            }
        iterations.append(
            {
                "round": iteration,
                "evaluation_sweep_samples": search_evaluation_config.sweep_samples,
                "evaluation_adaptive_sweep_levels": (
                    search_evaluation_config.adaptive_sweep_levels
                ),
                "max_penetration_m": after["max_penetration_m"],
                "simulation_status": after["status"],
                "fidelity": dict(fidelity),
                "parts": {link: len(patches[link]) for link in links},
                "combined_score": current_score,
                "surface_fidelity_reserve_exhausted": exhausted_reserve,
            }
        )
        logging.getLogger(__name__).info(
            "refit round=%s penetration_mm=%.3f parts=%s fidelity=%s",
            iteration,
            1000 * after["max_penetration_m"],
            {link: len(patches[link]) for link in links},
            {
                link: {
                    key: value[key]
                    for key in ("status", "bidirectional_sample_max_m", "worst_cell_coverage")
                }
                for link, value in fidelity.items()
            },
        )
        if _meets_penetration_target(after["max_penetration_m"], optimization_tolerance) and all(
            v["status"] == "PASSED" for v in fidelity.values()
        ):
            reason = "all fidelity and simulation gates passed with clearance safety margin"
            break
        if iteration == config.max_rounds:
            break

        # Open reconstructed surfaces need a directed shell at a moving seam.
        # First try the source-preserving reparameterization because it can remove
        # symmetric-skin overlap without consuming repeated support-plane cuts.
        # Accept it immediately only after exact GLB fidelity and MuJoCo sweeps;
        # otherwise retain the current state and continue with topology/shape search.
        sided = _contact_sided_refit_proposal(
            candidate_asset,
            patches,
            visuals,
            after,
            optimization_tolerance,
            config,
        )
        if sided is not None:
            sided_patches, sided_details = sided
            trial: dict[str, Any] = {
                "round": iteration,
                **sided_details,
            }
            try:
                sided_state, sided_preparation = _close_candidate_surface_fidelity(
                    store,
                    asset,
                    sided_patches,
                    visuals,
                    coverage,
                    links,
                    search_evaluation_config,
                    optimization_tolerance,
                    config,
                )
                sided_state["policy"] = "contact_sided_surface_refit"
                sided_state["details"] = sided_details
                trial.update(
                    {
                        "surface_fidelity_preparation": sided_preparation,
                        "max_penetration_m": sided_state["evaluation"]["max_penetration_m"],
                        "simulation_status": sided_state["evaluation"]["status"],
                        "fidelity": sided_state["fidelity"],
                        "combined_score": sided_state["score"],
                    }
                )
                sided_passed = all(
                    value["status"] == "PASSED" for value in sided_state["fidelity"].values()
                )
                sided_improved = (
                    sided_passed
                    and sided_state["score"] < current_score - 1e-8
                    and sided_state["evaluation"]["max_penetration_m"]
                    < after["max_penetration_m"] - 1e-6
                )
                trial["status"] = "ACCEPTED" if sided_improved else "REJECTED"
                if not sided_improved:
                    trial["reason"] = "exact candidate did not improve clearance and fidelity"
                contact_sided_refit_history.append(trial)
                iterations[-1]["contact_sided_candidate"] = {
                    key: trial[key]
                    for key in (
                        "status",
                        "changed_patches",
                        "changed_by_link",
                        "max_penetration_m",
                        "combined_score",
                    )
                }
                if sided_improved:
                    previous_penetration = float(after["max_penetration_m"])
                    patches = sided_state["patches"]
                    refined_collision_refs = sided_state["references"]
                    candidate_asset = sided_state["asset"]
                    after = sided_state["evaluation"]
                    fidelity = sided_state["fidelity"]
                    improvement = previous_penetration - float(after["max_penetration_m"])
                    minimum_progress = 0.05 * max(
                        0.0, previous_penetration - optimization_tolerance
                    )
                    topology_required = bool(_surface_reserve_exhausted(fidelity, config)) or (
                        improvement < minimum_progress
                    )
                    iterations[-1].update(
                        {
                            "selected_candidate": "contact_sided_surface_refit",
                            "selection_mode": "strict_improvement",
                            "penetration_improvement_m": improvement,
                            "minimum_shape_progress_m": minimum_progress,
                            "topology_required_next_round": topology_required,
                            "surface_fidelity_reserve_exhausted_after_selection": (
                                _surface_reserve_exhausted(fidelity, config)
                            ),
                        }
                    )
                    save_checkpoint(iteration + 1)
                    continue
            except (DomainValidationError, MethodExecutionError) as error:
                trial.update({"status": "REJECTED", "reason": str(error)})
                contact_sided_refit_history.append(trial)
                iterations[-1]["contact_sided_candidate"] = {
                    "status": "REJECTED",
                    "reason": str(error),
                    "changed_patches": sided_details["changed_patches"],
                }
        fraction_hint = float(shape_refit_history[-1]["fraction"]) if shape_refit_history else None
        current_shape_candidates = _shape_refit_candidates(
            candidate_asset,
            patches,
            visuals,
            coverage,
            after,
            optimization_tolerance,
            config,
            query=surface_query.nearest,
            fraction_hint=fraction_hint,
        )
        for candidate in current_shape_candidates:
            candidate["base_policy"] = "current"
        shape_preserves_fidelity = any(
            all(value["status"] == "PASSED" for value in candidate["fidelity"].values())
            for candidate in current_shape_candidates
        )
        defer_topology = shape_preserves_fidelity and not topology_required
        split_base_patches = patches
        recovery_from_shape = False
        if current_shape_candidates and not shape_preserves_fidelity:
            recovery = min(
                current_shape_candidates,
                key=lambda candidate: _fidelity_penalty(candidate["fidelity"], config),
            )
            recovery_selected: dict[str, set[int]] = {}
            recovery_priority: dict[str, dict[int, float]] = {}
            for link in links:
                _, recovery_selected[link], recovery_priority[link] = coverage[
                    link
                ].evaluate_with_priorities(
                    recovery["patches"][link],
                    max_error_m=config.max_local_error_m,
                    coverage_distance_m=config.coverage_distance_m,
                    min_coverage=config.min_patch_coverage,
                )
            if any(recovery_selected.values()):
                split_base_patches = recovery["patches"]
                selected = recovery_selected
                unsupported_parts = recovery_selected
                fidelity_priority = recovery_priority
                recovery_from_shape = True
                iterations[-1]["topology_trigger"] = "shape_fidelity_recovery"
        selected_for_split = selected if not defer_topology else {link: set() for link in links}
        if defer_topology:
            iterations[-1]["topology_policy"] = "deferred_until_shape_refit_stalls"
        elif topology_required:
            iterations[-1]["topology_trigger"] = "insufficient_shape_progress"
        progress = False
        next_patches: dict[str, list[SurfacePatch]] = {}
        rejected_splits: list[dict[str, Any]] = []
        deferred_splits: dict[str, int] = {}
        collision_priority: dict[str, dict[int, float]] = {link: {} for link in links}
        if not recovery_from_shape and not defer_topology:
            for pair in after["collision_pairs"]:
                depth = float(pair["max_penetration_m"])
                if depth <= optimization_tolerance:
                    continue
                for side in ("a", "b"):
                    link = str(pair[f"link_{side}"])
                    index = int(str(pair[f"geom_{side}"]).removeprefix(prefixes[link]))
                    collision_priority[link][index] = max(
                        depth, collision_priority[link].get(index, 0.0)
                    )
        for link in links:
            available_parts = config.max_parts_per_link - len(split_base_patches[link])
            fidelity_order = (
                []
                if defer_topology
                else sorted(
                    unsupported_parts[link],
                    key=lambda index: (
                        fidelity_priority[link].get(index, 0.0),
                        len(split_base_patches[link][index].faces),
                        -index,
                    ),
                    reverse=True,
                )
            )
            collision_order = sorted(
                collision_priority[link],
                key=lambda index: (
                    collision_priority[link][index] / optimization_tolerance,
                    len(split_base_patches[link][index].faces),
                    -index,
                ),
                reverse=True,
            )
            # Alternate the strongest geometry and collision violations. A single
            # global ordering used to spend the entire part budget on contacts and
            # indefinitely defer a severe outward surface error.
            ordered: list[int] = []
            seen: set[int] = set()
            for position in range(max(len(fidelity_order), len(collision_order))):
                for candidates in (fidelity_order, collision_order):
                    if position < len(candidates) and candidates[position] not in seen:
                        seen.add(candidates[position])
                        ordered.append(candidates[position])
            ordered.extend(
                sorted(
                    selected_for_split[link].difference(seen),
                    key=lambda index: (len(split_base_patches[link][index].faces), -index),
                    reverse=True,
                )
            )
            replacements: dict[int, list[SurfacePatch]] = {}
            update_limit = (
                config.max_patch_updates_per_link
                if unsupported_parts[link]
                else config.max_collision_patch_updates_per_link
            )
            for index in ordered[:update_limit]:
                patch = split_base_patches[link][index]
                if not (patch.depth == 0 or len(patch.faces) > 1):
                    continue
                children = _split_with_current_envelope(
                    visuals[link], patch, config, query=surface_query.nearest
                )
                if children is None:
                    rejected_splits.append(
                        {
                            "link": link,
                            "part": index,
                            "depth": patch.depth,
                            "source_faces": len(patch.faces),
                            "reason": "degenerate or envelope-incompatible convex subdivision",
                        }
                    )
                    continue
                added_parts = len(children) - 1
                if added_parts > available_parts:
                    continue
                replacements[index] = children
                available_parts -= max(0, added_parts)
            deferred_splits[link] = len(selected_for_split[link].difference(replacements))
            updated: list[SurfacePatch] = []
            for index, patch in enumerate(split_base_patches[link]):
                if index in replacements:
                    updated.extend(replacements[index])
                    progress = True
                else:
                    updated.append(patch)
            next_patches[link] = updated
        if rejected_splits:
            iterations[-1]["rejected_patch_splits"] = rejected_splits
        iterations[-1]["deferred_patch_splits"] = deferred_splits
        proposals: list[dict[str, Any]] = []
        split_state: dict[str, Any] | None = None
        if progress and all(
            values and len(values) <= config.max_parts_per_link for values in next_patches.values()
        ):
            for link in links:
                ids = np.concatenate([patch.faces for patch in next_patches[link]])
                if not np.array_equal(np.sort(ids), np.arange(len(visuals[link].faces))):
                    raise MethodExecutionError(
                        "refinement dropped or duplicated source face ownership"
                    )
            try:
                split_state, split_preparation = _close_candidate_surface_fidelity(
                    store,
                    asset,
                    next_patches,
                    visuals,
                    coverage,
                    links,
                    search_evaluation_config,
                    optimization_tolerance,
                    config,
                )
                split_state["policy"] = "topology_supported_split"
                split_state["details"] = {"surface_fidelity_preparation": split_preparation}
                proposals.append(split_state)
            except (DomainValidationError, MethodExecutionError) as error:
                iterations[-1]["split_candidate"] = {
                    "status": "REJECTED",
                    "reason": str(error),
                }
        elif progress:
            iterations[-1]["split_candidate"] = {
                "status": "REJECTED",
                "reason": "convex part budget exhausted",
                "parts": {link: len(next_patches[link]) for link in links},
            }

        if split_state is not None:
            split_shape_candidates = _shape_refit_candidates(
                split_state["asset"],
                split_state["patches"],
                visuals,
                coverage,
                split_state["evaluation"],
                optimization_tolerance,
                config,
                query=surface_query.nearest,
                fraction_hint=fraction_hint,
            )
            for candidate in split_shape_candidates:
                candidate["base_policy"] = "topology_supported_split"
            current_shape_candidates.extend(split_shape_candidates)

        # Jointly refit support planes every round. Previously this path ran only
        # after subdivision stopped, so a finite split budget could never invoke it.
        # Preserve accumulated support-plane trims. Splitting and refitting from the
        # visual surface is a separate candidate, not the base of every shape step.
        failed_shape_candidates: list[dict[str, Any]] = []
        materialized_shape_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        current_shape_candidates = _shape_candidate_order(current_shape_candidates)
        for shape_candidate in current_shape_candidates:
            if any(
                _definitely_fails_fidelity(value, config)
                for value in shape_candidate["fidelity"].values()
            ):
                failed_shape_candidates.append(shape_candidate)
                shape_refit_trials.append(
                    {
                        "round": iteration,
                        "base_policy": shape_candidate["base_policy"],
                        "policy": shape_candidate["policy"],
                        "fraction": shape_candidate["fraction"],
                        "status": "REJECTED",
                        "reason": "in-memory surface fidelity gate",
                        "fidelity": shape_candidate["fidelity"],
                    }
                )
                continue
            try:
                trial_state = _materialize_fidelity_candidate(
                    store,
                    asset,
                    shape_candidate["patches"],
                    coverage,
                    links,
                    config,
                )
            except (DomainValidationError, MethodExecutionError) as error:
                shape_refit_trials.append(
                    {
                        "round": iteration,
                        "base_policy": shape_candidate["base_policy"],
                        "policy": shape_candidate["policy"],
                        "fraction": shape_candidate["fraction"],
                        "status": "REJECTED",
                        "reason": str(error),
                    }
                )
                continue
            if not all(value["status"] == "PASSED" for value in trial_state["fidelity"].values()):
                failed_shape_candidates.append(
                    {
                        **shape_candidate,
                        "patches": trial_state["patches"],
                        "fidelity": trial_state["fidelity"],
                    }
                )
                shape_refit_trials.append(
                    {
                        "round": iteration,
                        "base_policy": shape_candidate["base_policy"],
                        "policy": shape_candidate["policy"],
                        "fraction": shape_candidate["fraction"],
                        "status": "REJECTED",
                        "reason": "materialized surface fidelity gate",
                        "fidelity": trial_state["fidelity"],
                    }
                )
                continue
            materialized_shape_candidates.append((shape_candidate, trial_state))
            if len(materialized_shape_candidates) >= config.max_materialized_candidates_per_round:
                break

        # MuJoCo model compilation dominates candidate scoring for dense compound
        # proxies. Treat support fractions as a trust-region line search and score
        # the bounded exact-candidate set concurrently. Each candidate owns its
        # MjModel/MjData instances; the final selected result still receives a
        # separate high-resolution sweep and dynamic rollouts.
        materialized_by_candidate = {
            id(candidate): (candidate, state) for candidate, state in materialized_shape_candidates
        }
        exact_candidates = [
            materialized_by_candidate[id(candidate)]
            for candidate in _shape_candidate_order(
                [candidate for candidate, _ in materialized_shape_candidates]
            )[: config.max_exact_candidates_per_round]
        ]

        def score_exact_candidate(item: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
            _, materialized_state = item
            return _score_materialized_candidate(
                store,
                materialized_state,
                search_evaluation_config,
                optimization_tolerance,
                config,
            )

        if len(exact_candidates) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=len(exact_candidates), thread_name_prefix="exact-candidate"
            ) as exact_pool:
                exact_states = list(exact_pool.map(score_exact_candidate, exact_candidates))
            iterations[-1]["exact_candidate_parallelism"] = len(exact_candidates)
        else:
            exact_states = [score_exact_candidate(item) for item in exact_candidates]
            iterations[-1]["exact_candidate_parallelism"] = len(exact_candidates)

        for (shape_candidate, _), trial_state in zip(exact_candidates, exact_states, strict=True):
            trial_state["policy"] = (
                "topology_support_plane_refit"
                if shape_candidate["base_policy"] == "topology_supported_split"
                else "joint_support_plane_refit"
            )
            trial_state["details"] = shape_candidate
            shape_refit_trials.append(
                {
                    "round": iteration,
                    "base_policy": shape_candidate["base_policy"],
                    "policy": shape_candidate["policy"],
                    "fraction": shape_candidate["fraction"],
                    "step_m": shape_candidate["step_m"],
                    "max_adjacency_gap_m": shape_candidate["max_adjacency_gap_m"],
                    "min_volume_ratio": shape_candidate["min_volume_ratio"],
                    "minimum_thickness_m": shape_candidate["minimum_thickness_m"],
                    "constraint_count": shape_candidate["constraint_count"],
                    "support_plane_count": shape_candidate["support_plane_count"],
                    "linear_constraint_residual_m": shape_candidate["linear_constraint_residual_m"],
                    "mean_constraint_residual_ratio": shape_candidate[
                        "mean_constraint_residual_ratio"
                    ],
                    "satisfied_constraint_fraction": shape_candidate[
                        "satisfied_constraint_fraction"
                    ],
                    "lp_slack_max_m": shape_candidate["lp_slack_max_m"],
                    "active_set_rounds": shape_candidate["active_set_rounds"],
                    "rejected_patch_count": shape_candidate["rejected_patch_count"],
                    "rejected_patches": shape_candidate["rejected_patches"],
                    "contact_regions": shape_candidate["contact_regions"],
                    "contact_normal_stabilization": shape_candidate["contact_normal_stabilization"],
                    "max_penetration_m": trial_state["evaluation"]["max_penetration_m"],
                    "simulation_status": trial_state["evaluation"]["status"],
                    "fidelity": trial_state["fidelity"],
                    "combined_score": trial_state["score"],
                    "status": "EVALUATED",
                }
            )
            proposals.append(trial_state)

        has_admissible_improvement = any(
            proposal["score"] < current_score - 1e-8
            and all(value["status"] == "PASSED" for value in proposal["fidelity"].values())
            for proposal in proposals
        )
        failed_shape = (
            min(
                failed_shape_candidates,
                key=lambda candidate: (
                    _fidelity_penalty(candidate["fidelity"], config),
                    float(candidate["linear_constraint_residual_m"]),
                    -float(candidate["fraction"]),
                ),
            )
            if failed_shape_candidates
            else None
        )
        if failed_shape is not None and not has_admissible_improvement:
            try:
                recovery_state, recovery_history = _close_candidate_surface_fidelity(
                    store,
                    asset,
                    failed_shape["patches"],
                    visuals,
                    coverage,
                    links,
                    search_evaluation_config,
                    optimization_tolerance,
                    config,
                )
                recovery_state["policy"] = "topology_supported_split"
                recovery_state["details"] = {
                    "surface_fidelity_preparation": recovery_history,
                    "source": "materialized_shape_fidelity_recovery",
                }
                proposals.append(recovery_state)
                iterations[-1]["topology_trigger"] = "materialized_shape_fidelity_recovery"
            except (DomainValidationError, MethodExecutionError) as error:
                iterations[-1]["materialized_shape_recovery"] = {
                    "status": "REJECTED",
                    "reason": str(error),
                }

        iterations[-1]["candidate_scores"] = [
            {
                "policy": proposal["policy"],
                "combined_score": proposal["score"],
                "max_penetration_m": proposal["evaluation"]["max_penetration_m"],
                "fidelity_passed": all(
                    value["status"] == "PASSED" for value in proposal["fidelity"].values()
                ),
                "linear_constraint_residual_m": proposal.get("details", {}).get(
                    "linear_constraint_residual_m"
                ),
                "satisfied_constraint_fraction": proposal.get("details", {}).get(
                    "satisfied_constraint_fraction"
                ),
                "parts": {link: len(proposal["patches"][link]) for link in links},
            }
            for proposal in proposals
        ]
        admissible = [
            proposal
            for proposal in proposals
            if all(value["status"] == "PASSED" for value in proposal["fidelity"].values())
        ]
        improving = [
            proposal for proposal in admissible if proposal["score"] < current_score - 1e-8
        ]
        selection_mode = "strict_improvement"
        if not improving:
            # A topology split can briefly increase the surface merit before the
            # next local fit. Permit only a bounded step that already reduces
            # penetration, while retaining the global best state for publication.
            improving = [
                proposal
                for proposal in admissible
                if proposal["evaluation"]["max_penetration_m"] < after["max_penetration_m"] - 1e-6
                and proposal["score"] <= current_score * 1.05
            ]
            selection_mode = "bounded_exploration"
        if not improving and split_state is not None:
            improving = _topology_refit_exploration_candidates(
                admissible,
                split_state,
                current_penetration_m=float(after["max_penetration_m"]),
                optimization_tolerance_m=optimization_tolerance,
            )
            if improving:
                selection_mode = "topology_refit_exploration"
        if (
            not improving
            and split_state is not None
            and any(proposal is split_state for proposal in admissible)
            and split_state["score"] <= current_score * 1.25
        ):
            # If a support step is blocked by local thickness/coverage, a bounded
            # source-driven split can add degrees of freedom for the next round.
            improving = [split_state]
            selection_mode = "topology_exploration"
        if not improving:
            reason = "no joint-constrained candidate improved geometry and clearance"
            break
        best = min(improving, key=lambda proposal: proposal["score"])
        previous_penetration = float(after["max_penetration_m"])
        patches = best["patches"]
        refined_collision_refs = best["references"]
        candidate_asset = best["asset"]
        after = best["evaluation"]
        fidelity = best["fidelity"]
        iterations[-1]["selected_candidate"] = best["policy"]
        iterations[-1]["selection_mode"] = selection_mode
        progress_requires_topology, improvement, minimum_shape_progress = (
            _shape_progress_requires_topology(
                best["policy"],
                previous_penetration,
                float(after["max_penetration_m"]),
                optimization_tolerance,
            )
        )
        exhausted_after_selection = _surface_reserve_exhausted(fidelity, config)
        topology_required = progress_requires_topology or bool(exhausted_after_selection)
        iterations[-1]["penetration_improvement_m"] = improvement
        iterations[-1]["minimum_shape_progress_m"] = minimum_shape_progress
        iterations[-1]["topology_required_next_round"] = topology_required
        iterations[-1]["surface_fidelity_reserve_exhausted_after_selection"] = (
            exhausted_after_selection
        )
        if best["policy"] in {
            "joint_support_plane_refit",
            "topology_support_plane_refit",
        }:
            details = best["details"]
            shape_refit_history.append(
                {
                    "round": iteration,
                    "policy": details["policy"],
                    "base_policy": details["base_policy"],
                    "fraction": details["fraction"],
                    "step_m": details["step_m"],
                    "max_adjacency_gap_m": details["max_adjacency_gap_m"],
                    "min_volume_ratio": details["min_volume_ratio"],
                    "minimum_thickness_m": details["minimum_thickness_m"],
                    "constraint_count": details["constraint_count"],
                    "support_plane_count": details["support_plane_count"],
                    "linear_constraint_residual_m": details["linear_constraint_residual_m"],
                    "mean_constraint_residual_ratio": details["mean_constraint_residual_ratio"],
                    "satisfied_constraint_fraction": details["satisfied_constraint_fraction"],
                    "lp_slack_max_m": details["lp_slack_max_m"],
                    "active_set_rounds": details["active_set_rounds"],
                    "rejected_patch_count": details["rejected_patch_count"],
                    "rejected_patches": details["rejected_patches"],
                    "contact_regions": details["contact_regions"],
                    "contact_normal_stabilization": details["contact_normal_stabilization"],
                    "max_penetration_m": after["max_penetration_m"],
                    "status": after["status"],
                    "combined_score": best["score"],
                }
            )
            shape_refit_rounds += 1
        save_checkpoint(iteration + 1)
    if best_state is not None and best_state["score"] < _candidate_score(
        after, fidelity, patches, optimization_tolerance, config
    ):
        patches = best_state["patches"]
        refined_collision_refs = best_state["references"]
        candidate_asset = best_state["asset"]
        after = best_state["evaluation"]
        fidelity = best_state["fidelity"]
    final_state = _score_materialized_candidate(
        store,
        {
            "patches": patches,
            "references": refined_collision_refs,
            "asset": candidate_asset,
            "fidelity": fidelity,
        },
        search_evaluation_config,
        optimization_tolerance,
        config,
        run_rollouts=True,
    )
    after = final_state["evaluation"]
    save_checkpoint(config.max_rounds + 1)
    passed = (
        after["status"] == "PASSED"
        and _meets_penetration_target(after["max_penetration_m"], optimization_tolerance)
        and all(v["status"] == "PASSED" for v in fidelity.values())
    )
    if passed:
        reason = "all fidelity and simulation gates passed with clearance safety margin"
    elif (
        not passed
        and reason == "all fidelity and simulation gates passed with clearance safety margin"
    ):
        reason = "clearance target reached but final dynamic stability validation failed"
    unresolved = []
    for pair in after["collision_pairs"]:
        if _meets_penetration_target(pair["max_penetration_m"], optimization_tolerance):
            continue
        region = dict(pair)
        for side in ("a", "b"):
            link = pair[f"link_{side}"]
            index = int(pair[f"geom_{side}"].rsplit("/", 1)[-1])
            region[f"source_faces_{side}"] = patches[link][index].faces.tolist()
        unresolved.append(region)
    common = {
        "schema": "physrealx.articulated-collision-repair/5",
        "status": "PASSED" if passed else "FAILED",
        "reason": reason,
        "evidence_scope": "collision_consistency_hypothesis",
        "acceptance_scope": "local_surface_coverage_and_simulation_feasibility",
        "physical_validation": False,
        "independent_observation_validation": False,
        "source_assembly": assembly_ref.model_dump(mode="json"),
        "source_feasibility": feasibility_ref.model_dump(mode="json"),
        "joint": joint.name,
        "locked_range": before["locked_range"],
        "links": list(links),
        "before": {"max_penetration_m": before["max_penetration_m"]},
        "after": {"max_penetration_m": after["max_penetration_m"]},
        "optimization_penetration_target_m": optimization_tolerance,
        "penetration_numeric_atol_m": _PENETRATION_NUMERIC_ATOL_M,
        "fidelity": fidelity,
        "unresolved_collisions": unresolved,
        "iterations": iterations,
        "visual_geometry_unchanged": True,
        "joint_model_unchanged": True,
        "parent_child_collision_enabled": True,
        "collision_geometry_refined": True,
        "source_face_ownership_preserved": True,
        "source_asset_lock": {
            "source_geometry": request.reconstructed_geometry.model_dump(mode="json")
            if request.reconstructed_geometry is not None
            else None,
            "visual_geometries": visual_refs,
            "joint": expected_motion,
            "reconstructed_motion": reconstructed_motion,
            "preserved_exactly": True,
        },
        "shape_refit_rounds": shape_refit_rounds,
        "shape_refit_history": shape_refit_history,
        "shape_refit_trials": shape_refit_trials,
        "contact_sided_refit_history": contact_sided_refit_history,
        "surface_fidelity_preparation": surface_fidelity_preparation,
        "config": config.model_dump(mode="json"),
        "surface_query": surface_query.describe(),
        "search_evaluation": {
            "sweep_samples": search_evaluation_config.sweep_samples,
            "adaptive_sweep_levels": search_evaluation_config.adaptive_sweep_levels,
            "matches_final_validation": True,
        },
        "visual_motion_audit": visual_motion_audit,
        "checkpoint": (
            checkpoint_store.latest.model_dump(mode="json")
            if checkpoint_store.latest is not None
            else None
        ),
        "resumed_from_checkpoint": resumed is not None,
        "limitations": [
            "sampled surface coverage and discrete motion sweep, not continuous certification",
            "local support-plane candidates use sampled contacts followed by "
            "exact sweep validation",
            "one-sided shells are selected from sampled contact directions and "
            "then independently revalidated",
            "fixed proxy skin is a collision model, not measured material thickness",
        ],
    }
    if not passed:
        return store.put_json(
            {
                **common,
                "candidate_assembly_input": None,
                "diagnostic_collision_geometries": {
                    link: ref.model_dump(mode="json")
                    for link, ref in refined_collision_refs.items()
                },
            }
        )
    if len(iterations) == 1 and not surface_prepared:
        return store.put_json(
            {
                **common,
                "collision_geometry_refined": False,
                "candidate_assembly_input": source_input.model_dump(mode="json"),
                "collision_refinement": None,
                "collision_reports": [r.model_dump(mode="json") for r in request.collision_reports],
            }
        )
    kept_parts = {link: [p.mesh for p in patches[link]] for link in links}
    surface_error_m = {link: fidelity[link]["surface_p95_m"] for link in links}
    if request.registration is None:
        raise DomainValidationError("articulated collision refinement requires registration")
    collision_refinement_ref = store.put_json(
        {
            "schema": "physrealx.articulated-collision-refinement/5",
            "status": "CANDIDATE",
            "evidence_scope": "collision_consistency_hypothesis",
            "physical_validation": False,
            "independent_observation_validation": False,
            "source_assembly": assembly_ref.model_dump(mode="json"),
            "source_feasibility": feasibility_ref.model_dump(mode="json"),
            "source_reference": request.reference_model.model_dump(mode="json"),
            "source_geometry": request.reconstructed_geometry.model_dump(mode="json")
            if request.reconstructed_geometry is not None
            else None,
            "upstream_video_articulation_refinement": (
                request.video_articulation_refinement.model_dump(mode="json")
            )
            if request.video_articulation_refinement is not None
            else None,
            "registration": request.registration.model_dump(mode="json"),
            "reconstructed_motion": reconstructed_motion,
            "visual_geometries": visual_refs,
            "source_collision_reports": {
                link: report_sources[link][0].model_dump(mode="json") for link in links
            },
            "refined_collision_geometries": {
                link: refined_collision_refs[link].model_dump(mode="json") for link in links
            },
            "links": list(links),
            "joint": joint.name,
            "locked_range": before["locked_range"],
            "fidelity": fidelity,
            "config": config.model_dump(mode="json"),
            "source_face_ownership_preserved": True,
            "selection_policy": "adaptive_source_surface_support_refit",
            "visual_geometry_changed": False,
            "collision_geometry_changed": True,
            "joint_model_changed": False,
            "parent_child_collision_disabled": False,
        }
    )

    replacement_reports: dict[str, ArtifactRef] = {}
    for link in links:
        source_report_ref, source_report = report_sources[link]
        report = {
            "schema": "physrealx.collision-build-report/1",
            "status": "PASSED",
            "unit": "m",
            "link": link,
            "source_geometry": source_report["source_geometry"],
            "collision_geometry": refined_collision_refs[link].model_dump(mode="json"),
            "method": source_report["method"],
            "preprocessing": {
                "mode": "adaptive_source_surface_support_refit",
                "source_collision_report": source_report_ref.model_dump(mode="json"),
                "source_closed_positive_volume": bool(visuals[link].is_volume),
                "volume_comparison_scope": "not_evaluated_after_proxy_refinement",
            },
            "metrics": {
                "convex_parts": float(len(kept_parts[link])),
                "bidirectional_surface_p95_m": surface_error_m[link],
                "bidirectional_sample_max_m": fidelity[link]["bidirectional_sample_max_m"],
                "worst_cell_coverage": fidelity[link]["worst_cell_coverage"],
                "contact_coverage": fidelity[link]["contact_coverage"],
            },
            "gates": {
                "convex": True,
                "closed_positive_volume": True,
                "roundtrip": True,
                "local_surface_error": True,
                "contact_surface_coverage": True,
                "source_face_ownership_preserved": True,
                "visual_collision_surface_fidelity": True,
                "visual_geometry_unchanged": True,
                "parent_child_collision_preserved": True,
            },
            "articulation_refinement": {
                "schema": "physrealx.articulated-collision-clearance/5",
                "source_collision_report": source_report_ref.model_dump(mode="json"),
                "source_assembly": assembly_ref.model_dump(mode="json"),
                "source_feasibility": feasibility_ref.model_dump(mode="json"),
                "collision_refinement": collision_refinement_ref.model_dump(mode="json"),
                "source_geometry": source_report["source_geometry"],
                "joint": joint.name,
                "locked_range": before["locked_range"],
                "policy": "adaptive_source_surface_support_refit",
                "visual_geometry_changed": False,
                "collision_geometry_changed": True,
                "joint_model_changed": False,
                "parent_child_collision_disabled": False,
            },
        }
        replacement_reports[link] = store.put_json(report)

    candidate_request = request.model_copy(
        update={
            "articulated_collision_refinement": collision_refinement_ref,
            "collision_reports": tuple(
                replacement_reports.get(
                    str(read_json(store, ref, "physrealx.collision-build-report/1")["link"]), ref
                )
                for ref in request.collision_reports
            ),
        }
    )
    candidate_input = store.put_json(candidate_request.model_dump(mode="json", by_alias=True))
    candidate_asset = reference_prior_asset(candidate_input, store)
    after = evaluate_asset_feasibility(store, candidate_asset, config=search_evaluation_config)
    if (
        asset.joints != candidate_asset.joints
        or asset.root_pose != candidate_asset.root_pose
        or asset.contact_exclusions != candidate_asset.contact_exclusions
        or [g for g in asset.geometries if g.role is GeometryRole.VISUAL]
        != [g for g in candidate_asset.geometries if g.role is GeometryRole.VISUAL]
    ):
        raise MethodExecutionError("collision refinement changed immutable asset geometry or poses")
    if after["status"] != "PASSED" or not _meets_penetration_target(
        after["max_penetration_m"], optimization_tolerance
    ):
        raise MethodExecutionError("published refinement did not reproduce simulation acceptance")
    return store.put_json(
        {
            **common,
            "candidate_assembly_input": candidate_input.model_dump(mode="json"),
            "collision_refinement": collision_refinement_ref.model_dump(mode="json"),
            "collision_reports": [
                ref.model_dump(mode="json") for ref in candidate_request.collision_reports
            ],
        }
    )
