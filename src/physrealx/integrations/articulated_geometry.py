"""Collision-only refinement over immutable module-one start/end motion."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel, GeometryRepresentation, GeometryRole
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations._collision_refit import (
    SurfaceCoverage,
    SurfacePatch,
    bind_surface,
    clip_patch_support,
    fit_patches,
)
from physrealx.integrations.assembly import assembled_asset
from physrealx.integrations.asset_prior import ReferenceAssemblyInput, reference_prior_asset
from physrealx.integrations.collision import load_meshes, mesh_reference
from physrealx.integrations.physical_feasibility import (
    FeasibilityConfig,
    evaluate_asset_feasibility,
)

_PENETRATION_NUMERIC_ATOL_M = 1e-7


def _meets_penetration_target(value_m: float, target_m: float) -> bool:
    """Compare solver penetration with sub-micrometre numerical tolerance."""
    return value_m <= target_m + _PENETRATION_NUMERIC_ATOL_M


class ArticulatedCollisionRefinementConfig(DomainModel):
    """Finite refinement budget and local surface/contact fidelity gates."""

    max_local_error_m: float = Field(default=0.003, gt=0, le=0.01)
    coverage_distance_m: float = Field(default=0.002, gt=0, le=0.01)
    min_patch_coverage: float = Field(default=0.99, gt=0, le=1)
    patch_size_m: float = Field(default=0.01, gt=0, le=0.1)
    proxy_skin_m: float = Field(default=0.00025, gt=0, le=0.002)
    shape_trust_region_m: float = Field(default=0.0018, gt=0, le=0.005)
    min_proxy_volume_ratio: float = Field(default=0.1, gt=0, le=1)
    min_proxy_thickness_ratio: float = Field(default=0.25, gt=0, le=1)
    penetration_safety_margin_m: float = Field(default=5e-5, ge=0, le=0.0001)
    max_rounds: int = Field(default=6, ge=1, le=20)
    max_parts_per_link: int = Field(default=768, ge=2, le=2048)
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
        return self


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


def _publish_parts(
    store: ArtifactStore, link: str, parts: list[Any]
) -> tuple[ArtifactRef, list[int]]:
    import hashlib
    from collections import defaultdict

    import numpy as np
    import trimesh

    def signature(mesh: Any) -> str:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        vertices = vertices[np.lexsort(vertices.T)]
        return hashlib.sha256(vertices.tobytes()).hexdigest()

    scene: Any = trimesh.Scene()
    order: dict[str, list[int]] = defaultdict(list)
    for index, part in enumerate(parts):
        scene.add_geometry(part, node_name=f"{link}_{index:05d}", geom_name=f"convex_{index:05d}")
        order[signature(part)].append(index)
    ref = store.put_bytes(scene.export(file_type="glb"), media_type="model/gltf-binary")
    decoded = load_meshes(store, ref)
    invalid = [
        {"index": index, "is_volume": bool(part.is_volume), "is_convex": bool(part.is_convex)}
        for index, part in enumerate(decoded)
        if not part.is_volume or not part.is_convex
    ]
    if len(decoded) != len(parts) or invalid:
        raise MethodExecutionError(
            "refitted collision proxy failed convex GLB roundtrip: "
            f"decoded={len(decoded)} expected={len(parts)} invalid={invalid[:3]}"
        )
    permutation = []
    for part in decoded:
        matches = order.get(signature(part), [])
        if not matches:
            raise MethodExecutionError("collision roundtrip changed a source patch")
        permutation.append(matches.pop())
    return ref, permutation


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


def _shape_refit_candidates(
    asset: Any,
    patches: dict[str, list[SurfacePatch]],
    visuals: dict[str, Any],
    coverage: dict[str, SurfaceCoverage],
    evaluation: dict[str, Any],
    tolerance: float,
    config: ArticulatedCollisionRefinementConfig,
) -> list[dict[str, Any]]:
    """Clip local convex supports using every failed pose in the joint sweep."""
    import numpy as np
    import trimesh
    from scipy.optimize import linprog

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

    def contact_regions() -> dict[str, list[list[int]]]:
        regions: dict[str, list[list[int]]] = {}
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
            regions[link] = sorted(components, key=lambda value: value[0])
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
    fractions = (
        (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625)
        if remaining_penetration <= 0.0002
        else (0.25, 0.125, 0.0625, 0.5, 0.75, 1.0)
    )
    for fraction in fractions:
        count = len(variable_order)
        rows: list[Any] = []
        bounds: list[float] = []
        for target, terms in linear_constraints:
            row = np.zeros(count + 1)
            for key, coefficient in terms:
                row[variable_index[key]] -= coefficient
            rows.append(row)
            bounds.append(-fraction * target)
        for index in range(count):
            row = np.zeros(count + 1)
            row[index], row[-1] = 1.0, -1.0
            rows.append(row)
            bounds.append(0.0)
        objective = np.r_[np.full(count, 1e-6 / count), 1.0]
        solution = linprog(
            objective,
            A_ub=np.asarray(rows),
            b_ub=np.asarray(bounds),
            bounds=[(0.0, config.shape_trust_region_m)] * (count + 1),
            method="highs",
        )
        if not solution.success:
            logging.getLogger(__name__).info(
                "support optimization rejected fraction=%s reason=%s",
                fraction,
                solution.message,
            )
            continue
        trims = solution.x[:count]
        cuts: dict[str, dict[int, list[tuple[Any, float]]]] = {link: {} for link in patches}
        for key, trim in zip(variable_order, trims, strict=True):
            if trim <= 1e-9:
                continue
            link, index, unit, support = support_variables[key]
            cuts[link].setdefault(index, []).append((unit, support - float(trim)))

        proposal = {link: list(values) for link, values in patches.items()}
        shape_metrics: dict[str, dict[int, dict[str, Any]]] = {link: {} for link in patches}
        valid = True
        for link, indexed in cuts.items():
            for index, halfspaces in indexed.items():
                clipped, clip_metrics = clip_patch_support(
                    patches[link][index],
                    visuals[link],
                    halfspaces,
                    trust_region_m=config.shape_trust_region_m,
                    min_thickness_ratio=config.min_proxy_thickness_ratio,
                    min_volume_ratio=config.min_proxy_volume_ratio,
                )
                if clipped is None:
                    logging.getLogger(__name__).info(
                        "support candidate rejected fraction=%s link=%s patch=%s gate=%s",
                        fraction,
                        link,
                        index,
                        clip_metrics,
                    )
                    valid = False
                    break
                proposal[link][index], shape_metrics[link][index] = clipped, clip_metrics
            if not valid:
                break
        if not valid or not any(shape_metrics.values()):
            continue

        max_adjacency_gap = 0.0
        for link, visual in visuals.items():
            face_to_patch = {
                int(face): index
                for index, patch in enumerate(patches[link])
                for face in patch.faces
            }
            for left, right in visual.face_adjacency:
                a, b = face_to_patch[int(left)], face_to_patch[int(right)]
                if a == b or (a not in shape_metrics[link] and b not in shape_metrics[link]):
                    continue
                common = np.intersect1d(visual.faces[int(left)], visual.faces[int(right)])
                probes = visual.vertices[common]
                if len(probes) == 2:
                    probes = np.vstack([probes, probes.mean(axis=0)])
                neighborhood = trimesh.util.concatenate(
                    [proposal[link][a].mesh, proposal[link][b].mesh]
                )
                distance = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
                    neighborhood, probes
                )[1]
                max_adjacency_gap = max(
                    max_adjacency_gap,
                    float(distance.max()),
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
                "support candidate rejected fraction=%s fidelity=%s",
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
            continue
        accepted_metrics = [value for values in shape_metrics.values() for value in values.values()]
        linear_residual = max(
            (
                fraction * target
                - sum(coefficient * trims[variable_index[key]] for key, coefficient in terms)
                for target, terms in linear_constraints
            ),
            default=0.0,
        )
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
                "support_plane_count": int(np.count_nonzero(trims > 1e-9)),
                "linear_constraint_residual_m": max(0.0, float(linear_residual)),
                "contact_regions": regions,
                "fidelity": fidelity,
            }
        )
        # Keep ordered backtracking candidates until GLB round-trip and the exact
        # sweep have run.  A geometrically valid in-memory hull can still become
        # non-volumetric at float32 serialization and must not end the line search.
    return candidates


def refine_articulated_collision(
    store: ArtifactStore,
    assembly_ref: ArtifactRef,
    feasibility_ref: ArtifactRef,
    *,
    config: ArticulatedCollisionRefinementConfig,
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
    if asset.contact_exclusions:
        raise DomainValidationError("collision refinement requires enabled parent/child collision")
    evaluation_config = FeasibilityConfig.model_validate(feasibility["config"])
    if config.penetration_safety_margin_m >= evaluation_config.penetration_tolerance_m:
        raise DomainValidationError(
            "collision refinement safety margin must be below the feasibility tolerance"
        )
    optimization_tolerance = (
        evaluation_config.penetration_tolerance_m - config.penetration_safety_margin_m
    )
    before = evaluate_asset_feasibility(
        store,
        asset,
        config=evaluation_config,
        retain_contact_constraints=True,
        contact_constraint_tolerance_m=optimization_tolerance,
    )
    if (
        before["status"] != feasibility["status"]
        or before["locked_range"] != feasibility.get("locked_range")
        or not np.isclose(before["max_penetration_m"], feasibility["max_penetration_m"])
    ):
        raise DomainValidationError("collision evidence is stale or non-reproducible")

    source_input = ArtifactRef.model_validate(assembly["source_input"])
    request = ReferenceAssemblyInput.model_validate(
        read_json(store, source_input, "physrealx.reference-assembly-input/1")
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

    conflicts = _conflicting_parts(before, prefixes, evaluation_config.penetration_tolerance_m)
    if any(i >= len(parts[link]) for link in links for i in conflicts[link]):
        raise DomainValidationError("failed sweep references an unavailable collision part")
    patches = {link: bind_surface(visuals[link], parts[link]) for link in links}
    coverage = {
        link: SurfaceCoverage(
            visuals[link],
            patches[link],
            conflicts[link],
            cell_size_m=config.patch_size_m,
            seed=config.seed,
        )
        for link in links
    }
    iterations: list[dict[str, Any]] = []
    fidelity: dict[str, Any] = {}
    refined_collision_refs = {
        link: mesh_reference(collision_geometries[link][1].uri, store) for link in links
    }
    candidate_asset = asset
    after = before
    reason = "refinement budget exhausted"
    shape_refit_rounds = 0
    shape_refit_history: list[dict[str, Any]] = []
    shape_refit_trials: list[dict[str, Any]] = []
    for iteration in range(config.max_rounds + 1):
        selected = _conflicting_parts(after, prefixes, evaluation_config.penetration_tolerance_m)
        for link in links:
            fidelity[link], unsupported = coverage[link].evaluate(
                patches[link],
                max_error_m=config.max_local_error_m,
                coverage_distance_m=config.coverage_distance_m,
                min_coverage=config.min_patch_coverage,
            )
            selected[link].update(unsupported)
        iterations.append(
            {
                "round": iteration,
                "max_penetration_m": after["max_penetration_m"],
                "simulation_status": after["status"],
                "fidelity": dict(fidelity),
                "parts": {link: len(patches[link]) for link in links},
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
        progress = False
        next_patches = {}
        rejected_splits = []
        for link in links:
            updated = []
            for index, patch in enumerate(patches[link]):
                if index in selected[link] and (patch.depth == 0 or len(patch.faces) > 1):
                    children = fit_patches(visuals[link], patch, config.proxy_skin_m)
                    if children is None:
                        updated.append(patch)
                        rejected_splits.append(
                            {
                                "link": link,
                                "part": index,
                                "depth": patch.depth,
                                "source_faces": len(patch.faces),
                                "reason": "degenerate convex subdivision",
                            }
                        )
                    else:
                        updated.extend(children)
                        progress = True
                else:
                    updated.append(patch)
            next_patches[link] = updated
        if rejected_splits:
            iterations[-1]["rejected_patch_splits"] = rejected_splits
        if not progress:
            shape_candidates = _shape_refit_candidates(
                candidate_asset,
                patches,
                visuals,
                coverage,
                after,
                optimization_tolerance,
                config,
            )
            admissible = []
            for shape_candidate in shape_candidates:
                try:
                    candidate_patches, candidate_refs, trial_asset = _materialize_candidate(
                        store, asset, shape_candidate["patches"]
                    )
                except (DomainValidationError, MethodExecutionError) as error:
                    shape_refit_trials.append(
                        {
                            "policy": shape_candidate["policy"],
                            "fraction": shape_candidate["fraction"],
                            "status": "REJECTED",
                            "reason": str(error),
                        }
                    )
                    continue
                trial_fidelity = {
                    link: coverage[link].evaluate(
                        candidate_patches[link],
                        max_error_m=config.max_local_error_m,
                        coverage_distance_m=config.coverage_distance_m,
                        min_coverage=config.min_patch_coverage,
                    )[0]
                    for link in links
                }
                if not all(value["status"] == "PASSED" for value in trial_fidelity.values()):
                    continue
                trial = evaluate_asset_feasibility(
                    store,
                    trial_asset,
                    config=evaluation_config,
                    retain_contact_constraints=True,
                    contact_constraint_tolerance_m=optimization_tolerance,
                )
                shape_refit_trials.append(
                    {
                        "policy": shape_candidate["policy"],
                        "fraction": shape_candidate["fraction"],
                        "step_m": shape_candidate["step_m"],
                        "max_adjacency_gap_m": shape_candidate["max_adjacency_gap_m"],
                        "min_volume_ratio": shape_candidate["min_volume_ratio"],
                        "minimum_thickness_m": shape_candidate["minimum_thickness_m"],
                        "constraint_count": shape_candidate["constraint_count"],
                        "support_plane_count": shape_candidate["support_plane_count"],
                        "linear_constraint_residual_m": shape_candidate[
                            "linear_constraint_residual_m"
                        ],
                        "contact_regions": shape_candidate["contact_regions"],
                        "max_penetration_m": trial["max_penetration_m"],
                        "status": trial["status"],
                        "fidelity": trial_fidelity,
                    }
                )
                if trial["max_penetration_m"] < after["max_penetration_m"] - 1e-9:
                    admissible.append(
                        (
                            candidate_patches,
                            candidate_refs,
                            trial_asset,
                            trial,
                            trial_fidelity,
                            shape_candidate,
                        )
                    )
                    break
            if not admissible:
                reason = "support-plane candidates cannot satisfy shape and fidelity gates"
                break
            best = admissible[0]
            patches, refined_collision_refs, candidate_asset, after, fidelity = best[:5]
            shape_refit_history.append(
                {
                    "policy": best[5]["policy"],
                    "fraction": best[5]["fraction"],
                    "step_m": best[5]["step_m"],
                    "max_adjacency_gap_m": best[5]["max_adjacency_gap_m"],
                    "min_volume_ratio": best[5]["min_volume_ratio"],
                    "minimum_thickness_m": best[5]["minimum_thickness_m"],
                    "constraint_count": best[5]["constraint_count"],
                    "support_plane_count": best[5]["support_plane_count"],
                    "linear_constraint_residual_m": best[5]["linear_constraint_residual_m"],
                    "contact_regions": best[5]["contact_regions"],
                    "max_penetration_m": after["max_penetration_m"],
                    "status": after["status"],
                }
            )
            shape_refit_rounds += 1
            continue
        if any(not v or len(v) > config.max_parts_per_link for v in next_patches.values()):
            reason = "convex part budget exhausted without relaxing fidelity"
            break
        for link in links:
            ids = np.concatenate([p.faces for p in next_patches[link]])
            if not np.array_equal(np.sort(ids), np.arange(len(visuals[link].faces))):
                raise MethodExecutionError("refinement dropped or duplicated source face ownership")
        patches, refined_collision_refs, candidate_asset = _materialize_candidate(
            store, asset, next_patches
        )
        after = evaluate_asset_feasibility(
            store,
            candidate_asset,
            config=evaluation_config,
            retain_contact_constraints=True,
            contact_constraint_tolerance_m=optimization_tolerance,
        )
    passed = (
        after["status"] == "PASSED"
        and _meets_penetration_target(after["max_penetration_m"], optimization_tolerance)
        and all(v["status"] == "PASSED" for v in fidelity.values())
    )
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
        "schema": "physrealx.articulated-collision-repair/4",
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
        "shape_refit_rounds": shape_refit_rounds,
        "shape_refit_history": shape_refit_history,
        "shape_refit_trials": shape_refit_trials,
        "config": config.model_dump(mode="json"),
        "limitations": [
            "sampled surface coverage and discrete motion sweep, not continuous certification",
            "local support-plane candidates use sampled contacts followed by "
            "exact sweep validation",
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
    if len(iterations) == 1:
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
    motion = asset.metadata.get("reconstructed_motion")
    if not isinstance(motion, dict) or motion.get("range") != before["locked_range"]:
        raise DomainValidationError("collision refinement changed module-one motion")
    visual_refs = {
        link: mesh_reference(visual_geometries[link].uri, store).model_dump(mode="json")
        for link in links
    }
    collision_refinement_ref = store.put_json(
        {
            "schema": "physrealx.articulated-collision-refinement/4",
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
            "upstream_geometry_refinement": request.geometry_refinement.model_dump(mode="json")
            if request.geometry_refinement is not None
            else None,
            "registration": request.registration.model_dump(mode="json"),
            "reconstructed_motion": motion,
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
                "schema": "physrealx.articulated-collision-clearance/4",
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
    after = evaluate_asset_feasibility(store, candidate_asset, config=evaluation_config)
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
