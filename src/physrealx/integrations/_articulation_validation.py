"""Validate a coupled candidate without publishing rejected diagnostic geometry."""

from __future__ import annotations

import numpy as np

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, AssetSpec
from physrealx.errors import DomainValidationError

from ._artifacts import read_json
from .physical_feasibility import FeasibilityConfig, evaluate_asset_feasibility


def validate_articulation_candidate(
    store: ArtifactStore,
    refinement: ArtifactRef,
    source_assembly: ArtifactRef,
    collision_reports: list[ArtifactRef],
    config: FeasibilityConfig,
) -> ArtifactRef:
    result = read_json(store, refinement, "physrealx.video-articulation-refinement/2")
    from ._collision_shell import certified_surface_budget

    allowed_error, surface_certificate = certified_surface_budget(
        store, result, config.minimum_clearance_m
    )
    bundle = read_json(store, source_assembly, "physrealx.asset-assembly-bundle/1")
    asset = bundle["asset_spec"]
    evidence_ids = {r["artifact_id"] for r in asset["evidence"]}
    if (
        not {result["source_reference"]["artifact_id"], result["source_geometry"]["artifact_id"]}
        <= evidence_ids
    ):
        raise DomainValidationError(
            "diagnostic physics prior does not belong to source reconstruction"
        )
    motion = result["optimized_motion"]
    original = result["source_motion"]
    if asset["metadata"]["reconstructed_motion"] != original or len(asset["joints"]) != 1:
        raise DomainValidationError("diagnostic source assembly has a different motion")
    if config.minimum_clearance_m < result["config"]["minimum_clearance_m"]:
        raise DomainValidationError(
            "physics validation may not relax the surface clearance requirement"
        )
    reports = {}
    for ref in collision_reports:
        report = read_json(store, ref, "physrealx.collision-build-report/1")
        name = report["link"]
        if (
            name in reports
            or report["status"] != "PASSED"
            or report["source_geometry"] != result["diagnostic_geometries"].get(name)
            or report.get("source_refinement") != refinement.model_dump(mode="json")
        ):
            raise DomainValidationError(
                "rebuilt collision proxy does not match candidate visual surface"
            )
        reports[name] = report
    if set(reports) != set(asset["links"]):
        raise DomainValidationError(
            "candidate collision reports must cover every link exactly once"
        )
    joint = asset["joints"][0]
    if [joint["lower_limit"], joint["upper_limit"]] != motion["range"]:
        raise DomainValidationError("candidate physics validation changed source range")
    joint["axis"] = motion["axis"]
    joint["parent_to_joint"]["translation_m"] = motion["origin_m"]
    for geometry in asset["geometries"]:
        name = geometry["link"]
        role = geometry["role"]
        if geometry["local_pose"]["rotation_xyzw"] != [0.0, 0.0, 0.0, 1.0] or geometry["scale"] != [
            1.0,
            1.0,
            1.0,
        ]:
            raise DomainValidationError("candidate geometry requires the registered common frame")
        ref = (
            result["diagnostic_geometries"][name]
            if role == "visual"
            else reports[name]["collision_geometry"]
        )
        geometry["uri"] = "artifact://" + ref["artifact_id"]
        geometry["local_pose"]["translation_m"] = (
            -np.array(motion["origin_m"]) if name == motion["child"] else np.zeros(3)
        ).tolist()
    for physics in asset["physics"]:
        if physics["link"] == motion["child"]:
            physics["center_of_mass_m"] = (
                np.array(physics["center_of_mass_m"])
                + np.array(original["origin_m"])
                - motion["origin_m"]
            ).tolist()
    asset["metadata"]["reconstructed_motion"] = motion
    asset["metadata"]["diagnostic_only"] = True
    asset["evidence"].extend(
        [
            refinement.model_dump(mode="json"),
            *[r.model_dump(mode="json") for r in collision_reports],
        ]
    )
    # Keep the original VLM physical priors explicit; this is a feasibility rollout,
    # not an identification of real mass, inertia, friction or external hand forces.
    from ._collision_refit import SurfaceCoverage, bind_surface
    from ._surface_query import create_surface_query
    from .collision import load_meshes

    fidelity = {}
    query = create_surface_query(
        backend="auto", gpu_devices=(), require_accelerator=False, cpu_workers=1, batch_size=65536
    )
    try:
        for name, report in reports.items():
            visual = load_meshes(
                store, ArtifactRef.model_validate(result["diagnostic_geometries"][name])
            )[0]
            parts = load_meshes(store, ArtifactRef.model_validate(report["collision_geometry"]))
            patches = bind_surface(visual, parts, query=query.nearest)
            delta = visual.triangles_center - np.asarray(motion["origin_m"])
            axis = np.asarray(motion["axis"])
            radial = np.linalg.norm(delta - np.outer(delta @ axis, axis), axis=1)
            contact_faces = np.flatnonzero(radial < result["config"]["hinge_band_m"])
            coverage = SurfaceCoverage(
                visual,
                patches,
                set(),
                cell_size_m=0.01,
                seed=2026,
                query=query.nearest,
                contact_faces=contact_faces,
            )
            budget = report.get("metrics", {}).get("surface_error_budget_m")
            if budget is None or not np.isfinite(budget) or not 0 < budget <= allowed_error:
                raise DomainValidationError(
                    "candidate proxy error budget exceeds the certified clearance allocation"
                )
            fidelity[name], _ = coverage.evaluate(
                patches, max_error_m=budget, coverage_distance_m=budget, min_coverage=1.0
            )
        import trimesh

        from ._surface_clearance import swept_clearance

        proxy_meshes = {
            n: trimesh.util.concatenate(
                load_meshes(store, ArtifactRef.model_validate(r["collision_geometry"]))
            )
            for n, r in reports.items()
        }
        proxy_sweep = swept_clearance(
            proxy_meshes[motion["parent"]],
            proxy_meshes[motion["child"]],
            motion,
            query,
            clearance=config.minimum_clearance_m,
            samples=61,
            temporal_levels=5,
        )
    finally:
        query.close()
    candidate = AssetSpec.model_validate(asset)
    feasibility = evaluate_asset_feasibility(store, candidate, config=config)
    return store.put_json(
        {
            "schema": "physrealx.articulation-candidate-validation/1",
            "status": "PASSED"
            if result["status"] == feasibility["status"] == proxy_sweep["status"] == "PASSED"
            and all(row["status"] == "PASSED" for row in fidelity.values())
            else "FAILED",
            "source_refinement": refinement.model_dump(mode="json"),
            "source_assembly": source_assembly.model_dump(mode="json"),
            "collision_reports": [r.model_dump(mode="json") for r in collision_reports],
            "diagnostic_asset_spec": candidate.model_dump(mode="json"),
            "surface_status": result["status"],
            "construction_surface_clearance": surface_certificate,
            "proxy_surface_fidelity": fidelity,
            "continuous_proxy_clearance": proxy_sweep,
            "physics": feasibility,
            "config": config.model_dump(mode="json"),
            "range_preserved": True,
            "physical_parameters": "source_VLM_priors_not_independently_measured",
            "published_asset": False,
        }
    )
