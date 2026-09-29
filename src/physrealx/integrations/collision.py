"""Measured-unit convex collision proxies using the official CoACD binding."""

from __future__ import annotations

import io
import math
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import ArtifactRef, DomainModel, MethodCapability, OperationSpec
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import dependency_probe, input_artifact, publish
from physrealx.methods import builtin_method


def mesh_reference(uri: str, store: ArtifactStore) -> Any:
    from physrealx.domain import ArtifactRef

    if not uri.startswith("artifact://"):
        raise DomainValidationError("simulation meshes require immutable artifact:// URIs")
    identifier = uri[len("artifact://") :]
    if not store.verify(identifier):
        raise DomainValidationError("simulation mesh artifact is unavailable")
    payload = store.get_bytes(identifier)
    # Assembly publishes GLB collision and model GLB/PLY visual geometry.
    media = (
        "model/gltf-binary"
        if payload.startswith(b"glTF")
        else "model/ply"
        if payload.startswith(b"ply")
        else "model/obj"
    )
    return ArtifactRef(artifact_id=identifier, size_bytes=len(payload), media_type=media)


def load_meshes(store: ArtifactStore, reference: ArtifactRef, *, process: bool = True) -> list[Any]:
    from importlib import import_module

    import numpy as np

    trimesh = import_module("trimesh")

    formats = {"model/gltf-binary": "glb", "model/ply": "ply", "model/obj": "obj"}
    if reference.media_type not in formats or reference.size_bytes > 512 * 1024 * 1024:
        raise DomainValidationError("mesh must be a self-contained GLB, PLY or OBJ under 512 MiB")
    try:
        scene = trimesh.load(
            io.BytesIO(store.get_bytes(reference)),
            file_type=formats[reference.media_type],
            force="scene",
            resolver={},
            process=process,
        )
        meshes = list(scene.dump())
    except (ValueError, OSError, KeyError) as error:
        raise DomainValidationError("mesh artifact cannot be decoded") from error
    if not meshes:
        raise DomainValidationError("mesh artifact is empty")
    for mesh in meshes:
        if (
            not isinstance(mesh, trimesh.Trimesh)
            or len(mesh.vertices) < 4
            or len(mesh.faces) < 4
            or not np.isfinite(mesh.vertices).all()
        ):
            raise DomainValidationError("mesh must contain finite triangulated geometry")
    return meshes


def _exterior_samples(
    parts: list[Any], proxy: Any, seed: int, *, include_vertices: bool = False
) -> Any:
    """Sample the proxy union boundary, excluding buried convex partition faces."""
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull

    samples, faces = trimesh.sample.sample_surface(proxy, 5000, seed=seed)
    epsilon = max(float(np.ptp(proxy.vertices, axis=0).max()) * 1e-7, 1e-9)
    normals = proxy.face_normals[faces]
    if include_vertices:
        # Include every candidate vertex and face center so small boundary defects
        # cannot move out of a random sample after each subdivision.
        samples = np.concatenate(
            [samples, *[p.vertices for p in parts], *[p.triangles_center for p in parts]]
        )
        normals = np.concatenate(
            [normals, *[p.vertex_normals for p in parts], *[p.face_normals for p in parts]]
        )
    outside = samples + normals * epsilon
    buried = np.zeros(len(samples), dtype=bool)
    for part in parts:
        bounds = part.bounds
        candidates = np.flatnonzero(
            (~buried) & np.all(outside > bounds[0], axis=1) & np.all(outside < bounds[1], axis=1)
        )
        if not len(candidates):
            continue
        equations = ConvexHull(part.vertices).equations
        buried[candidates] |= np.all(
            outside[candidates] @ equations[:, :3].T + equations[:, 3] < -epsilon / 10,
            axis=1,
        )
    if int((~buried).sum()) < 500:
        raise MethodExecutionError("collision proxy has insufficient exterior surface samples")
    return samples[~buried]


def _allocate_component_budgets(areas: list[float], total: int) -> list[int]:
    """Split one global convex-part budget across disconnected source surfaces."""
    if not areas:
        return []
    if total < len(areas):
        raise MethodExecutionError(
            f"collision input has {len(areas)} non-convex components but only {total} parts"
        )
    remaining = total - len(areas)
    positive = [max(float(area), 0.0) for area in areas]
    area_sum = sum(positive)
    weights = positive if area_sum > 0 else [1.0] * len(areas)
    weight_sum = sum(weights)
    shares = [remaining * weight / weight_sum for weight in weights]
    extras = [math.floor(share) for share in shares]
    for index in sorted(
        range(len(areas)), key=lambda item: (shares[item] - extras[item], -item), reverse=True
    )[: remaining - sum(extras)]:
        extras[index] += 1
    return [1 + extra for extra in extras]


class CollisionConfig(DomainModel):
    source_unit: Literal["m"]
    link: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    threshold_m: float = Field(default=0.005, gt=0, allow_inf_nan=False)
    max_parts: int = Field(default=64, ge=1, le=1024)
    max_volume_error: float = Field(default=0.15, gt=0, le=1)
    seed: int = Field(default=1, ge=1, le=2_147_483_647)
    weld_tolerance_m: float = Field(default=0.0, ge=0, le=0.001, allow_inf_nan=False)
    preprocessing: Literal["strict", "coacd"] = "strict"


class CollisionAdapter:
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "coacd",
        input_contract="VisualMesh/1",
        output_contract="CollisionGeometry/1+CollisionBuildReport/1",
        python=">=3.10,<3.13",
        version="1.5.0",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe(
            {
                "numpy": ("numpy", "1.26.4"),
                "trimesh": ("trimesh", "4.5.1"),
                "coacd": ("coacd", "1.0.9"),
                "rtree": ("rtree", "1.4.1"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        from importlib import import_module

        import coacd
        import numpy as np

        trimesh = import_module("trimesh")

        config = CollisionConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "asset", artifacts)
        meshes = load_meshes(artifacts, source)
        # UV and normal seams duplicate vertices without creating a physical gap.
        # Collision topology ignores appearance and merges only exactly equal XYZ.
        topology = []
        for mesh in meshes:
            vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
            topology.append(
                trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
            )
        meshes = topology
        welded_vertices = 0
        if config.weld_tolerance_m:
            # Explicitly bounded seam welding; never fill missing surfaces or cabinet openings.
            digits = math.ceil(-math.log10(config.weld_tolerance_m / math.sqrt(3)))
            meshes = [trimesh.util.concatenate(meshes)]
            for mesh in meshes:
                original_count = len(mesh.vertices)
                mesh.merge_vertices(digits_vertex=digits, merge_tex=True, merge_norm=True)
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.remove_unreferenced_vertices()
                welded_vertices += original_count - len(mesh.vertices)
        source_closed = all(mesh.is_volume for mesh in meshes)
        if not source_closed and config.preprocessing == "strict":
            raise DomainValidationError(
                "collision input must be watertight, consistently wound positive volumes"
            )
        components = [part for original in meshes for part in original.split(only_watertight=False)]
        fixed_parts = []
        decomposable = []
        for mesh in components:
            hull = mesh.convex_hull
            convex_volume_agrees = mesh.volume > 0 and math.isclose(
                float(hull.volume), float(mesh.volume), rel_tol=1e-6
            )
            if mesh.is_volume and (mesh.is_convex or convex_volume_agrees):
                fixed_parts.append(hull)
            else:
                decomposable.append(mesh)
        remaining_budget = config.max_parts - len(fixed_parts)
        component_budgets = _allocate_component_budgets(
            [float(mesh.area) for mesh in decomposable], remaining_budget
        )
        parts = list(fixed_parts)
        for mesh, component_budget in zip(decomposable, component_budgets, strict=True):
            decomposition = coacd.run_coacd(
                coacd.Mesh(mesh.vertices, mesh.faces),
                threshold=config.threshold_m,
                real_metric=True,
                max_convex_hull=component_budget,
                preprocess_mode="off" if mesh.is_volume else "on",
                preprocess_resolution=max(
                    50,
                    min(
                        256,
                        math.ceil(
                            math.sqrt(3)
                            * float(np.ptp(mesh.vertices, axis=0).max())
                            / config.threshold_m
                        ),
                    ),
                )
                if not mesh.is_volume
                else 50,
                seed=config.seed,
            )
            # Rebuild each convex hull from its vertices. Native triangulation can
            # contain coplanar degeneracies after float conversion; hulls are the
            # actual collision representation, not the tessellation of that hull.
            parts.extend(
                trimesh.Trimesh(vertices=v, faces=f, process=True).convex_hull
                for v, f in decomposition
            )
        # GLB stores positions as float32. Build topology at that precision so
        # quantization cannot turn nearly coplanar hull faces non-convex on reload.
        parts = [trimesh.convex.convex_hull(part.vertices.astype(np.float32)) for part in parts]
        invalid = sum(not part.is_volume or not part.is_convex for part in parts)
        if not parts or len(parts) > config.max_parts or invalid:
            raise MethodExecutionError(
                f"CoACD collision gates failed: {len(parts)} parts (limit {config.max_parts}), "
                f"{invalid} invalid convex volumes"
            )
        quality_metrics: dict[str, float] = {}
        if source_closed:
            volume = sum(float(mesh.volume) for mesh in meshes)
            volume_error = abs(sum(float(part.volume) for part in parts) - volume) / volume
            if not np.isfinite(volume_error) or volume_error > config.max_volume_error:
                raise MethodExecutionError(
                    f"collision proxy volume error {volume_error:.6g} exceeds "
                    f"tolerance {config.max_volume_error:.6g}"
                )
            quality_metrics["relative_volume_error"] = volume_error
        else:
            # An open surface has no defined enclosed volume. Verify geometry
            # directly instead of comparing signed, origin-dependent estimates.
            source_mesh = trimesh.util.concatenate(meshes)
            proxy_mesh = trimesh.util.concatenate(parts)
            source_samples, _ = trimesh.sample.sample_surface(source_mesh, 5000, seed=config.seed)
            proxy_samples = _exterior_samples(parts, proxy_mesh, config.seed)
            source_distance = trimesh.proximity.closest_point(proxy_mesh, source_samples)[1]
            proxy_distance = trimesh.proximity.closest_point(source_mesh, proxy_samples)[1]
            p95 = max(
                float(np.quantile(distance, 0.95)) for distance in (source_distance, proxy_distance)
            )
            if not np.isfinite(p95) or p95 > config.threshold_m:
                raise MethodExecutionError(
                    f"collision proxy surface P95 {p95:.6g} m exceeds "
                    f"threshold {config.threshold_m:.6g} m"
                )
            quality_metrics["bidirectional_surface_p95_m"] = p95
        scene = trimesh.Scene()
        for index, part in enumerate(parts):
            scene.add_geometry(
                part, node_name=f"{config.link}_{index}", geom_name=f"convex_{index}"
            )
        collision = artifacts.put_bytes(
            scene.export(file_type="glb"), media_type="model/gltf-binary"
        )
        # Reopen the actual exported bytes, rather than reporting only in-memory quality.
        decoded = load_meshes(artifacts, collision)
        if len(decoded) != len(parts) or any(not p.is_volume or not p.is_convex for p in decoded):
            raise MethodExecutionError("collision GLB roundtrip changed valid convex geometry")
        metrics = {
            "convex_parts": float(len(parts)),
            **quality_metrics,
            "welded_vertices": float(welded_vertices),
        }
        report = {
            "schema": "physrealx.collision-build-report/1",
            "status": "PASSED",
            "unit": "m",
            "link": config.link,
            "source_geometry": source.model_dump(mode="json"),
            "collision_geometry": collision.model_dump(mode="json"),
            "method": self.descriptor.model_dump(mode="json"),
            "preprocessing": {
                "mode": config.preprocessing,
                "weld_tolerance_m": config.weld_tolerance_m,
                "source_closed_positive_volume": source_closed,
                "volume_comparison_scope": "closed_surface"
                if source_closed
                else "not_applicable_open_surface",
                "source_surface_samples": 0 if source_closed else len(source_samples),
                "proxy_surface_samples": 0 if source_closed else len(proxy_samples),
                "component_part_budgets": component_budgets,
            },
            "metrics": metrics,
            "gates": {
                "convex": True,
                "closed_positive_volume": True,
                ("volume_error" if source_closed else "surface_error"): True,
                "roundtrip": True,
            },
        }
        return publish(
            spec,
            artifacts,
            {"collision_build_report": report},
            artifacts={"collision_geometry": collision},
            metrics=metrics,
        )
