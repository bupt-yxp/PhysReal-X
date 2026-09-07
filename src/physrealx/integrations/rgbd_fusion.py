"""CPU reconstruction of observed object surfaces from registered RGB-D views."""

from __future__ import annotations

import io
import zipfile
from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    DomainModel,
    MethodCapability,
    OperationSpec,
    canonical_json_bytes,
)
from physrealx.errors import MethodExecutionError
from physrealx.integrations._artifacts import (
    dependency_probe,
    input_artifact,
    publish,
    read_json,
)
from physrealx.integrations.rgbd import MetricRGBDInput, load_rgbd
from physrealx.methods import builtin_method


class RGBDFusionConfig(DomainModel):
    max_edge_m: float = Field(default=0.035, gt=0, le=0.1)
    max_holdout_median_m: float = Field(default=0.04, gt=0, le=0.1)
    min_view_points: int = Field(default=150, ge=30)
    min_state_views: int = Field(default=3, ge=3)


def surface_mesh(points: Any, colors: Any, mask: Any, max_edge_m: float) -> Any:
    import numpy as np
    import trimesh

    height, width = mask.shape
    ids = np.arange(height * width).reshape(height, width)
    a, b, c, d = (
        ids[:-1, :-1].ravel(),
        ids[:-1, 1:].ravel(),
        ids[1:, :-1].ravel(),
        ids[1:, 1:].ravel(),
    )
    faces = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    points, colors = points.reshape(-1, 3), colors.reshape(-1, 3)
    faces = faces[mask.ravel()[faces].all(axis=1)]
    edges = points[faces] - points[np.roll(faces, 1, axis=1)]
    faces = faces[(np.linalg.norm(edges, axis=2) <= max_edge_m).all(axis=1)]
    mesh = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


class RGBDFusionAdapter:
    descriptor = builtin_method(
        MethodCapability.GEOMETRY_RECONSTRUCT,
        "rgbd_fusion",
        input_contract="MetricMultiStateRGBDBundle/1",
        output_contract="VisualGeometryBundle/1+GeometryReconstructionReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        return dependency_probe(
            {
                "numpy": ("numpy", "1.26.4"),
                "scipy": ("scipy", "1.13.1"),
                "trimesh": ("trimesh", "4.5.1"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree

        config = RGBDFusionConfig.model_validate(spec.method.config)
        source_ref = input_artifact(spec, "source", artifacts)
        source = MetricRGBDInput.model_validate(
            read_json(artifacts, source_ref, "physrealx.metric-multistate-rgbd-bundle/1")
        )
        camera, _, views = load_rgbd(source, artifacts)
        reference = read_json(artifacts, source.reference_model, "physrealx.usd-reference/1")
        groups: dict[str, list[dict[str, Any]]] = {}
        for view in views:
            mask = view.mask & (view.depth > 0)
            if mask.sum() < config.min_view_points:
                continue
            points = (view.rays * view.depth[..., None] - view.pose[:, 3]) @ view.pose[:, :3]
            groups.setdefault(view.stream, []).append(
                {"frame_id": view.frame_id, "points": points, "rgb": view.rgb, "mask": mask}
            )
        if set(groups) != {s["stream_name"] for s in camera["streams"]}:
            raise MethodExecutionError("every capture state must have supported object depth")
        records: list[dict[str, Any]] = []
        payloads = []
        evaluations: dict[str, dict[str, Any]] = {}
        for state, frames in sorted(groups.items()):
            if len(frames) < config.min_state_views:
                raise MethodExecutionError(
                    f"{state} needs at least {config.min_state_views} supported views"
                )
            fit = [frame for index, frame in enumerate(frames) if index % 3 != 0]
            holdout = [frame for index, frame in enumerate(frames) if index % 3 == 0]
            cloud = np.concatenate([f["points"][f["mask"]] for f in fit])
            test_cloud = np.concatenate([frame["points"][frame["mask"]] for frame in holdout])
            distance = cKDTree(cloud).query(test_cloud, workers=1)[0]
            median = float(np.median(distance))
            if median > config.max_holdout_median_m:
                raise MethodExecutionError(
                    f"{state} held-out visible surface median error {median:.4f} m exceeds gate"
                )
            mesh = trimesh.util.concatenate(
                [surface_mesh(f["points"], f["rgb"], f["mask"], config.max_edge_m) for f in fit]
            )
            if len(mesh.faces) < 100:
                raise MethodExecutionError("RGB-D fusion did not produce a usable observed surface")
            mesh.merge_vertices(digits_vertex=4)
            payload = mesh.export(file_type="ply")
            mesh_ref = artifacts.put_bytes(payload, media_type="model/ply")
            path = f"surfaces/{state}.ply"
            payloads.append((path, payload))
            records.append(
                {
                    "state": state,
                    "artifact": mesh_ref.model_dump(mode="json"),
                    "path": path,
                    "vertex_count": len(mesh.vertices),
                    "triangle_count": len(mesh.faces),
                    "fit_frame_ids": [f["frame_id"] for f in fit],
                    "holdout_frame_ids": [frame["frame_id"] for frame in holdout],
                }
            )
            evaluations[state] = {
                "heldout_visible_surface_median_m": median,
                "heldout_visible_surface_p90_m": float(np.percentile(distance, 90)),
                "holdout_points": len(distance),
                "fit_points": len(cloud),
            }
        metrics = {
            "state_count": float(len(records)),
            "triangle_count": float(sum(r["triangle_count"] for r in records)),
            "max_heldout_surface_median_m": max(
                v["heldout_visible_surface_median_m"] for v in evaluations.values()
            ),
        }
        report = {
            "schema": "physrealx.geometry-reconstruction-report/1",
            "status": "OBSERVED_SURFACE_RECONSTRUCTED",
            "evidence_scope": "model_scaled_rgbd_visible_surfaces",
            "unit": "m",
            "coordinate_frame": camera["coordinate_frame"],
            "source_camera_artifact_id": source.camera_rig.artifact_id,
            "source_depth_artifact_id": source.enhanced_depth.artifact_id,
            "source_reference_artifact_id": source.reference_model.artifact_id,
            "observed_surfaces": records,
            "evaluations": evaluations,
            "metrics": metrics,
            "reference_link_count": len(reference["links"]),
            "gates": {
                "heldout_views": True,
                "independent_metric_truth": False,
                "complete_unobserved_geometry": False,
                "automatic_part_segmentation": False,
            },
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", canonical_json_bytes(report))
            for path, payload in payloads:
                archive.writestr(path, payload)
        return publish(
            spec,
            artifacts,
            {"geometry_reconstruction_report": report},
            artifacts={
                "visual_geometry_bundle": artifacts.put_bytes(
                    buffer.getvalue(), media_type="application/zip"
                )
            },
            metrics=metrics,
        )
