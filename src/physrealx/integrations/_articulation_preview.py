"""Measured before/after geometry, video fit, and swept-clearance diagnostics."""

from __future__ import annotations

import io
from importlib import import_module
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image

from physrealx.domain import ArtifactRef
from physrealx.reconstruction import joint_transform

from ._artifacts import read_json
from ._surface_query import create_surface_query
from ._triangle_contacts import triangle_pairs
from .collision import load_meshes


def preview_video_articulation(
    store: Any, refinement: ArtifactRef, *, frames: int = 24
) -> dict[str, ArtifactRef]:
    if type(frames) is not int or not 2 <= frames <= 120:
        raise ValueError("preview frames must be between 2 and 120")
    Poly3DCollection = import_module("mpl_toolkits.mplot3d.art3d").Poly3DCollection
    report = read_json(store, refinement, "physrealx.video-articulation-refinement/2")
    source = read_json(
        store,
        ArtifactRef.model_validate(report["source_geometry"]),
        "physrealx.registered-reconstruction-geometry/4",
    )
    before = {
        name: load_meshes(store, ArtifactRef.model_validate(ref), process=False)[0]
        for name, ref in source["geometries"].items()
    }
    after = {
        name: load_meshes(store, ArtifactRef.model_validate(ref), process=False)[0]
        for name, ref in report["diagnostic_geometries"].items()
    }
    motion0, motion = report["source_motion"], report["optimized_motion"]
    positions = np.linspace(motion["range"][0], motion["range"][1], frames)
    allpoints = []
    for meshes, m in ((before, motion0), (after, motion)):
        for q in positions:
            for name, mesh in meshes.items():
                M = (
                    joint_transform("hinge", m["axis"], m["origin_m"], q)
                    if name == m["child"]
                    else np.eye(4)
                )
                allpoints.append(mesh.vertices @ M[:3, :3].T + M[:3, 3])
    points = np.concatenate(allpoints)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    extent = np.max(np.ptp(points, axis=0)) * 0.55
    query = create_surface_query(
        backend="auto", gpu_devices=(), require_accelerator=False, cpu_workers=4, batch_size=65536
    )
    try:
        distances = {
            name: query.nearest(before[name], mesh.triangles_center)[0] * 1000
            for name, mesh in after.items()
        }
    finally:
        query.close()
    norm = Normalize(
        0,
        max(
            report["config"]["max_surface_displacement_m"] * 1000,
            max(float(values.max()) for values in distances.values()),
        ),
    )
    figure = Figure(figsize=(13.5, 4.5), dpi=100)
    canvas: Any = FigureCanvasAgg(figure)
    axes = [figure.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    artists = []
    heatmap = {}
    for index, (ax, meshes) in enumerate(zip(axes, [before, after, after], strict=True)):
        for name, mesh in meshes.items():
            colors = (
                np.asarray(mesh.visual.vertex_colors)[mesh.faces, :3].mean(axis=1) / 255
                if index < 2
                else colormaps["magma"](norm(distances[name]))[:, :3]
            )
            poly = Poly3DCollection(
                mesh.triangles, facecolors=colors, edgecolors="none", linewidths=0
            )
            ax.add_collection3d(poly)
            artists.append((poly, name, mesh, motion0 if index == 0 else motion))
            if index == 2:
                heatmap[name] = poly, colors
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, strict=True):
            setter(c - extent, c + extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-68)
        ax.set_axis_off()
        ax.set_title(
            [
                "Module 1 · original RGB surface",
                "Module 2 · candidate RGB surface",
                "Distance to source surface · mm\nRed = intersecting / touching triangles",
            ][index],
            fontsize=11,
        )
    from matplotlib.cm import ScalarMappable

    figure.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=axes[2], shrink=0.65, pad=-0.08)
    title = figure.suptitle("")
    contact_note = figure.text(0.5, 0.025, "", ha="center", fontsize=9)
    max_shift = max(row["max_displacement_m"] for row in report["links"].values()) * 1000
    shift_limit = report["config"]["max_surface_displacement_m"] * 1000
    image_status = "PASSED" if report["heldout_images"]["passed"] else "FAILED"
    figure.subplots_adjust(left=0, right=1, bottom=0.09, top=0.86, wspace=0)

    def render(q: float) -> tuple[Any, int, Any]:
        for poly, name, mesh, m in artists:
            M = (
                joint_transform("hinge", m["axis"], m["origin_m"], q)
                if name == m["child"]
                else np.eye(4)
            )
            poly.set_verts(mesh.triangles @ M[:3, :3].T + M[:3, 3])
        moved = after[motion["child"]].copy()
        moved.apply_transform(joint_transform("hinge", motion["axis"], motion["origin_m"], q))
        parent_faces, child_faces, _, gaps = triangle_pairs(
            after[motion["parent"]], moved, margin=1e-9
        )
        crossing = gaps <= 0
        for name, faces in (
            (motion["parent"], parent_faces[crossing]),
            (motion["child"], child_faces[crossing]),
        ):
            poly, colors = heatmap[name]
            marked = colors.copy()
            marked[np.unique(faces)] = [1.0, 0.08, 0.03]
            poly.set_facecolors(marked)
        contact_note.set_text(
            f"Triangle pairs intersecting / touching at this pose: {crossing.sum():,} "
            "· full-range clearance is checked separately\n"
            f"Maximum source vertex displacement: {max_shift:.2f} mm / limit {shift_limit:g} mm "
            f"· held-out RGB / mask / depth images: {image_status}"
        )
        title.set_text(
            f"Asset validation: {report['status']} · "
            f"full-range gap: {report['swept_surface_clearance']['status']} | "
            f"q={np.degrees(q):.1f}° | original start/end preserved"
        )
        canvas.draw()
        contact_center = np.asarray(motion["origin_m"])
        if np.any(crossing):
            witness = int(np.argmin(gaps))
            contact_center = (
                after[motion["parent"]].triangles[parent_faces[witness]].mean(axis=0)
                + moved.triangles[child_faces[witness]].mean(axis=0)
            ) / 2
        return (
            Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy()),
            int(crossing.sum()),
            contact_center,
        )

    images = []
    detail_count, detail_q, detail_center = -1, float(positions[0]), center
    for q in positions:
        frame, count, contact_center = render(float(q))
        images.append(frame)
        if count > detail_count:
            detail_count, detail_q, detail_center = count, float(q), contact_center

    stream = io.BytesIO()
    images[0].save(
        stream,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        disposal=2,
    )
    animation = store.put_bytes(stream.getvalue(), media_type="image/gif")
    stream = io.BytesIO()
    images[len(images) // 2].save(stream, format="PNG")
    comparison = store.put_bytes(stream.getvalue(), media_type="image/png")
    radius = report["config"]["hinge_band_m"] * 0.5
    for ax in axes:
        for setter, value in zip(
            (ax.set_xlim, ax.set_ylim, ax.set_zlim), detail_center, strict=True
        ):
            setter(value - radius, value + radius)
    detail, _, _ = render(detail_q)
    detail.close()
    title.set_text(
        f"Local contact detail · {report['status']} | q={np.degrees(detail_q):.1f}° | "
        f"{detail_count:,} intersecting / touching triangle pairs at this pose"
    )
    canvas.draw()
    detail_stream = io.BytesIO()
    figure.savefig(detail_stream, format="png")
    contact_detail = store.put_bytes(detail_stream.getvalue(), media_type="image/png")
    for im in images:
        im.close()
    figure.clear()
    figure = Figure(figsize=(12, 7), dpi=130, layout="constrained")
    ax = figure.subplots(2, 3).ravel()
    h = report["optimizer"].get("history", [])
    if h:
        ax[0].plot(
            [r["iteration"] for r in h],
            [r["maximum_constraint_violation_m"] * 1000 for r in h],
            label="Deficit to clearance + optimization reserve",
        )
        ax[0].axhline(0, color="green", linestyle="--", label="Feasible active constraints")
        ax[0].set(title="Optimization · active target deficit", xlabel="Iteration", ylabel="mm")
    else:
        links = report["optimizer"]["links"]
        ax[0].bar(list(links), [row["max_displacement_m"] * 1000 for row in links.values()])
        ax[0].axhline(
            report["config"]["max_surface_displacement_m"] * 1000,
            color="red",
            linestyle="--",
            label="Allowed displacement",
        )
        ax[0].set(title="Change from immutable source · maximum", ylabel="mm")
    ax[0].legend(fontsize=7)
    names = list(report["heldout"])
    x = np.arange(len(names))
    for shift, label, key in ((-0.18, "Original", "before"), (0.18, "Candidate", "candidate")):
        ax[1].bar(
            x + shift,
            [report["heldout"][n][key]["p90_m"] * 1000 for n in names],
            width=0.35,
            label=label,
        )
    ax[1].set(xticks=x, xticklabels=names, title="Held-out video fit · P90", ylabel="mm")
    ax[1].legend()
    rows = report["swept_surface_clearance"]["intervals"]
    ax[2].plot(
        [np.degrees(r["q"]) for r in rows],
        [r["lower_bound_m"] * 1000 for r in rows],
        label="Conservative lower bound",
    )
    ax[2].plot(
        [np.degrees(r["q"]) for r in rows],
        [r["upper_bound_m"] * 1000 for r in rows],
        label="Witness upper bound",
    )
    ax[2].axhline(
        report["config"]["minimum_clearance_m"] * 1000,
        color="red",
        linestyle="--",
        label="Required gap",
    )
    ax[2].set(title="Full-range surface clearance", xlabel="Joint angle (deg)", ylabel="mm")
    ax[2].legend(fontsize=7)
    images = report["heldout_images"]["frames"]
    for chart, (key, chart_title, unit, multiplier) in zip(
        ax[3:],
        (
            ("mask_iou", "Held-out silhouette overlap", "IoU", 1),
            ("rgb_mae", "Held-out RGB error", "MAE [0, 1]", 1),
            ("depth_p90_m", "Held-out rendered depth · P90", "mm", 1000),
        ),
        strict=True,
    ):
        for label, field in (("Original", "before"), ("Candidate", "candidate")):
            chart.plot(
                [
                    multiplier * r[field][key] if r[field][key] is not None else np.nan
                    for r in images
                ],
                ".-",
                markersize=3,
                label=label,
            )
        failed = [i for i, r in enumerate(images) if not r["passed"]]
        for i in failed:
            chart.axvspan(i - 0.35, i + 0.35, color="red", alpha=0.08)
        chart.set(title=chart_title, xlabel="Held-out view index · red = failed view", ylabel=unit)
        chart.legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.2)
    stream = io.BytesIO()
    figure.savefig(stream, format="png")
    figure.clear()
    return {
        "comparison": comparison,
        "contact_detail": contact_detail,
        "animation": animation,
        "metrics": store.put_bytes(stream.getvalue(), media_type="image/png"),
    }
