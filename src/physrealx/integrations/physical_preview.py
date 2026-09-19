"""Render one synchronized visual/collision preview for an assembled physical asset."""

from __future__ import annotations

import io
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, AssetSpec, GeometryRole
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations._asset_preview import asset_geometry, link_transforms


def preview_physics_video(
    store: ArtifactStore, directory: Path, *, frames: int = 24, fps: int = 8
) -> ArtifactRef:
    """Animate source RGB, depth and reviewed masks without changing the observations."""
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image, ImageDraw

    if type(frames) is not int or not 2 <= frames <= 60:
        raise DomainValidationError("video preview frames must be an integer between 2 and 60")
    if type(fps) is not int or not 1 <= fps <= 30:
        raise DomainValidationError("video preview fps must be an integer between 1 and 30")
    paths = {
        "rgb": {p.stem: p for p in (directory / "img").glob("*.png")},
        "depth": {p.stem: p for p in (directory / "depth").glob("*.npy")},
        "mask": {p.stem: p for p in (directory / "mask").glob("*.png")},
        "link": {p.stem: p for p in (directory / "link_mask").glob("*.png")},
    }
    names = set(paths["rgb"])
    if not names or any(set(value) != names for value in paths.values()):
        raise DomainValidationError("video preview requires synchronized RGB/depth/mask/link_mask")
    try:
        ordered = sorted(names, key=int)
    except ValueError as error:
        raise DomainValidationError("video frame names must be integer indices") from error
    selected_indices = (
        np.linspace(0, len(ordered) - 1, min(frames, len(ordered))).round().astype(int)
    )
    selected = [ordered[index] for index in np.unique(selected_indices)]
    panel_size = (480, 270)
    palette = np.asarray([[0, 0, 0], [0, 208, 190], [229, 58, 145]], dtype=np.uint8)
    outputs = []
    for name in selected:
        rgb = np.asarray(Image.open(paths["rgb"][name]).convert("RGB"))
        depth = np.load(paths["depth"][name], allow_pickle=False)
        binary_mask = np.asarray(Image.open(paths["mask"][name]))
        mask = binary_mask > 0
        links = np.asarray(Image.open(paths["link"][name]))
        if (
            depth.shape != rgb.shape[:2]
            or mask.shape != depth.shape
            or links.shape != depth.shape
            or not np.isfinite(depth).all()
            or np.any(depth < 0)
            or not set(np.unique(binary_mask)) <= {0, 255}
            or not set(np.unique(links)) <= {0, 1, 2}
            or not np.array_equal(mask, links > 0)
        ):
            raise DomainValidationError("video preview found invalid synchronized observations")
        overlay = rgb.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 208, 190])).astype(
            np.uint8
        )
        valid = depth > 0
        if not np.any(valid):
            depth_rgb = np.zeros_like(rgb)
        else:
            lower, upper = np.quantile(depth[valid], [0.02, 0.98])
            normalized = np.clip((depth.astype(float) - lower) / max(upper - lower, 1e-12), 0, 1)
            depth_rgb = (colormaps["turbo_r"](normalized)[..., :3] * 255).astype(np.uint8)
            depth_rgb[~valid] = 0
        link_rgb = palette[np.clip(links.astype(int), 0, len(palette) - 1)]
        panels = [rgb, overlay, depth_rgb, link_rgb]
        captions = ["RGB", "Foreground mask", "Depth", "Link mask"]
        canvas = Image.new("RGB", (960, 596), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 6), f"Video observations · frame {name}", fill="black")
        for index, (array, caption) in enumerate(zip(panels, captions)):
            image = Image.fromarray(array).resize(panel_size, Image.Resampling.LANCZOS)
            x, y = (index % 2) * 480, 28 + (index // 2) * 284
            canvas.paste(image, (x, y + 14))
            draw.text((x + 8, y), caption, fill="black")
        outputs.append(canvas)
    stream = io.BytesIO()
    try:
        outputs[0].save(
            stream,
            format="GIF",
            save_all=True,
            append_images=outputs[1:],
            duration=round(1000 / fps),
            loop=0,
            disposal=2,
            comment=json.dumps(
                {"source_directory": str(directory), "frame_ids": selected, "fps": fps}
            ).encode(),
        )
        return store.put_bytes(stream.getvalue(), media_type="image/gif")
    finally:
        for image in outputs:
            image.close()


def preview_physics_report(
    store: ArtifactStore,
    prior: ArtifactRef,
    feasibility: ArtifactRef,
    readiness: ArtifactRef,
    initial_feasibility: ArtifactRef | None = None,
) -> ArtifactRef:
    """Show uncertainty and measured simulation diagnostics without implying video fitting ran."""
    import numpy as np
    from matplotlib.figure import Figure

    prediction = read_json(store, prior, "physrealx.visual-physics-prior/1")
    report = read_json(store, feasibility, "physrealx.physical-feasibility-report/1")
    video = read_json(store, readiness, "physrealx.video-physics-readiness/1")
    assembly = read_json(
        store,
        ArtifactRef.model_validate(report["source_assembly"]),
        "physrealx.asset-assembly-bundle/1",
    )
    if prior.artifact_id not in {ref["artifact_id"] for ref in assembly["asset_spec"]["evidence"]}:
        raise DomainValidationError("physics preview prior does not belong to audited assembly")
    figure = Figure(figsize=(10, 3.2), dpi=140, layout="constrained")
    left, right = figure.subplots(1, 2)
    bodies = prediction["prediction"]["bodies"]
    masses = [v["mass_kg"] for v in bodies.values()]
    values = np.array([v["value"] for v in masses])
    errors = np.array(
        [[v["value"] - v["lower"] for v in masses], [v["upper"] - v["value"] for v in masses]]
    )
    left.errorbar(list(bodies), values, yerr=errors, fmt="o", capsize=6, color="#2563eb")
    implementation = prediction.get("prior_implementation", "VLM")
    left.set(title=f"VLM mass priors · {implementation.split('/')[-1]}", ylabel="kg")
    left.grid(axis="y", alpha=0.2)
    rows = report["sweep"]
    if initial_feasibility is not None:
        initial = read_json(
            store, initial_feasibility, "physrealx.physical-feasibility-report/1"
        )
        initial_rows = initial["sweep"]
        if initial["locked_range"] != report["locked_range"]:
            raise DomainValidationError("collision comparison changed the locked joint range")
        right.plot(
            [v["q"] for v in initial_rows],
            [1000 * v["penetration_m"] for v in initial_rows],
            color="#dc2626",
            label="initial",
        )
    right.plot(
        [v["q"] for v in rows],
        [1000 * v["penetration_m"] for v in rows],
        color="#059669",
        label="refined" if initial_feasibility is not None else "sweep",
    )
    right.axhline(1000 * report["config"]["penetration_tolerance_m"], linestyle="--", color="gray")
    right.set(
        title=f"Collision sweep: {report['status']}",
        xlabel="Joint position (native unit)",
        ylabel="Penetration (mm)",
    )
    right.legend()
    right.grid(alpha=0.2)
    figure.suptitle(
        f"Fixed start/end | Video: {video['geometry_status']} | Physics remains PRIOR", fontsize=10
    )
    stream = io.BytesIO()
    figure.savefig(stream, format="png")
    figure.clear()
    return store.put_bytes(stream.getvalue(), media_type="image/png")


def preview_physical_asset(
    store: ArtifactStore, assembly: ArtifactRef, *, frames: int = 24, fps: int = 12
) -> ArtifactRef:
    """Animate visual and collision geometry through the asset's declared joint limits."""
    import numpy as np
    import trimesh
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    Poly3DCollection = import_module("mpl_toolkits.mplot3d.art3d").Poly3DCollection
    if type(frames) is not int or not 2 <= frames <= 120:
        raise DomainValidationError("preview frames must be an integer between 2 and 120")
    if type(fps) is not int or not 1 <= fps <= 50:
        raise DomainValidationError("preview fps must be an integer between 1 and 50")
    document = read_json(store, assembly, "physrealx.asset-assembly-bundle/1")
    try:
        asset = AssetSpec.model_validate(document["asset_spec"])
    except (KeyError, ValueError) as error:
        raise DomainValidationError("asset assembly has no valid asset specification") from error
    geometry = asset_geometry(store, asset, roles=(GeometryRole.VISUAL, GeometryRole.COLLISION))
    timeline = np.linspace(0, 1, frames)
    states = [
        link_transforms(
            asset,
            {
                joint.name: joint.lower_limit
                + float(fraction) * (joint.upper_limit - joint.lower_limit)
                for joint in asset.joints
            },
        )
        for fraction in timeline
    ]

    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for transforms in states:
        for meshes in geometry.values():
            for link, mesh in meshes:
                vertices = trimesh.transform_points(mesh.vertices, transforms[link])
                lower = np.minimum(lower, vertices.min(axis=0))
                upper = np.maximum(upper, vertices.max(axis=0))
    if not np.isfinite([lower, upper]).all() or np.max(upper - lower) <= 0:
        raise DomainValidationError("physical asset preview requires finite nonempty geometry")
    center = (lower + upper) / 2
    extent = upper - lower
    longest = float(np.max(extent))
    padded_extent = np.maximum(extent, longest * 0.06) * 1.1
    box_aspect = tuple(float(value) for value in padded_extent)

    figure = Figure(figsize=(9, 3.6), dpi=100, facecolor="white")
    canvas: Any = FigureCanvasAgg(figure)
    collections = []
    for index, title in enumerate(("Visual geometry", "Collision geometry")):
        axis = figure.add_subplot(1, 2, index + 1, projection="3d")
        axis.set_title(title, fontsize=11, pad=-2, y=0.99)
        axis.set(
            xlim=(center[0] - padded_extent[0] / 2, center[0] + padded_extent[0] / 2),
            ylim=(center[1] - padded_extent[1] / 2, center[1] + padded_extent[1] / 2),
            zlim=(center[2] - padded_extent[2] / 2, center[2] + padded_extent[2] / 2),
        )
        axis.set_box_aspect(box_aspect, zoom=1.22)
        axis.view_init(elev=20, azim=-40)
        axis.set_axis_off()
        axis.set_position((index * 0.5 + 0.015, 0.045, 0.47, 0.91))
        collection = Poly3DCollection([], edgecolors="none", linewidths=0)
        axis.add_collection3d(collection)
        collections.append(collection)

    visual_palette = (
        (0.65, 0.69, 0.74),
        (0.16, 0.55, 0.84),
        (0.4, 0.72, 0.48),
        (0.72, 0.52, 0.82),
    )
    visual_colors = {
        link: visual_palette[index % len(visual_palette)]
        for index, link in enumerate(sorted(asset.links))
    }
    collision_palette = ((0.95, 0.55, 0.18), (0.84, 0.34, 0.24), (0.62, 0.38, 0.72))
    images = []
    try:
        for transforms in states:
            for role, collection in zip((GeometryRole.VISUAL, GeometryRole.COLLISION), collections):
                triangles, face_colors = [], []
                for mesh_index, (link, mesh) in enumerate(geometry[role]):
                    transform = transforms[link]
                    vertices = trimesh.transform_points(mesh.vertices, transform)
                    triangles.append(vertices[mesh.faces])
                    normals = mesh.face_normals @ transform[:3, :3].T
                    brightness = 0.45 + 0.55 * np.clip(
                        normals @ (np.array([1, -1, 2]) / np.sqrt(6)), 0, 1
                    )
                    if (
                        role is GeometryRole.VISUAL
                        and getattr(mesh.visual, "kind", None) == "vertex"
                        and len(mesh.visual.vertex_colors) == len(mesh.vertices)
                    ):
                        base = np.asarray(mesh.visual.vertex_colors)[mesh.faces, :3].mean(axis=1)
                        base = base / 255.0
                    else:
                        color = (
                            visual_colors[link]
                            if role is GeometryRole.VISUAL
                            else collision_palette[mesh_index % len(collision_palette)]
                        )
                        base = np.tile(color, (len(mesh.faces), 1))
                    face_colors.append(brightness[:, None] * base)
                collection.set_verts(np.concatenate(triangles))
                collection.set_facecolor(np.concatenate(face_colors))
            canvas.draw()
            images.append(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB"))
        output = io.BytesIO()
        images[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / fps),
            loop=0,
            disposal=2,
            comment=json.dumps(
                {
                    "assembly": assembly.model_dump(mode="json"),
                    "joints": [joint.name for joint in asset.joints],
                    "motion": "declared lower-to-upper joint limits",
                    "fps": fps,
                }
            ).encode(),
        )
        return store.put_bytes(output.getvalue(), media_type="image/gif")
    finally:
        figure.clear()
        for image in images:
            image.close()
