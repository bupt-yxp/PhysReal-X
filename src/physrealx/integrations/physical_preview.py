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
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([0, 208, 190])).astype(np.uint8)
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

    prediction = read_json(store, prior, "physrealx.visual-physics-prior/3")
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
        initial = read_json(store, initial_feasibility, "physrealx.physical-feasibility-report/1")
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
        "Module-one predicted joint and start/end fixed | "
        f"Video: {video['geometry_status']} | Physics remains PRIOR",
        fontsize=10,
    )
    stream = io.BytesIO()
    figure.savefig(stream, format="png")
    figure.clear()
    return store.put_bytes(stream.getvalue(), media_type="image/png")


def preview_physical_asset(
    store: ArtifactStore,
    assembly: ArtifactRef,
    *,
    frames: int = 24,
    fps: int = 12,
    rejected_refinement: ArtifactRef | None = None,
) -> ArtifactRef:
    """Animate geometry while marking only measured proxy contacts as contacts."""
    import numpy as np
    import trimesh
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from PIL import Image

    from physrealx.integrations._collision_refit import nearest
    from physrealx.integrations.physical_feasibility import (
        FeasibilityConfig,
        evaluate_asset_feasibility,
    )

    Poly3DCollection = import_module("mpl_toolkits.mplot3d.art3d").Poly3DCollection
    proj3d = import_module("mpl_toolkits.mplot3d.proj3d")
    if type(frames) is not int or not 2 <= frames <= 120:
        raise DomainValidationError("preview frames must be an integer between 2 and 120")
    if type(fps) is not int or not 1 <= fps <= 50:
        raise DomainValidationError("preview fps must be an integer between 1 and 50")
    document = read_json(store, assembly, "physrealx.asset-assembly-bundle/1")
    try:
        asset = AssetSpec.model_validate(document["asset_spec"])
    except (KeyError, ValueError) as error:
        raise DomainValidationError("asset assembly has no valid asset specification") from error
    if rejected_refinement is not None:
        report = read_json(store, rejected_refinement, "physrealx.articulated-collision-repair/4")
        if (
            report.get("status") != "FAILED"
            or report.get("source_assembly") != assembly.model_dump(mode="json")
            or set(report.get("diagnostic_collision_geometries", {})) != set(asset.links)
        ):
            raise DomainValidationError(
                "diagnostic preview requires this assembly's rejected refinement"
            )
        replacements = {
            link: ArtifactRef.model_validate(ref)
            for link, ref in report["diagnostic_collision_geometries"].items()
        }
        asset = asset.model_copy(
            update={
                "geometries": tuple(
                    g.model_copy(update={"uri": f"artifact://{replacements[g.link].artifact_id}"})
                    if g.role is GeometryRole.COLLISION
                    else g
                    for g in asset.geometries
                )
            }
        )
    geometry = asset_geometry(store, asset, roles=(GeometryRole.VISUAL, GeometryRole.COLLISION))
    # Proxy colors identify link ownership. They are not collision highlights.
    # Sample MuJoCo at these exact animation states so a small red point, rather
    # than an entire proxy, carries the collision meaning.
    evidence_samples = max(3, frames)
    collision_evidence = evaluate_asset_feasibility(
        store,
        asset,
        config=FeasibilityConfig(sweep_samples=evidence_samples),
    )
    # Use per-face maximum vertex/centroid distance to expose small missing patches.
    fit_colors = []
    color_limit_mm = 3.0
    for link, mesh in geometry[GeometryRole.VISUAL]:
        proxy = trimesh.util.concatenate(
            [p for name, p in geometry[GeometryRole.COLLISION] if name == link]
        )
        vertex_distance, _ = nearest(proxy, mesh.vertices)
        face_distance, _ = nearest(proxy, mesh.triangles_center)
        fit_error = np.maximum(vertex_distance[mesh.faces].max(axis=1), face_distance) * 1000
        fit_colors.append(colormaps["YlOrRd"](np.clip(fit_error / color_limit_mm, 0, 1)))
    if len(asset.joints) != 1:
        raise DomainValidationError("physical preview currently requires one joint")
    joint = asset.joints[0]
    motion = asset.metadata.get("reconstructed_motion")
    reconstructed_playback = (
        isinstance(motion, dict)
        and motion.get("joint") == joint.name
        and motion.get("range") == [float(joint.lower_limit), float(joint.upper_limit)]
        and len(motion.get("positions", ())) == 2
    )
    if not reconstructed_playback:
        motion = {
            "joint": joint.name,
            "positions": [float(joint.lower_limit), float(joint.upper_limit)],
            "states": ["lower_limit", "upper_limit"],
        }
    timeline = np.linspace(0, 1, frames)
    playback_positions = [
        (1 - float(fraction)) * motion["positions"][0]
        + float(fraction) * motion["positions"][1]
        for fraction in timeline
    ]
    states = [
        link_transforms(asset, {motion["joint"]: float(position)})
        for position in playback_positions
    ]
    sweep_positions = np.asarray([row["q"] for row in collision_evidence["sweep"]])
    evidence_indices = [
        int(np.argmin(np.abs(sweep_positions - position))) for position in playback_positions
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

    figure = Figure(figsize=(13.5, 3.6), dpi=100, facecolor="white")
    joint_ranges = {
        joint.name: [float(joint.lower_limit), float(joint.upper_limit)] for joint in asset.joints
    }
    start, end = motion["positions"]
    range_text = (
        f"{joint.name}: {np.degrees(start):.1f}° → {np.degrees(end):.1f}°"
        if joint.joint_type.value == "revolute"
        else f"{joint.name}: {start:.4g} → {end:.4g} m"
    )
    if rejected_refinement is not None:
        figure.suptitle(
            f"REJECTED candidate · diagnostic only · {range_text}",
            fontsize=11,
            color="#b91c1c",
            y=0.995,
        )
    elif reconstructed_playback:
        figure.suptitle(
            f"Module-one reconstructed start → end · {range_text}", fontsize=11, y=0.995
        )
    else:
        figure.suptitle(f"Declared joint limits · {range_text}", fontsize=11, y=0.995)
    canvas: Any = FigureCanvasAgg(figure)
    axes = []
    collections = []
    for index, title in enumerate(
        (
            "Visual geometry · RGB surface",
            "Proxy shapes · link colors are not contact",
            "Proxy ↔ visual surface error · mm",
        )
    ):
        axis = figure.add_subplot(1, 3, index + 1, projection="3d")
        axis.set_title(title, fontsize=11, pad=-2, y=0.99)
        axis.set(
            xlim=(center[0] - padded_extent[0] / 2, center[0] + padded_extent[0] / 2),
            ylim=(center[1] - padded_extent[1] / 2, center[1] + padded_extent[1] / 2),
            zlim=(center[2] - padded_extent[2] / 2, center[2] + padded_extent[2] / 2),
        )
        axis.set_box_aspect(box_aspect, zoom=1.22)
        axis.view_init(elev=20, azim=-40)
        axis.set_axis_off()
        axis.set_position((index / 3 + 0.010, 0.065, 0.313, 0.89))
        collection = Poly3DCollection([], edgecolors="none", linewidths=0)
        axis.add_collection3d(collection)
        axes.append(axis)
        collections.append(collection)

    contact_marker = axes[1].text2D(
        0,
        0,
        "",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="#dc2626",
        fontweight="bold",
        clip_on=True,
    )
    contact_label = axes[1].text2D(
        0.5,
        0.01,
        "",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#991b1b",
    )
    color_axis = figure.add_axes((0.72, 0.085, 0.23, 0.025))
    figure.colorbar(
        ScalarMappable(norm=Normalize(0, color_limit_mm), cmap="YlOrRd"),
        cax=color_axis,
        orientation="horizontal",
        extend="max",
    )
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
    collision_palette = ((0.95, 0.55, 0.18), (0.62, 0.38, 0.72), (0.84, 0.34, 0.24))
    collision_colors = {
        link: collision_palette[index % len(collision_palette)]
        for index, link in enumerate(sorted(asset.links))
    }
    color_names = ("orange", "purple", "brown")
    ownership_legend = " · ".join(
        f"{color_names[index % len(color_names)]} = {link} proxy"
        for index, link in enumerate(sorted(asset.links))
    )
    figure.text(
        0.5,
        0.005,
        f"{ownership_legend} · red point = deepest MuJoCo contact",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4b5563",
    )
    images = []
    try:
        for frame_index, transforms in enumerate(states):
            for panel, (role, collection) in enumerate(
                zip((GeometryRole.VISUAL, GeometryRole.COLLISION, GeometryRole.VISUAL), collections)
            ):
                triangles, face_colors = [], []
                for link, mesh in geometry[role]:
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
                            else collision_colors[link]
                        )
                        base = np.tile(color, (len(mesh.faces), 1))
                    face_colors.append(brightness[:, None] * base)
                collection.set_verts(np.concatenate(triangles))
                collection.set_facecolor(np.concatenate(fit_colors if panel == 2 else face_colors))
            row = collision_evidence["sweep"][int(evidence_indices[frame_index])]
            contact = row["deepest_contact"]
            if contact is None:
                contact_marker.set_text("")
                contact_label.set_text("No penetrating proxy contact")
            else:
                x, y, z = contact["position_m"]
                projected_x, projected_y, _ = proj3d.proj_transform(x, y, z, axes[1].get_proj())
                display_xy = axes[1].transData.transform((projected_x, projected_y))
                axes_xy = axes[1].transAxes.inverted().transform(display_xy)
                contact_marker.set_position(axes_xy)
                contact_marker.set_text("●")
                contact_label.set_text(
                    f"Deepest proxy penetration: {1000 * row['penetration_m']:.3f} mm"
                )
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
                    "rejected_refinement": rejected_refinement.model_dump(mode="json")
                    if rejected_refinement
                    else None,
                    "surface_fit_error_color_limit_mm": color_limit_mm,
                    "contact_annotation": {
                        "backend": "mujoco",
                        "meaning": "red point is the deepest penetrating proxy contact per frame",
                        "proxy_link_colors_are_contact_highlights": False,
                        "max_penetration_m": collision_evidence["max_penetration_m"],
                    },
                    "assembly": assembly.model_dump(mode="json"),
                    "joints": [joint.name for joint in asset.joints],
                    "joint_ranges": joint_ranges,
                    "motion": (
                        "module-one predicted start-to-end"
                        if reconstructed_playback
                        else "declared lower-to-upper joint limits"
                    ),
                    "states": motion["states"],
                    "playback_positions": playback_positions,
                    "fps": fps,
                }
            ).encode(),
        )
        return store.put_bytes(output.getvalue(), media_type="image/gif")
    finally:
        figure.clear()
        for image in images:
            image.close()
