"""Render one synchronized visual/collision preview for an assembled physical asset."""

from __future__ import annotations

import io
import json
from importlib import import_module
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, AssetSpec, GeometryRole
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations._asset_preview import asset_geometry, link_transforms


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
                    base = (
                        visual_colors[link]
                        if role is GeometryRole.VISUAL
                        else collision_palette[mesh_index % len(collision_palette)]
                    )
                    face_colors.append(brightness[:, None] * np.asarray(base))
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
