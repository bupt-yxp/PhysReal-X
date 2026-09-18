"""Render synchronized reconstructed/reference geometry without retraining or simulation."""

from __future__ import annotations

import io
import json
import zipfile
from importlib import import_module
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.collision import load_meshes
from physrealx.integrations.reconstruction_evaluation import (
    ReferenceEvaluationPose,
    _prediction,
)
from physrealx.integrations.usd_reference import reference_link_transforms
from physrealx.reconstruction import joint_transform


def _appearance_frames(
    store: ArtifactStore, reconstruction: dict[str, Any]
) -> tuple[list[Any], list[float], list[str]]:
    """Load and tightly crop optional appearance frames published by a method."""
    import numpy as np
    from PIL import Image

    if not reconstruction.get("bundle"):
        return [], [], []
    bundle = ArtifactRef.model_validate(reconstruction["bundle"])
    images: list[Any] = []
    with zipfile.ZipFile(io.BytesIO(store.get_bytes(bundle))) as archive:
        try:
            manifest = json.loads(archive.read("bundle-manifest.json"))
        except (KeyError, json.JSONDecodeError) as error:
            raise DomainValidationError("reconstruction bundle manifest is invalid") from error
        records = manifest.get("files")
        if not isinstance(records, list):
            raise DomainValidationError("reconstruction bundle file inventory is invalid")
        paths = sorted(
            record["path"]
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and record["path"].startswith("appearance/")
            and record.get("media_type") in {"image/png", "image/jpeg"}
        )
        if not paths:
            return [], [], []
        if len(paths) > 120 or any(path not in archive.namelist() for path in paths):
            raise DomainValidationError("reconstruction appearance inventory is invalid")
        for path in paths:
            try:
                with Image.open(io.BytesIO(archive.read(path))) as image:
                    images.append(image.convert("RGB"))
            except (OSError, ValueError) as error:
                raise DomainValidationError(
                    f"reconstruction appearance frame is invalid: {path}"
                ) from error
        states = np.linspace(0, 1, len(images)).tolist()
        if "appearance/states.json" in archive.namelist():
            try:
                declared = json.loads(archive.read("appearance/states.json"))["states"]
                states = [float(value) for value in declared]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise DomainValidationError(
                    "reconstruction appearance states are invalid"
                ) from error
    if len({image.size for image in images}) != 1:
        for image in images:
            image.close()
        raise DomainValidationError("reconstruction appearance frames have inconsistent sizes")
    if (
        len(states) != len(images)
        or not np.isfinite(states).all()
        or any(not 0 <= value <= 1 for value in states)
        or any(left >= right for left, right in zip(states, states[1:]))
    ):
        for image in images:
            image.close()
        raise DomainValidationError("reconstruction appearance states are invalid")

    arrays = [np.asarray(image) for image in images]
    foreground = np.zeros(arrays[0].shape[:2], dtype=bool)
    for array in arrays:
        border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
        background = np.median(border, axis=0)
        foreground |= np.max(np.abs(array.astype(float) - background), axis=2) > 10
    if foreground.any():
        y, x = np.where(foreground)
        padding = max(2, round(max(np.ptp(x), np.ptp(y)) * 0.06))
        box = (
            max(0, int(x.min()) - padding),
            max(0, int(y.min()) - padding),
            min(images[0].width, int(x.max()) + padding + 1),
            min(images[0].height, int(y.max()) + padding + 1),
        )
        cropped = [image.crop(box) for image in images]
        for image in images:
            image.close()
        images = cropped
    return images, states, paths


def preview_reconstruction(
    store: ArtifactStore, evaluation: ArtifactRef, *, frames: int = 24, fps: int = 12
) -> ArtifactRef:
    """Interpolate each side's own observed motion; GIF time is playback, not capture time.

    Full meshes are rendered with group colors. No per-side rescaling, geometry
    decimation or use of GT motion to animate predicted parts.
    """
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
    report = read_json(store, evaluation, "physrealx.reconstruction-evaluation-report/2")
    source = report["source"]
    reconstruction = read_json(
        store,
        ArtifactRef.model_validate(source["reconstruction_report"]),
        "physrealx.geometry-reconstruction-report/1",
    )
    appearance_images, appearance_states, appearance_paths = _appearance_frames(
        store, reconstruction
    )
    reference = read_json(
        store, ArtifactRef.model_validate(source["reference_model"]), "physrealx.usd-reference/1"
    )
    pose = ReferenceEvaluationPose.model_validate(report["reference_pose"])
    pair = tuple(report["state_pair"])
    predicted, joints, _, _, segmented = _prediction(store, reconstruction, (pair[0], pair[1]))
    if not segmented or reconstruction["prediction"].get("motion_pair") != list(pair):
        raise DomainValidationError(
            "continuous preview requires a declared articulated motion pair"
        )
    rotation, translation = np.asarray(pose.rotation), np.asarray(pose.translation_m)
    if (
        reconstruction.get("unit") != "m"
        or reference.get("unit") != "m"
        or pose.source_frame != reconstruction["coordinate_frame"]
        or pose.target_frame != reference["coordinate_frame"]
        or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)
    ):
        raise DomainValidationError("preview requires the evaluation's metric rigid registration")
    gt_meshes = {
        row["name"]: trimesh.util.concatenate(
            load_meshes(store, ArtifactRef.model_validate(row["geometry"]), process=False)
        )
        for row in reference["links"]
    }
    registration = np.eye(4)
    registration[:3, :3], registration[:3, 3] = rotation, translation

    def transforms(fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
        recovered: dict[str, Any] = {}
        remaining = dict(joints)
        while remaining:
            ready = [
                name
                for name, joint in remaining.items()
                if joint["parent"] is None or joint["parent"] in recovered
            ]
            if not ready:
                raise DomainValidationError("predicted joint graph is disconnected or cyclic")
            for name in ready:
                joint = remaining.pop(name)
                recovered[name] = recovered.get(joint["parent"], np.eye(4)) @ joint_transform(
                    joint["kind"], joint["axis"], joint["origin"], joint["span"] * fraction
                )
        positions = {
            name: (1 - fraction) * start + fraction * pose.end_positions[name]
            for name, start in pose.start_positions.items()
        }
        return (
            {name: registration @ matrix for name, matrix in recovered.items()},
            reference_link_transforms(reference, positions),
        )

    timeline = np.linspace(0, 1, frames)
    matrices = [transforms(float(fraction)) for fraction in timeline]
    # One camera extent covers both geometries over the entire animation.
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for state in matrices:
        for meshes, transforms_at_state in zip((predicted, gt_meshes), state):
            for name, mesh in meshes.items():
                vertices = trimesh.transform_points(mesh.vertices, transforms_at_state[name])
                lower = np.minimum(lower, vertices.min(axis=0))
                upper = np.maximum(upper, vertices.max(axis=0))
    if not np.isfinite([lower, upper]).all() or np.max(upper - lower) <= 0:
        raise DomainValidationError("preview requires finite nonempty geometry")
    center = (lower + upper) / 2
    extent = upper - lower
    longest = float(np.max(extent))
    # Preserve one shared metric camera while avoiding the large empty volume
    # introduced by forcing flat or elongated assets into a cube.
    padded_extent = np.maximum(extent, longest * 0.06) * 1.08
    box_aspect = tuple(float(value) for value in padded_extent)
    palette = [(0.65, 0.69, 0.74), (0.16, 0.55, 0.84), (0.96, 0.59, 0.22), (0.4, 0.72, 0.48)]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(sorted(predicted))}
    gt_colors = {
        name: next(
            (colors[group] for group, links in pose.part_mapping.items() if name in links),
            (0.8, 0.45, 0.48),
        )
        for name in gt_meshes
    }
    panel_count = 3 if appearance_images else 2
    figure = Figure(figsize=(panel_count * 4, 3.6), dpi=100, facecolor="white")
    canvas: Any = FigureCanvasAgg(figure)
    appearance_artist = None
    if appearance_images:
        appearance_axis = figure.add_subplot(1, panel_count, 1)
        appearance_axis.set_title("Appearance", fontsize=11, color="white", pad=-14, y=0.98)
        appearance_axis.set_axis_off()
        appearance_axis.set_position((0.022, 0.09, 0.29, 0.82))
        appearance_artist = appearance_axis.imshow(appearance_images[0], interpolation="bilinear")
    collections = []
    panel_margin = 0.008 if appearance_images else 0
    for index, title in enumerate(("Reconstruction", "GT / supplied USD")):
        column = index + bool(appearance_images)
        axis = figure.add_subplot(1, panel_count, column + 1, projection="3d")
        axis.set_title(title, fontsize=11, pad=-2, y=0.99)
        axis.set(
            xlim=(center[0] - padded_extent[0] / 2, center[0] + padded_extent[0] / 2),
            ylim=(center[1] - padded_extent[1] / 2, center[1] + padded_extent[1] / 2),
            zlim=(center[2] - padded_extent[2] / 2, center[2] + padded_extent[2] / 2),
        )
        # Matplotlib normalizes mutable arrays in place; pass an immutable value
        # so both panels keep exactly the same limits and metric scale.
        axis.set_box_aspect(box_aspect, zoom=1.29)
        axis.view_init(elev=20, azim=-40)
        axis.set_axis_off()
        axis.set_position(
            (
                column / panel_count + panel_margin,
                0.045,
                1 / panel_count - 2 * panel_margin,
                0.91,
            )
        )
        # A single collection sorts triangles across parts, including open drawers.
        collection = Poly3DCollection([], edgecolors="none", linewidths=0)
        axis.add_collection3d(collection)
        collections.append(collection)
    images = []
    try:
        for fraction, state in zip(timeline, matrices):
            if appearance_artist is not None:
                appearance_index = int(np.argmin(np.abs(np.asarray(appearance_states) - fraction)))
                appearance_artist.set_data(appearance_images[appearance_index])
            for meshes, color_map, items, transform_at_state in zip(
                (predicted, gt_meshes), (colors, gt_colors), collections, state
            ):
                triangles, face_colors = [], []
                for name, mesh in meshes.items():
                    vertices = trimesh.transform_points(mesh.vertices, transform_at_state[name])
                    triangles.append(vertices[mesh.faces])
                    normals = mesh.face_normals @ transform_at_state[name][:3, :3].T
                    brightness = 0.45 + 0.55 * np.clip(
                        normals @ (np.array([1, -1, 2]) / np.sqrt(6)), 0, 1
                    )
                    face_colors.append(brightness[:, None] * np.asarray(color_map[name]))
                items.set_verts(np.concatenate(triangles))
                items.set_facecolor(np.concatenate(face_colors))
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
                    "evaluation": evaluation.model_dump(mode="json"),
                    "motion": "predicted joint interpolation versus reference state interpolation",
                    "appearance_frames": appearance_paths,
                    "geometry": "group-colored full meshes with shared view and scale",
                    "fps": fps,
                }
            ).encode(),
        )
        return store.put_bytes(output.getvalue(), media_type="image/gif")
    finally:
        figure.clear()
        for image in images:
            image.close()
        for image in appearance_images:
            image.close()
