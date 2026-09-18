"""One Blender PBR scene shared by task and synchronized sensor previews."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import AssetSpec, GeometryRole, SceneEntityRole, SceneSpec, VersionedRef
from physrealx.errors import DomainValidationError, MethodExecutionError, MethodUnavailableError
from physrealx.integrations._artifacts import load_resource
from physrealx.integrations._asset_preview import asset_geometry

OVERVIEW_CAMERA = (-2.6, -4.7, 2.35)
OVERVIEW_TARGET = (0.0, 0.0, 0.55)
FALLBACK_PBR_COLOURS = {
    "floor": (0.40, 0.22, 0.10),
    "wall": (0.82, 0.83, 0.80),
    "table": (0.31, 0.13, 0.045),
    "chair_left": (0.53, 0.25, 0.075),
    "chair_right": (0.53, 0.25, 0.075),
    "laptop": (0.055, 0.07, 0.095),
    "lamp": (0.12, 0.37, 0.22),
    "bottle": (0.08, 0.30, 0.55),
    "robot_arm": (0.68, 0.70, 0.72),
    "other": (0.38, 0.48, 0.58),
}


@dataclass(frozen=True)
class BlenderSensorCamera:
    """One calibrated moving camera rendered from the shared PBR scene."""

    name: str
    width: int
    height: int
    fovy_degrees: float
    max_depth_m: float
    positions: tuple[tuple[float, float, float], ...]
    rotations_xyzw: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class BlenderSceneFrames:
    """In-memory synchronized outputs produced by one Blender scene."""

    overview: tuple[Any, ...]
    rgb: tuple[Any, ...] = ()
    depth: tuple[Any, ...] = ()
    mask: tuple[Any, ...] = ()
    labels: dict[int, str] | None = None


def _validate_states(states: list[dict[str, dict[str, Any]]]) -> None:
    import numpy as np

    if not 2 <= len(states) <= 60:
        raise DomainValidationError("Blender scene rendering requires between 2 and 60 states")
    for state in states:
        if not state:
            raise DomainValidationError("Blender scene rendering received an empty state")
        for links in state.values():
            for transform in links.values():
                matrix = np.asarray(transform)
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise DomainValidationError(
                        "Blender scene transforms must be finite 4x4 arrays"
                    )


def _validate_camera(camera: BlenderSensorCamera | None, count: int) -> None:
    import math

    if camera is None:
        return
    if (
        not camera.name
        or not 16 <= camera.width <= 1920
        or not 16 <= camera.height <= 1080
        or not 0 < camera.fovy_degrees < 180
        or not math.isfinite(camera.max_depth_m)
        or camera.max_depth_m <= 0
        or len(camera.positions) != count
        or len(camera.rotations_xyzw) != count
    ):
        raise DomainValidationError("Blender sensor camera configuration is inconsistent")


def render_blender_scene_frames(
    store: ArtifactStore,
    scene_ref: VersionedRef,
    states: list[dict[str, dict[str, Any]]],
    *,
    sensor: BlenderSensorCamera | None = None,
) -> BlenderSceneFrames:
    """Render PBR overview and calibrated sensor passes from identical scene states."""
    import numpy as np
    import trimesh
    from PIL import Image

    _validate_states(states)
    _validate_camera(sensor, len(states))
    blender = shutil.which("blender")
    if blender is None:
        raise MethodUnavailableError("PBR scene rendering requires the Blender executable")
    scene = load_resource(store, scene_ref, SceneSpec)
    assets = {
        entity.name: load_resource(store, entity.asset, AssetSpec) for entity in scene.entities
    }
    geometry = {
        name: asset_geometry(store, asset, roles=(GeometryRole.VISUAL,))[GeometryRole.VISUAL]
        for name, asset in assets.items()
    }
    environment = next(
        (entity for entity in scene.entities if entity.role is SceneEntityRole.ENVIRONMENT), None
    )
    room_dimensions = None
    if environment is not None:
        dimensions = assets[environment.name].metadata.get("dimensions_m")
        if isinstance(dimensions, list) and len(dimensions) == 3:
            numeric_dimensions: list[float] = []
            for value in dimensions:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    break
                numeric_dimensions.append(float(value))
            if len(numeric_dimensions) == 3:
                room_dimensions = numeric_dimensions
    procedural_environment = environment.name if environment and room_dimensions else None
    expected = {
        entity.name: {link for link, _ in geometry[entity.name]}
        for entity in scene.entities
        if entity.name != procedural_environment
    }
    for state in states:
        for entity_name, links in expected.items():
            if entity_name not in state or not links <= set(state[entity_name]):
                raise DomainValidationError(
                    f"Blender scene state is missing visual links for {entity_name!r}"
                )

    label_names = []
    if environment is not None and procedural_environment is not None:
        label_names.append(f"{environment.name}/{assets[environment.name].root_link}")
    for entity in scene.entities:
        if entity.name == procedural_environment:
            continue
        label_names.extend(f"{entity.name}/{link}" for link in sorted(expected[entity.name]))
    labels = {index + 1: name for index, name in enumerate(dict.fromkeys(label_names))}
    label_indices = {name: index for index, name in labels.items()}

    renderer = Path(__file__).with_name("_blender_multimodal_renderer.py")
    with tempfile.TemporaryDirectory(prefix="physrealx-blender-scene-") as temporary:
        root = Path(temporary)
        output = root / "rendered"
        output.mkdir()
        rendered = trimesh.Scene()
        for scene_entity in scene.entities:
            if scene_entity.name == procedural_environment:
                continue
            for geometry_index, (link, mesh) in enumerate(geometry[scene_entity.name]):
                name = f"{scene_entity.name}__{link}__{geometry_index}"
                rendered.add_geometry(mesh.copy(), node_name=name, geom_name=name)
        source = root / "scene.glb"
        payload = rendered.export(file_type="glb")  # type: ignore[no-untyped-call]
        source.write_bytes(payload if isinstance(payload, bytes) else payload.encode())

        configuration: dict[str, Any] = {
            "scene": str(source),
            "states": [
                {
                    entity_name: {
                        link: np.asarray(transform, dtype=float).tolist()
                        for link, transform in links.items()
                    }
                    for entity_name, links in state.items()
                }
                for state in states
            ],
            "output_dir": str(output),
            "fallback_colours": FALLBACK_PBR_COLOURS,
            "labels": label_indices,
            "room_dimensions": room_dimensions,
            "room_label_index": (
                label_indices[f"{environment.name}/{assets[environment.name].root_link}"]
                if environment is not None and procedural_environment is not None
                else 0
            ),
            "overview": {
                "width": 720,
                "height": 480,
                "position": OVERVIEW_CAMERA,
                "target": OVERVIEW_TARGET,
            },
        }
        if sensor is not None:
            configuration["sensor"] = {
                "name": sensor.name,
                "width": sensor.width,
                "height": sensor.height,
                "fovy_degrees": sensor.fovy_degrees,
                "max_depth_m": sensor.max_depth_m,
                "positions": sensor.positions,
                "rotations_xyzw": sensor.rotations_xyzw,
            }
        config_path = root / "render.json"
        config_path.write_text(json_dumps(configuration))
        process_environment = os.environ.copy()
        process_environment["SDL_AUDIODRIVER"] = "dummy"
        result = subprocess.run(
            [blender, "-b", "--factory-startup", "-P", str(renderer), "--", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(180, len(states) * (45 if sensor is not None else 20)),
            env=process_environment,
            check=False,
        )
        overview_paths = sorted((output / "overview").glob("*.png"))
        if len(overview_paths) != len(states):
            tail = "\n".join(result.stdout.splitlines()[-30:])
            raise MethodExecutionError(f"Blender PBR scene rendering failed:\n{tail}")

        def images(paths: list[Path]) -> tuple[Any, ...]:
            rendered_images = []
            for path in paths:
                with Image.open(path) as image:
                    rendered_images.append(image.convert("RGB").copy())
            return tuple(rendered_images)

        overview = images(overview_paths)
        if sensor is None:
            return BlenderSceneFrames(overview=overview)
        rgb_paths = sorted((output / "rgb").glob("*.png"))
        depth_paths = sorted((output / "depth").glob("*.npy"))
        mask_paths = sorted((output / "mask").glob("*.npy"))
        if not all(len(paths) == len(states) for paths in (rgb_paths, depth_paths, mask_paths)):
            tail = "\n".join(result.stdout.splitlines()[-30:])
            raise MethodExecutionError(f"Blender PBR sensor rendering failed:\n{tail}")
        rgbs = images(rgb_paths)
        depths = tuple(np.load(path, allow_pickle=False) for path in depth_paths)
        masks = tuple(np.load(path, allow_pickle=False) for path in mask_paths)
        shape = (sensor.height, sensor.width)
        if any(
            array.shape != shape
            or array.dtype != np.float32
            or not np.isfinite(array).all()
            or np.any(array < 0)
            for array in depths
        ):
            raise MethodExecutionError("Blender PBR depth output violates its camera contract")
        valid_labels = {0, *labels}
        if any(
            array.shape != shape
            or array.dtype != np.int32
            or not set(np.unique(array)) <= valid_labels
            for array in masks
        ):
            raise MethodExecutionError("Blender PBR mask output violates its camera contract")
        if any(
            np.asarray(image).shape != (*shape, 3) or np.asarray(image).dtype != np.uint8
            for image in rgbs
        ):
            raise MethodExecutionError("Blender PBR RGB output violates its camera contract")
        if any(np.any(mask[depth <= 0]) for depth, mask in zip(depths, masks, strict=True)):
            raise MethodExecutionError("Blender PBR mask is not aligned to valid metric depth")
        return BlenderSceneFrames(
            overview=overview,
            rgb=rgbs,
            depth=depths,
            mask=masks,
            labels=labels,
        )


def json_dumps(value: Any) -> str:
    """Emit stable compact JSON without importing project modules inside Blender."""
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))
