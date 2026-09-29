# mypy: ignore-errors
"""Blender-side PBR renderer for synchronized overview and sensor passes."""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def _material(name, color, roughness, metallic=0.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return material


def _look_at(item, target):
    item.rotation_euler = (Vector(target) - item.location).to_track_quat("-Z", "Y").to_euler()


def _box(name, dimensions, location, material, pass_index):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    item = bpy.context.object
    item.name = name
    item.dimensions = dimensions
    item.pass_index = pass_index
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    item.data.materials.append(material)


def _camera(name):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = name
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    return camera


def _resolution(scene, width, height):
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100


def _file_output(scene, socket, directory, prefix):
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    output = nodes.new("CompositorNodeOutputFile")
    output.base_path = directory
    output.file_slots[0].path = prefix
    output.format.file_format = "OPEN_EXR"
    output.format.color_depth = "32"
    output.format.color_mode = "RGB"
    output.format.exr_codec = "ZIP"
    scene.node_tree.links.new(layers.outputs[socket], output.inputs[0])


def _read_pass(directory, prefix):
    matches = glob.glob(os.path.join(directory, prefix + "*.exr"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix} EXR pass, received {matches!r}")
    image = bpy.data.images.load(matches[0], check_existing=False)
    try:
        values = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(values)
        values = values.reshape(image.size[1], image.size[0], image.channels)
        return values[::-1, :, 0].copy()
    finally:
        bpy.data.images.remove(image)
        os.unlink(matches[0])


def _entity_link(name):
    base, separator, suffix = name.rpartition(".")
    if not separator or not suffix.isdigit():
        base = name
    parts = base.split("__")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _configure_scene(config):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 32
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 2.0
    scene.eevee.gtao_factor = 0.85
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.78, 0.79, 0.76, 1.0)
    background.inputs["Strength"].default_value = 0.30

    colours = config["fallback_colours"]
    materials = {
        "floor": _material("Floor", tuple(colours["floor"]), 0.78),
        "wall": _material("Wall", tuple(colours["wall"]), 0.90),
        "table": _material("Table fallback", tuple(colours["table"]), 0.66),
        "chair_left": _material("Left chair fallback", tuple(colours["chair_left"]), 0.68),
        "chair_right": _material("Right chair fallback", tuple(colours["chair_right"]), 0.68),
        "laptop": _material("Laptop fallback", tuple(colours["laptop"]), 0.28, 0.34),
        "lamp": _material("Lamp fallback", tuple(colours["lamp"]), 0.52),
        "bottle": _material("Bottle fallback", tuple(colours["bottle"]), 0.28, 0.06),
        "robot_arm": _material("Robot fallback", tuple(colours["robot_arm"]), 0.42, 0.08),
        "other": _material("Object fallback", tuple(colours["other"]), 0.58),
    }
    room = config.get("room_dimensions")
    if room is not None:
        width, depth, height = (float(value) for value in room)
        thickness = 0.04
        room_index = int(config["room_label_index"])
        _box(
            "Floor",
            (width, depth, thickness),
            (0, 0, -thickness / 2),
            materials["floor"],
            room_index,
        )
        _box(
            "Back wall",
            (width, thickness, height),
            (0, depth / 2, height / 2),
            materials["wall"],
            room_index,
        )
        _box(
            "Left wall",
            (thickness, depth, height),
            (-width / 2, 0, height / 2),
            materials["wall"],
            room_index,
        )
    for name, location, energy, size in (
        ("Key", (-2.7, -3.0, 4.0), 320, 4.5),
        ("Fill", (3.0, -2.0, 2.8), 220, 4.0),
        ("Room", (0.0, 1.0, 3.6), 260, 4.0),
    ):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        _look_at(light, (0.0, 0.0, 0.55))
    return scene, materials


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    with open(arguments[0], encoding="utf-8") as stream:
        config = json.load(stream)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene, fallback_materials = _configure_scene(config)
    overview_config = config["overview"]
    overview = _camera("Overview camera")
    overview.location = tuple(overview_config["position"])
    overview.data.lens = 52
    overview.data.sensor_width = 36
    _look_at(overview, tuple(overview_config["target"]))
    sensor_config = config.get("sensor")
    sensor = None
    if sensor_config is not None:
        sensor = _camera(sensor_config["name"] + " sensor")
        sensor.data.sensor_fit = "VERTICAL"
        sensor.data.angle_y = math.radians(sensor_config["fovy_degrees"])

    output_root = config["output_dir"]
    for directory in ("overview", "rgb", "depth", "mask", "passes"):
        os.makedirs(os.path.join(output_root, directory), exist_ok=True)
    labels = config["labels"]
    # trimesh exports Z-up scenes through glTF's Y-up convention.  Blender's
    # importer converts mesh coordinates.  Compose its inverse after every
    # recorded world transform so that one imported PBR scene can be reused.
    axis_correction = Matrix.Rotation(-math.pi / 2, 4, "X")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=config["scene"])
    imported = [item for item in bpy.data.objects if item not in before]
    visual_items = []
    for item in imported:
        if item.type != "MESH":
            continue
        parsed = _entity_link(item.name)
        if parsed is None:
            continue
        entity, link = parsed
        item.pass_index = labels[f"{entity}/{link}"]
        if not item.data.materials:
            item.data.materials.append(fallback_materials.get(entity, fallback_materials["other"]))
        for material in item.data.materials:
            if material is not None:
                material.use_backface_culling = False
        visual_items.append((item, entity, link))

    for frame_index, state in enumerate(config["states"]):
        for item, entity, link in visual_items:
            item.matrix_world = Matrix(state[entity][link]) @ axis_correction

        scene.use_nodes = False
        scene.render.engine = "BLENDER_EEVEE"
        scene.camera = overview
        _resolution(scene, overview_config["width"], overview_config["height"])
        scene.render.filepath = os.path.join(output_root, "overview", f"{frame_index:03d}.png")
        bpy.ops.render.render(write_still=True)

        if sensor is not None:
            position = sensor_config["positions"][frame_index]
            x, y, z, w = sensor_config["rotations_xyzw"][frame_index]
            sensor.matrix_world = Quaternion((w, x, y, z)).to_matrix().to_4x4()
            sensor.matrix_world.translation = Vector(position)
            scene.camera = sensor
            _resolution(scene, sensor_config["width"], sensor_config["height"])
            pass_directory = os.path.join(output_root, "passes")
            scene.view_layers[0].use_pass_z = True
            _file_output(scene, "Depth", pass_directory, "depth_")
            scene.render.filepath = os.path.join(output_root, "rgb", f"{frame_index:03d}.png")
            bpy.ops.render.render(write_still=True)
            depth = _read_pass(pass_directory, "depth_")
            depth[(depth <= 0) | (depth > sensor_config["max_depth_m"])] = 0
            np.save(
                os.path.join(output_root, "depth", f"{frame_index:03d}.npy"),
                depth.astype(np.float32),
            )

            scene.render.engine = "CYCLES"
            scene.cycles.samples = 1
            scene.cycles.use_denoising = False
            scene.view_layers[0].use_pass_object_index = True
            _file_output(scene, "IndexOB", pass_directory, "index_")
            bpy.ops.render.render(write_still=False)
            mask = np.rint(_read_pass(pass_directory, "index_")).astype(np.int32)
            mask[depth <= 0] = 0
            np.save(os.path.join(output_root, "mask", f"{frame_index:03d}.npy"), mask)


if __name__ == "__main__":
    main()
