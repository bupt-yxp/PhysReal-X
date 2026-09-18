# mypy: ignore-errors
"""Blender-side RGB point-cloud renderer launched in a background process."""

from __future__ import annotations

import json
import os
import sys

import bpy
from mathutils import Vector


def _look_at(item, target) -> None:
    item.rotation_euler = (Vector(target) - item.location).to_track_quat("-Z", "Y").to_euler()


def _material():
    material = bpy.data.materials.new("Point RGB")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    material.node_tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    with open(arguments[0], encoding="utf-8") as stream:
        config = json.load(stream)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = config["width"]
    scene.render.resolution_y = config["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    # PLY vertex colours already contain display-referred RGB sampled from the
    # Blender beauty pass.  A neutral look preserves those byte values.
    scene.view_settings.look = "None"
    scene.display.shading.light = "STUDIO"
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.96, 0.97, 0.98, 1.0)
    background.inputs["Strength"].default_value = 0.8

    bpy.ops.object.camera_add(location=tuple(config["camera"]))
    camera = bpy.context.object
    camera.data.lens = 52
    camera.data.sensor_width = 36
    camera.data.clip_end = 100
    _look_at(camera, tuple(config["target"]))
    scene.camera = camera
    material = _material()
    for frame_index, source in enumerate(config["frames"]):
        bpy.ops.import_mesh.ply(filepath=source)
        item = bpy.context.object
        item.data.materials.append(material)
        scene.render.filepath = os.path.join(config["output_dir"], f"{frame_index:03d}.png")
        bpy.ops.render.render(write_still=True)
        mesh = item.data
        bpy.data.objects.remove(item, do_unlink=True)
        bpy.data.meshes.remove(mesh)


if __name__ == "__main__":
    main()
