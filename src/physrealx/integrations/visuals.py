"""Portable mesh appearance at the USD, glTF and simulator boundaries.

Authored textures and UVs are preserved. OmniPBR box projection is baked into
face-corner UVs; procedural MDL lobes remain explicitly identified approximations.
"""

from __future__ import annotations

import io
from hashlib import sha256
from typing import Any

from physrealx.errors import DomainValidationError


def png_bytes(image: Any) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def usd_visual_mesh(
    prim: Any, vertices: Any, faces: Any, corners: Any
) -> tuple[Any, dict[str, Any] | None]:
    from importlib import import_module

    import numpy as np
    from PIL import Image
    from pxr import Ar, UsdGeom, UsdShade

    trimesh = import_module("trimesh")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
        raise DomainValidationError("face material subsets must be split before visual import")
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return mesh, None
    shaders = [
        UsdShade.Shader(p)
        for p in material.GetPrim().GetChildren()
        if p.IsA(UsdShade.Shader)
        and (
            UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"
            or UsdShade.Shader(p).GetImplementationSourceAttr().Get() == "sourceAsset"
        )
    ]
    if len(shaders) != 1:
        raise DomainValidationError("visual import requires a single supported surface shader")
    shader = shaders[0]

    def value(name: str, default: Any = None) -> Any:
        result = shader.GetInput(name).Get()
        return default if result is None else result

    shader_id = shader.GetIdAttr().Get()
    mdl = str(shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() or "")
    preview = shader_id == "UsdPreviewSurface"
    is_metal = mdl == "scratched_metal_v2"
    texture_asset = value("diffuse_texture")
    if preview:
        source = shader.GetInput("diffuseColor").GetConnectedSource()
        if source:
            texture_shader = UsdShade.Shader(source[0].GetPrim())
            if texture_shader.GetIdAttr().Get() != "UsdUVTexture":
                raise DomainValidationError("diffuseColor must use a portable USD UV texture")
            texture_asset = texture_shader.GetInput("file").Get()
    if not preview and not is_metal and texture_asset is None:
        raise DomainValidationError(f"unsupported authored material shader: {shader_id or mdl}")

    def texture(asset: Any) -> Any:
        if asset is None:
            return None
        if not asset.resolvedPath:
            raise DomainValidationError("material texture has no resolved local source")
        resolved = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(asset.resolvedPath))
        if resolved is None:
            raise DomainValidationError("material texture cannot be resolved")
        with Image.open(io.BytesIO(bytes(resolved.GetBuffer()))) as image:
            if max(image.size) > 8192:
                raise DomainValidationError("material texture exceeds the 8192-pixel limit")
            return image.convert("RGB").copy()

    base = texture(texture_asset)
    color = (
        value("diffuseColor", (1.0, 1.0, 1.0))
        if preview
        else value("metal_color" if is_metal else "diffuse_tint", (1.0, 1.0, 1.0))
    )
    color = np.asarray((*color, value("opacity", 1.0)), dtype=float)
    if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise DomainValidationError("material base color must be finite RGBA in [0, 1]")
    uv = None
    projection = "none"
    if base is not None:
        primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if bool(value("project_uvw", False)):
            triangles = np.asarray(vertices)[faces]
            # Split seams before projecting each face onto its dominant plane.
            dominant = np.abs(mesh.face_normals).argmax(axis=1)
            coordinates = ((1, 2), (0, 2), (0, 1))
            uv = np.stack(
                [points[:, coordinates[axis]] for points, axis in zip(triangles, dominant)]
            )
            uv = uv.reshape(-1, 2)
            projection = "baked_metric_box_projection"
        elif primvar and primvar.HasValue():
            values = np.asarray(primvar.ComputeFlattened(), dtype=float)
            interpolation = str(primvar.GetInterpolation())
            if interpolation in ("vertex", "varying"):
                uv = values[np.asarray(faces).reshape(-1)]
            elif interpolation == "faceVarying":
                uv = values[np.asarray(corners).reshape(-1)]
            elif interpolation == "constant":
                uv = np.repeat(values[:1], len(faces) * 3, axis=0)
            else:
                raise DomainValidationError(f"unsupported texture interpolation: {interpolation}")
            projection = "authored_uv"
        else:
            raise DomainValidationError(
                "textured surface requires authored UVs or explicit projection"
            )
        if uv.shape != (len(faces) * 3, 2) or not np.isfinite(uv).all():
            raise DomainValidationError("texture coordinates differ from the mesh face corners")
        if not preview:
            angle = np.deg2rad(float(value("texture_rotate", 0.0)))
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            uv = (uv * np.asarray(value("texture_scale", (1.0, 1.0)))) @ rotation.T
            uv += np.asarray(value("texture_translate", (0.0, 0.0)))
        normals = mesh.vertex_normals[mesh.faces.reshape(-1)]
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices)[np.asarray(faces).reshape(-1)],
            faces=np.arange(len(faces) * 3).reshape(-1, 3),
            vertex_normals=normals,
            process=False,
        )
    pbr = trimesh.visual.material.PBRMaterial(
        name=material.GetPrim().GetName(),
        baseColorFactor=color,
        baseColorTexture=base,
        normalTexture=texture(value("normalmap_texture")),
        metallicRoughnessTexture=texture(value("ORM_texture")),
        metallicFactor=float(
            value("metallic", 1.0 if is_metal else value("metallic_constant", 0.0))
        ),
        roughnessFactor=float(value("roughness", value("reflection_roughness_constant", 0.5))),
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr)
    return mesh, {
        "material": str(material.GetPath()),
        "surface": shader_id or mdl,
        "projection": projection,
        "base_color_texture": base is not None,
        "approximation": "constant PBR metal; procedural scratches not evaluated"
        if is_metal
        else "portable PBR; MDL lighting model not evaluated"
        if not preview
        else None,
    }


def write_usd_visual(stage: Any, usd_mesh: Any, mesh: Any, files: dict[str, bytes]) -> None:
    """Export glTF PBR appearance with package-relative texture dependencies."""
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    if mesh.visual.kind != "texture":
        return
    pbr = mesh.visual.material
    if hasattr(pbr, "to_pbr"):
        pbr = pbr.to_pbr()
    factor = pbr.baseColorFactor
    color = np.asarray(factor if factor is not None else [255] * 4, dtype=float) / 255
    path = str(usd_mesh.GetPath()) + "/Appearance"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(color[3]))
    for name, value in (("metallic", pbr.metallicFactor), ("roughness", pbr.roughnessFactor)):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(
            float(value if value is not None else 0.5)
        )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)
    if pbr.baseColorTexture is None:
        return
    uv = np.asarray(mesh.visual.uv)
    if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
        raise DomainValidationError("USD visual export requires finite per-vertex UVs")
    UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set(uv.tolist())
    reader = UsdShade.Shader.Define(stage, path + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for name, image, space in (
        ("BaseColor", pbr.baseColorTexture, "sRGB"),
        ("Normal", pbr.normalTexture, "raw"),
        ("ORM", pbr.metallicRoughnessTexture, "raw"),
    ):
        if image is None:
            continue
        payload = png_bytes(image.convert("RGB"))
        filename = "textures/" + sha256(payload).hexdigest() + ".png"
        files[filename] = payload
        node = UsdShade.Shader.Define(stage, path + "/" + name)
        node.CreateIdAttr("UsdUVTexture")
        node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(filename))
        node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        for channel in ("s", "t"):
            node.CreateInput("wrap" + channel.upper(), Sdf.ValueTypeNames.Token).Set("repeat")
        if name == "BaseColor":
            node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color[:3], 1))
            node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.GetInput("diffuseColor").ConnectToSource(node.ConnectableAPI(), "rgb")
        elif name == "Normal":
            node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            node.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                node.ConnectableAPI(), "rgb"
            )
        else:
            node.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(
                    1,
                    float(shader.GetInput("roughness").Get()),
                    float(shader.GetInput("metallic").Get()),
                    1,
                )
            )
            for channel, target in (("g", "roughness"), ("b", "metallic")):
                node.CreateOutput(channel, Sdf.ValueTypeNames.Float)
                shader.GetInput(target).ConnectToSource(node.ConnectableAPI(), channel)


def validate_usd_visual(stage: Any, usd_mesh: Any, mesh: Any) -> None:
    import numpy as np
    from pxr import Ar, UsdGeom, UsdShade

    if mesh.visual.kind != "texture":
        return
    material, _ = UsdShade.MaterialBindingAPI(usd_mesh.GetPrim()).ComputeBoundMaterial()
    if not material:
        raise DomainValidationError("USD roundtrip lost a visual material")
    pbr = mesh.visual.material
    if hasattr(pbr, "to_pbr"):
        pbr = pbr.to_pbr()
    path = str(material.GetPath())
    surface = UsdShade.Shader(stage.GetPrimAtPath(path + "/Surface"))
    factor = pbr.baseColorFactor
    expected = np.asarray(factor if factor is not None else [255] * 4, dtype=float) / 255
    actual = [*surface.GetInput("diffuseColor").Get(), surface.GetInput("opacity").Get()]
    if not np.allclose(actual, expected, atol=1e-6):
        raise DomainValidationError("USD roundtrip changed the base color")
    if pbr.baseColorTexture is None:
        return
    uv = UsdGeom.PrimvarsAPI(usd_mesh).GetPrimvar("st").ComputeFlattened()
    if uv is None or not np.allclose(uv, mesh.visual.uv, atol=1e-6):
        raise DomainValidationError("USD roundtrip changed texture coordinates")
    for name, image in (
        ("BaseColor", pbr.baseColorTexture),
        ("Normal", pbr.normalTexture),
        ("ORM", pbr.metallicRoughnessTexture),
    ):
        if image is None:
            continue
        node = UsdShade.Shader(stage.GetPrimAtPath(path + "/" + name))
        asset = node.GetInput("file").Get()
        resolved = (
            Ar.GetResolver().OpenAsset(Ar.ResolvedPath(asset.resolvedPath)) if asset else None
        )
        if resolved is None or bytes(resolved.GetBuffer()) != png_bytes(image.convert("RGB")):
            raise DomainValidationError("USD roundtrip lost or changed texture pixels")
