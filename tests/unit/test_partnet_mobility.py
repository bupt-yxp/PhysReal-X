import json
import shutil
from pathlib import Path

import pytest

from physrealx import PhysRealX, SceneSpec
from physrealx.domain import GeometryRole, JointType
from physrealx.errors import DomainValidationError
from physrealx.integrations._asset_preview import asset_geometry
from physrealx.integrations.partnet_mobility import retrieve_partnet_scene_assets
from physrealx.integrations.scenes import scene_bounds


def partnet_fixture(root: Path) -> Path:
    import trimesh

    model = root / "123"
    (model / "textured_objs").mkdir(parents=True)
    (model / "textured_objs/cube.obj").write_text(
        trimesh.creation.box(extents=(2.0, 1.0, 0.5)).export(file_type="obj")
    )
    (model / "textured_objs/panel.mtl").write_text(
        "newmtl front\nKd 0.8 0.8 0.8\nnewmtl back\nKd 0.6 0.6 0.6\n"
    )
    (model / "textured_objs/panel.obj").write_text(
        """mtllib panel.mtl
v -1.0 -0.5 -0.20
v  1.0 -0.5 -0.20
v  1.0  0.5 -0.20
v -1.0  0.5 -0.20
v -1.0 -0.5 -0.21
v -1.0  0.5 -0.21
v  1.0  0.5 -0.21
v  1.0 -0.5 -0.21
usemtl front
f 1 2 3
f 1 3 4
usemtl back
f 5 6 7
f 5 7 8
"""
    )
    (model / "meta.json").write_text(json.dumps({"model_cat": "Laptop"}))
    (model / "bounding_box.json").write_text(
        json.dumps({"min": [-1.0, -0.5, -0.25], "max": [1.0, 0.5, 0.25]})
    )
    (model / "mobility.urdf").write_text(
        """<robot name="fixture">
<link name="base"/>
<link name="body">
  <visual><geometry><mesh filename="textured_objs/cube.obj"/></geometry></visual>
  <visual><geometry><mesh filename="textured_objs/panel.obj"/></geometry></visual>
  <collision><geometry><mesh filename="textured_objs/cube.obj"/></geometry></collision>
</link>
<joint name="mount" type="fixed">
  <parent link="base"/><child link="body"/><origin xyz="0 0 0"/>
</joint>
<link name="screen">
  <visual>
    <origin xyz="0 0 0.5"/>
    <geometry><mesh filename="textured_objs/cube.obj"/></geometry>
  </visual>
  <collision>
    <origin xyz="0 0 0.5"/>
    <geometry><mesh filename="textured_objs/cube.obj"/></geometry>
  </collision>
</link>
<joint name="hinge" type="revolute">
  <parent link="body"/><child link="screen"/><axis xyz="0 1 0"/>
  <limit lower="-1.0" upper="1.0"/>
</joint>
</robot>"""
    )
    return root


def test_partnet_mobility_import_is_metric_articulated_and_idempotent(tmp_path):
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    catalog = partnet_fixture(tmp_path / "catalog")
    reference = px.assets.import_partnet_mobility(
        catalog,
        model_id="123",
        resource_id="partnet.laptop.123",
        target_extent_m=0.4,
    )
    repeated = px.assets.import_partnet_mobility(
        catalog,
        model_id="123",
        resource_id="partnet.laptop.123",
        target_extent_m=0.4,
    )
    asset = px.assets.get(reference)

    assert repeated == reference
    assert asset.metadata["category"] == "Laptop"
    assert asset.metadata["uniform_scale"] == pytest.approx(0.2)
    assert asset.metadata["collision_proxy"] == "per_link_axis_aligned_box"
    assert asset.metadata["orientation"] == "urdf_joint_frames"
    assert asset.metadata["joint_dynamics"] == "geometry_derived_holding_prior"
    body = next(
        mesh
        for link, mesh in asset_geometry(px.artifacts, asset, roles=(GeometryRole.VISUAL,))[
            GeometryRole.VISUAL
        ]
        if link == "body"
    )
    assert body.extents == pytest.approx((0.4, 0.2, 0.1))
    assert len(body.faces) == 16
    assert asset.metadata["source_faces"]["visual"] == 28
    assert asset.metadata["imported_faces"]["visual"] == 28
    assert [joint.joint_type for joint in asset.joints] == [JointType.FIXED, JointType.REVOLUTE]
    assert next(joint for joint in asset.joints if joint.name == "hinge").friction > 0.01
    assert {geometry.role.value for geometry in asset.geometries} == {"visual", "collision"}
    assert len(asset.contact_exclusions) == 3
    assert asset.metadata["self_collision"] == "disabled_for_conservative_link_proxies"
    with pytest.raises(DomainValidationError, match="different import settings"):
        px.assets.import_partnet_mobility(
            catalog,
            model_id="123",
            resource_id="partnet.laptop.123",
            target_extent_m=0.5,
        )


def test_scenesmith_searches_full_library_and_returns_reproducible_candidates(tmp_path):
    catalog = partnet_fixture(tmp_path / "catalog")
    for model_id in ("124", "125"):
        shutil.copytree(catalog / "123", catalog / model_id)
    (catalog / "catalog.json").write_text(
        json.dumps(
            {
                "schema": "physrealx.scenesmith-partnet-catalog/1",
                "profiles": {
                    "home": [
                        {
                            "name": "laptop",
                            "query": "articulated laptop",
                            "categories": ["Laptop"],
                            "target_extent_m": 0.5,
                            "role": "target",
                            "required_joint_types": ["revolute"],
                        }
                    ]
                },
            }
        )
    )

    first = retrieve_partnet_scene_assets(
        catalog,
        scene_type="home",
        instruction="关闭笔记本电脑屏幕",
        seed=13,
        candidate_count=2,
    )
    second = retrieve_partnet_scene_assets(
        catalog,
        scene_type="home",
        instruction="关闭笔记本电脑屏幕",
        seed=13,
        candidate_count=2,
    )

    assert first == second
    assert first.evaluated_models == 3
    assert first.valid_models == 3
    assert len(first.candidates) == 2
    assert len({item.assets[0].model_id for item in first.candidates}) == 2
    assert all(item.assets[0].score_components["function"] == 1.0 for item in first.candidates)

    px = PhysRealX.open(tmp_path / "retrieval-workspace", discover_plugins=False)
    selection = px.assets.retrieve_partnet_mobility(
        catalog,
        scene_type="home",
        instruction="关闭笔记本电脑屏幕",
        method="SceneSmith  ",
        seed=13,
        candidate_count=2,
    )
    report = json.loads(px.artifacts.get_bytes(selection.retrieval_report))
    assert len(selection.candidates) == 2
    assert report["schema"] == "physrealx.scenesmith-retrieval-report/1"
    assert report["method"] == "scenesmith"
    assert report["library"] == {
        "evaluated_models": 3,
        "valid_models": 3,
        "rejected_models": 0,
    }
    assert len(report["selected"]) == 2

    with pytest.raises(DomainValidationError, match="does not define"):
        retrieve_partnet_scene_assets(
            catalog,
            scene_type="industrial",
            instruction="关闭保险柜",
            seed=13,
        )


def test_scenesmith_resolves_support_orientation_and_composes(tmp_path):
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    catalog = partnet_fixture(tmp_path / "catalog")
    table = px.assets.import_partnet_mobility(
        catalog,
        model_id="123",
        resource_id="partnet.table.123",
        target_extent_m=1.0,
    )
    laptop = px.assets.import_partnet_mobility(
        catalog,
        model_id="123",
        resource_id="partnet.laptop.123",
        target_extent_m=0.2,
    )
    room = px.assets.create_room_shell(
        resource_id="environment.test-room", width_m=4.0, depth_m=4.0, height_m=2.4
    )
    operation = px.scenes.generate(
        {
            "resource_id": "semantic.scene",
            "version": "1.0.0",
            "display_name": "Semantic scene",
            "domain": "tabletop",
            "environment": room.model_dump(mode="json"),
            "objects": [
                {"name": "table", "asset": table.model_dump(mode="json")},
                {
                    "name": "laptop",
                    "asset": laptop.model_dump(mode="json"),
                    "yaw_rad": 0.5,
                },
            ],
            "relations": [
                {"subject": "laptop", "relation": "on", "object": "table", "clearance_m": 0.0}
            ],
            "xy_bounds_m": [-2.0, -2.0, 2.0, 2.0],
            "interaction_target": "laptop",
            "seed": 7,
        },
        method="SceneSmith  ",
    )
    composed = px.scenes.compose(operation.output("scene_draft"), method="collision_constraints")
    scene = SceneSpec.model_validate_json(px.artifacts.get_bytes(composed.output("scene_spec")))
    bounds = scene_bounds(scene, px.artifacts)

    assert operation.metrics == {
        "entities": 3.0,
        "relations": 1.0,
        "reachability_clearance_m": 0.3,
    }
    assert scene.entities[0].name == "room"
    assert scene.entities[0].role.value == "environment"
    assert bounds["laptop"][0, 2] == pytest.approx(bounds["table"][1, 2])
    assert scene.metadata["layout_method"] == "scenesmith"
    assert scene.metadata["method_basis"]["project"] == "SceneSmith"
    assert scene.metadata["interaction_corridor"]["target"] == "laptop"
    assert scene.entities[2].pose.rotation_xyzw[2] == pytest.approx(0.247403959)


def test_canonical_room_and_open_source_robot_keep_provenance_and_geometry(tmp_path):
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    room = px.assets.get(
        px.assets.create_room_shell(
            resource_id="environment.room", width_m=3.6, depth_m=3.6, height_m=2.4
        )
    )
    robot = px.assets.get(
        px.assets.import_mujoco_model(
            Path(__file__).parents[2] / "examples/scene_and_task_generation/input/MuJoCo-Menagerie/"
            "franka_emika_panda/panda.xml",
            resource_id="robot.franka-emika-panda",
            display_name="Franka Emika Panda",
            source_name="MuJoCo Menagerie/franka_emika_panda",
            source_revision="8161bba264d7fa7c99ca301e91e7fb44737676ad",
            license_name="Apache-2.0",
        )
    )

    assert {item.name for item in room.geometries if item.role.value == "collision"} == {
        "floor_collision"
    }
    assert {item.name for item in room.geometries if item.role.value == "visual"} == {
        "floor_visual",
        "back_wall_visual",
        "left_wall_visual",
    }
    assert robot.metadata["source"] == "MuJoCo Menagerie/franka_emika_panda"
    assert robot.metadata["license"] == "Apache-2.0"
    assert robot.metadata["degrees_of_freedom"] == 9
    assert [item.name for item in robot.joints if item.joint_type is JointType.REVOLUTE] == [
        f"joint{index}" for index in range(1, 8)
    ]
    assert {item.role.value for item in robot.geometries} == {"visual", "collision"}
    assert robot.metadata["home_positions"]["joint4"] == pytest.approx(-1.57079)
