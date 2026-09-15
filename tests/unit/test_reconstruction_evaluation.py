from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from physrealx import PhysRealX, ReconstructionEvaluationInput, ReferenceEvaluationPose
from physrealx.errors import DomainValidationError
from physrealx.integrations.reconstruction_evaluation import (
    ReconstructionEvaluationConfig,
    chamfer_distance,
    evaluate_reconstruction,
    filter_prediction,
    joint_errors,
)


def test_chamfer_has_squared_metric_units_and_both_directions():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
    result = chamfer_distance(a, b)
    assert result["cd"] == pytest.approx(35.0)
    assert chamfer_distance(a * 100, b * 100)["cd"] == pytest.approx(result["cd"] * 10000)


@pytest.mark.parametrize("invalid", [np.empty((0, 3)), np.ones((4, 2)), np.full((2, 3), np.nan)])
def test_chamfer_rejects_invalid_clouds(invalid):
    with pytest.raises(DomainValidationError):
        chamfer_distance(invalid, np.zeros((1, 3)))


def test_hinge_parallel_skew_and_sign_equivalent_axes():
    row = joint_errors("hinge", [0, 0, -1], [1, 0, 8], -np.pi / 2, [0, 0, 1], [0, 0, 0], np.pi / 2)
    assert row["axis_ang_deg"] == pytest.approx(0)
    assert row["axis_pos"] == pytest.approx(10)
    assert row["part_motion"] == pytest.approx(0)
    skew = joint_errors("hinge", [1, 0, 0], [0, 0, 2], 1, [0, 1, 0], [0, 0, 0], 1)
    assert skew["axis_pos"] == pytest.approx(20)
    assert skew["axis_ang_deg"] == pytest.approx(90)


def test_slide_position_is_unidentifiable_but_wrong_motion_sign_is_detected():
    row = joint_errors("slide", [-1, 0, 0], [100, 20, 30], 0.2, [1, 0, 0], [0, 0, 0], 0.2)
    assert row["axis_ang_deg"] == pytest.approx(0)
    assert row["axis_pos"] is None
    assert row["part_motion"] == pytest.approx(0.4)


def test_cd_agrees_with_dtart_pytorch3d_formula():
    torch = pytest.importorskip("torch")
    loss = pytest.importorskip("pytorch3d.loss")
    random = np.random.default_rng(13)
    a, b = random.normal(size=(71, 3)), random.normal(size=(89, 3))
    expected, _ = loss.chamfer_distance(
        torch.tensor(a[None], dtype=torch.float32),
        torch.tensor(b[None], dtype=torch.float32),
        batch_reduction=None,
    )
    assert chamfer_distance(a, b)["cd"] == pytest.approx(expected.item() * 1000, rel=1e-6)


def test_triangle_filter_agrees_with_dtart_open3d_and_preserves_input():
    o3d = pytest.importorskip("open3d")

    def fan(count, offset):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        vertices = np.vstack(
            ([0.0, 0.0, 0.0], np.c_[np.cos(angles), np.sin(angles), np.zeros(count)])
        )
        vertices += offset
        faces = [[0, i + 1, (i + 1) % count + 1] for i in range(count)]
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # DTArt uses face counts, not component area. Coincident components must
    # stay distinct; equality at the 10% cutoff must be retained.
    mesh = trimesh.util.concatenate([fan(100, 0), fan(10, 0), fan(9, 3)])
    original = mesh.faces.copy()
    native = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
    )
    labels, sizes, _ = native.cluster_connected_triangles()
    expected = np.asarray(sizes)[np.asarray(labels)] >= max(sizes) * 0.1
    filtered, stats = filter_prediction(mesh)
    assert np.array_equal(mesh.faces, original)
    assert stats == {
        "input_triangles": 119,
        "retained_triangles": 110,
        "input_components": 3,
        "retained_components": 2,
    }
    assert len(filtered.faces) == int(expected.sum())
    assert np.allclose(filtered.triangles, mesh.triangles[expected])


@pytest.fixture
def evaluation_case(tmp_path):
    """Known metre meshes, nonidentity pose, a hinge and two independent sliders."""
    px = PhysRealX.open(tmp_path / "workspace")
    kinds = ["fixed", "hinge", "slide", "slide"]
    axes = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    origins = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    spans = [0.0, np.pi / 3, 0.2, -0.15]
    rotation = Rotation.from_euler("xyz", [20, -30, 40], degrees=True).as_matrix()
    translation = np.array([0.2, -0.1, 0.4])
    links, reference_joints, parts, meshes = [], [], {}, {}
    for i, kind in enumerate(kinds):
        mesh = trimesh.creation.box(extents=[0.2, 0.3, 0.1])
        mesh.apply_translation([i * 0.4, 0, 0.2])
        reference_mesh = px.artifacts.put_bytes(
            mesh.export(file_type="glb"), media_type="model/gltf-binary"
        )
        name = f"part_{i}"
        links.append({"name": name, "geometry": reference_mesh.model_dump(mode="json")})
        mesh.vertices = (mesh.vertices - translation) @ rotation
        ref = px.artifacts.put_bytes(mesh.export(file_type="ply"), media_type="model/ply")
        meshes[str(i)] = ref.model_dump(mode="json")
        parts[str(i)] = dict(
            kind=kind,
            parent="0" if i else None,
            axis=(rotation.T @ axes[i]).tolist(),
            origin=(rotation.T @ (np.array(origins[i]) - translation)).tolist(),
            span=spans[i],
        )
        if i:
            reference_joints.append(
                dict(
                    name=f"joint_{i}",
                    parent="part_0",
                    child=name,
                    joint_type=kind,
                    axis=axes[i],
                    origin_m=origins[i],
                    lower_limit=-2.0,
                    upper_limit=2.0,
                    limit_unit="rad" if kind == "hinge" else "m",
                )
            )
    reconstruction = px.artifacts.put_json(
        {
            "schema": "physrealx.geometry-reconstruction-report/1",
            "unit": "m",
            "coordinate_frame": "capture",
            "method": {"name": "independent_test_method"},
            "prediction": {
                "parts": parts,
                "states": {"start": meshes, "end": {}},
                "motion_pair": ["start", "end"],
            },
        }
    )
    reference = px.artifacts.put_json(
        {
            "schema": "physrealx.usd-reference/1",
            "unit": "m",
            "coordinate_frame": "reference",
            "root_link": "part_0",
            "links": links,
            "joints": reference_joints,
        }
    )
    pose = ReferenceEvaluationPose(
        rotation=rotation.tolist(),
        translation_m=translation.tolist(),
        source_frame="capture",
        target_frame="reference",
        evidence_scope="capture_annotations",
        part_mapping={str(i): (f"part_{i}",) for i in range(4)},
        start_positions={f"joint_{i}": 0.0 for i in range(1, 4)},
        end_positions={f"joint_{i}": spans[i] for i in range(1, 4)},
    )
    return px, ReconstructionEvaluationInput(
        reconstruction_report=reconstruction, reference_model=reference, pose=pose
    )


def test_public_api_evaluates_every_part_joint_and_both_states(evaluation_case):
    px, request = evaluation_case
    report_ref = px.assets.evaluate_reconstruction(px.artifacts.put_json(request))
    report = json.loads(px.artifacts.get_bytes(report_ref))
    assert report["coverage"]["individually_evaluated_parts"] == 4
    assert report["coverage"]["evaluated_joints"] == 3
    for state in ("start", "end"):
        assert report["geometry"][state]["whole"]["cd"] < 0.5
        assert all(
            part["status"] == "measured" for part in report["geometry"][state]["parts"].values()
        )
    for row in report["joints"].values():
        assert row["status"] == "measured"
        assert row["axis_ang_deg"] == pytest.approx(0, abs=1e-5)
        assert row["part_motion"] == pytest.approx(0, abs=1e-6)
    assert report["joints"]["joint_1"]["axis_pos"] == pytest.approx(0, abs=1e-6)


def test_merged_missing_and_extra_geometry_are_not_silently_excluded(evaluation_case):
    px, request = evaluation_case
    pose = request.pose.model_copy(
        update={"part_mapping": {"0": ("part_0", "part_1"), "2": ("part_2",)}}
    )
    request = request.model_copy(update={"pose": pose})
    config = ReconstructionEvaluationConfig()
    report = evaluate_reconstruction(px.artifacts, request, config)
    assert report == evaluate_reconstruction(px.artifacts, request, config)
    assert report["coverage"]["missing_reference_parts"] == ["part_3"]
    assert report["coverage"]["unmapped_predicted_groups"] == ["1", "3"]
    assert report["geometry"]["start"]["parts"]["part_1"]["cd"] is None
    assert report["geometry"]["start"]["parts"]["part_3"]["status"] == "missing"
    assert report["joints"]["joint_1"]["status"] == "unobserved_or_merged"
    # Whole-object score still uses ALL predicted and reference meshes, even unmapped parts.
    assert report["geometry"]["end"]["whole"]["cd"] < 2


@pytest.mark.parametrize(
    "change",
    [
        {"rotation": ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
        {"rotation": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
        {"translation_m": (float("nan"), 0.0, 0.0)},
        {"source_frame": "wrong"},
        {"part_mapping": {"0": ("part_0",), "1": ("part_0",)}},
        {"part_mapping": {"9": ("part_0",)}},
        {"end_positions": {"joint_1": 50.0, "joint_2": 0.2, "joint_3": -0.15}},
    ],
)
def test_invalid_pose_correspondence_and_state_rejected(evaluation_case, change):
    px, request = evaluation_case
    with pytest.raises((DomainValidationError, ValidationError)):
        request = request.model_copy(update={"pose": request.pose.model_copy(update=change)})
        evaluate_reconstruction(px.artifacts, request, ReconstructionEvaluationConfig())


def test_exactly_one_alignment_source(evaluation_case):
    _, request = evaluation_case
    for pose, registration in ((None, None), (request.pose, request.reference_model)):
        with pytest.raises(ValidationError, match="exactly one"):
            ReconstructionEvaluationInput(
                reconstruction_report=request.reconstruction_report,
                reference_model=request.reference_model,
                pose=pose,
                registration=registration,
            )


def test_nested_motion_and_cyclic_predicted_graph(evaluation_case):
    px, request = evaluation_case
    model = json.loads(px.artifacts.get_bytes(request.reference_model))
    model["joints"][1]["parent"] = "part_1"
    reconstruction = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    parts = reconstruction["prediction"]["parts"]
    parts["2"]["parent"] = "1"
    for cyclic in (False, True):
        if cyclic:
            parts["1"]["parent"] = "2"
        nested = request.model_copy(
            update={
                "reference_model": px.artifacts.put_json(model),
                "reconstruction_report": px.artifacts.put_json(reconstruction),
            }
        )
        if cyclic:
            with pytest.raises(DomainValidationError, match="cyclic"):
                evaluate_reconstruction(px.artifacts, nested, ReconstructionEvaluationConfig())
        else:
            report = evaluate_reconstruction(px.artifacts, nested, ReconstructionEvaluationConfig())
            assert report["coverage"]["evaluated_joints"] == 3
            assert report["geometry"]["end"]["whole"]["cd"] < 2


@pytest.mark.parametrize(
    "mutation", ["unit", "unknown_mesh", "reference_unit", "reference_duplicate"]
)
def test_malformed_reports_are_rejected(evaluation_case, mutation):
    px, request = evaluation_case
    reconstruction = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    reference = json.loads(px.artifacts.get_bytes(request.reference_model))
    if mutation == "unit":
        reconstruction["unit"] = "mm"
    elif mutation == "unknown_mesh":
        meshes = reconstruction["prediction"]["states"]["start"]
        meshes["unknown"] = meshes["0"]
    elif mutation == "reference_unit":
        reference["joints"][0]["limit_unit"] = "deg"
    else:
        reference["joints"].append(reference["joints"][0])
    request = request.model_copy(
        update={
            "reconstruction_report": px.artifacts.put_json(reconstruction),
            "reference_model": px.artifacts.put_json(reference),
        }
    )
    with pytest.raises(DomainValidationError):
        evaluate_reconstruction(px.artifacts, request, ReconstructionEvaluationConfig())


def test_evaluation_is_independent_of_reconstruction_method(evaluation_case, monkeypatch):
    from examples._display import show_reconstruction_evaluation

    px, request = evaluation_case
    document = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    results = []
    tables = []
    monkeypatch.setattr("IPython.display.display", lambda table: tables.append(table.data))
    presentations = []
    for method, diagnostics in (
        ("artgs", {"validation_fg_psnr": 1, "validation_alpha_iou": 0}),
        ("dtart", {"validation_surface_median_m": 10}),
        ("other_open_source_method", {}),
        (None, None),
    ):
        if method is None:
            document.pop("method")
        else:
            document["method"] = {"name": method}
        if diagnostics is None:
            document.pop("metrics")
        else:
            document["metrics"] = diagnostics
        source = request.model_copy(
            update={"reconstruction_report": px.artifacts.put_json(document)}
        )
        result = json.loads(
            px.artifacts.get_bytes(px.assets.evaluate_reconstruction(px.artifacts.put_json(source)))
        )
        results.append((result["geometry"], result["joints"], result["protocol"]))
        tables.clear()
        show_reconstruction_evaluation(result)
        presentations.append(tuple(tables))
    assert all(result == results[0] for result in results)
    assert len(presentations[0]) == 2
    assert all(tables == presentations[0] for tables in presentations)
    with pytest.raises(TypeError, match="method"):
        px.assets.evaluate_reconstruction(px.artifacts.put_json(request), method="dtart")


def test_native_end_geometry_is_evaluated_not_overwritten_by_motion(evaluation_case):
    px, request = evaluation_case
    document = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    prediction = document["prediction"]
    # Actual end geometry deliberately disagrees with predicted articulation.
    prediction["states"]["end"] = prediction["states"]["start"]
    request = request.model_copy(update={"reconstruction_report": px.artifacts.put_json(document)})
    report = evaluate_reconstruction(px.artifacts, request, ReconstructionEvaluationConfig())
    assert report["geometry"]["end"]["whole"]["cd"] > report["geometry"]["start"]["whole"]["cd"]
    assert set(report["protocol"]["end_geometry"].values()) == {"published state mesh"}
    assert report["joints"]["joint_1"]["part_motion"] == pytest.approx(0, abs=1e-6)


def test_unsegmented_multistate_output_still_reports_whole_cd_and_missing_metrics(evaluation_case):
    px, request = evaluation_case
    document = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    mesh = document["prediction"]["states"]["start"]["0"]
    document["prediction"] = {
        "parts": {"whole": {}},
        "segmented": False,
        "states": {state: {"whole": mesh} for state in ("first", "second", "third")},
    }
    request = request.model_copy(
        update={
            "reconstruction_report": px.artifacts.put_json(document),
            "pose": request.pose.model_copy(update={"part_mapping": {}}),
        }
    )
    with pytest.raises(DomainValidationError, match="state_pair"):
        evaluate_reconstruction(px.artifacts, request, ReconstructionEvaluationConfig())
    request = request.model_copy(update={"state_pair": ("second", "third")})
    report = evaluate_reconstruction(px.artifacts, request, ReconstructionEvaluationConfig())
    assert report["state_pair"] == ["second", "third"]
    assert report["coverage"]["evaluated_joints"] == 0
    for state in ("start", "end"):
        assert report["geometry"][state]["whole"]["cd"] > 0
        assert report["geometry"][state]["static"]["cd"] is None
        assert all(
            row["status"] == "unsegmented" and row["cd"] is None
            for row in report["geometry"][state]["parts"].values()
        )
    assert all(
        row["status"] == "unsegmented" and row["axis_ang_deg"] is None
        for row in report["joints"].values()
    )


def test_missing_normalized_output_fails_before_operation_publication(evaluation_case):
    from physrealx.domain import OperationOutput, OperationOutputRequirement
    from physrealx.errors import MethodExecutionError
    from physrealx.operation_validation import validate_operation_output

    px, request = evaluation_case
    document = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    document.pop("prediction")
    with pytest.raises(MethodExecutionError, match="reconstruction contract"):
        validate_operation_output(
            px.artifacts,
            OperationOutput(
                name="geometry_reconstruction_report",
                contract="GeometryReconstructionReport/1",
                artifact=px.artifacts.put_json(document),
            ),
            OperationOutputRequirement(
                name="geometry_reconstruction_report",
                contract="GeometryReconstructionReport/1",
                content_kind="json",
                schema_discriminator="physrealx.geometry-reconstruction-report/1",
            ),
        )


def test_joint_selection_skips_other_part_scores_but_preserves_full_geometry(
    evaluation_case, monkeypatch
):
    import physrealx.integrations.reconstruction_evaluation as evaluation

    px, request = evaluation_case
    full = evaluate_reconstruction(px.artifacts, request, ReconstructionEvaluationConfig())
    calls = []
    original = evaluation.chamfer_distance

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(evaluation, "chamfer_distance", count)
    selected = request.model_copy(update={"joint_names": ("joint_2",)})
    report = evaluate_reconstruction(px.artifacts, selected, ReconstructionEvaluationConfig())
    assert len(calls) == 6  # One selected child, whole and static, in each state.
    assert list(report["joints"]) == ["joint_2"]
    for state in ("start", "end"):
        assert list(report["geometry"][state]["parts"]) == ["part_2"]
        for key in ("whole", "static"):
            assert report["geometry"][state][key] == full["geometry"][state][key]
    assert report["coverage"]["reference_joints"] == 3
    assert report["coverage"]["evaluated_joints"] == 1
    assert report["coverage"]["individually_evaluated_parts"] == 1
    assert report["source"]["joint_names"] == ["joint_2"]
    calls.clear()
    report = evaluate_reconstruction(
        px.artifacts,
        request.model_copy(update={"joint_names": ()}),
        ReconstructionEvaluationConfig(),
    )
    assert len(calls) == 4
    assert report["joints"] == report["geometry"]["start"]["parts"] == {}


def test_joint_selection_does_not_hide_a_selected_prediction_failure(evaluation_case):
    px, request = evaluation_case
    pose = request.pose.model_copy(update={"part_mapping": {"0": ("part_0",)}})
    selected = request.model_copy(update={"joint_names": ("joint_2",), "pose": pose})
    report = evaluate_reconstruction(px.artifacts, selected, ReconstructionEvaluationConfig())
    assert report["joints"]["joint_2"]["status"] == "missing"
    assert report["geometry"]["start"]["parts"]["part_2"]["status"] == "missing"
    assert report["geometry"]["start"]["parts"]["part_2"]["cd"] is None
    with pytest.raises(DomainValidationError, match="unknown evaluation joints"):
        evaluate_reconstruction(
            px.artifacts,
            request.model_copy(update={"joint_names": ("typo",)}),
            ReconstructionEvaluationConfig(),
        )
    with pytest.raises(ValidationError, match="unique"):
        ReconstructionEvaluationInput.model_validate(
            {
                **request.model_dump(),
                "joint_names": ["joint_2", "joint_2"],
            }
        )


def test_animation_uses_predicted_motion_and_reference_states_independently(
    evaluation_case, monkeypatch
):
    import io

    from PIL import Image

    import physrealx.integrations.reconstruction_preview as preview

    px, request = evaluation_case
    document = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    document["prediction"]["parts"]["2"]["span"] = 0.1
    request = request.model_copy(update={"reconstruction_report": px.artifacts.put_json(document)})
    evaluation = px.assets.evaluate_reconstruction(px.artifacts.put_json(request))
    spans, states = [], []
    motion, reference = preview.joint_transform, preview.reference_link_transforms

    def record_motion(kind, axis, origin, span):
        spans.append((kind, span))
        return motion(kind, axis, origin, span)

    def record_reference(model, positions):
        states.append(dict(positions))
        return reference(model, positions)

    monkeypatch.setattr(preview, "joint_transform", record_motion)
    monkeypatch.setattr(preview, "reference_link_transforms", record_reference)
    animation = px.assets.preview_reconstruction(evaluation, frames=3, fps=10)
    assert animation.media_type == "image/gif"
    assert ("slide", 0.1) in spans
    assert ("slide", 0.2) not in spans  # GT displacement must not drive reconstruction.
    assert [state["joint_2"] for state in states] == pytest.approx([0, 0.1, 0.2])
    with Image.open(io.BytesIO(px.artifacts.get_bytes(animation))) as gif:
        assert gif.n_frames == 3
        assert gif.size == (1000, 500)
        assert gif.info["duration"] == 100
        assert (
            json.loads(gif.info["comment"])["evaluation"]["artifact_id"] == evaluation.artifact_id
        )
        first = np.asarray(gif.convert("RGB"))
        gif.seek(2)
        last = np.asarray(gif.convert("RGB"))
        assert np.any(first[:450, :500] != last[:450, :500])
        assert np.any(first[:450, 500:] != last[:450, 500:])
    with pytest.raises(DomainValidationError, match="frames"):
        px.assets.preview_reconstruction(evaluation, frames=True)
    with pytest.raises(DomainValidationError, match="fps"):
        px.assets.preview_reconstruction(evaluation, fps=0)


def test_appearance_preview_uses_every_published_frame(evaluation_case):
    import io
    import zipfile

    from PIL import Image

    px, request = evaluation_case
    frames = []
    for value in (32, 224, 96):
        output = io.BytesIO()
        Image.fromarray(np.full((18, 24, 3), value, np.uint8)).save(output, format="PNG")
        frames.append(output.getvalue())
    bundle = io.BytesIO()
    inventory = [
        {"path": f"appearance/{index:02d}.png", "media_type": "image/png"}
        for index in range(len(frames))
    ]
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("bundle-manifest.json", json.dumps({"files": inventory}))
        for record, payload in zip(inventory, frames):
            archive.writestr(record["path"], payload)
    report = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    report["bundle"] = px.artifacts.put_bytes(
        bundle.getvalue(), media_type="application/zip"
    ).model_dump(mode="json")
    report_ref = px.artifacts.put_json(report)

    animation = px.assets.preview_reconstruction_appearance(report_ref, fps=4)

    with Image.open(io.BytesIO(px.artifacts.get_bytes(animation))) as gif:
        assert gif.n_frames == 3
        assert gif.size == (24, 18)
        assert gif.info["duration"] == 250
        assert json.loads(gif.info["comment"])["appearance_frames"] == [
            "appearance/00.png",
            "appearance/01.png",
            "appearance/02.png",
        ]
    with pytest.raises(DomainValidationError, match="fps"):
        px.assets.preview_reconstruction_appearance(report_ref, fps=True)
