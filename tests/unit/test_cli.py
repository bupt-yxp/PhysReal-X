from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from physrealx import cli
from physrealx.diagnostics import DoctorReport
from physrealx.domain import ResourceType, TrainingSpec, VersionedRef
from physrealx.platform import PhysRealX
from physrealx.reference import reference_asset, reference_run, reference_task


def _invoke(
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> tuple[int, str, str]:
    result = cli.main(arguments)
    captured = capsys.readouterr()
    return result, captured.out, captured.err


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_cli_runs_complete_local_workspace_workflow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    asset = reference_asset()
    task = reference_task(asset)
    run = reference_run(asset, task, episode_count=2, base_seed=19)
    asset_path = tmp_path / "asset.json"
    task_path = tmp_path / "task.yaml"
    run_path = tmp_path / "run.json"
    _write_json(asset_path, asset)
    _write_yaml(task_path, task)
    _write_json(run_path, run)

    status, output, error = _invoke(capsys, ["init", str(workspace)])
    assert status == 0
    assert Path(output.strip()) == workspace.resolve()
    assert not error
    assert (workspace / "registry.sqlite3").is_file()

    status, output, error = _invoke(
        capsys,
        ["plugins", "--workspace", str(workspace)],
    )
    assert status == 0
    assert not error
    plugins = json.loads(output)
    assert {item["name"] for item in plugins} == {
        "reference_drawer",
        "scripted_drawer",
        "torch_bc",
    }

    status, output, error = _invoke(
        capsys,
        ["add", "asset", str(asset_path), "--workspace", str(workspace)],
    )
    assert status == 0
    assert not error
    asset_ref = json.loads(output)
    assert asset_ref["resource_type"] == "asset"
    assert asset_ref["digest"].startswith("sha256:")

    status, output, error = _invoke(
        capsys,
        ["add", "task", str(task_path), "--workspace", str(workspace)],
    )
    assert status == 0
    assert not error
    task_ref = json.loads(output)
    assert task_ref["resource_type"] == "task"

    for resource_type, expected_id in (
        ("asset", asset.resource_id),
        ("task", task.resource_id),
    ):
        status, output, error = _invoke(
            capsys,
            ["list", resource_type, "--workspace", str(workspace)],
        )
        assert status == 0
        assert not error
        assert [item["resource_id"] for item in json.loads(output)] == [expected_id]

    status, output, error = _invoke(
        capsys,
        ["run", str(run_path), "--workspace", str(workspace)],
    )
    assert status == 0
    assert not error
    record = json.loads(output)
    assert record["run_id"] == run.run_id
    assert record["status"] == "succeeded"
    assert record["result"]["episode_count"] == 2

    status, output, error = _invoke(
        capsys,
        ["status", run.run_id, "--workspace", str(workspace)],
    )
    assert status == 0
    assert not error
    assert json.loads(output) == record


@pytest.mark.parametrize("kind", ["asset", "task", "run", "training"])
def test_cli_validates_every_supported_contract_kind(
    kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset = reference_asset()
    values = {
        "asset": asset,
        "task": reference_task(asset),
        "run": reference_run(asset, reference_task(asset), episode_count=1),
        "training": TrainingSpec(
            resource_id="drawer.validation",
            version="1.0.0",
            dataset=VersionedRef(
                resource_type=ResourceType.DATASET,
                resource_id="drawer.dataset",
                version="1.0.0",
                digest=f"sha256:{'a' * 64}",
            ),
            trainer="torch_bc",
        ),
    }
    path = tmp_path / f"{kind}.json"
    _write_json(path, values[kind])

    status, output, error = _invoke(capsys, ["validate", kind, str(path)])

    assert status == 0
    assert not error
    assert json.loads(output)["schema_version"] == 1


def test_cli_doctor_supports_json_human_and_strict_modes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = DoctorReport(
        python="3.11.15",
        platform="test-platform",
        conda_environment="PhysReal-X",
        torch_installed=True,
        torch_version="2.5.1+cu121",
        torch_cuda_runtime="12.1",
        cuda_available=False,
        cuda_device_count=0,
        cuda_devices=(),
        cu121_ready=False,
        issues=("CUDA is intentionally unavailable",),
    )
    probes: list[bool] = []

    def inspect(*, probe_cuda: bool) -> DoctorReport:
        probes.append(probe_cuda)
        return report

    monkeypatch.setattr(cli, "inspect_runtime", inspect)

    status, output, error = _invoke(capsys, ["doctor", "--json", "--no-cuda"])
    assert status == 0
    assert not error
    assert json.loads(output)["torch_cuda_runtime"] == "12.1"

    status, output, error = _invoke(capsys, ["doctor", "--strict"])
    assert status == 1
    assert "Python:          3.11.15" in output
    assert "warning: CUDA is intentionally unavailable" in output
    assert not error
    assert probes == [False, True]


def test_cli_demo_and_expected_error_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "demo"
    status, output, error = _invoke(
        capsys,
        [
            "demo",
            "--workspace",
            str(workspace),
            "--episodes",
            "2",
            "--base-seed",
            "41",
        ],
    )
    assert status == 0
    assert not error
    demo = json.loads(output)
    assert demo["workspace"] == str(workspace.resolve())
    assert demo["run"]["status"] == "succeeded"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    status, output, error = _invoke(capsys, ["validate", "asset", str(invalid)])
    assert status == 2
    assert not output
    assert "physrealx: invalid_input:" in error
    assert "configuration root must be a mapping" in error

    empty_workspace = tmp_path / "empty"
    task_path = tmp_path / "task.json"
    _write_json(task_path, reference_task())
    status, output, error = _invoke(
        capsys,
        ["add", "task", str(task_path), "--workspace", str(empty_workspace)],
    )
    assert status == 2
    assert not output
    assert "physrealx: resource_not_found:" in error

    missing_id = "00000000-0000-0000-0000-000000000099"
    status, output, error = _invoke(
        capsys,
        ["status", missing_id, "--workspace", str(empty_workspace)],
    )
    assert status == 2
    assert not output
    assert "physrealx: resource_not_found:" in error

    with pytest.raises(FileNotFoundError):
        cli.main(["--debug", "validate", "asset", str(tmp_path / "missing.json")])


@pytest.mark.integration
def test_cli_trains_and_evaluates_registered_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    workspace = tmp_path / "learning"
    platform = PhysRealX.open(workspace, discover_plugins=False)
    asset = reference_asset()
    task = reference_task(asset)
    platform.register_asset(asset)
    platform.register_task(task)
    run = reference_run(asset, task, episode_count=4, base_seed=73)
    record = platform.execute(run)
    assert record.result is not None and record.result.dataset is not None

    training_path = tmp_path / "training.json"
    _write_json(
        training_path,
        TrainingSpec(
            resource_id="drawer.cli.bc",
            version="1.0.0",
            dataset=record.result.dataset,
            trainer="torch_bc",
            config={"device": "cpu", "epochs": 20, "seed": 13},
        ),
    )
    status, output, error = _invoke(
        capsys,
        ["train", str(training_path), "--workspace", str(workspace)],
    )
    assert status == 0
    assert not error
    model = VersionedRef.model_validate_json(output)
    assert model.resource_type is ResourceType.MODEL

    run_path = tmp_path / "evaluation-run.json"
    _write_json(run_path, run)
    status, output, error = _invoke(
        capsys,
        [
            "evaluate",
            model.resource_id,
            model.version,
            str(run_path),
            "--id",
            "drawer.cli.eval",
            "--version",
            "1.0.0",
            "--episodes",
            "1",
            "--workspace",
            str(workspace),
        ],
    )
    assert status == 0
    assert not error
    evaluation = VersionedRef.model_validate_json(output)
    assert evaluation.resource_type is ResourceType.EVALUATION
    assert platform.load_evaluation(evaluation).metrics["episode_count"] == 1.0
