from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.domain import (
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationOutputRequirement,
    ResourceType,
    sha256_digest,
)
from physrealx.errors import DomainValidationError, MethodExecutionError, ResourceNotFoundError
from physrealx.external_workers import ExternalMethodWorker, ProcessLaunch
from physrealx.methods import builtin_method
from physrealx.platform import PhysRealX
from physrealx.recipe_executor import (
    CompletedStepOutput,
    ExternalMethodRecipe,
    InitialRecipeInput,
    RecipeExecutionSpec,
    RecipeExecutor,
    RecipeFinalOutput,
    RecipeInputBinding,
    RecipeStep,
)


def _external_descriptor(
    capability: MethodCapability,
    name: str,
    digest: str,
    *,
    input_contract: str = "Artifact/1",
    output_contract: str,
) -> MethodDescriptor:
    return MethodDescriptor(
        capability=capability,
        name=name,
        version="1.0.0",
        api_version="1",
        source_uri="https://example.com/physrealx-recipe-test-worker",
        source_revision="v1.0.0",
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            python=">=3.8,<3.13",
            environment_digest=digest,
        ),
        input_contract=input_contract,
        output_contract=output_contract,
    )


def _write_worker(path: Path, *, fail: bool = False) -> None:
    path.write_text(
        f"""\
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--response", required=True)
arguments = parser.parse_args()
request_path = Path(arguments.request)
request_bytes = request_path.read_bytes()
request = json.loads(request_bytes)
if {fail!r}:
    raise SystemExit(23)
root = request_path.parent
source = root / request["inputs"][0]["path"]
source_bytes = source.read_bytes()
outputs = []
for index, requirement in enumerate(request["operation"]["outputs"]):
    output = Path(arguments.response).parent / (str(index) + ".json")
    output.write_text(
        json.dumps(
            {{
                "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "method": request["operation"]["method"]["descriptor"]["name"],
            }}
        ),
        encoding="utf-8",
    )
    outputs.append(
        {{
            "name": requirement["name"],
            "contract": requirement["contract"],
            "path": output.name,
            "media_type": "application/json",
        }}
    )
descriptor_bytes = json.dumps(
    request["operation"]["method"]["descriptor"],
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
response = {{
    "protocol_version": "2",
    "request_digest": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
    "descriptor_digest": "sha256:" + hashlib.sha256(descriptor_bytes).hexdigest(),
    "outputs": outputs,
    "metrics": {{"ran": 1.0}},
}}
Path(arguments.response).write_text(json.dumps(response), encoding="utf-8")
""",
        encoding="utf-8",
    )


def _install_worker(
    platform: PhysRealX,
    root: Path,
    *,
    capability: MethodCapability,
    name: str,
    input_contract: str,
    output_contract: str,
    fail: bool = False,
) -> MethodDescriptor:
    script = root / f"{name}.py"
    _write_worker(script, fail=fail)
    runtime_files = (Path(sys.executable).resolve(), script.resolve())
    environment = root / f"{name}.environment.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "2",
                "runtime_files": [
                    {"path": str(path), "sha256": sha256_digest(path.read_bytes())}
                    for path in runtime_files
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = _external_descriptor(
        capability,
        name,
        sha256_digest(environment.read_bytes()),
        input_contract=input_contract,
        output_contract=output_contract,
    )
    platform.plugins.register_worker(
        ExternalMethodWorker(
            descriptor,
            ProcessLaunch(
                command=(str(runtime_files[0]), str(runtime_files[1])),
                environment_manifest=environment,
            ),
        )
    )
    return descriptor


def _spec(
    source: ArtifactRef,
    first: MethodDescriptor,
    terminal: MethodDescriptor,
) -> RecipeExecutionSpec:
    return RecipeExecutionSpec(
        resource_id="cabinet.recipe",
        version="1.0.0",
        inputs=(OperationInput(name="capture", artifact=source),),
        recipe=ExternalMethodRecipe(
            name="cabinet.digital-twin",
            version="1.0.0",
            steps=(
                RecipeStep(
                    name="prepare",
                    method=first.bind(),
                    inputs=(
                        RecipeInputBinding(
                            name="source",
                            contract=first.input_contract,
                            source=InitialRecipeInput(input_name="capture"),
                        ),
                    ),
                    outputs=(
                        OperationOutputRequirement(
                            name="prepared",
                            contract="PreparedCapture/1",
                        ),
                    ),
                ),
                RecipeStep(
                    name="package",
                    method=terminal.bind(),
                    inputs=(
                        RecipeInputBinding(
                            name="prepared",
                            contract=terminal.input_contract,
                            source=CompletedStepOutput(step="prepare", output="prepared"),
                        ),
                    ),
                    outputs=(
                        OperationOutputRequirement(
                            name="asset_package",
                            contract="PackagedAsset/1",
                        ),
                    ),
                ),
            ),
            outputs=(
                RecipeFinalOutput(
                    name="build",
                    source=CompletedStepOutput(
                        step="package",
                        output="asset_package",
                    ),
                ),
            ),
        ),
    )


def test_recipe_rejects_forward_references_and_in_process_methods() -> None:
    in_process = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "in_process",
        input_contract="Artifact/1",
        output_contract="Value/1",
    ).bind()
    with pytest.raises(ValidationError, match="process or container"):
        RecipeStep(
            name="invalid",
            method=in_process,
            inputs=(
                RecipeInputBinding(
                    name="artifact",
                    contract="Artifact/1",
                    source=InitialRecipeInput(input_name="artifact"),
                ),
            ),
            outputs=(OperationOutputRequirement(name="value", contract="Value/1"),),
        )

    digest = f"sha256:{'a' * 64}"
    descriptor = _external_descriptor(
        MethodCapability.COLLISION_BUILD,
        "external",
        digest,
        output_contract="Value/1",
    )
    with pytest.raises(ValidationError, match="unavailable prior output"):
        ExternalMethodRecipe(
            name="invalid-graph",
            version="1.0.0",
            steps=(
                RecipeStep(
                    name="first",
                    method=descriptor.bind(),
                    inputs=(
                        RecipeInputBinding(
                            name="future",
                            contract="Artifact/1",
                            source=CompletedStepOutput(step="second", output="value"),
                        ),
                    ),
                    outputs=(OperationOutputRequirement(name="value", contract="Value/1"),),
                ),
            ),
            outputs=(
                RecipeFinalOutput(
                    name="value",
                    source=CompletedStepOutput(step="first", output="value"),
                ),
            ),
        )


def test_recipe_rejects_input_contract_relabeling() -> None:
    descriptor = _external_descriptor(
        MethodCapability.ASSET_PACKAGE,
        "external",
        f"sha256:{'a' * 64}",
        input_contract="AssetAssemblyBundle/1",
        output_contract="UsdAssetBundle/1",
    )

    with pytest.raises(ValidationError, match=r"exactly match.*input contract"):
        RecipeStep(
            name="package",
            method=descriptor.bind(),
            inputs=(
                RecipeInputBinding(
                    name="asset",
                    contract="IdentifiedJointPhysics/1",
                    source=InitialRecipeInput(input_name="asset"),
                ),
            ),
            outputs=(
                OperationOutputRequirement(
                    name="usd",
                    contract="UsdAssetBundle/1",
                ),
            ),
        )


def test_recipe_rejects_incompatible_upstream_output_contract() -> None:
    digest = f"sha256:{'a' * 64}"
    producer = _external_descriptor(
        MethodCapability.ASSET_NORMALIZE,
        "producer",
        digest,
        input_contract="Capture/1",
        output_contract="PreparedCapture/1",
    )
    consumer = _external_descriptor(
        MethodCapability.ASSET_PACKAGE,
        "consumer",
        digest,
        input_contract="AssetAssemblyBundle/1",
        output_contract="UsdAssetBundle/1",
    )

    with pytest.raises(ValidationError, match="producer publishes"):
        ExternalMethodRecipe(
            name="invalid-contract-edge",
            version="1.0.0",
            steps=(
                RecipeStep(
                    name="prepare",
                    method=producer.bind(),
                    inputs=(
                        RecipeInputBinding(
                            name="capture",
                            contract="Capture/1",
                            source=InitialRecipeInput(input_name="capture"),
                        ),
                    ),
                    outputs=(
                        OperationOutputRequirement(
                            name="prepared",
                            contract="PreparedCapture/1",
                        ),
                    ),
                ),
                RecipeStep(
                    name="package",
                    method=consumer.bind(),
                    inputs=(
                        RecipeInputBinding(
                            name="asset",
                            contract="AssetAssemblyBundle/1",
                            source=CompletedStepOutput(
                                step="prepare",
                                output="prepared",
                            ),
                        ),
                    ),
                    outputs=(
                        OperationOutputRequirement(
                            name="usd",
                            contract="UsdAssetBundle/1",
                        ),
                    ),
                ),
            ),
            outputs=(
                RecipeFinalOutput(
                    name="usd",
                    source=CompletedStepOutput(step="package", output="usd"),
                ),
            ),
        )


def test_recipe_rejects_undeclared_or_unused_initial_evidence() -> None:
    digest = f"sha256:{'a' * 64}"
    descriptor = _external_descriptor(
        MethodCapability.COLLISION_BUILD,
        "external",
        digest,
        output_contract="Asset/1",
    )
    recipe = ExternalMethodRecipe(
        name="one-step",
        version="1.0.0",
        steps=(
            RecipeStep(
                name="build",
                method=descriptor.bind(),
                inputs=(
                    RecipeInputBinding(
                        name="source",
                        contract="Artifact/1",
                        source=InitialRecipeInput(input_name="capture"),
                    ),
                ),
                outputs=(OperationOutputRequirement(name="asset", contract="Asset/1"),),
            ),
        ),
        outputs=(
            RecipeFinalOutput(
                name="asset",
                source=CompletedStepOutput(step="build", output="asset"),
            ),
        ),
    )
    artifact = ArtifactRef(
        digest=f"sha256:{'b' * 64}",
        size_bytes=1,
        media_type="application/octet-stream",
    )
    with pytest.raises(ValidationError, match="undeclared initial"):
        RecipeExecutionSpec(
            resource_id="missing",
            version="1.0.0",
            recipe=recipe,
            inputs=(OperationInput(name="different", artifact=artifact),),
        )


def test_executor_runs_registered_external_steps_and_records_real_operations(
    tmp_path: Path,
) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    first = _install_worker(
        platform,
        tmp_path,
        capability=MethodCapability.ASSET_NORMALIZE,
        name="prepare_worker",
        input_contract="CapturedEvidence/1",
        output_contract="PreparedCapture/1",
    )
    terminal = _install_worker(
        platform,
        tmp_path,
        capability=MethodCapability.ASSET_PACKAGE,
        name="package_worker",
        input_contract="PreparedCapture/1",
        output_contract="PackagedAsset/1",
    )
    source = platform.artifacts.put_json({"capture": "cabinet2"})
    spec = _spec(source, first, terminal)

    result = RecipeExecutor(platform).execute(spec)

    assert result.execution.resource_type is ResourceType.RECIPE_EXECUTION
    assert result.execution.resolved
    assert (
        platform.repository.get_resource(
            ResourceType.RECIPE_EXECUTION.value,
            spec.resource_id,
            spec.version,
        )
        == result.execution
    )
    assert tuple(item.step for item in result.operations) == ("prepare", "package")
    assert all(item.operation.resolved for item in result.operations)
    assert all(item.operation.resource_type is ResourceType.OPERATION for item in result.operations)
    manifests = tuple(platform.operations.get(item) for item in result.operation_refs)
    assert tuple(item.spec.method.name for item in manifests) == (
        "prepare_worker",
        "package_worker",
    )
    assert manifests[1].spec.inputs[0].artifact == manifests[0].outputs[0].artifact
    assert result.outputs[0].artifact == manifests[1].outputs[0].artifact
    assert platform.artifacts.verify(result.outputs[0].artifact)

    # Retry reuses the immutable registered operations and yields the same receipt.
    assert RecipeExecutor(platform).execute(spec) == result


def test_failed_prefix_never_registers_terminal_build_operation(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    first = _install_worker(
        platform,
        tmp_path,
        capability=MethodCapability.ASSET_NORMALIZE,
        name="prepare_worker",
        input_contract="CapturedEvidence/1",
        output_contract="PreparedCapture/1",
    )
    terminal = _install_worker(
        platform,
        tmp_path,
        capability=MethodCapability.ASSET_PACKAGE,
        name="failing_package_worker",
        input_contract="PreparedCapture/1",
        output_contract="PackagedAsset/1",
        fail=True,
    )
    spec = _spec(
        platform.artifacts.put_bytes(
            b"captured evidence",
            media_type="application/octet-stream",
        ),
        first,
        terminal,
    )
    executor = RecipeExecutor(platform)

    with pytest.raises(MethodExecutionError):
        executor.execute(spec)

    assert platform.repository.list_resources(resource_type="recipe_execution") == []
    registered = platform.operations.list()
    assert len(registered) == 1
    assert registered[0].resource_id == executor.operation_resource_id(
        spec,
        spec.recipe.steps[0],
    )
    with pytest.raises(ResourceNotFoundError):
        platform.repository.get_resource(
            ResourceType.OPERATION.value,
            executor.operation_resource_id(spec, spec.recipe.steps[-1]),
            spec.version,
        )


def test_unavailable_initial_artifact_blocks_before_any_step(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    first = _install_worker(
        platform,
        tmp_path,
        capability=MethodCapability.ASSET_NORMALIZE,
        name="prepare_worker",
        input_contract="CapturedEvidence/1",
        output_contract="PreparedCapture/1",
    )
    terminal = _install_worker(
        platform,
        tmp_path,
        capability=MethodCapability.ASSET_PACKAGE,
        name="package_worker",
        input_contract="PreparedCapture/1",
        output_contract="PackagedAsset/1",
    )
    missing = ArtifactRef(
        digest=f"sha256:{'f' * 64}",
        size_bytes=1,
        media_type="application/octet-stream",
    )

    with pytest.raises(DomainValidationError, match="unavailable or corrupt"):
        RecipeExecutor(platform).execute(_spec(missing, first, terminal))
    assert platform.operations.list() == []
    assert platform.repository.list_resources(resource_type="recipe_execution") == []
