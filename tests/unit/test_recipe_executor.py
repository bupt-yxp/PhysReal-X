from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.contracts import ArtifactStore, MethodExecutionOutput
from physrealx.domain import (
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    OperationInput,
    OperationOutput,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
)
from physrealx.errors import DomainValidationError, MethodExecutionError, ResourceNotFoundError
from physrealx.methods import builtin_method
from physrealx.platform import PhysRealX
from physrealx.recipe_executor import (
    CompletedStepOutput,
    InitialRecipeInput,
    MethodRecipe,
    RecipeExecutionSpec,
    RecipeExecutor,
    RecipeFinalOutput,
    RecipeInputBinding,
    RecipeStep,
)


def _descriptor(
    capability: MethodCapability,
    name: str,
    *,
    input_contract: str,
    output_contract: str,
) -> MethodDescriptor:
    return builtin_method(
        capability,
        name,
        input_contract=input_contract,
        output_contract=output_contract,
    )


@dataclass(frozen=True)
class _JsonAdapter:
    descriptor: MethodDescriptor
    fail: bool = False

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if self.fail:
            raise MethodExecutionError(f"{self.descriptor.name} failed")
        source = spec.inputs[0]
        if source.artifact is None:
            raise DomainValidationError("fixture adapter requires one artifact input")
        payload = {
            "method": self.descriptor.name,
            "source_artifact_id": source.artifact.artifact_id,
        }
        outputs = tuple(
            OperationOutput(
                name=requirement.name,
                contract=requirement.contract,
                artifact=artifacts.put_json(payload),
            )
            for requirement in spec.outputs
        )
        return MethodExecutionOutput(outputs=outputs, metrics={"ran": 1.0})


def _register(
    platform: PhysRealX,
    capability: MethodCapability,
    name: str,
    *,
    input_contract: str,
    output_contract: str,
    fail: bool = False,
) -> MethodDescriptor:
    descriptor = _descriptor(
        capability,
        name,
        input_contract=input_contract,
        output_contract=output_contract,
    )
    platform.plugins.register_adapter(_JsonAdapter(descriptor=descriptor, fail=fail))
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
        recipe=MethodRecipe(
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
                    source=CompletedStepOutput(step="package", output="asset_package"),
                ),
            ),
        ),
    )


def test_recipe_rejects_forward_references() -> None:
    descriptor = _descriptor(
        MethodCapability.COLLISION_BUILD,
        "embedded",
        input_contract="Artifact/1",
        output_contract="Value/1",
    )
    with pytest.raises(ValidationError, match="unavailable prior output"):
        MethodRecipe(
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
    descriptor = _descriptor(
        MethodCapability.ASSET_PACKAGE,
        "embedded",
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
            outputs=(OperationOutputRequirement(name="usd", contract="UsdAssetBundle/1"),),
        )


def test_executor_runs_registered_in_process_steps_and_records_operations(
    tmp_path: Path,
) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    first = _register(
        platform,
        MethodCapability.ASSET_NORMALIZE,
        "prepare_adapter",
        input_contract="CapturedEvidence/1",
        output_contract="PreparedCapture/1",
    )
    terminal = _register(
        platform,
        MethodCapability.ASSET_PACKAGE,
        "package_adapter",
        input_contract="PreparedCapture/1",
        output_contract="PackagedAsset/1",
    )
    spec = _spec(platform.artifacts.put_json({"capture": "cabinet2"}), first, terminal)

    result = RecipeExecutor(platform).execute(spec)

    assert result.execution.resource_type is ResourceType.RECIPE_EXECUTION
    assert result.execution.resolved
    assert tuple(item.step for item in result.operations) == ("prepare", "package")
    manifests = tuple(platform.operations.get(item) for item in result.operation_refs)
    assert manifests[1].spec.inputs[0].artifact == manifests[0].outputs[0].artifact
    assert result.outputs[0].artifact == manifests[1].outputs[0].artifact
    assert platform.artifacts.verify(result.outputs[0].artifact)
    assert RecipeExecutor(platform).execute(spec) == result


def test_failed_prefix_never_registers_terminal_operation(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    first = _register(
        platform,
        MethodCapability.ASSET_NORMALIZE,
        "prepare_adapter",
        input_contract="CapturedEvidence/1",
        output_contract="PreparedCapture/1",
    )
    terminal = _register(
        platform,
        MethodCapability.ASSET_PACKAGE,
        "failing_package_adapter",
        input_contract="PreparedCapture/1",
        output_contract="PackagedAsset/1",
        fail=True,
    )
    spec = _spec(
        platform.artifacts.put_bytes(b"capture", media_type="application/octet-stream"),
        first,
        terminal,
    )
    executor = RecipeExecutor(platform)

    with pytest.raises(MethodExecutionError):
        executor.execute(spec)

    assert platform.repository.list_resources(resource_type="recipe_execution") == []
    assert len(platform.operations.list()) == 1
    with pytest.raises(ResourceNotFoundError):
        platform.repository.get_resource(
            ResourceType.OPERATION.value,
            executor.operation_resource_id(spec, spec.recipe.steps[-1]),
            spec.version,
        )


def test_unavailable_initial_artifact_blocks_before_any_step(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    first = _register(
        platform,
        MethodCapability.ASSET_NORMALIZE,
        "prepare_adapter",
        input_contract="CapturedEvidence/1",
        output_contract="PreparedCapture/1",
    )
    terminal = _register(
        platform,
        MethodCapability.ASSET_PACKAGE,
        "package_adapter",
        input_contract="PreparedCapture/1",
        output_contract="PackagedAsset/1",
    )
    missing = ArtifactRef(
        artifact_id=f"art_{'f' * 32}",
        size_bytes=1,
        media_type="application/octet-stream",
    )

    with pytest.raises(DomainValidationError, match="unavailable or corrupt"):
        RecipeExecutor(platform).execute(_spec(missing, first, terminal))
    assert platform.operations.list() == []
