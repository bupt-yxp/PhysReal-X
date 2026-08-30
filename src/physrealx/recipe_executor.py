"""Strict stage-wise execution for externally installed method recipes.

The recipe executor is intentionally smaller than a workflow engine.  It only
resolves an immutable directed chain, invokes already registered external
workers through :class:`~physrealx.domain.OperationSpec`, and returns an
aggregate receipt made exclusively from registered operation evidence.

Planned output names are never represented as artifacts.  A downstream input
becomes resolvable only after the producing worker returned a valid
``OperationManifest`` and the platform registered that manifest.  Consequently
a failed step may leave reusable, auditable prefix operations, but it can never
publish the terminal build operation or a successful recipe receipt.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated

from physrealx.domain import (
    DomainModel,
    JSONValue,
    MethodExecutionMode,
    MethodSpec,
    OperationInput,
    OperationManifest,
    OperationOutput,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
    VersionedRef,
    sha256_digest,
)
from physrealx.errors import (
    DomainValidationError,
    PluginError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from physrealx.external_workers import ExternalMethodWorker

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_VERSION = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def _identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{label} must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_' or '-'"
        )
    return value


def _semver(value: str) -> str:
    if _VERSION.fullmatch(value) is None:
        raise ValueError("version must be an explicit semantic version such as '1.0.0'")
    return value


class InitialRecipeInput(DomainModel):
    """Select one immutable input declared by :class:`RecipeExecutionSpec`."""

    kind: Literal["initial"] = "initial"
    input_name: str

    @field_validator("input_name")
    @classmethod
    def validate_input_name(cls, value: str) -> str:
        return _identifier(value, "initial input name")


class CompletedStepOutput(DomainModel):
    """Select one output from an earlier, successfully completed step."""

    kind: Literal["step_output"] = "step_output"
    step: str
    output: str

    @field_validator("step", "output")
    @classmethod
    def validate_names(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))


RecipeInputSource = Annotated[
    Union[InitialRecipeInput, CompletedStepOutput],
    Field(discriminator="kind"),
]


class RecipeInputBinding(DomainModel):
    """Bind one operation input name to explicit immutable evidence."""

    name: str
    contract: str = Field(min_length=1, max_length=127)
    source: RecipeInputSource

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "operation input name")


class RecipeStep(DomainModel):
    """One externally executed method with exact inputs and output contracts."""

    name: str
    method: MethodSpec
    inputs: tuple[RecipeInputBinding, ...] = ()
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    outputs: tuple[OperationOutputRequirement, ...]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "recipe step name")

    @field_validator("outputs")
    @classmethod
    def validate_outputs(
        cls,
        values: tuple[OperationOutputRequirement, ...],
    ) -> tuple[OperationOutputRequirement, ...]:
        if not values:
            raise ValueError("a recipe step requires at least one output")
        if len(values) != len({item.name for item in values}):
            raise ValueError("recipe step output names must be unique")
        return values

    @model_validator(mode="after")
    def validate_step(self) -> RecipeStep:
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError("recipe step input names must be unique")
        declared_inputs = tuple(self.method.descriptor.input_contract.split("+"))
        bound_inputs = tuple(item.contract for item in self.inputs)
        if bound_inputs != declared_inputs:
            raise ValueError(
                "recipe step inputs must exactly match the selected method input contract"
            )
        declared_contracts = tuple(self.method.descriptor.output_contract.split("+"))
        requested_contracts = tuple(item.contract for item in self.outputs)
        if sorted(requested_contracts) != sorted(declared_contracts):
            raise ValueError(
                "recipe step outputs must exactly match the selected method output contract"
            )
        if self.method.descriptor.runtime.mode not in {
            MethodExecutionMode.PROCESS,
            MethodExecutionMode.CONTAINER,
        }:
            raise ValueError("recipe steps must use process or container method workers")
        return self


class RecipeFinalOutput(DomainModel):
    """Rename one real terminal-step output into the aggregate receipt."""

    name: str
    source: CompletedStepOutput

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "recipe final output name")


class ExternalMethodRecipe(DomainModel):
    """A topologically ordered recipe with a single explicit terminal step."""

    schema_version: Literal[1] = 1
    name: str
    version: str
    steps: tuple[RecipeStep, ...]
    outputs: tuple[RecipeFinalOutput, ...]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "recipe name")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_graph(self) -> ExternalMethodRecipe:
        if not self.steps:
            raise ValueError("an external method recipe requires at least one step")
        step_names = [item.name for item in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("recipe step names must be unique")
        if not self.outputs:
            raise ValueError("an external method recipe requires at least one final output")
        output_names = [item.name for item in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError("recipe final output names must be unique")
        output_sources = [(item.source.step, item.source.output) for item in self.outputs]
        if len(output_sources) != len(set(output_sources)):
            raise ValueError("recipe final outputs must reference distinct terminal outputs")

        available: dict[tuple[str, str], str] = {}
        for step in self.steps:
            for binding in step.inputs:
                source = binding.source
                if isinstance(source, CompletedStepOutput):
                    key = (source.step, source.output)
                    if key not in available:
                        raise ValueError(
                            f"step {step.name!r} references unavailable prior output "
                            f"{source.step!r}:{source.output!r}"
                        )
                    if binding.contract != available[key]:
                        raise ValueError(
                            f"step {step.name!r} binds {source.step!r}:{source.output!r} "
                            f"as {binding.contract!r}, but the producer publishes "
                            f"{available[key]!r}"
                        )
            available.update(((step.name, output.name), output.contract) for output in step.outputs)

        terminal = self.steps[-1].name
        for output in self.outputs:
            key = (output.source.step, output.source.output)
            if key not in available:
                raise ValueError(
                    f"final output {output.name!r} references unavailable output "
                    f"{output.source.step!r}:{output.source.output!r}"
                )
            if output.source.step != terminal:
                raise ValueError("final outputs must come from the terminal recipe step")
        return self


class RecipeExecutionSpec(DomainModel):
    """Immutable execution identity, initial evidence and method recipe."""

    schema_version: Literal[1] = 1
    resource_id: str
    version: str
    recipe: ExternalMethodRecipe
    inputs: tuple[OperationInput, ...]

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "recipe execution resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_initial_inputs(self) -> RecipeExecutionSpec:
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError("recipe execution input names must be unique")
        declared = set(names)
        referenced = {
            binding.source.input_name
            for step in self.recipe.steps
            for binding in step.inputs
            if isinstance(binding.source, InitialRecipeInput)
        }
        missing = sorted(referenced - declared)
        if missing:
            raise ValueError(f"recipe references undeclared initial inputs: {missing}")
        unused = sorted(declared - referenced)
        if unused:
            raise ValueError(f"recipe declares unused initial inputs: {unused}")
        return self


class RecipeOperationRecord(DomainModel):
    """Resolved registry reference for one successful step operation."""

    step: str
    operation: VersionedRef

    @field_validator("step")
    @classmethod
    def validate_step(cls, value: str) -> str:
        return _identifier(value, "recipe operation step")

    @model_validator(mode="after")
    def validate_operation(self) -> RecipeOperationRecord:
        if self.operation.resource_type is not ResourceType.OPERATION:
            raise ValueError("recipe operation records require operation references")
        if not self.operation.resolved:
            raise ValueError("recipe operation records require resolved references")
        return self


class RecipeExecutionManifest(DomainModel):
    """Aggregate receipt created only after the terminal operation succeeds."""

    schema_version: Literal[1] = 1
    resource_id: str
    version: str
    spec: RecipeExecutionSpec
    operations: tuple[RecipeOperationRecord, ...]
    outputs: tuple[OperationOutput, ...]
    started_at: datetime
    completed_at: datetime

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "recipe execution resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> RecipeExecutionManifest:
        if (self.resource_id, self.version) != (
            self.spec.resource_id,
            self.spec.version,
        ):
            raise ValueError("recipe execution manifest identity must match its specification")
        expected_steps = tuple(item.name for item in self.spec.recipe.steps)
        actual_steps = tuple(item.step for item in self.operations)
        if actual_steps != expected_steps:
            raise ValueError("recipe operation records must exactly match recipe step order")
        expected_outputs = tuple(item.name for item in self.spec.recipe.outputs)
        actual_outputs = tuple(item.name for item in self.outputs)
        if actual_outputs != expected_outputs:
            raise ValueError("recipe outputs must exactly match final output declarations")
        terminal_contracts = {
            item.name: item.contract for item in self.spec.recipe.steps[-1].outputs
        }
        expected_contracts = tuple(
            terminal_contracts[item.source.output] for item in self.spec.recipe.outputs
        )
        actual_contracts = tuple(item.contract for item in self.outputs)
        if actual_contracts != expected_contracts:
            raise ValueError("recipe output contracts must match terminal step contracts")
        if len({item.artifact.digest for item in self.outputs}) != len(self.outputs):
            raise ValueError("recipe outputs must reference distinct artifacts")
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("recipe timestamps must include a UTC offset")
        if self.completed_at < self.started_at:
            raise ValueError("recipe completion cannot precede its start")
        return self

    @property
    def operation_refs(self) -> tuple[VersionedRef, ...]:
        return tuple(item.operation for item in self.operations)

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.RECIPE_EXECUTION,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class RecipeExecutionResult(DomainModel):
    """Resolved registry reference and typed receipt for a completed recipe."""

    execution: VersionedRef
    manifest: RecipeExecutionManifest

    @model_validator(mode="after")
    def validate_result(self) -> RecipeExecutionResult:
        if self.execution.resource_type is not ResourceType.RECIPE_EXECUTION:
            raise ValueError("recipe execution result requires a recipe-execution reference")
        if not self.execution.resolved:
            raise ValueError("recipe execution result requires a resolved reference")
        if (
            self.execution.resource_id,
            self.execution.version,
        ) != (self.manifest.resource_id, self.manifest.version):
            raise ValueError("recipe execution result identity must match its manifest")
        if self.execution.digest != sha256_digest(self.manifest):
            raise ValueError("recipe execution reference digest must match its manifest")
        return self

    @property
    def operations(self) -> tuple[RecipeOperationRecord, ...]:
        return self.manifest.operations

    @property
    def operation_refs(self) -> tuple[VersionedRef, ...]:
        return self.manifest.operation_refs

    @property
    def outputs(self) -> tuple[OperationOutput, ...]:
        return self.manifest.outputs


class RecipeExecutor:
    """Execute an :class:`ExternalMethodRecipe` through the platform kernel."""

    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    @staticmethod
    def operation_resource_id(spec: RecipeExecutionSpec, step: RecipeStep) -> str:
        """Return the deterministic, retry-safe identity of one recipe step."""

        identity = sha256_digest(
            {
                "execution": spec.model_dump(mode="json"),
                "step": step.name,
            }
        )[-24:]
        return f"operation.recipe.{identity}"

    def execute(self, spec: RecipeExecutionSpec) -> RecipeExecutionResult:
        """Run every step and return evidence only after the terminal step succeeds."""

        existing = self._existing(spec)
        if existing is not None:
            return existing
        self._preflight(spec)
        initial = {item.name: item for item in spec.inputs}
        completed: dict[tuple[str, str], OperationOutput] = {}
        manifests: list[OperationManifest] = []
        records: list[RecipeOperationRecord] = []

        for step in spec.recipe.steps:
            operation_inputs = tuple(
                self._resolve_binding(binding, initial=initial, completed=completed)
                for binding in step.inputs
            )
            operation = OperationSpec(
                resource_id=self.operation_resource_id(spec, step),
                version=spec.version,
                method=step.method,
                inputs=operation_inputs,
                arguments=step.arguments,
                outputs=step.outputs,
            )
            manifest = self._platform._execute_operation(operation)
            if manifest.spec != operation:
                raise DomainValidationError(
                    f"recipe step {step.name!r} returned evidence for another operation"
                )
            if any(
                not self._platform.artifacts.verify(output.artifact) for output in manifest.outputs
            ):
                raise DomainValidationError(
                    f"recipe step {step.name!r} references unavailable output artifacts"
                )
            reference = self._registered_operation_reference(manifest)
            manifests.append(manifest)
            records.append(RecipeOperationRecord(step=step.name, operation=reference))
            completed.update(((step.name, output.name), output) for output in manifest.outputs)

        final_outputs = tuple(
            self._final_output(declaration, completed=completed)
            for declaration in spec.recipe.outputs
        )
        recipe_manifest = RecipeExecutionManifest(
            resource_id=spec.resource_id,
            version=spec.version,
            spec=spec,
            operations=tuple(records),
            outputs=final_outputs,
            started_at=min(item.started_at for item in manifests),
            completed_at=max(item.completed_at for item in manifests),
        )
        reference = self._platform._register(
            recipe_manifest,
            recipe_manifest.reference(),
            "application/vnd.physrealx.recipe-execution+json",
        )
        return RecipeExecutionResult(execution=reference, manifest=recipe_manifest)

    def _existing(self, spec: RecipeExecutionSpec) -> RecipeExecutionResult | None:
        try:
            reference = self._platform.repository.get_resource(
                ResourceType.RECIPE_EXECUTION.value,
                spec.resource_id,
                spec.version,
            )
        except ResourceNotFoundError:
            return None
        manifest = self._platform._load_registered(reference, RecipeExecutionManifest)
        if manifest.spec != spec:
            raise ResourceConflictError(
                "recipe execution identity already exists for a different specification"
            )
        if any(not self._platform.artifacts.verify(output.artifact) for output in manifest.outputs):
            raise DomainValidationError(
                "registered recipe execution references unavailable output artifacts"
            )
        for operation in manifest.operation_refs:
            self._platform._load_registered(operation, OperationManifest)
        return RecipeExecutionResult(execution=reference, manifest=manifest)

    def _preflight(self, spec: RecipeExecutionSpec) -> None:
        for item in spec.inputs:
            if item.resource is not None:
                registered = self._platform.repository.get_resource(
                    item.resource.resource_type.value,
                    item.resource.resource_id,
                    item.resource.version,
                )
                if registered.digest != item.resource.digest:
                    raise DomainValidationError(
                        f"recipe input {item.name!r} does not match registered evidence"
                    )
                if registered.digest is None or not self._platform.artifacts.verify(
                    registered.digest
                ):
                    raise DomainValidationError(
                        f"recipe input resource {item.name!r} is unavailable or corrupt"
                    )
            elif item.artifact is None or not self._platform.artifacts.verify(item.artifact):
                raise DomainValidationError(
                    f"recipe input artifact {item.name!r} is unavailable or corrupt"
                )

        for step in spec.recipe.steps:
            self._platform.plugins.resolve_method(step.method)
            worker = self._platform.plugins.worker(step.method)
            if not isinstance(worker, ExternalMethodWorker):
                raise PluginError(f"recipe step {step.name!r} must resolve to ExternalMethodWorker")

    @staticmethod
    def _resolve_binding(
        binding: RecipeInputBinding,
        *,
        initial: dict[str, OperationInput],
        completed: dict[tuple[str, str], OperationOutput],
    ) -> OperationInput:
        source = binding.source
        if isinstance(source, InitialRecipeInput):
            evidence = initial[source.input_name]
            return evidence.model_copy(update={"name": binding.name})
        try:
            output = completed[(source.step, source.output)]
        except KeyError as error:  # Defensive: the schema already rejects forward references.
            raise DomainValidationError(
                f"recipe input {binding.name!r} has no completed producing operation"
            ) from error
        return OperationInput(name=binding.name, artifact=output.artifact)

    def _registered_operation_reference(self, manifest: OperationManifest) -> VersionedRef:
        registered = self._platform.repository.get_resource(
            ResourceType.OPERATION.value,
            manifest.resource_id,
            manifest.version,
        )
        loaded = self._platform._load_registered(registered, OperationManifest)
        if loaded != manifest:
            raise DomainValidationError("registered operation evidence changed after execution")
        return registered

    @staticmethod
    def _final_output(
        declaration: RecipeFinalOutput,
        *,
        completed: dict[tuple[str, str], OperationOutput],
    ) -> OperationOutput:
        source = declaration.source
        try:
            output = completed[(source.step, source.output)]
        except KeyError as error:  # Defensive: only real completed outputs are accepted.
            raise DomainValidationError(
                f"final output {declaration.name!r} was not produced"
            ) from error
        return OperationOutput(
            name=declaration.name,
            contract=output.contract,
            artifact=output.artifact,
        )


__all__ = [
    "CompletedStepOutput",
    "ExternalMethodRecipe",
    "InitialRecipeInput",
    "RecipeExecutionManifest",
    "RecipeExecutionResult",
    "RecipeExecutionSpec",
    "RecipeExecutor",
    "RecipeFinalOutput",
    "RecipeInputBinding",
    "RecipeOperationRecord",
    "RecipeStep",
]
