"""Validate actual adapter output bytes before publishing operation evidence."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from physrealx.contracts import ArtifactStore
from physrealx.domain import OperationOutput, OperationOutputRequirement
from physrealx.errors import MethodExecutionError
from physrealx.evidence import strict_json_object


def validate_operation_output(
    store: ArtifactStore, output: OperationOutput, requirement: OperationOutputRequirement
) -> None:
    reference = output.artifact
    if requirement.media_types and reference.media_type not in requirement.media_types:
        raise MethodExecutionError(f"method output {output.name!r} uses an unexpected media type")
    if requirement.max_size_bytes is not None and reference.size_bytes > requirement.max_size_bytes:
        raise MethodExecutionError(f"method output {output.name!r} exceeds its declared size limit")
    kind = requirement.content_kind
    if kind == "json" or requirement.schema_discriminator is not None:
        try:
            document = strict_json_object(store.get_bytes(reference), output.name)
        except ValueError as error:
            raise MethodExecutionError(f"method output {output.name!r} is invalid JSON") from error
        if (
            requirement.schema_discriminator
            and document.get("schema") != requirement.schema_discriminator
        ):
            raise MethodExecutionError(
                f"method output {output.name!r} has the wrong schema discriminator"
            )
        if requirement.schema_discriminator == "physrealx.geometry-reconstruction-report/1":
            from physrealx.reconstruction import ReconstructionPrediction

            try:
                ReconstructionPrediction.model_validate(document.get("prediction"))
                if document.get("unit") != "m" or not document.get("coordinate_frame"):
                    raise ValueError("reconstruction requires metre geometry and a capture frame")
            except ValueError as error:
                raise MethodExecutionError(
                    f"method output {output.name!r} violates the reconstruction contract: {error}"
                ) from error
    elif kind == "zip":
        with tempfile.TemporaryDirectory(prefix="physrealx-validate-") as directory:
            path = Path(directory) / "output.zip"
            store.materialize(reference, path)
            try:
                with zipfile.ZipFile(path) as archive:
                    entries = archive.infolist()
                    names = [entry.filename for entry in entries]
                    if not entries or len(set(names)) != len(names):
                        raise ValueError("archive is empty or contains duplicate paths")
                    for entry in entries:
                        name = PurePosixPath(entry.filename)
                        if (
                            name.is_absolute()
                            or ".." in name.parts
                            or "\\" in entry.filename
                            or (entry.external_attr >> 16) & 0o170000 == 0o120000
                        ):
                            raise ValueError("unsafe archive member")
                    limit = requirement.max_size_bytes or 32 * 1024**3
                    if sum(entry.file_size for entry in entries) > limit:
                        raise ValueError("uncompressed archive exceeds its output size limit")
                    if archive.testzip() is not None:
                        raise ValueError("archive CRC validation failed")
            except (ValueError, zipfile.BadZipFile, RuntimeError) as error:
                raise MethodExecutionError(
                    f"method output {output.name!r} is not a valid bounded ZIP"
                ) from error
