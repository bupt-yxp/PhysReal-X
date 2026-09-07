from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from physrealx.domain import OperationOutput, OperationOutputRequirement
from physrealx.errors import MethodExecutionError
from physrealx.operation_validation import validate_operation_output
from physrealx.storage import LocalArtifactStore


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema":"expected","x":NaN}',
        b'{"schema":"expected","x":1e999}',
        b'{"schema":"expected","x":1,"x":2}',
        b'{"schema":"wrong"}',
    ],
)
def test_rejects_ambiguous_nonfinite_or_mislabelled_operation_json(
    tmp_path: Path, payload: bytes
) -> None:
    store = LocalArtifactStore(tmp_path)
    output = OperationOutput(
        name="report",
        contract="Report/1",
        artifact=store.put_bytes(payload, media_type="application/json"),
    )
    requirement = OperationOutputRequirement(
        name="report", contract="Report/1", content_kind="json", schema_discriminator="expected"
    )
    with pytest.raises(MethodExecutionError):
        validate_operation_output(store, output, requirement)


def test_rejects_traversal_archive_before_publication(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("../escape", b"payload")
    output = OperationOutput(
        name="bundle",
        contract="Bundle/1",
        artifact=store.put_bytes(stream.getvalue(), media_type="application/zip"),
    )
    with pytest.raises(MethodExecutionError, match="bounded ZIP"):
        validate_operation_output(
            store,
            output,
            OperationOutputRequirement(name="bundle", contract="Bundle/1", content_kind="zip"),
        )
