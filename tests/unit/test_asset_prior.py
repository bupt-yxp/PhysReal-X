"""Version identity tests for refined physical assets."""

from physrealx.integrations.asset_prior import _refined_asset_version
from physrealx.storage import LocalArtifactStore


def test_refined_asset_version_is_evidence_addressed_and_idempotent(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    first_evidence = artifacts.put_json({"evidence": 1})
    duplicate_evidence = artifacts.put_json({"evidence": 1})
    changed_evidence = artifacts.put_json({"evidence": 2})
    stages = (("collision-proxy", first_evidence),)

    first = _refined_asset_version("1.0.0", stages)

    assert first.startswith("1.0.1+evidence.")
    assert _refined_asset_version("1.0.0", stages) == first
    assert _refined_asset_version("1.0.0", (("collision-proxy", duplicate_evidence),)) != first
    assert _refined_asset_version("1.0.0", (("collision-proxy", changed_evidence),)) != first
    assert _refined_asset_version(
        "1.0.0",
        (
            ("video-articulation", first_evidence),
            ("collision-proxy", changed_evidence),
        ),
    ).startswith("1.0.2+evidence.")


def test_unrefined_asset_keeps_declared_version():
    assert _refined_asset_version("1.2.3+capture.source", ()) == "1.2.3+capture.source"
