"""Published demo media must retain their exported content and provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("demo_gallery", ROOT / "docs/_hooks/demo_gallery.py")
assert SPEC is not None and SPEC.loader is not None
GALLERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GALLERY)


def fixture_manifest(tmp_path: Path) -> tuple[Path, dict]:
    media = tmp_path / "preview.gif"
    media.write_bytes(b"saved-demo")
    demos = GALLERY.demo_catalog()
    for demo in demos:
        guide = tmp_path / demo.guide
        guide.parent.mkdir(parents=True, exist_ok=True)
        guide.write_text("# Demo")
    manifest = {
        "schema": "physrealx.website-demo-snapshots/1",
        "demos": [
            {
                "slug": demo.slug,
                "title": demo.title,
                "guide": demo.guide,
                "dependencies": list(demo.dependencies),
                "media": [
                    {"path": "preview.gif", "sha256": GALLERY.sha256(media), "size_bytes": 10}
                ],
            }
            for demo in demos
        ],
    }
    path = tmp_path / GALLERY.MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_published_snapshots_match_catalog_and_content() -> None:
    from examples.catalog import DEMOS

    manifest = GALLERY.validate_manifest(ROOT / "docs")
    assert [(d["slug"], d["title"], d["guide"]) for d in manifest["demos"]] == [
        (d.slug, d.title, d.guide) for d in DEMOS
    ]


def test_changed_media_is_rejected(tmp_path: Path) -> None:
    fixture_manifest(tmp_path)
    (tmp_path / "preview.gif").write_bytes(b"other-demo")
    with pytest.raises(ValueError, match="without refreshing provenance"):
        GALLERY.validate_manifest(tmp_path)


def test_media_outside_docs_is_rejected(tmp_path: Path) -> None:
    path, manifest = fixture_manifest(tmp_path)
    manifest["demos"][0]["media"][0]["path"] = "../outside.gif"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="must stay inside"):
        GALLERY.validate_manifest(tmp_path)


def test_repeated_module_is_rejected(tmp_path: Path) -> None:
    path, manifest = fixture_manifest(tmp_path)
    manifest["demos"].append(manifest["demos"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="repeated demo"):
        GALLERY.validate_manifest(tmp_path)


@pytest.mark.parametrize("change", ["missing", "order", "title", "guide", "dependencies"])
def test_catalog_drift_is_rejected(tmp_path: Path, change: str) -> None:
    path, manifest = fixture_manifest(tmp_path)
    demos = manifest["demos"]
    if change == "missing":
        demos.pop()
    elif change == "order":
        demos.reverse()
    elif change == "dependencies":
        demos[1]["dependencies"] = []
    else:
        demos[0][change] = "outdated"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"differs from examples\.catalog"):
        GALLERY.validate_manifest(tmp_path)


def completed_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    import base64

    from examples import run_all

    docs = tmp_path / "docs"
    snapshots = docs / GALLERY.MANIFEST
    snapshots.parent.mkdir(parents=True)
    snapshots.write_text("previous published snapshots")
    report_path = tmp_path / "examples/output/notebook-acceptance.json"
    report_path.parent.mkdir(parents=True)
    report_path.with_name(".notebook-acceptance.lock").touch()
    receipts = {}
    for demo, selection in zip(GALLERY.demo_catalog(), GALLERY.SNAPSHOTS):
        guide = docs / demo.guide
        guide.parent.mkdir(parents=True, exist_ok=True)
        guide.write_text("# Demo")
        directory = tmp_path / "examples" / demo.slug / "output"
        directory.mkdir(parents=True)
        results, outputs = {}, []
        for filename, key in selection:
            payload = (demo.slug + filename).encode()
            mime = "image/gif" if filename.endswith(".gif") else "image/png"
            (directory / filename).write_bytes(payload)
            outputs.append({"data": {mime: base64.b64encode(payload).decode()}})
            if key:
                results[key] = {"size_bytes": len(payload), "media_type": mime}
        (directory / "results.json").write_text(json.dumps(results))
        (directory / "demo.executed.ipynb").write_text(
            json.dumps({"cells": [{"outputs": outputs}]})
        )
        receipt = {
            "schema": "physrealx.notebook-execution/3",
            "module": demo.slug,
            "status": "COMPLETED",
            "finished_at": "2026-09-21T00:00:00+00:00",
            "source_sha256": "0" * 64,
            "implementation_sha256": "1" * 64,
            "results_sha256": GALLERY.sha256(directory / "results.json"),
            "executed_sha256": GALLERY.sha256(directory / "demo.executed.ipynb"),
            "upstream_results": {
                slug: receipts[slug]["results_sha256"] for slug in demo.dependencies
            },
        }
        (directory / "execution.json").write_text(json.dumps(receipt))
        receipts[demo.slug] = receipt
    report_path.write_text(
        json.dumps(
            {
                "schema": "physrealx.notebook-acceptance/3",
                "execution_status": "COMPLETED",
                "accepted_modules": list(receipts),
                "notebooks": list(receipts.values()),
            }
        )
    )
    monkeypatch.setattr(
        run_all, "_load_current_receipt", lambda repo, slug, **kwargs: receipts[slug]
    )
    return report_path, receipts


def test_refresh_binds_exports_to_completed_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed_exports(tmp_path, monkeypatch)
    GALLERY.refresh(tmp_path)
    manifest = GALLERY.validate_manifest(tmp_path / "docs")
    assert manifest["verification"] == "completed_execution_and_embedded_media"
    assert all(d["execution"]["executed_sha256"] for d in manifest["demos"])


def test_refresh_rejects_same_size_stale_gif_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed_exports(tmp_path, monkeypatch)
    path = tmp_path / "examples/scene_and_task_generation/output/task-preview-home.gif"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="not displayed by this execution"):
        GALLERY.refresh(tmp_path)
    assert (tmp_path / "docs" / GALLERY.MANIFEST).read_text() == "previous published snapshots"
    assert not list((tmp_path / "docs").rglob("*.gif"))


@pytest.mark.parametrize("status", ["RUNNING", "FAILED"])
def test_refresh_rejects_incomplete_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    report_path, _ = completed_exports(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text())
    report["execution_status"] = status
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed notebook acceptance"):
        GALLERY.refresh(tmp_path)


def test_refresh_rejects_concurrent_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    report_path, _ = completed_exports(tmp_path, monkeypatch)
    with report_path.with_name(".notebook-acceptance.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="execution is active"):
            GALLERY.refresh(tmp_path)
