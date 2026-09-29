"""Publish verified demo snapshots without requiring datasets at site-build time."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import html
import json
import posixpath
import re
import sys
from pathlib import Path
from typing import Any

MANIFEST = "assets/demos/manifest.json"
SNAPSHOTS = (
    (("reconstruction-comparison.gif", "reconstruction-comparison.gif"),),
    (("physical-asset-preview.gif", "preview"),),
    (
        ("task-preview-home.gif", None),
        ("task-preview-tabletop.gif", None),
        ("task-preview-shelf.gif", None),
        ("task-preview-industrial.gif", None),
    ),
    (("simulation-preview.gif", "preview"),),
    (
        ("world-model-preview.gif", "world_model_preview"),
        ("dataset-preview.png", "dataset_preview"),
    ),
    (("feedback-cycle.gif", "feedback_cycle"), ("feedback-dashboard.png", "feedback_dashboard")),
)
_manifest: dict[str, Any] = {}


def demo_catalog() -> tuple[Any, ...]:
    repo = str(Path(__file__).resolve().parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from examples.catalog import DEMOS

    return DEMOS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_media_hashes(path: Path) -> dict[str, set[str]]:
    """Identify the exact image bytes displayed by an accepted notebook execution."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    hashes: dict[str, set[str]] = {"image/gif": set(), "image/png": set()}
    for cell in notebook["cells"]:
        for output in cell.get("outputs", ()):
            for mime, encoded in output.get("data", {}).items():
                if mime not in hashes:
                    continue
                text = "".join(encoded) if isinstance(encoded, list) else encoded
                payload = base64.b64decode("".join(text.split()), validate=True)
                hashes[mime].add(hashlib.sha256(payload).hexdigest())
    return hashes


def confined_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Demo path must stay inside {root}: {relative}")
    return path


def validate_manifest(docs_dir: Path) -> dict[str, Any]:
    manifest = json.loads((docs_dir / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != "physrealx.website-demo-snapshots/1":
        raise ValueError("Unsupported website demo manifest")
    seen = set()
    for demo in manifest["demos"]:
        slug = demo["slug"]
        if slug in seen or not re.fullmatch(r"[a-z_]+", slug):
            raise ValueError(f"Invalid or repeated demo: {slug}")
        seen.add(slug)
    actual = [
        (d["slug"], d["title"], d["guide"], tuple(d["dependencies"])) for d in manifest["demos"]
    ]
    expected = [(d.slug, d.title, d.guide, d.dependencies) for d in demo_catalog()]
    if actual != expected:
        raise ValueError("Website demo manifest differs from examples.catalog; refresh snapshots")
    if "verification" in manifest:
        if manifest["verification"] != "completed_execution_and_embedded_media":
            raise ValueError("Unknown demo snapshot verification")
        by_slug = {d["slug"]: d for d in manifest["demos"]}
        implementations = set()
        for demo in manifest["demos"]:
            execution = demo["execution"]
            upstream = {slug: by_slug[slug]["results_sha256"] for slug in demo["dependencies"]}
            if (
                execution.get("schema") != "physrealx.notebook-execution/3"
                or execution.get("status") != "COMPLETED"
                or execution.get("results_sha256") != demo["results_sha256"]
                or execution.get("upstream_results") != upstream
            ):
                raise ValueError(f"Inconsistent demo execution evidence: {demo['slug']}")
            implementations.add(execution["implementation_sha256"])
        if len(implementations) != 1:
            raise ValueError("Demo snapshots were produced by different implementations")
    for demo in manifest["demos"]:
        slug = demo["slug"]
        if not confined_path(docs_dir, demo["guide"]).is_file():
            raise ValueError(f"Missing demo guide: {slug}")
        if not demo["media"]:
            raise ValueError(f"Missing demo preview: {slug}")
        for media in demo["media"]:
            path = confined_path(docs_dir, media["path"])
            if path.stat().st_size != media["size_bytes"] or sha256(path) != media["sha256"]:
                raise ValueError(f"Demo snapshot changed without refreshing provenance: {path}")
    return manifest


def refresh(repo: Path) -> None:
    """Publish one coherent set of current executions, never intermediate output."""
    lock_path = repo / "examples/output/.notebook-acceptance.lock"
    with lock_path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Notebook execution is active; refresh after it completes"
            ) from error
        _refresh_completed(repo)


def _refresh_completed(repo: Path) -> None:
    # The catalog is the only source of module names, guides and dependencies.
    demos_catalog = demo_catalog()
    from examples.run_all import _load_current_receipt

    if len(demos_catalog) != len(SNAPSHOTS):
        raise ValueError("Update the demo snapshot selection for the module catalog")
    acceptance_path = repo / "examples/output/notebook-acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if (
        acceptance.get("schema") != "physrealx.notebook-acceptance/3"
        or acceptance.get("execution_status") != "COMPLETED"
    ):
        raise ValueError("A completed notebook acceptance run is required before refreshing")
    accepted = acceptance.get("accepted_modules", ())
    if not accepted or [r.get("module") for r in acceptance["notebooks"]] != accepted:
        raise ValueError("Acceptance summary does not identify its completed notebooks")
    current: dict[str, dict[str, Any]] = {}
    receipts = {
        demo.slug: _load_current_receipt(repo, demo.slug, current=current) for demo in demos_catalog
    }
    for recorded in acceptance["notebooks"]:
        if recorded != receipts.get(recorded.get("module")):
            raise ValueError("Acceptance summary and current notebook receipts disagree")
    docs = repo / "docs"
    demos = []
    copies = []
    for demo, selection in zip(demos_catalog, SNAPSHOTS):
        source_dir = repo / "examples" / demo.slug
        results_path = source_dir / "output/results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        receipt_path = source_dir / "output/execution.json"
        receipt = receipts[demo.slug]
        executed_path = source_dir / "output/demo.executed.ipynb"
        embedded_images = notebook_media_hashes(executed_path)
        media = []
        for filename, result_key in selection:
            source = source_dir / "output" / filename
            reference = results[result_key] if result_key else None
            payload = source.read_bytes()
            size = len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            mime = "image/gif" if source.suffix == ".gif" else "image/png"
            if reference and reference["size_bytes"] != size:
                raise ValueError(f"Result index and exported preview disagree: {source}")
            if reference and reference["media_type"] != mime:
                raise ValueError(f"Result index and exported media type disagree: {source}")
            if digest not in embedded_images[mime]:
                raise ValueError(f"Exported preview was not displayed by this execution: {source}")
            destination = f"assets/demos/{demo.slug}/{filename}"
            media.append(
                {
                    "path": destination,
                    "source": source.relative_to(repo).as_posix(),
                    "sha256": digest,
                    "size_bytes": size,
                    "result_key": result_key,
                    "artifact": reference,
                    "notebook_media_type": mime,
                }
            )
            copies.append((payload, docs / destination))
        demos.append(
            {
                "slug": demo.slug,
                "title": demo.title,
                "guide": demo.guide,
                "dependencies": list(demo.dependencies),
                "notebook": (source_dir / "demo.ipynb").relative_to(repo).as_posix(),
                "results_sha256": sha256(results_path),
                "execution": {
                    "schema": receipt["schema"],
                    "status": receipt["status"],
                    "finished_at": receipt["finished_at"],
                    "source_sha256": receipt["source_sha256"],
                    "implementation_sha256": receipt["implementation_sha256"],
                    "results_sha256": receipt["results_sha256"],
                    "executed_sha256": receipt["executed_sha256"],
                    "upstream_results": receipt["upstream_results"],
                    "receipt_sha256": sha256(receipt_path),
                },
                "media": media,
            }
        )
    # Validate every source before replacing any published snapshot.
    for payload, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    manifest = {
        "schema": "physrealx.website-demo-snapshots/1",
        "verification": "completed_execution_and_embedded_media",
        "acceptance_sha256": sha256(acceptance_path),
        "demos": demos,
    }
    (docs / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    validate_manifest(docs)
    total_bytes = sum(m["size_bytes"] for d in demos for m in d["media"])
    print(f"Refreshed {len(demos)} demo snapshots ({total_bytes:,} bytes)")


def on_config(config: Any) -> None:
    global _manifest
    _manifest = validate_manifest(Path(config["docs_dir"]))


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    from mkdocs.utils import get_relative_url

    def relative(path: str) -> str:
        return html.escape(get_relative_url(path, page.url), quote=True)

    def preview(demo: dict[str, Any], media: dict[str, Any]) -> str:
        image = relative(media["path"])
        title = html.escape(demo["title"])
        source = html.escape(f"{config['repo_url']}/blob/main/{demo['notebook']}", quote=True)
        guide = files.get_file_from_path(demo["guide"])
        if guide is None:
            raise ValueError(f"Demo guide is not included in the site: {demo['guide']}")
        caption = (
            f'<figcaption>点击图像查看原尺寸 · <a href="{source}">Notebook 源码</a>'
            f' · <a href="{relative(MANIFEST)}">来源清单</a></figcaption>'
        )
        return (
            f'<figure class="prx-demo-preview"><a href="{image}" aria-label="查看{title}原图">'
            f'<img src="{image}" alt="{title} Demo 输出" loading="lazy"></a>{caption}</figure>'
        )

    demos = {demo["slug"]: demo for demo in _manifest["demos"]}
    if "<!-- demo-verification -->" in markdown:
        receipt_note = ""
        if _manifest.get("verification") == "completed_execution_and_embedded_media":
            finished = max(d["execution"]["finished_at"] for d in demos.values())
            receipt_note = (
                '<p class="prx-snapshot-note">六模块运行快照 · '
                f"{html.escape(finished[:10])} UTC · 完成回执与图像已校核</p>"
            )
        markdown = markdown.replace(
            "<!-- demo-verification -->",
            receipt_note,
        )

    def replace(match: re.Match[str]) -> str:
        demo = demos[match.group(2)]
        filename = match.group(3)
        media = (
            next((m for m in demo["media"] if posixpath.basename(m["path"]) == filename), None)
            if filename
            else demo["media"][0]
        )
        if media is None:
            raise ValueError(f"Unknown demo media: {match.group(0)}")
        return preview(demo, media)

    return re.sub(r"<!-- demo-(preview|image):([a-z_]+)(?::([\w.-]+))? -->", replace, markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", required=True)
    parser.parse_args()
    refresh(Path(__file__).resolve().parents[2])
