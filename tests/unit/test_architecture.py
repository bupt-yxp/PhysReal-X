from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).parents[2] / "src" / "physrealx"


def test_domain_contract_has_no_outward_dependencies() -> None:
    tree = ast.parse((PACKAGE / "domain.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))

    forbidden = {
        name
        for name in imports
        if name.startswith(("physrealx", "torch", "fastapi", "sqlalchemy", "sqlite3"))
    }
    assert forbidden == set()


def test_import_is_side_effect_free_and_does_not_load_torch(tmp_path: Path) -> None:
    probe = (
        "import json, pathlib, sys; import physrealx; "
        "print(json.dumps({'torch': 'torch' in sys.modules, "
        "'files': [p.name for p in pathlib.Path.cwd().iterdir()]}))"
    )
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("COV_CORE_"):
            environment.pop(name)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert json.loads(completed.stdout) == {"torch": False, "files": []}


def test_no_compatibility_namespaces_are_present() -> None:
    forbidden = {"compat", "legacy", "v0", "v1", "physreal", "physreal_x"}
    names = {path.stem for path in PACKAGE.rglob("*.py")} | {
        part for path in PACKAGE.rglob("*") for part in path.parts
    }

    assert names.isdisjoint(forbidden)


def test_domain_services_do_not_import_the_facade_or_storage_at_runtime() -> None:
    """Composition injects the workspace; domain services must not construct one."""

    def runtime_imports(node: ast.AST) -> set[str]:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            return set()
        if isinstance(node, ast.Import):
            return {alias.name for alias in node.names}
        if isinstance(node, ast.ImportFrom):
            return {node.module or ""}
        return set().union(*(runtime_imports(child) for child in ast.iter_child_nodes(node)))

    for path in (PACKAGE / "services").glob("*.py"):
        imports = runtime_imports(ast.parse(path.read_text(encoding="utf-8")))
        assert imports.isdisjoint({"physrealx.platform", "physrealx.storage"}), path.name
