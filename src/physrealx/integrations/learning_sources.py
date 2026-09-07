"""Load the small pinned algorithm subsets shipped with source and wheel distributions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from physrealx.errors import MethodExecutionError


def algorithm_module(name: str, *, package: bool = False) -> ModuleType:
    if name not in {"act", "dinov2", "dino_wm"}:
        raise MethodExecutionError("unknown packaged learning algorithm")
    module_name = f"physrealx._algorithm_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    roots = (
        Path(__file__).resolve().parents[3] / "algorithms" / name,
        Path(__file__).resolve().parents[1] / "_embedded" / name,
    )
    filename = "__init__.py" if package else "model.py"
    path = next((root / filename for root in roots if (root / filename).is_file()), None)
    if path is None:
        raise MethodExecutionError(f"packaged {name} source is missing")
    spec = importlib.util.spec_from_file_location(
        module_name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    if spec is None or spec.loader is None:
        raise MethodExecutionError(f"cannot load packaged {name} source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
