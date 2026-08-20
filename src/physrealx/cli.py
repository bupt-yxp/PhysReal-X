"""Thin command-line interface for the PhysReal-X application facade."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from pydantic import BaseModel, ValidationError

from physrealx.config import load_mapping
from physrealx.diagnostics import inspect_runtime
from physrealx.domain import (
    AssetSpec,
    ResourceType,
    RunSpec,
    TaskSpec,
    TrainingSpec,
)
from physrealx.errors import PhysRealXError
from physrealx.platform import PhysRealX
from physrealx.reference import reference_asset, reference_run, reference_task


def _workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, default=Path(".physrealx"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physrealx",
        description="Reproducible embodied simulation, data, and learning workflows.",
    )
    parser.add_argument("--debug", action="store_true", help="show tracebacks for failures")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="initialize a local workspace")
    init.add_argument("workspace", type=Path, nargs="?", default=Path(".physrealx"))

    doctor = commands.add_parser("doctor", help="inspect Python and optional CUDA runtime")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.add_argument("--no-cuda", action="store_true")
    doctor.add_argument("--strict", action="store_true")

    plugins = commands.add_parser("plugins", help="list registered plugins")
    _workspace(plugins)

    validate = commands.add_parser("validate", help="validate a JSON/YAML specification")
    validate.add_argument("kind", choices=("asset", "task", "run", "training"))
    validate.add_argument("path", type=Path)

    add = commands.add_parser("add", help="register an immutable asset or task")
    add.add_argument("kind", choices=("asset", "task"))
    add.add_argument("path", type=Path)
    _workspace(add)

    listing = commands.add_parser("list", help="list versioned resources")
    listing.add_argument("kind", choices=tuple(item.value for item in ResourceType))
    _workspace(listing)

    run = commands.add_parser("run", help="execute a RunSpec")
    run.add_argument("path", type=Path)
    _workspace(run)

    status = commands.add_parser("status", help="show a persisted run")
    status.add_argument("run_id")
    _workspace(status)

    train = commands.add_parser("train", help="train a model from a Dataset manifest")
    train.add_argument("path", type=Path)
    _workspace(train)

    evaluate = commands.add_parser("evaluate", help="evaluate a registered model")
    evaluate.add_argument("model_id")
    evaluate.add_argument("model_version")
    evaluate.add_argument("run", type=Path, help="base RunSpec used for held-out rollouts")
    evaluate.add_argument("--id", required=True, dest="evaluation_id")
    evaluate.add_argument("--version", required=True)
    evaluate.add_argument("--episodes", type=int, default=16)
    evaluate.add_argument("--base-seed", type=int, default=100_000)
    _workspace(evaluate)

    demo = commands.add_parser("demo", help="run the reference vertical slice")
    demo.add_argument("--episodes", type=int, default=8)
    demo.add_argument("--base-seed", type=int, default=7)
    demo.add_argument("--train", action="store_true")
    demo.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    _workspace(demo)
    return parser


def _print_model(model: object) -> None:
    if hasattr(model, "model_dump_json"):
        print(model.model_dump_json(indent=2))
    else:
        print(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True))


def _platform(path: Path) -> PhysRealX:
    return PhysRealX.open(path)


def _run(args: argparse.Namespace) -> int:
    command = args.command
    if command == "init":
        platform = _platform(args.workspace)
        print(platform.workspace)
        return 0
    if command == "doctor":
        report = inspect_runtime(probe_cuda=not args.no_cuda)
        if args.as_json:
            _print_model(report.to_dict())
        else:
            print(f"Python:          {report.python}")
            print(f"Conda env:       {report.conda_environment or '-'}")
            print(f"Torch:           {report.torch_version or 'not installed'}")
            print(f"Torch CUDA:      {report.torch_cuda_runtime or '-'}")
            print(f"CUDA available:  {report.cuda_available}")
            print(f"CUDA devices:    {', '.join(report.cuda_devices) or '-'}")
            print(f"cu121 ready:     {report.cu121_ready}")
            for issue in report.issues:
                print(f"warning: {issue}")
        return 1 if args.strict and report.issues else 0
    if command == "plugins":
        platform = _platform(args.workspace)
        _print_model([asdict(descriptor) for descriptor in platform.plugins.descriptors()])
        return 0
    if command == "validate":
        models: dict[str, type[BaseModel]] = {
            "asset": AssetSpec,
            "task": TaskSpec,
            "run": RunSpec,
            "training": TrainingSpec,
        }
        model = models[args.kind].model_validate(load_mapping(args.path))
        _print_model(model)
        return 0
    if command == "add":
        platform = _platform(args.workspace)
        if args.kind == "asset":
            reference = platform.register_asset(AssetSpec.model_validate(load_mapping(args.path)))
        else:
            reference = platform.register_task(TaskSpec.model_validate(load_mapping(args.path)))
        _print_model(reference)
        return 0
    if command == "list":
        platform = _platform(args.workspace)
        references = platform.list_resources(ResourceType(args.kind))
        _print_model([item.model_dump(mode="json") for item in references])
        return 0
    if command == "run":
        record = _platform(args.workspace).execute(RunSpec.model_validate(load_mapping(args.path)))
        _print_model(record)
        return 0
    if command == "status":
        _print_model(_platform(args.workspace).repository.get_run(args.run_id))
        return 0
    if command == "train":
        spec = TrainingSpec.model_validate(load_mapping(args.path))
        _print_model(_platform(args.workspace).train(spec))
        return 0
    if command == "evaluate":
        platform = _platform(args.workspace)
        model = platform.repository.get_model(args.model_id, args.model_version)
        reference = platform.evaluate(
            resource_id=args.evaluation_id,
            version=args.version,
            model=model,
            run=RunSpec.model_validate(load_mapping(args.run)),
            episode_count=args.episodes,
            base_seed=args.base_seed,
        )
        _print_model(reference)
        return 0
    if command == "demo":
        return _demo(args)
    raise AssertionError(f"unhandled command: {command}")


def _demo(args: argparse.Namespace) -> int:
    platform = _platform(args.workspace)
    asset = reference_asset()
    task = reference_task(asset)
    platform.register_asset(asset)
    platform.register_task(task)
    record = platform.execute(
        reference_run(
            asset,
            task,
            episode_count=args.episodes,
            base_seed=args.base_seed,
        )
    )
    output: dict[str, object] = {
        "workspace": str(platform.workspace),
        "run": record.model_dump(mode="json"),
    }
    if args.train:
        if record.result is None or record.result.dataset is None:
            raise RuntimeError("reference run did not publish a dataset")
        training = TrainingSpec(
            resource_id="drawer.bc",
            version="1.0.0",
            dataset=record.result.dataset,
            trainer="torch_bc",
            config={"device": args.device},
        )
        model = platform.train(training)
        evaluation = platform.evaluate(
            resource_id="drawer.bc.eval",
            version="1.0.0",
            model=model,
            run=reference_run(asset, task, episode_count=args.episodes, base_seed=args.base_seed),
            episode_count=max(4, args.episodes),
        )
        output["model"] = model.model_dump(mode="json")
        output["evaluation"] = platform.load_evaluation(evaluation).model_dump(mode="json")
    _print_model(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (PhysRealXError, ValidationError, ValueError, OSError) as error:
        if args.debug:
            raise
        code = error.code if isinstance(error, PhysRealXError) else "invalid_input"
        print(f"physrealx: {code}: {error}", file=sys.stderr)
        return 2


def _entrypoint() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _entrypoint()
