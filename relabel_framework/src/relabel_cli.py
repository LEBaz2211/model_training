from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.batch import generate_labels
    from src.goose_ex import build_goose_ex_index, framework_root, inspect_label_values, resolve_dataset_root
    from src.taxonomy import load_yaml, validate_scheme_dict, validate_with_pydantic
else:
    from .batch import generate_labels
    from .goose_ex import build_goose_ex_index, framework_root, inspect_label_values, resolve_dataset_root
    from .taxonomy import load_yaml, validate_scheme_dict, validate_with_pydantic


try:
    import typer

    app = typer.Typer(help="Goose-EX relabeling workflow")
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    typer = None
    app = None


def _config_path(config: str | Path) -> Path:
    path = Path(config)
    return path if path.is_absolute() else framework_root() / path


def _validate_impl(config: str | Path) -> None:
    path = _config_path(config)
    data = load_yaml(path)
    errors = validate_scheme_dict(data)
    if errors:
        raise ValueError("; ".join(errors))
    validate_with_pydantic(data)

    dataset_root = resolve_dataset_root(data["dataset"]["root_dir"], framework_root())
    label_version = data["dataset"].get("label_version", "base")
    index = build_goose_ex_index(dataset_root, label_version=label_version)
    stats = inspect_label_values(index.pairs)
    print(f"Config: {path}")
    print(f"Dataset: {dataset_root}")
    print(f"Pairs: {len(index.pairs)}")
    print(f"Splits: {', '.join(index.splits) if index.splits else '(none)'}")
    print(f"Label versions: {index.label_versions}")
    print(f"Sample label IDs: {list(stats.unique_values[:30])}")
    for warning in index.warnings:
        print(f"Warning: {warning}")


def _generate_impl(config: str | Path, splits: str | None, num_workers: int, overwrite: bool) -> None:
    split_values = [part.strip() for part in splits.split(",") if part.strip()] if splits else None
    summary = generate_labels(
        config,
        splits=split_values,
        num_workers=num_workers,
        overwrite=overwrite,
        workspace_root=framework_root(),
    )
    print(f"Generated {summary.label_count} labels")
    print(f"Output: {summary.output_dir}")
    for list_file in summary.list_files:
        print(f"List: {list_file}")
    for warning in summary.warnings:
        print(f"Warning: {warning}")


def _app_impl() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(framework_root() / "app.py")], check=True)


if typer is not None:

    @app.command("validate")
    def validate_command(
        config: Path = typer.Option(..., "--config", "-c", help="Relabeling YAML config"),
    ) -> None:
        _validate_impl(config)

    @app.command("generate")
    def generate_command(
        config: Path = typer.Option(..., "--config", "-c", help="Relabeling YAML config"),
        splits: str | None = typer.Option(None, "--splits", help="Comma-separated split filter, e.g. val or train,val"),
        num_workers: int = typer.Option(4, "--num-workers", "-j", min=1),
        overwrite: bool = typer.Option(False, "--overwrite", help="Regenerate existing labels"),
    ) -> None:
        _generate_impl(config, splits, num_workers, overwrite)

    @app.command("app")
    def app_command() -> None:
        _app_impl()


def argparse_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Goose-EX relabeling workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", "-c", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", "-c", required=True)
    generate_parser.add_argument("--splits", default=None)
    generate_parser.add_argument("--num-workers", "-j", type=int, default=4)
    generate_parser.add_argument("--overwrite", action="store_true")

    subparsers.add_parser("app")

    args = parser.parse_args(argv)
    if args.command == "validate":
        _validate_impl(args.config)
    elif args.command == "generate":
        _generate_impl(args.config, args.splits, args.num_workers, args.overwrite)
    elif args.command == "app":
        _app_impl()


def main() -> None:
    if typer is not None:
        app()
    else:
        argparse_main()


if __name__ == "__main__":
    main()

