from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image

from .goose_ex import GooseExPair, build_goose_ex_index, framework_root, resolve_dataset_root
from .preview import load_label_array, remap_label_array
from .runtime_export import export_runtime_configs
from .taxonomy import groups_from_config, load_yaml, validate_scheme_dict, validate_with_pydantic


@dataclass(frozen=True)
class GeneratedLabel:
    split: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class GenerationSummary:
    config_path: Path
    output_dir: Path
    label_count: int
    list_files: tuple[Path, ...]
    warnings: tuple[str, ...]


def _resolve_output_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def _relpath(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def output_label_path(pair: GooseExPair, annotations_dir: Path, suffix: str) -> Path:
    return annotations_dir / pair.scene / f"{pair.stem}_labelids{suffix}.png"


def _process_one(task: tuple[str, str, list[int], bool]) -> tuple[str, bool]:
    label_path_text, output_path_text, groups, overwrite = task
    output_path = Path(output_path_text)
    if output_path.exists() and not overwrite:
        return output_path_text, False
    label_arr = load_label_array(label_path_text)
    relabeled = remap_label_array(label_arr, groups)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(relabeled).save(output_path)
    return output_path_text, True


def generate_labels(
    config_path: str | Path,
    splits: Iterable[str] | None = None,
    num_workers: int = 4,
    overwrite: bool = False,
    workspace_root: str | Path | None = None,
) -> GenerationSummary:
    root = Path(workspace_root) if workspace_root is not None else framework_root()
    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute():
        config_path_obj = root / config_path_obj
    config = load_yaml(config_path_obj)
    errors = validate_scheme_dict(config)
    if errors:
        raise ValueError("; ".join(errors))
    validate_with_pydantic(config)

    dataset_config = config["dataset"]
    dataset_root = resolve_dataset_root(dataset_config["root_dir"], root)
    label_version = dataset_config.get("label_version", "base")
    index = build_goose_ex_index(dataset_root, label_version=label_version, splits=splits)
    selected_pairs = index.pairs_for_splits(splits)
    if not selected_pairs:
        raise ValueError(f"No Goose-EX image/label pairs found under {dataset_root}")

    groups = groups_from_config(config)
    output_config = config["output"]
    suffix = output_config["new_annotation_suffix"]
    output_dirs = output_config["output_dirs"]
    annotations_dir = _resolve_output_path(output_dirs["annotations"], root)
    lists_dir = _resolve_output_path(output_dirs["lists"], root)
    visuals_dir = _resolve_output_path(output_dirs["visuals"], root)
    lists_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, list[int], bool]] = []
    generated: list[GeneratedLabel] = []
    for pair in selected_pairs:
        label_out = output_label_path(pair, annotations_dir, suffix)
        tasks.append((str(pair.label_path), str(label_out), groups, overwrite))
        generated.append(GeneratedLabel(pair.split, pair.image_path, label_out))

    if num_workers <= 1:
        for task in tasks:
            _process_one(task)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_one, task) for task in tasks]
            for future in as_completed(futures):
                future.result()

    split_to_rows: dict[str, list[GeneratedLabel]] = {}
    for item in generated:
        split_to_rows.setdefault(item.split, []).append(item)

    list_files: list[Path] = []
    for split, rows in sorted(split_to_rows.items()):
        list_path = lists_dir / f"{output_config['new_list_prefix']}{split}.txt"
        with open(list_path, "w") as handle:
            for row in sorted(rows, key=lambda item: (_relpath(item.image_path, root), _relpath(item.label_path, root))):
                handle.write(f"{_relpath(row.image_path, root)} {_relpath(row.label_path, root)}\n")
        list_files.append(list_path)

    manifest_path = lists_dir.parent / "manifest.yaml"
    runtime_paths = export_runtime_configs(config, lists_dir.parent / "runtime")
    manifest = {
        "config": _relpath(config_path_obj, root),
        "dataset_root": _relpath(dataset_root, root),
        "source_label_version": label_version,
        "labels": len(generated),
        "lists": [_relpath(path, root) for path in list_files],
        "runtime_configs": {name: _relpath(path, root) for name, path in runtime_paths.items()},
        "warnings": list(index.warnings),
    }
    with open(manifest_path, "w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    return GenerationSummary(
        config_path=config_path_obj,
        output_dir=lists_dir.parent,
        label_count=len(generated),
        list_files=tuple(list_files),
        warnings=index.warnings,
    )
