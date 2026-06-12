from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import yaml


DEFAULT_BRIGHT_PALETTE = [
    [128, 128, 128],
    [0, 255, 0],
    [255, 230, 0],
    [255, 122, 0],
    [255, 0, 80],
    [0, 170, 255],
    [190, 0, 255],
    [0, 255, 210],
    [255, 0, 220],
    [140, 255, 0],
]

DEFAULT_TRAVERSABILITY_SCORES = [
    -1.0,
    1.0,
    0.7,
    0.5,
    0.1,
    0.0,
]


try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

    PYDANTIC_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional deps
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]
    PYDANTIC_AVAILABLE = False


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, width=120)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "scheme"


def make_experiment_slug(dataset_name: str, experiment_name: str, class_count: int, created_at: datetime | None = None) -> str:
    timestamp = (created_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    prefix = slugify(dataset_name)
    body = slugify(experiment_name) if experiment_name else f"{class_count}cls"
    return f"{prefix}_{body}_{class_count}cls_{timestamp}"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_v{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free path near {path}")


def normalize_palette(palette: list[Any], count: int) -> list[list[int]]:
    normalized: list[list[int]] = []
    for color in palette[:count]:
        if isinstance(color, str):
            normalized.append(hex_to_rgb(color))
        else:
            normalized.append([int(color[0]), int(color[1]), int(color[2])])
    while len(normalized) < count:
        normalized.append(DEFAULT_BRIGHT_PALETTE[len(normalized) % len(DEFAULT_BRIGHT_PALETTE)])
    return normalized


def rgb_to_hex(color: list[int] | tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def hex_to_rgb(color: str) -> list[int]:
    text = color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)]


def superclass_names(config: dict[str, Any]) -> list[str]:
    mapping = config.get("superclasses", {}).get("mapping", {})
    if isinstance(mapping, dict):
        return [str(mapping[str(index)]) for index in sorted(int(key) for key in mapping.keys())]
    raise ValueError("superclasses.mapping must be a dictionary")


def normalize_traversability_scores(scores: Any, count: int) -> list[float]:
    normalized: list[float] = []
    if isinstance(scores, dict):
        for index in range(count):
            value = scores.get(str(index), scores.get(index, DEFAULT_TRAVERSABILITY_SCORES[index % len(DEFAULT_TRAVERSABILITY_SCORES)]))
            normalized.append(float(value))
    elif isinstance(scores, list):
        for index, value in enumerate(scores[:count]):
            normalized.append(float(value))
    while len(normalized) < count:
        normalized.append(float(DEFAULT_TRAVERSABILITY_SCORES[len(normalized) % len(DEFAULT_TRAVERSABILITY_SCORES)]))
    return normalized[:count]


def traversability_scores(config: dict[str, Any]) -> list[float]:
    names = superclass_names(config)
    scores = config.get("superclasses", {}).get("traversability", {})
    return normalize_traversability_scores(scores, len(names))


def groups_from_config(config: dict[str, Any]) -> list[int]:
    groups = config.get("classes", {}).get("groups", [])
    return [int(value) for value in groups]


def id_mapping_from_groups(groups: list[int]) -> dict[int, int]:
    return {index: int(group) for index, group in enumerate(groups)}


def validate_scheme_dict(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    dataset = config.get("dataset", {})
    classes = config.get("classes", {})
    superclasses = config.get("superclasses", {})
    output = config.get("output", {})

    for key in ("name", "root_dir"):
        if not dataset.get(key):
            errors.append(f"dataset.{key} is required")
    base_classes = classes.get("base_classes", [])
    groups = classes.get("groups", [])
    if len(base_classes) != len(groups):
        errors.append("classes.base_classes and classes.groups must have the same length")
    mapping = superclasses.get("mapping", {})
    palette = superclasses.get("palette", [])
    if len(mapping) != len(palette):
        errors.append("superclasses.mapping and superclasses.palette must have the same length")
    traversability = superclasses.get("traversability", {})
    if traversability:
        if isinstance(traversability, dict) and len(traversability) != len(mapping):
            errors.append("superclasses.traversability and superclasses.mapping must have the same length")
        if isinstance(traversability, list) and len(traversability) != len(mapping):
            errors.append("superclasses.traversability and superclasses.mapping must have the same length")
    if groups and mapping:
        max_group = max(int(value) for value in groups)
        if max_group >= len(mapping):
            errors.append("classes.groups references a superclass id that does not exist")
    output_dirs = output.get("output_dirs", {})
    for key in ("annotations", "lists", "visuals"):
        if not output_dirs.get(key):
            errors.append(f"output.output_dirs.{key} is required")
    return errors


if PYDANTIC_AVAILABLE:

    class DatasetConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

        name: str
        root_dir: str
        images_dir: str = "images"
        labels_dir: str = "labels"
        list_files: dict[str, str] = Field(default_factory=dict)
        label_version: str = "base"

    class ClassesConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

        base_classes: list[str]
        palette: list[list[int]]
        groups: list[int]
        id_mapping: dict[int, int] | None = None
        id_seq: dict[int, int] | None = None

        @model_validator(mode="after")
        def check_lengths(self) -> "ClassesConfig":
            if len(self.base_classes) != len(self.groups):
                raise ValueError("base_classes and groups must have the same length")
            return self

    class SuperclassesConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

        mapping: dict[str, str]
        palette: list[list[int]]
        traversability: dict[str, float] | list[float] | None = None

        @field_validator("mapping", mode="before")
        @classmethod
        def stringify_mapping_keys(cls, value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): val for key, val in value.items()}
            return value

        @model_validator(mode="after")
        def check_lengths(self) -> "SuperclassesConfig":
            if len(self.mapping) != len(self.palette):
                raise ValueError("mapping and palette must have the same length")
            if self.traversability is not None and len(self.traversability) != len(self.mapping):
                raise ValueError("mapping and traversability must have the same length")
            return self

    class OutputDirs(BaseModel):
        annotations: str
        lists: str
        visuals: str

    class OutputConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

        new_annotation_suffix: str
        new_list_prefix: str
        output_dirs: OutputDirs

    class RelabelConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

        dataset: DatasetConfig
        classes: ClassesConfig
        superclasses: SuperclassesConfig
        output: OutputConfig


def validate_with_pydantic(config: dict[str, Any]) -> None:
    if not PYDANTIC_AVAILABLE:
        return
    RelabelConfig.model_validate(config)  # type: ignore[name-defined]


def build_experiment_config(
    template_config: dict[str, Any],
    dataset_root: str,
    experiment_name: str,
    groups: list[int],
    superclass_names_input: list[str],
    superclass_palette: list[list[int]],
    traversability_scores_input: list[float] | None = None,
    label_version: str = "base",
    output_root: str = "output/goose_ex",
    created_at: datetime | None = None,
) -> tuple[dict[str, Any], str]:
    config = deepcopy(template_config)
    class_count = len(superclass_names_input)
    slug = make_experiment_slug(config.get("dataset", {}).get("name", "goose_ex"), experiment_name, class_count, created_at)
    normalized_palette = normalize_palette(superclass_palette, class_count)
    normalized_scores = normalize_traversability_scores(traversability_scores_input or {}, class_count)

    config.setdefault("dataset", {})
    config["dataset"]["root_dir"] = dataset_root
    config["dataset"]["images_dir"] = config["dataset"].get("images_dir", "images")
    config["dataset"]["labels_dir"] = config["dataset"].get("labels_dir", "labels")
    config["dataset"]["label_version"] = label_version

    config.setdefault("classes", {})
    config["classes"]["groups"] = [int(value) for value in groups]
    config["classes"]["id_mapping"] = id_mapping_from_groups(groups)
    config["classes"]["id_seq"] = {index: index for index in range(len(groups))}

    config["superclasses"] = {
        "mapping": {str(index): name for index, name in enumerate(superclass_names_input)},
        "palette": normalized_palette,
        "traversability": {str(index): float(score) for index, score in enumerate(normalized_scores)},
    }

    annotation_suffix = f"_{slug}"
    config["output"] = {
        "new_annotation_suffix": annotation_suffix,
        "new_list_prefix": f"{slug}_",
        "output_dirs": {
            "annotations": f"{output_root}/{slug}/labels",
            "lists": f"{output_root}/{slug}/lists",
            "visuals": f"{output_root}/{slug}/previews",
        },
    }
    config["experiment"] = {
        "name": experiment_name or f"{class_count} class scheme",
        "slug": slug,
        "created_at": (created_at or datetime.now()).isoformat(timespec="seconds"),
        "source_label_version": label_version,
    }
    return config, slug
