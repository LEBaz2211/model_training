from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable

import numpy as np
from PIL import Image


LABEL_RE = re.compile(r"^(?P<stem>.+)_labelids(?P<suffix>_[A-Za-z0-9][A-Za-z0-9_-]*)?\.png$")
IMAGE_SUFFIX_RE = re.compile(
    r"_(camera_left|camera_right|windshield_vis|windshield_nir|realsense|front|rgb|color)$"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
IMAGE_PREFERENCE = ("camera_left", "windshield_vis", "color", "rgb", "front", "camera_right", "realsense", "windshield_nir")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    rel_path: str
    split: str
    scene: str
    stem: str
    sensor: str


@dataclass(frozen=True)
class LabelRecord:
    path: Path
    rel_path: str
    split: str
    scene: str
    stem: str
    version: str


@dataclass(frozen=True)
class GooseExPair:
    split: str
    scene: str
    stem: str
    image_path: Path
    label_path: Path
    image_rel: str
    label_rel: str
    label_version: str


@dataclass(frozen=True)
class LabelValueStats:
    sample_count: int
    unique_values: tuple[int, ...]
    max_value: int
    looks_grouped: bool


@dataclass(frozen=True)
class GooseExIndex:
    root: Path
    selected_label_version: str
    pairs: tuple[GooseExPair, ...]
    label_versions: dict[str, int]
    image_count: int
    label_count: int
    warnings: tuple[str, ...]

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted({pair.split for pair in self.pairs}))

    def pairs_for_splits(self, splits: Iterable[str] | None = None) -> list[GooseExPair]:
        if not splits:
            return list(self.pairs)
        allowed = set(splits)
        return [pair for pair in self.pairs if pair.split in allowed]


@dataclass(frozen=True)
class StemInventory:
    split: str
    image_stems: tuple[str, ...]
    label_stems: tuple[str, ...]

    @property
    def paired_count(self) -> int:
        return len(set(self.image_stems) & set(self.label_stems))

    @property
    def missing_labels(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.image_stems) - set(self.label_stems)))

    @property
    def missing_images(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.label_stems) - set(self.image_stems)))


def framework_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_root(root_dir: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve Goose-EX roots and tolerate goose-ex/goose_ex spelling drift."""
    base = Path(base_dir) if base_dir is not None else framework_root()
    raw = Path(root_dir)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(base / raw)
        candidates.append(raw)

    spellings: list[Path] = []
    seen_spellings: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        for spelling in (Path(text.replace("goose-ex", "goose_ex")), Path(text.replace("goose_ex", "goose-ex"))):
            spelling_text = spelling.as_posix()
            if spelling_text not in seen_spellings:
                spellings.append(spelling)
                seen_spellings.add(spelling_text)
    candidates.extend(spellings)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base / raw).resolve() if not raw.is_absolute() else raw.resolve()


def label_version_from_suffix(suffix: str | None) -> str:
    return suffix[1:] if suffix else "base"


def parse_label_path(path: Path, root: Path) -> LabelRecord | None:
    match = LABEL_RE.match(path.name)
    if not match:
        return None
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    split = "unknown"
    scene = path.parent.name
    if len(parts) >= 4 and parts[0] == "labels" and parts[1] in {"train", "val", "test"}:
        split = parts[1]
        scene = parts[2]
    return LabelRecord(
        path=path,
        rel_path=rel,
        split=split,
        scene=scene,
        stem=match.group("stem"),
        version=label_version_from_suffix(match.group("suffix")),
    )


def parse_image_path(path: Path, root: Path) -> ImageRecord | None:
    if path.suffix.lower() not in IMAGE_EXTS:
        return None
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    if len(parts) < 3 or parts[0] != "images":
        return None

    split = "unknown"
    scene_index = 1
    if len(parts) >= 4 and parts[1] in {"train", "val", "test"}:
        split = parts[1]
        scene_index = 2
    scene = parts[scene_index]
    stem = path.stem
    sensor = ""
    match = IMAGE_SUFFIX_RE.search(stem)
    if match:
        sensor = match.group(1)
        stem = stem[: match.start()]
    return ImageRecord(path=path, rel_path=rel, split=split, scene=scene, stem=stem, sensor=sensor)


def discover_label_versions(root: str | Path) -> dict[str, int]:
    root_path = Path(root)
    labels_dir = root_path / "labels"
    versions: Counter[str] = Counter()
    if not labels_dir.exists():
        return {}
    for path in labels_dir.rglob("*.png"):
        record = parse_label_path(path, root_path)
        if record is not None:
            versions[record.version] += 1
    return dict(sorted(versions.items()))


def choose_label_version(versions: dict[str, int], requested: str | None = None) -> str:
    if requested and requested in versions:
        return requested
    if "base" in versions:
        return "base"
    if not versions:
        return requested or "base"
    return max(versions.items(), key=lambda item: item[1])[0]


def _image_preference_key(record: ImageRecord) -> tuple[int, str]:
    try:
        score = IMAGE_PREFERENCE.index(record.sensor)
    except ValueError:
        score = len(IMAGE_PREFERENCE)
    return score, record.rel_path


def build_goose_ex_index(
    root: str | Path,
    label_version: str | None = None,
    splits: Iterable[str] | None = None,
) -> GooseExIndex:
    root_path = Path(root).resolve()
    requested_splits = set(splits or [])
    warnings: list[str] = []

    images_dir = root_path / "images"
    labels_dir = root_path / "labels"
    if not images_dir.exists():
        warnings.append(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        warnings.append(f"Labels directory not found: {labels_dir}")

    images_by_stem: dict[str, list[ImageRecord]] = defaultdict(list)
    image_count = 0
    if images_dir.exists():
        for path in images_dir.rglob("*"):
            record = parse_image_path(path, root_path)
            if record is None:
                continue
            if requested_splits and record.split not in requested_splits:
                continue
            images_by_stem[record.stem].append(record)
            image_count += 1

    labels_by_stem: dict[str, list[LabelRecord]] = defaultdict(list)
    versions: Counter[str] = Counter()
    label_count = 0
    if labels_dir.exists():
        for path in labels_dir.rglob("*.png"):
            record = parse_label_path(path, root_path)
            if record is None:
                continue
            versions[record.version] += 1
            label_count += 1
            labels_by_stem[record.stem].append(record)

    versions_dict = dict(sorted(versions.items()))
    selected_version = choose_label_version(versions_dict, label_version)
    if label_version and label_version not in versions_dict:
        warnings.append(
            f"Requested label version '{label_version}' was not found; using '{selected_version}' instead."
        )
    if selected_version != "base":
        warnings.append(
            f"Selected labels are '{selected_version}', not unsuffixed Goose-EX base labels. "
            "They may already be relabeled outputs."
        )

    pairs: list[GooseExPair] = []
    missing_images = 0
    for stem, label_records in labels_by_stem.items():
        selected_labels = [record for record in label_records if record.version == selected_version]
        if not selected_labels:
            continue
        image_records = sorted(images_by_stem.get(stem, []), key=_image_preference_key)
        if not image_records:
            missing_images += len(selected_labels)
            continue
        for label_record in selected_labels:
            if label_record.split != "unknown":
                matching_images = [record for record in image_records if record.split == label_record.split]
                if not matching_images:
                    missing_images += 1
                    continue
                image_record = matching_images[0]
            else:
                image_record = image_records[0]
            pairs.append(
                GooseExPair(
                    split=image_record.split,
                    scene=label_record.scene,
                    stem=stem,
                    image_path=image_record.path,
                    label_path=label_record.path,
                    image_rel=image_record.rel_path,
                    label_rel=label_record.rel_path,
                    label_version=selected_version,
                )
            )

    if missing_images:
        warnings.append(f"{missing_images} selected label files did not have a matching image.")
    if requested_splits and not pairs:
        warnings.append(f"No paired samples found for splits: {', '.join(sorted(requested_splits))}.")

    pairs.sort(key=lambda pair: (pair.split, pair.scene, pair.stem, pair.label_rel))
    return GooseExIndex(
        root=root_path,
        selected_label_version=selected_version,
        pairs=tuple(pairs),
        label_versions=versions_dict,
        image_count=image_count,
        label_count=label_count,
        warnings=tuple(warnings),
    )


def inventory_stems(root: str | Path, label_version: str = "base") -> tuple[StemInventory, ...]:
    root_path = Path(root).resolve()
    images_by_split: dict[str, set[str]] = defaultdict(set)
    labels_by_split: dict[str, set[str]] = defaultdict(set)

    images_dir = root_path / "images"
    if images_dir.exists():
        for path in images_dir.rglob("*"):
            record = parse_image_path(path, root_path)
            if record is not None:
                images_by_split[record.split].add(record.stem)

    labels_dir = root_path / "labels"
    if labels_dir.exists():
        for path in labels_dir.rglob("*.png"):
            record = parse_label_path(path, root_path)
            if record is not None and record.version == label_version:
                if record.split == "unknown":
                    for split in images_by_split:
                        labels_by_split[split].add(record.stem)
                else:
                    labels_by_split[record.split].add(record.stem)

    split_names = sorted(images_by_split.keys() | labels_by_split.keys() | {"train", "val"})
    return tuple(
        StemInventory(
            split=split,
            image_stems=tuple(sorted(images_by_split.get(split, set()))),
            label_stems=tuple(sorted(labels_by_split.get(split, set()))),
        )
        for split in split_names
    )


def sample_pairs(pairs: Iterable[GooseExPair], count: int = 1, seed: int | None = None) -> list[GooseExPair]:
    pool = list(pairs)
    if not pool:
        return []
    rng = random.Random(seed)
    count = min(count, len(pool))
    return rng.sample(pool, count)


def inspect_label_values(pairs: Iterable[GooseExPair], sample_count: int = 8) -> LabelValueStats:
    selected = list(pairs)[:sample_count]
    values: set[int] = set()
    for pair in selected:
        arr = np.array(Image.open(pair.label_path).convert("L"))
        values.update(int(value) for value in np.unique(arr))
    max_value = max(values) if values else 0
    return LabelValueStats(
        sample_count=len(selected),
        unique_values=tuple(sorted(values)),
        max_value=max_value,
        looks_grouped=bool(values) and max_value <= 10 and len(values) <= 11,
    )
