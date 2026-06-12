from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import shutil
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.goose_ex import (
        build_goose_ex_index,
        framework_root,
        inventory_stems,
        resolve_dataset_root,
    )
else:
    from .goose_ex import build_goose_ex_index, framework_root, inventory_stems, resolve_dataset_root


LEGACY_OUTPUT_NAMES = (
    "goose-ex",
    "goose/test1",
    "goose/test1_annotations",
    "goose/test1_lists",
    "goose/colorized_old",
)


@dataclass(frozen=True)
class ArchiveSummary:
    moved_labels: int
    moved_outputs: int
    archive_dir: Path


def _version_from_label_name(path: Path) -> str | None:
    stem = path.stem
    marker = "_labelids_"
    if marker not in stem:
        return None
    return stem.split(marker, 1)[1]


def archive_generated_labels(
    dataset_root: str | Path,
    archive_dir: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[int, Path]:
    root = Path(dataset_root).resolve()
    labels_dir = root / "labels"
    archive_root = Path(archive_dir) if archive_dir else root / "archive" / "generated_labels"
    archive_root = archive_root.resolve()
    moved = 0

    if not labels_dir.exists():
        return 0, archive_root

    for label_path in sorted(labels_dir.rglob("*_labelids_*.png")):
        if archive_root in label_path.parents:
            continue
        version = _version_from_label_name(label_path)
        if not version:
            continue
        rel = label_path.relative_to(labels_dir)
        destination = archive_root / version / "labels" / rel
        moved += 1
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(label_path), str(destination))

    return moved, archive_root


def archive_legacy_outputs(
    output_root: str | Path,
    archive_dir: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[int, Path]:
    root = Path(output_root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = Path(archive_dir).resolve() if archive_dir else root / "archive" / f"legacy_{stamp}"
    moved = 0

    for name in LEGACY_OUTPUT_NAMES:
        source = root / name
        if not source.exists():
            continue
        destination = archive_root / name
        moved += 1
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    return moved, archive_root


def archive_generated(
    dataset_root: str | Path,
    output_root: str | Path | None = None,
    dry_run: bool = False,
) -> ArchiveSummary:
    resolved_dataset = resolve_dataset_root(dataset_root, framework_root())
    moved_labels, label_archive = archive_generated_labels(resolved_dataset, dry_run=dry_run)
    moved_outputs = 0
    archive_dir = label_archive
    if output_root:
        moved_outputs, archive_dir = archive_legacy_outputs(output_root, dry_run=dry_run)
    return ArchiveSummary(moved_labels=moved_labels, moved_outputs=moved_outputs, archive_dir=archive_dir)


def build_pair_lists(
    dataset_root: str | Path,
    label_version: str = "base",
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    splits: tuple[str, ...] = ("train", "val"),
    dry_run: bool = False,
) -> dict[str, Path]:
    root = resolve_dataset_root(dataset_root, framework_root())
    index = build_goose_ex_index(root, label_version=label_version, splits=splits)
    list_dir = Path(output_dir) if output_dir else root / "lists"
    list_dir = list_dir.resolve()
    list_prefix = prefix if prefix is not None else (label_version if label_version != "base" else "base")
    written: dict[str, Path] = {}

    for split in splits:
        pairs = index.pairs_for_splits([split])
        path = list_dir / f"{list_prefix}_{split}.txt"
        written[split] = path
        if dry_run:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            for pair in pairs:
                handle.write(f"{pair.image_rel} {pair.label_rel}\n")

    combined_pairs = []
    for split in splits:
        combined_pairs.extend(index.pairs_for_splits([split]))
    combined_path = list_dir / f"{list_prefix}_{'_'.join(splits)}.txt"
    written["_".join(splits)] = combined_path
    if not dry_run:
        with open(combined_path, "w") as handle:
            for pair in combined_pairs:
                handle.write(f"{pair.image_rel} {pair.label_rel}\n")

    return written


def verify_dataset(
    dataset_root: str | Path,
    label_version: str = "base",
    splits: tuple[str, ...] = ("train", "val"),
    max_examples: int = 8,
) -> int:
    root = resolve_dataset_root(dataset_root, framework_root())
    index = build_goose_ex_index(root, label_version=label_version, splits=splits)
    print(f"Dataset: {root}")
    print(f"Selected label version: {index.selected_label_version}")
    print(f"Image files indexed: {index.image_count}")
    print(f"Label files indexed: {index.label_count}")
    print(f"Label versions: {index.label_versions}")
    for split in splits:
        print(f"{split} pairs: {len(index.pairs_for_splits([split]))}")

    inventories = {item.split: item for item in inventory_stems(root, index.selected_label_version)}
    failures = 0
    for split in splits:
        item = inventories.get(split)
        if not item:
            print(f"{split}: no image inventory")
            failures += 1
            continue
        print(
            f"{split}: {len(item.image_stems)} image stems, {len(item.label_stems)} label stems, "
            f"{item.paired_count} possible pairs"
        )
        if item.missing_labels:
            failures += 1
            examples = ", ".join(item.missing_labels[:max_examples])
            print(f"{split}: missing labels for {len(item.missing_labels)} image stems. Examples: {examples}")
        if item.missing_images:
            examples = ", ".join(item.missing_images[:max_examples])
            print(f"{split}: labels without {split} images: {len(item.missing_images)}. Examples: {examples}")

    for warning in index.warnings:
        print(f"Warning: {warning}")
    return failures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Goose-EX data/list maintenance")
    parser.add_argument("--dataset-root", default="data/goose_ex")
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive_parser = subparsers.add_parser("archive-generated")
    archive_parser.add_argument("--output-root", default="output")
    archive_parser.add_argument("--dry-run", action="store_true")
    archive_parser.add_argument("--skip-output", action="store_true")

    build_parser = subparsers.add_parser("build-lists")
    build_parser.add_argument("--label-version", default="base")
    build_parser.add_argument("--output-dir", default=None)
    build_parser.add_argument("--prefix", default=None)
    build_parser.add_argument("--splits", default="train,val")
    build_parser.add_argument("--dry-run", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--label-version", default="base")
    verify_parser.add_argument("--splits", default="train,val")
    verify_parser.add_argument("--max-examples", type=int, default=8)

    args = parser.parse_args(argv)
    splits = tuple(part.strip() for part in getattr(args, "splits", "train,val").split(",") if part.strip())

    if args.command == "archive-generated":
        summary = archive_generated(
            args.dataset_root,
            None if args.skip_output else args.output_root,
            dry_run=args.dry_run,
        )
        mode = "Would move" if args.dry_run else "Moved"
        print(f"{mode} {summary.moved_labels} generated source labels")
        print(f"{mode} {summary.moved_outputs} legacy output directories")
    elif args.command == "build-lists":
        paths = build_pair_lists(
            args.dataset_root,
            label_version=args.label_version,
            output_dir=args.output_dir,
            prefix=args.prefix,
            splits=splits,
            dry_run=args.dry_run,
        )
        mode = "Would write" if args.dry_run else "Wrote"
        for split, path in paths.items():
            print(f"{mode} {split}: {path}")
    elif args.command == "verify":
        failures = verify_dataset(
            args.dataset_root,
            label_version=args.label_version,
            splits=splits,
            max_examples=args.max_examples,
        )
        raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

