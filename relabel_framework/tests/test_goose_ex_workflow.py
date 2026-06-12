from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from PIL import Image


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from src.batch import generate_labels
from src.distribution import compute_base_distribution, remap_distribution
from src.goose_ex import build_goose_ex_index, resolve_dataset_root
from src.goose_ex_maint import archive_generated_labels, build_pair_lists
from src.taxonomy import build_experiment_config, write_yaml


def save_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((3, 4, 3), dtype=np.uint8)
    arr[:, :, 0] = 80
    arr[:, :, 1] = 120
    arr[:, :, 2] = 160
    Image.fromarray(arr).save(path)


def save_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(arr).save(path)


def make_dataset(root: Path) -> None:
    scene = "alice_scenario02"
    stem = "alice_scenario02_sequence07_0003_1697208285946810000"
    save_rgb(root / "images" / "val" / scene / f"{stem}_camera_left.png")
    save_label(root / "labels" / scene / f"{stem}_labelids.png")


def template_config(root_dir: str) -> dict:
    return {
        "dataset": {
            "name": "Goose-EX",
            "root_dir": root_dir,
            "images_dir": "images",
            "labels_dir": "labels",
            "list_files": {"val": "val.txt"},
        },
        "classes": {
            "base_classes": ["undefined", "asphalt", "grass", "obstacle"],
            "palette": [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
            "groups": [0, 1, 2, 3],
        },
        "superclasses": {
            "mapping": {"0": "Background", "1": "Stable", "2": "Vegetation", "3": "Obstacle"},
            "palette": [[128, 128, 128], [0, 255, 0], [255, 255, 0], [255, 0, 80]],
        },
        "output": {
            "new_annotation_suffix": "_old",
            "new_list_prefix": "old_",
            "output_dirs": {
                "annotations": "output/old/labels",
                "lists": "output/old/lists",
                "visuals": "output/old/previews",
            },
        },
    }


def test_resolve_dataset_root_accepts_goose_ex_spelling(tmp_path: Path) -> None:
    root = tmp_path / "data" / "goose_ex"
    root.mkdir(parents=True)
    assert resolve_dataset_root("data/goose-ex", tmp_path) == root.resolve()


def test_goose_ex_index_pairs_val_images_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "data" / "goose_ex"
    make_dataset(root)

    index = build_goose_ex_index(root)

    assert index.selected_label_version == "base"
    assert index.splits == ("val",)
    assert len(index.pairs) == 1
    pair = index.pairs[0]
    assert pair.scene == "alice_scenario02"
    assert pair.image_rel.startswith("images/val/")
    assert pair.label_rel.startswith("labels/")


def test_generate_labels_writes_index_png_and_framework_relative_list(tmp_path: Path) -> None:
    root = tmp_path / "data" / "goose_ex"
    make_dataset(root)
    config, slug = build_experiment_config(
        template_config("data/goose_ex"),
        dataset_root="data/goose_ex",
        experiment_name="terrain test",
        groups=[0, 1, 1, 2],
        superclass_names_input=["Background", "Traversable", "Blocked"],
        superclass_palette=[[128, 128, 128], [0, 255, 0], [255, 0, 80]],
        traversability_scores_input=[-1.0, 1.0, 0.0],
        label_version="base",
        created_at=datetime(2026, 5, 12, 14, 30, 0),
    )
    config_path = tmp_path / "config" / "experiments" / f"{slug}.yaml"
    write_yaml(config, config_path)

    summary = generate_labels(config_path, splits=["val"], num_workers=1, workspace_root=tmp_path)

    assert summary.label_count == 1
    output_label = next((tmp_path / "output" / "goose_ex" / slug / "labels").rglob("*.png"))
    arr = np.array(Image.open(output_label).convert("L"))
    assert arr.tolist() == [[0, 1, 1, 2], [1, 1, 2, 0], [1, 2, 0, 1]]
    list_text = summary.list_files[0].read_text()
    assert "data/goose_ex/images/val/alice_scenario02/" in list_text
    assert f"output/goose_ex/{slug}/labels/alice_scenario02/" in list_text
    runtime_params = (summary.output_dir / "runtime" / "ros_params.yaml").read_text()
    assert "class_palette:" in runtime_params
    assert "semantic_class_scores:" in runtime_params
    assert "- -1.0" in runtime_params
    assert "- 1.0" in runtime_params
    assert "- 0.0" in runtime_params


def test_archive_generated_labels_and_build_base_lists(tmp_path: Path) -> None:
    root = tmp_path / "data" / "goose_ex"
    make_dataset(root)
    scene = "alice_scenario02"
    stem = "alice_scenario02_sequence07_0003_1697208285946810000"
    save_label(root / "labels" / scene / f"{stem}_labelids_test2.png")

    moved, archive_root = archive_generated_labels(root)

    assert moved == 1
    assert not (root / "labels" / scene / f"{stem}_labelids_test2.png").exists()
    assert (archive_root / "test2" / "labels" / scene / f"{stem}_labelids_test2.png").exists()
    assert (root / "labels" / scene / f"{stem}_labelids.png").exists()

    paths = build_pair_lists(root, label_version="base", splits=("val",))

    assert paths["val"].exists()
    assert paths["val"].read_text().strip() == (
        f"images/val/{scene}/{stem}_camera_left.png labels/{scene}/{stem}_labelids.png"
    )


def test_distribution_counts_and_remaps_new_classes(tmp_path: Path) -> None:
    root = tmp_path / "data" / "goose_ex"
    make_dataset(root)
    label_path = next(root.rglob("*_labelids.png"))

    base = compute_base_distribution([label_path], class_count=4, num_workers=1)
    remapped = remap_distribution(base.counts, groups=[0, 1, 1, 2], target_count=3)

    assert base.counts == (3, 3, 3, 3)
    assert remapped.counts == (3, 6, 3)
