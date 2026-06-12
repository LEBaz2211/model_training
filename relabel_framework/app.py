from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

FRAMEWORK_ROOT = Path(__file__).resolve().parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.batch import generate_labels
from src.class_index import build_presence_index
from src.distribution import compute_base_distribution, remap_distribution
from src.goose_ex import (
    build_goose_ex_index,
    choose_label_version,
    discover_label_versions,
    inspect_label_values,
    resolve_dataset_root,
    sample_pairs,
)
from src.preview import (
    load_label_array,
    make_relabel_preview,
    remap_label_array,
    resize_image_for_preview,
    resize_label_for_preview,
)
from src.runtime_export import export_runtime_configs
from src.taxonomy import (
    build_experiment_config,
    groups_from_config,
    hex_to_rgb,
    load_yaml,
    normalize_palette,
    rgb_to_hex,
    superclass_names,
    traversability_scores,
    unique_path,
    validate_scheme_dict,
    write_yaml,
)


DEFAULT_CONFIG = FRAMEWORK_ROOT / "config" / "goose_ex_test_1.yaml"
AUTOSAVE_CONFIG = FRAMEWORK_ROOT / "config" / "drafts" / "autosave.yaml"
BASE_DISTRIBUTION_CACHE = FRAMEWORK_ROOT / "config" / "drafts" / "base_distribution_cache.json"


def _relative_to_framework(path: Path) -> str:
    try:
        return path.resolve().relative_to(FRAMEWORK_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def path_from_text(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else FRAMEWORK_ROOT / path


@st.cache_data(show_spinner=False)
def load_config_cached(path_text: str, mtime_ns: int) -> dict:
    _ = mtime_ns
    return load_yaml(path_from_text(path_text))


@st.cache_data(show_spinner=False)
def build_index_cached(root_text: str, label_version: str):
    resolved = resolve_dataset_root(root_text, FRAMEWORK_ROOT)
    return build_goose_ex_index(resolved, label_version=label_version)


@st.cache_data(show_spinner=False)
def label_present_ids_cached(label_path: str, mtime_ns: int) -> tuple[int, ...]:
    _ = mtime_ns
    return tuple(int(value) for value in np.unique(load_label_array(label_path)))


def clean_superclasses(df: pd.DataFrame) -> tuple[list[str], list[list[int]], list[float]]:
    names: list[str] = []
    palette: list[list[int]] = []
    scores: list[float] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        color_text = str(row.get("color", "#808080")).strip()
        try:
            color = hex_to_rgb(color_text)
        except ValueError:
            color = [128, 128, 128]
        try:
            score = float(row.get("traversability", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(-1.0, min(1.0, score))
        names.append(name)
        palette.append(color)
        scores.append(score)
    return names, palette, scores


def config_file_mtime(path_text: str) -> int:
    path = path_from_text(path_text)
    return path.stat().st_mtime_ns if path.exists() else 0


def editor_config_signature(config_rel: str, config: dict) -> str:
    payload = {
        "config_path": config_rel,
        "base_classes": config.get("classes", {}).get("base_classes", []),
        "groups": config.get("classes", {}).get("groups", []),
        "superclasses": config.get("superclasses", {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def initialize_editor_state(config_rel: str, config: dict) -> None:
    signature = editor_config_signature(config_rel, config)
    if st.session_state.get("editor_config_signature") == signature:
        return
    names = superclass_names(config)
    palette = config.get("superclasses", {}).get("palette", [])
    scores = traversability_scores(config)
    st.session_state.superclass_rows = [
        {
            "name": name,
            "color": rgb_to_hex(palette[index]) if index < len(palette) else "#808080",
            "traversability": float(scores[index]) if index < len(scores) else 0.0,
        }
        for index, name in enumerate(names)
    ]
    groups = groups_from_config(config)
    base_count = len(config.get("classes", {}).get("base_classes", []))
    if len(groups) < base_count:
        groups.extend([0] * (base_count - len(groups)))
    st.session_state.mapping_groups = [int(value) for value in groups[:base_count]]
    st.session_state.editor_config_signature = signature
    st.session_state.pop("superclass_editor", None)
    for key in list(st.session_state.keys()):
        if key.startswith("mapping_table_"):
            st.session_state.pop(key, None)


def build_superclass_editor(config: dict) -> tuple[list[str], list[list[int]], list[float]]:
    initial = pd.DataFrame(st.session_state.get("superclass_rows", []))
    edited = st.data_editor(
        initial,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Class"),
            "color": st.column_config.TextColumn("Color"),
            "traversability": st.column_config.NumberColumn(
                "Traversability",
                min_value=-1.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                help="-1 ignores the class in semantic traversability; 0 is blocked, 1 is fully traversable.",
            ),
        },
        key="superclass_editor",
    )
    target_names, target_palette, target_scores = clean_superclasses(edited)
    st.session_state.superclass_rows = [
        {"name": name, "color": rgb_to_hex(target_palette[index]), "traversability": float(target_scores[index])}
        for index, name in enumerate(target_names)
    ]
    return target_names, target_palette, target_scores


def build_mapping_editor(config: dict, target_names: list[str]) -> list[int]:
    base_classes = [str(name) for name in config["classes"]["base_classes"]]
    if not target_names:
        target_names = ["Background"]
    option_labels = [f"{index}: {name}" for index, name in enumerate(target_names)]
    mapping_groups = list(st.session_state.get("mapping_groups", groups_from_config(config)))
    if len(mapping_groups) < len(base_classes):
        mapping_groups.extend([0] * (len(base_classes) - len(mapping_groups)))

    for class_id, group_id in enumerate(mapping_groups):
        if group_id < 0 or group_id >= len(target_names):
            mapping_groups[class_id] = 0
    st.session_state.mapping_groups = mapping_groups[: len(base_classes)]

    page_ids = list(range(len(base_classes)))

    rows = []
    for class_id in page_ids:
        group_id = min(max(int(st.session_state.mapping_groups[class_id]), 0), len(target_names) - 1)
        rows.append(
            {
                "id": class_id,
                "source_class": base_classes[class_id],
                "new_class": option_labels[group_id],
            }
        )

    target_signature = hashlib.sha1("|".join(option_labels).encode("utf-8")).hexdigest()[:10]
    table_key = f"mapping_table_{target_signature}_{hashlib.sha1('|'.join(map(str, page_ids)).encode('utf-8')).hexdigest()[:10]}"
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        height=520,
        disabled=["id", "source_class"],
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "source_class": st.column_config.TextColumn("Goose-EX class"),
            "new_class": st.column_config.SelectboxColumn("New class", options=option_labels),
        },
        key=table_key,
    )

    for _, row in edited.iterrows():
        class_id = int(row["id"])
        try:
            group_id = int(str(row["new_class"]).split(":", 1)[0])
        except ValueError:
            group_id = 0
        st.session_state.mapping_groups[class_id] = min(max(group_id, 0), len(target_names) - 1)

    return list(st.session_state.mapping_groups)


def config_with_current_edits(
    config: dict,
    dataset_root_text: str,
    label_version: str,
    groups: list[int],
    target_names: list[str],
    target_palette: list[list[int]],
    target_scores: list[float],
) -> dict:
    updated = dict(config)
    updated["dataset"] = dict(config.get("dataset", {}))
    updated["dataset"]["root_dir"] = dataset_root_text
    updated["dataset"]["label_version"] = label_version
    updated["classes"] = dict(config.get("classes", {}))
    updated["classes"]["groups"] = [int(value) for value in groups]
    updated["classes"]["id_mapping"] = {index: int(value) for index, value in enumerate(groups)}
    updated["superclasses"] = {
        "mapping": {str(index): name for index, name in enumerate(target_names)},
        "palette": normalize_palette(target_palette, len(target_names)),
        "traversability": {
            str(index): float(target_scores[index]) if index < len(target_scores) else 0.0
            for index in range(len(target_names))
        },
    }
    updated["draft"] = {
        "autosaved_by": "relabel_framework/app.py",
        "note": "This file is rewritten automatically so UI progress survives refreshes and restarts.",
    }
    return updated


def autosave_draft(
    config: dict,
    dataset_root_text: str,
    label_version: str,
    groups: list[int],
    target_names: list[str],
    target_palette: list[list[int]],
    target_scores: list[float],
) -> None:
    draft_config = config_with_current_edits(
        config,
        dataset_root_text,
        label_version,
        groups,
        target_names,
        target_palette,
        target_scores,
    )
    write_yaml(draft_config, AUTOSAVE_CONFIG)


def render_legend(target_names: list[str], target_palette: list[list[int]]) -> None:
    items = []
    for index, name in enumerate(target_names):
        color = rgb_to_hex(target_palette[index])
        items.append(
            f"<div class='legend-item'>"
            f"<span class='legend-swatch' style='background:{color};'></span>"
            f"<span class='legend-label'>{index}: {name}</span>"
            f"</div>"
        )
    st.markdown(
        "<style>"
        ".legend-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 12px;margin-top:4px;}"
        ".legend-item{display:flex;align-items:center;gap:8px;min-width:0;}"
        ".legend-swatch{width:18px;height:18px;border:1px solid #222;display:inline-block;flex:0 0 18px;}"
        ".legend-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
        "</style>"
        f"<div class='legend-grid'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


def _plotly_rgb(color: list[int]) -> str:
    return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"


def categorical_colorscale(palette: list[list[int]]) -> list[list[float | str]]:
    if len(palette) == 1:
        return [[0.0, _plotly_rgb(palette[0])], [1.0, _plotly_rgb(palette[0])]]
    scale: list[list[float | str]] = []
    count = len(palette)
    for index, color in enumerate(palette):
        start = index / count
        end = (index + 1) / count
        scale.append([start, _plotly_rgb(color)])
        scale.append([end, _plotly_rgb(color)])
    return scale


def hover_text_for_labels(label_arr: np.ndarray, names: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in label_arr:
        text_row: list[str] = []
        for value in row:
            class_id = int(value)
            name = names[class_id] if 0 <= class_id < len(names) else "unknown"
            text_row.append(f"{class_id}: {name}")
        rows.append(text_row)
    return rows


def render_hover_mask(
    title: str,
    label_arr: np.ndarray,
    names: list[str],
    palette: list[list[int]],
    height: int,
) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=label_arr,
            text=hover_text_for_labels(label_arr, names),
            colorscale=categorical_colorscale(palette),
            zmin=-0.5,
            zmax=max(len(palette) - 0.5, 0.5),
            showscale=False,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 0, "r": 0, "t": 38, "b": 0},
        xaxis={"visible": False, "constrain": "domain"},
        yaxis={"visible": False, "scaleanchor": "x", "autorange": "reversed"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def render_raw_image(title: str, image, height: int) -> None:
    fig = go.Figure(data=go.Image(z=np.array(image)))
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 0, "r": 0, "t": 38, "b": 0},
        xaxis={"visible": False, "constrain": "domain"},
        yaxis={"visible": False, "scaleanchor": "x"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_new_class_distribution(
    pairs,
    groups: list[int],
    target_names: list[str],
    target_palette: list[list[int]],
    base_class_count: int,
    num_workers: int,
    max_files: int | None,
) -> None:
    selected_pairs = list(pairs)
    if max_files is not None:
        selected_pairs = selected_pairs[:max_files]
    label_paths = tuple(str(pair.label_path) for pair in selected_pairs)
    if not label_paths:
        st.info("No labels available for distribution.")
        return

    label_signature = [
        {
            "path": path,
            "mtime_ns": Path(path).stat().st_mtime_ns,
            "size": Path(path).stat().st_size,
        }
        for path in label_paths
    ]
    cache_key_data = {
        "class_count": int(base_class_count),
        "labels": label_signature,
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_data, sort_keys=True).encode("utf-8")).hexdigest()
    cache = st.session_state.setdefault("base_distribution_cache", {})
    disk_cache = _read_json_cache(BASE_DISTRIBUTION_CACHE)
    controls = st.columns([0.22, 0.78])
    refresh = controls[0].button("Recount Base Distribution")
    if refresh or cache_key not in cache:
        if not refresh and cache_key in disk_cache:
            cache[cache_key] = tuple(int(value) for value in disk_cache[cache_key]["counts"])
            controls[1].caption(
                f"Loaded stored base 64-class counts for {len(label_paths)} files; taxonomy edits only remap these counts."
            )
        else:
            with st.spinner(f"Counting base 64-class pixels in {len(label_paths)} label files..."):
                counts = compute_base_distribution(
                    label_paths,
                    base_class_count,
                    num_workers=max(1, int(num_workers)),
                ).counts
            cache[cache_key] = counts
            disk_cache[cache_key] = {
                "counts": list(counts),
                "class_count": int(base_class_count),
                "file_count": len(label_paths),
            }
            _write_json_cache(BASE_DISTRIBUTION_CACHE, disk_cache)
    else:
        controls[1].caption(
            f"Using cached base 64-class counts for {len(label_paths)} files; taxonomy edits only remap these counts."
        )

    base_counts = cache[cache_key]
    new_dist = remap_distribution(base_counts, groups, len(target_names))
    percentages = new_dist.percentages()
    table = pd.DataFrame(
        {
            "id": list(range(len(target_names))),
            "class": target_names,
            "pixels": list(new_dist.counts),
            "percent": [round(value, 4) for value in percentages],
            "color": [rgb_to_hex(color) for color in target_palette],
        }
    )
    table = table.sort_values("pixels", ascending=False)

    fig = go.Figure(
        data=go.Bar(
            x=table["class"],
            y=table["percent"],
            marker_color=table["color"],
            customdata=np.stack([table["id"], table["pixels"]], axis=-1),
            hovertemplate="Class %{customdata[0]}: %{x}<br>%{y:.4f}%<br>%{customdata[1]:,} pixels<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        yaxis_title="Pixels (%)",
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    st.dataframe(
        table[["id", "class", "pixels", "percent"]],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Counted {len(label_paths)} files, {new_dist.total:,} pixels. Distribution is recomputed after mapping changes.")


def pair_present_ids(pair) -> tuple[int, ...]:
    return label_present_ids_cached(str(pair.label_path), pair.label_path.stat().st_mtime_ns)


def pair_matches_classes(
    pair,
    taxonomy: str,
    selected_ids: set[int],
    groups: list[int],
    match_all: bool,
    presence_index: dict[str, tuple[int, ...]] | None = None,
) -> bool:
    if presence_index is not None and pair.label_rel in presence_index:
        base_ids = set(presence_index[pair.label_rel])
    else:
        base_ids = set(pair_present_ids(pair))
    if taxonomy == "New taxonomy":
        present = {groups[class_id] for class_id in base_ids if 0 <= class_id < len(groups)}
    else:
        present = base_ids
    return selected_ids.issubset(present) if match_all else bool(selected_ids & present)


def find_pair_with_classes(
    pairs,
    taxonomy: str,
    selected_ids: set[int],
    groups: list[int],
    match_all: bool,
    seed: int,
    presence_index: dict[str, tuple[int, ...]] | None = None,
):
    candidates = list(pairs)
    random.Random(seed).shuffle(candidates)
    match_count = 0
    first_match = None
    for pair in candidates:
        if pair_matches_classes(pair, taxonomy, selected_ids, groups, match_all, presence_index=presence_index):
            match_count += 1
            if first_match is None:
                first_match = pair
    return first_match, match_count


def pair_by_key(pairs, key: str | None):
    if not key:
        return None
    for pair in pairs:
        if pair.label_rel == key:
            return pair
    return None


def pairs_by_keys(pairs, keys: list[str] | tuple[str, ...] | None):
    if not keys:
        return []
    pair_map = {pair.label_rel: pair for pair in pairs}
    return [pair_map[key] for key in keys if key in pair_map]


def current_preview_pool(pairs):
    filter_keys = st.session_state.get("class_filter_pair_keys")
    filtered = pairs_by_keys(pairs, filter_keys)
    return filtered if filter_keys is not None else list(pairs)


def history_current_pair(pairs):
    history = st.session_state.get("preview_history", [])
    index = st.session_state.get("preview_history_index", -1)
    if 0 <= index < len(history):
        return pair_by_key(pairs, history[index])
    return None


def push_preview_history(pair) -> None:
    if pair is None:
        return
    history = list(st.session_state.get("preview_history", []))
    index = int(st.session_state.get("preview_history_index", -1))
    if 0 <= index < len(history) and history[index] == pair.label_rel:
        return
    if index < len(history) - 1:
        history = history[: index + 1]
    history.append(pair.label_rel)
    if len(history) > 100:
        history = history[-100:]
    st.session_state.preview_history = history
    st.session_state.preview_history_index = len(history) - 1
    st.session_state.preview_pair_key = pair.label_rel


def move_preview_history(delta: int, pairs):
    history = st.session_state.get("preview_history", [])
    if not history:
        return None
    index = int(st.session_state.get("preview_history_index", len(history) - 1))
    index = min(max(index + delta, 0), len(history) - 1)
    st.session_state.preview_history_index = index
    st.session_state.preview_pair_key = history[index]
    return pair_by_key(pairs, history[index])


def choose_random_pair(pool, seed: int):
    if not pool:
        return None
    return sample_pairs(pool, 1, seed=seed)[0]


def presence_cache_path(dataset_root: Path, label_version: str) -> Path:
    return dataset_root / "cache" / f"class_presence_{label_version}.json"


def presence_index_key(pairs, dataset_root: Path, label_version: str) -> tuple[str, tuple[str, ...]]:
    return (str(presence_cache_path(dataset_root, label_version)), tuple(pair.label_rel for pair in pairs))


def ensure_presence_index(pairs, dataset_root: Path, label_version: str, workers: int, force: bool = False):
    cache_path = presence_cache_path(dataset_root, label_version)
    index, computed, total = build_presence_index(
        pairs,
        cache_path=cache_path,
        num_workers=max(1, int(workers)),
        force=force,
    )
    st.session_state.presence_index_key = presence_index_key(pairs, dataset_root, label_version)
    st.session_state.presence_index = index
    return index, computed, total, cache_path


def current_presence_index(pairs, dataset_root: Path, label_version: str):
    if st.session_state.get("presence_index_key") == presence_index_key(pairs, dataset_root, label_version):
        return st.session_state.get("presence_index")
    return None


def render_three_column_preview(
    pair,
    config: dict,
    groups: list[int],
    target_names: list[str],
    target_palette: list[list[int]],
    max_width: int,
    panel_height: int,
) -> None:
    source_names = [str(name) for name in config["classes"]["base_classes"]]
    source_palette = config["classes"]["palette"]
    raw = resize_image_for_preview(load_image(pair.image_path), max_width=max_width)
    source_labels = load_label_array(pair.label_path)
    relabeled = remap_label_array(source_labels, groups)
    source_preview = resize_label_for_preview(source_labels, max_width=max_width)
    relabeled_preview = resize_label_for_preview(relabeled, max_width=max_width)

    columns = st.columns(3, gap="small")
    with columns[0]:
        render_raw_image("Base image", raw, panel_height)
    with columns[1]:
        render_hover_mask("Relabeled mask", relabeled_preview, target_names, target_palette, panel_height)
    with columns[2]:
        render_hover_mask("Base Goose-EX mask", source_preview, source_names, source_palette, panel_height)


def render_large_preview(
    pair,
    config: dict,
    groups: list[int],
    target_names: list[str],
    target_palette: list[list[int]],
    max_width: int,
) -> None:
    source_names = [str(name) for name in config["classes"]["base_classes"]]
    source_palette = config["classes"]["palette"]
    raw = resize_image_for_preview(load_image(pair.image_path), max_width=max_width)
    source_labels = load_label_array(pair.label_path)
    relabeled = remap_label_array(source_labels, groups)
    source_preview = resize_label_for_preview(source_labels, max_width=max_width)
    relabeled_preview = resize_label_for_preview(relabeled, max_width=max_width)
    chart_height = max(320, int(0.5 * max_width))

    st.image(raw, caption="Base image", use_container_width=True)
    render_hover_mask("Relabeled mask (hover for class name)", relabeled_preview, target_names, target_palette, chart_height)
    render_hover_mask("Base Goose-EX mask (hover for class name)", source_preview, source_names, source_palette, chart_height)


def load_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def save_current_scheme(
    config: dict,
    dataset_root_text: str,
    experiment_name: str,
    groups: list[int],
    target_names: list[str],
    target_palette: list[list[int]],
    target_scores: list[float],
    label_version: str,
) -> Path:
    scheme_config, slug = build_experiment_config(
        config,
        dataset_root=dataset_root_text,
        experiment_name=experiment_name,
        groups=groups,
        superclass_names_input=target_names,
        superclass_palette=target_palette,
        traversability_scores_input=target_scores,
        label_version=label_version,
    )
    errors = validate_scheme_dict(scheme_config)
    if errors:
        raise ValueError("; ".join(errors))
    output_path = unique_path(FRAMEWORK_ROOT / "config" / "experiments" / f"{slug}.yaml")
    write_yaml(scheme_config, output_path)
    output_dir = path_from_text(scheme_config["output"]["output_dirs"]["lists"]).parent
    export_runtime_configs(scheme_config, output_dir / "runtime")
    return output_path


def main() -> None:
    st.set_page_config(page_title="Goose-EX Relabeler", layout="wide")
    st.title("Goose-EX Relabeler")

    default_config_rel = _relative_to_framework(AUTOSAVE_CONFIG if AUTOSAVE_CONFIG.exists() else DEFAULT_CONFIG)
    config_actions = st.sidebar.columns(2)
    if config_actions[0].button("Use Autosave", disabled=not AUTOSAVE_CONFIG.exists()):
        st.session_state.config_path = _relative_to_framework(AUTOSAVE_CONFIG)
        st.rerun()
    if config_actions[1].button("Reset Draft", disabled=not AUTOSAVE_CONFIG.exists()):
        AUTOSAVE_CONFIG.unlink()
        st.session_state.config_path = _relative_to_framework(DEFAULT_CONFIG)
        st.session_state.pop("editor_config_signature", None)
        st.rerun()
    config_rel = st.sidebar.text_input("Base config", default_config_rel, key="config_path")
    config = load_config_cached(config_rel, config_file_mtime(config_rel))
    initialize_editor_state(config_rel, config)

    default_root = config.get("dataset", {}).get("root_dir", "data/goose_ex")
    if not resolve_dataset_root(default_root, FRAMEWORK_ROOT).exists() and (FRAMEWORK_ROOT / "data" / "goose_ex").exists():
        default_root = "data/goose_ex"
    dataset_root_text = st.sidebar.text_input("Dataset root", default_root)
    resolved_root = resolve_dataset_root(dataset_root_text, FRAMEWORK_ROOT)

    versions = discover_label_versions(resolved_root)
    version_options = list(versions.keys()) or ["base"]
    selected_version = st.sidebar.selectbox(
        "Source labels",
        version_options,
        index=version_options.index(choose_label_version(versions)) if choose_label_version(versions) in version_options else 0,
    )
    index = build_index_cached(dataset_root_text, selected_version)
    split_options = list(index.splits) or ["val"]
    selected_splits = st.sidebar.multiselect(
        "Splits",
        split_options,
        default=["val"] if "val" in split_options else split_options,
    )
    pairs = index.pairs_for_splits(selected_splits)

    experiment_name = st.sidebar.text_input("Experiment name", "trav")
    alpha = st.sidebar.slider("Overlay", 0.15, 0.9, 0.58, 0.05)
    preview_mode = st.sidebar.radio("Preview layout", ["Three-column", "Large stacked", "Four-up"], index=0)
    preview_max_width = st.sidebar.slider("Preview detail", 700, 1800, 1200, 100)
    preview_panel_height = st.sidebar.slider("Preview panel height", 260, 620, 340, 20)
    distribution_mode = st.sidebar.radio("Distribution", ["Selected splits", "First N files"], index=0)
    distribution_limit = st.sidebar.number_input("Distribution N", min_value=10, max_value=20000, value=500, step=100)
    workers = st.sidebar.number_input("Workers", min_value=1, max_value=32, value=4, step=1)

    stat_cols = st.columns(4)
    stat_cols[0].metric("Pairs", len(pairs))
    stat_cols[1].metric("Images", index.image_count)
    stat_cols[2].metric("Labels", index.label_count)
    stat_cols[3].metric("Label versions", len(index.label_versions))

    for warning in index.warnings:
        st.warning(warning)
    if pairs:
        stats = inspect_label_values(pairs)
        if stats.looks_grouped and selected_version != "base":
            st.warning(
                f"Sampled labels only contain IDs up to {stats.max_value}. "
                "Use unsuffixed Goose-EX labelids for 64-class taxonomy experiments."
            )

    left, right = st.columns([0.38, 0.62], gap="large")
    with left:
        st.subheader("New Taxonomy")
        target_names, target_palette, target_scores = build_superclass_editor(config)
        render_legend(target_names, target_palette)

    with right:
        st.subheader("Class Mapping")
        groups = build_mapping_editor(config, target_names)

    try:
        autosave_draft(config, dataset_root_text, selected_version, groups, target_names, target_palette, target_scores)
        st.caption(f"Draft autosaved to {_relative_to_framework(AUTOSAVE_CONFIG)}")
    except Exception as exc:
        st.warning(f"Draft autosave failed: {exc}")

    with st.expander("New Class Distribution", expanded=True):
        max_files = None if distribution_mode == "Selected splits" else int(distribution_limit)
        render_new_class_distribution(
            pairs,
            groups,
            target_names,
            target_palette,
            base_class_count=len(config["classes"]["base_classes"]),
            num_workers=int(workers),
            max_files=max_files,
        )

    st.subheader("Preview")
    if "preview_seed" not in st.session_state:
        st.session_state.preview_seed = 1
    source_class_names = [str(name) for name in config["classes"]["base_classes"]]
    with st.expander("Show Example Containing Class", expanded=False):
        chooser_cols = st.columns([0.18, 0.34, 0.12, 0.14, 0.12, 0.1])
        example_taxonomy = chooser_cols[0].radio(
            "Taxonomy",
            ["New taxonomy", "Base Goose-EX taxonomy"],
            horizontal=False,
            key="example_taxonomy",
        )
        if example_taxonomy == "New taxonomy":
            class_options = list(range(len(target_names)))
            format_class = lambda class_id: f"{class_id}: {target_names[class_id]}"
        else:
            class_options = list(range(len(source_class_names)))
            format_class = lambda class_id: f"{class_id}: {source_class_names[class_id]}"
        selected_example_ids = chooser_cols[1].multiselect(
            "Classes",
            class_options,
            format_func=format_class,
            key="example_classes",
        )
        match_all = chooser_cols[2].checkbox("Require all", value=False)
        if chooser_cols[3].button("Rebuild Index", disabled=not pairs):
            with st.spinner("Building class presence index..."):
                _, computed, total, cache_path = ensure_presence_index(
                    pairs,
                    resolved_root,
                    selected_version,
                    int(workers),
                    force=True,
                )
            st.success(f"Indexed {total} labels, recomputed {computed}. Cache: {_relative_to_framework(cache_path)}")
        if chooser_cols[4].button("Apply Filter", disabled=not selected_example_ids or not pairs):
            presence_index = current_presence_index(pairs, resolved_root, selected_version)
            if presence_index is None:
                with st.spinner("Building class presence index for fast search..."):
                    presence_index, computed, total, cache_path = ensure_presence_index(
                        pairs,
                        resolved_root,
                        selected_version,
                        int(workers),
                    )
                st.caption(f"Indexed {total} labels, computed {computed} missing/stale entries.")
            with st.spinner("Searching class index..."):
                selected_ids = set(int(value) for value in selected_example_ids)
                filtered_keys = [
                    pair.label_rel
                    for pair in pairs
                    if pair_matches_classes(
                        pair,
                        example_taxonomy,
                        selected_ids,
                        groups,
                        match_all,
                        presence_index=presence_index,
                    )
                ]
            if not filtered_keys:
                st.warning("No selected sample contains those classes.")
                st.session_state.pop("class_filter_pair_keys", None)
                st.session_state.pop("class_filter_label", None)
            else:
                st.session_state.class_filter_pair_keys = filtered_keys
                selected_names = [format_class(class_id) for class_id in selected_example_ids]
                mode = "all" if match_all else "any"
                st.session_state.class_filter_label = f"{example_taxonomy}, {mode}: {', '.join(selected_names)}"
                st.session_state.preview_history = []
                st.session_state.preview_history_index = -1
                st.session_state.preview_pair_key = None
                st.success(f"Filter active: {len(filtered_keys)} matching samples.")
        if chooser_cols[5].button("Clear", disabled=st.session_state.get("class_filter_pair_keys") is None):
            st.session_state.pop("class_filter_pair_keys", None)
            st.session_state.pop("class_filter_label", None)
            st.session_state.preview_history = []
            st.session_state.preview_history_index = -1
            st.session_state.preview_pair_key = None
        loaded_presence_index = current_presence_index(pairs, resolved_root, selected_version)
        if loaded_presence_index is not None:
            st.caption(f"Fast class index loaded for {len(loaded_presence_index)} selected labels.")

    if not pairs:
        st.error("No image/label pairs found.")
    else:
        preview_pool = current_preview_pool(pairs)
        if st.session_state.get("class_filter_pair_keys") is not None:
            st.info(
                f"Filter active: {len(preview_pool)} matching samples"
                + (f" ({st.session_state.get('class_filter_label')})" if st.session_state.get("class_filter_label") else "")
            )
        preview_controls = st.columns([0.14, 0.14, 0.14, 0.16, 0.42])
        if preview_controls[0].button("Previous", disabled=st.session_state.get("preview_history_index", -1) <= 0):
            move_preview_history(-1, pairs)
        if preview_controls[1].button("Next", disabled=st.session_state.get("preview_history_index", -1) >= len(st.session_state.get("preview_history", [])) - 1):
            move_preview_history(1, pairs)
        if preview_controls[2].button("Random"):
            st.session_state.preview_seed += 1
            random_pair = choose_random_pair(preview_pool, int(st.session_state.preview_seed))
            push_preview_history(random_pair)
        st.session_state.preview_seed = preview_controls[3].number_input(
            "Seed", min_value=0, value=int(st.session_state.preview_seed), step=1
        )

        pair = history_current_pair(pairs)
        if pair is None:
            pair = choose_random_pair(preview_pool, int(st.session_state.preview_seed))
            push_preview_history(pair)
        source_palette = config["classes"]["palette"]
        if pair is None:
            st.warning("No samples available for the current filter.")
        else:
            st.caption(f"{pair.split} / {pair.scene} / {pair.stem}")
            if preview_mode == "Three-column":
                render_three_column_preview(
                    pair,
                    config,
                    groups,
                    target_names,
                    target_palette,
                    max_width=preview_max_width,
                    panel_height=preview_panel_height,
                )
            elif preview_mode == "Large stacked":
                render_large_preview(
                    pair,
                    config,
                    groups,
                    target_names,
                    target_palette,
                    max_width=preview_max_width,
                )
            else:
                previews = make_relabel_preview(
                    pair.image_path,
                    pair.label_path,
                    groups,
                    source_palette,
                    target_palette,
                    target_names,
                    alpha=alpha,
                )
                image_cols = st.columns(4)
                image_cols[0].image(previews["raw"], caption="Base image", use_container_width=True)
                image_cols[1].image(previews["overlay"], caption="Relabeled overlay", use_container_width=True)
                image_cols[2].image(previews["relabeled_mask"], caption="Relabeled mask", use_container_width=True)
                image_cols[3].image(previews["source_mask"], caption="Base Goose-EX labels", use_container_width=True)

    action_cols = st.columns([0.2, 0.2, 0.6])
    if action_cols[0].button("Save Scheme", type="primary"):
        try:
            saved_path = save_current_scheme(
                config,
                dataset_root_text,
                experiment_name,
                groups,
                target_names,
                target_palette,
                target_scores,
                selected_version,
            )
            st.session_state.saved_scheme_path = saved_path.as_posix()
            st.success(f"Saved {_relative_to_framework(saved_path)}")
        except Exception as exc:
            st.error(str(exc))

    saved_scheme = st.session_state.get("saved_scheme_path")
    if action_cols[1].button("Generate Labels"):
        try:
            if not saved_scheme:
                saved_path = save_current_scheme(
                    config,
                    dataset_root_text,
                    experiment_name,
                    groups,
                    target_names,
                    target_palette,
                    target_scores,
                    selected_version,
                )
                st.session_state.saved_scheme_path = saved_path.as_posix()
                saved_scheme = saved_path.as_posix()
            summary = generate_labels(
                saved_scheme,
                splits=selected_splits,
                num_workers=int(workers),
                overwrite=False,
                workspace_root=FRAMEWORK_ROOT,
            )
            st.success(
                f"Generated {summary.label_count} labels in {_relative_to_framework(summary.output_dir)}"
            )
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
