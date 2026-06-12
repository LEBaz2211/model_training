from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
from typing import Iterable

import numpy as np
from PIL import Image


def label_unique_ids(path: str | Path) -> list[int]:
    arr = np.array(Image.open(path).convert("L"))
    return [int(value) for value in np.unique(arr)]


def _compute_one(args: tuple[str, str, int]) -> tuple[str, dict]:
    key, path_text, mtime_ns = args
    return key, {"path": path_text, "mtime_ns": mtime_ns, "ids": label_unique_ids(path_text)}


def load_presence_cache(cache_path: str | Path) -> dict[str, dict]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        data = json.load(handle)
    entries = data.get("entries", data)
    return entries if isinstance(entries, dict) else {}


def save_presence_cache(cache_path: str | Path, entries: dict[str, dict]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {"version": 1, "entries": entries}
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
    tmp_path.replace(path)


def build_presence_index(
    pairs: Iterable,
    cache_path: str | Path,
    num_workers: int = 4,
    force: bool = False,
) -> tuple[dict[str, tuple[int, ...]], int, int]:
    """Return label_rel -> present ids, updating a persistent cache if needed."""
    entries = {} if force else load_presence_cache(cache_path)
    tasks: list[tuple[str, str, int]] = []
    requested_keys: set[str] = set()

    for pair in pairs:
        key = pair.label_rel
        requested_keys.add(key)
        mtime_ns = pair.label_path.stat().st_mtime_ns
        cached = entries.get(key)
        if force or not cached or cached.get("mtime_ns") != mtime_ns:
            tasks.append((key, str(pair.label_path), mtime_ns))

    if tasks:
        if num_workers <= 1:
            for task in tasks:
                key, value = _compute_one(task)
                entries[key] = value
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_compute_one, task) for task in tasks]
                for future in as_completed(futures):
                    key, value = future.result()
                    entries[key] = value
        save_presence_cache(cache_path, entries)

    index = {
        key: tuple(int(value) for value in entries[key].get("ids", []))
        for key in requested_keys
        if key in entries
    }
    return index, len(tasks), len(requested_keys)

