from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ClassDistribution:
    counts: tuple[int, ...]

    @property
    def total(self) -> int:
        return int(sum(self.counts))

    def percentages(self) -> tuple[float, ...]:
        total = self.total
        if total == 0:
            return tuple(0.0 for _ in self.counts)
        return tuple((count / total) * 100.0 for count in self.counts)


def count_label_file(path: str | Path, class_count: int) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    valid = arr[arr < class_count]
    return np.bincount(valid.reshape(-1), minlength=class_count).astype(np.int64)


def compute_base_distribution(
    label_paths: Iterable[str | Path],
    class_count: int,
    num_workers: int = 4,
) -> ClassDistribution:
    paths = [str(path) for path in label_paths]
    if not paths:
        return ClassDistribution(tuple(0 for _ in range(class_count)))

    total = np.zeros(class_count, dtype=np.int64)
    if num_workers <= 1:
        for path in paths:
            total += count_label_file(path, class_count)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for counts in executor.map(count_label_file, paths, [class_count] * len(paths), chunksize=16):
                total += counts
    return ClassDistribution(tuple(int(value) for value in total))


def remap_distribution(base_counts: Iterable[int], groups: list[int], target_count: int) -> ClassDistribution:
    target = np.zeros(target_count, dtype=np.int64)
    for source_id, count in enumerate(base_counts):
        if source_id >= len(groups):
            continue
        target_id = int(groups[source_id])
        if 0 <= target_id < target_count:
            target[target_id] += int(count)
    return ClassDistribution(tuple(int(value) for value in target))

