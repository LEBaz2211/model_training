from __future__ import annotations

from pathlib import Path
from typing import Any

from .taxonomy import normalize_palette, normalize_traversability_scores, superclass_names, write_yaml


def _flat_palette(palette: list[list[int]]) -> list[int]:
    return [int(channel) for color in palette for channel in color[:3]]


def runtime_payload(config: dict[str, Any]) -> dict[str, Any]:
    names = superclass_names(config)
    class_count = len(names)
    superclasses = config.get("superclasses", {})
    palette = normalize_palette(superclasses.get("palette", []), class_count)
    scores = normalize_traversability_scores(superclasses.get("traversability", {}), class_count)
    background_color = palette[0] if palette else [127, 127, 127]
    flat_palette = _flat_palette(palette)

    classes = [
        {
            "id": index,
            "name": names[index],
            "color": palette[index],
            "traversability": float(scores[index]),
        }
        for index in range(class_count)
    ]

    return {
        "classes": classes,
        "segmentation": {
            "class_palette": flat_palette,
        },
        "projection": {
            "background_color": background_color,
        },
        "traversability": {
            "semantic_class_colors": flat_palette,
            "semantic_class_scores": [float(score) for score in scores],
        },
    }


def ros_params_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = runtime_payload(config)
    class_palette = payload["segmentation"]["class_palette"]
    background_color = payload["projection"]["background_color"]
    semantic_colors = payload["traversability"]["semantic_class_colors"]
    semantic_scores = payload["traversability"]["semantic_class_scores"]

    def projector_params() -> dict[str, Any]:
        return {
            "ros__parameters": {
                "background_color": list(background_color),
            }
        }

    def traversability_params() -> dict[str, Any]:
        return {
            "ros__parameters": {
                "semantic_class_colors": list(semantic_colors),
                "semantic_class_scores": list(semantic_scores),
            }
        }

    return {
        "inference_node": {
            "ros__parameters": {
                "class_palette": class_palette,
            }
        },
        "trt_inference": {
            "ros__parameters": {
                "class_palette": class_palette,
            }
        },
        "projector_min": projector_params(),
        "projector": projector_params(),
        "point_projection_node": projector_params(),
        "dynamic_elevation_grid_map_node": traversability_params(),
        "terrain_traversability_node": traversability_params(),
    }


def export_runtime_configs(config: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    runtime_dir = Path(output_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    payload = runtime_payload(config)
    ros_params = ros_params_payload(config)
    paths = {
        "runtime": runtime_dir / "experiment_runtime.yaml",
        "ros_params": runtime_dir / "ros_params.yaml",
        "segmentation": runtime_dir / "segmentation_palette.yaml",
        "projection": runtime_dir / "projection_params.yaml",
        "traversability": runtime_dir / "traversability_semantics.yaml",
    }

    write_yaml(payload, paths["runtime"])
    write_yaml(ros_params, paths["ros_params"])
    write_yaml({"inference_node": ros_params["inference_node"], "trt_inference": ros_params["trt_inference"]}, paths["segmentation"])
    write_yaml(
        {
            "projector_min": ros_params["projector_min"],
            "projector": ros_params["projector"],
            "point_projection_node": ros_params["point_projection_node"],
        },
        paths["projection"],
    )
    write_yaml(
        {
            "dynamic_elevation_grid_map_node": ros_params["dynamic_elevation_grid_map_node"],
            "terrain_traversability_node": ros_params["terrain_traversability_node"],
        },
        paths["traversability"],
    )
    return paths
