# Relabel Framework

This folder contains tools for building, previewing, saving, and applying semantic segmentation relabeling schemes. The main target is Goose-EX, but the older script-based RUGD/Rellis workflows are still present.

The recommended workflow is:

1. Verify the Goose-EX source dataset.
2. Launch the Streamlit UI in a Python virtual environment.
3. Create and preview a taxonomy on random images.
4. Save the scheme as an experiment config.
5. Generate full relabeled index PNGs and train/val list files.

## Directory Layout

```text
relabel_framework/
├── app.py                         # Streamlit UI
├── config/                        # Base configs and saved experiment configs
├── data/goose_ex/                 # Goose-EX source dataset
│   ├── images/train/<scene>/
│   ├── images/val/<scene>/
│   ├── labels/train/<scene>/
│   ├── labels/val/<scene>/
│   ├── lists/                     # Generated source pair lists
│   └── archive/                   # Archived generated labels/results
├── output/goose_ex/<scheme_slug>/ # New relabeling experiment outputs
├── src/                           # Reusable Python modules and CLIs
├── tests/                         # Unit tests for Goose-EX workflow
└── requirements.txt
```

## Virtual Environment Setup

From the repo root:

```bash
cd model_training/relabel_framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use the helper script:

```bash
cd model_training/relabel_framework
bash setup_venv.sh
source .venv/bin/activate
```

Whenever you come back later:

```bash
cd model_training/relabel_framework
source .venv/bin/activate
```

## Goose-EX Expected Source Format

The active base dataset should look like this:

```text
data/goose_ex/
├── images/train/<scene>/<stem>_camera_left.png
├── images/val/<scene>/<stem>_windshield_vis.png
├── labels/train/<scene>/<stem>_labelids.png
└── labels/val/<scene>/<stem>_labelids.png
```

The tools also tolerate related image suffixes such as `_camera_right`, `_windshield_vis`, `_front`, and `_color`. Labels used as the base taxonomy must be unsuffixed `*_labelids.png` files. Generated labels such as `*_labelids_test2.png` should stay in `archive/` or `output/`, not in the active source label tree.

## Verify Dataset and Build Pair Lists

Run this after adding or changing Goose-EX files:

```bash
python -m src.goose_ex_maint --dataset-root data/goose_ex verify --label-version base --splits train,val
```

A healthy complete state should show nonzero train and val pairs with no missing examples. Current complete Goose-EX source should be about:

```text
train pairs: 11834
val pairs: 1369
```

Build explicit source pair lists for training code:

```bash
python -m src.goose_ex_maint --dataset-root data/goose_ex build-lists --label-version base --splits train,val
```

This writes:

```text
data/goose_ex/lists/base_train.txt
data/goose_ex/lists/base_val.txt
data/goose_ex/lists/base_train_val.txt
```

Each row is dataset-root-relative:

```text
images/train/<scene>/<stem>_camera_left.png labels/train/<scene>/<stem>_labelids.png
images/val/<scene>/<stem>_windshield_vis.png labels/val/<scene>/<stem>_labelids.png
```

The Streamlit UI and new generation CLI do not require these list files; they scan the dataset directly. The lists are for downstream training pipelines that expect txt pairs.

## Launch the UI

Activate the venv first:

```bash
cd model_training/relabel_framework
source .venv/bin/activate
python -m streamlit run app.py
```

In the browser:

- `Base config`: keep `config/goose_ex_test_1.yaml`
- `Dataset root`: use `data/goose_ex`
- `Source labels`: choose `base`
- `Splits`: start with `val` for fast testing, then add `train`
- Edit the superclass names, colors, and traversability scores.
- Traversability scores use `1.0` for fully traversable, `0.0` for blocked, and `-1.0` to ignore/unknown in the traversability node.
- Map the 64 Goose-EX classes to your new classes.
- Use `Preview layout: Three-column` to inspect the base image, relabeled mask, and base Goose-EX mask in one row.
- Use `Preview panel height` to fit all three panels on your screen without scrolling.
- Switch to `Large stacked` when you want one wide panel at a time.
- Hover over either mask to see the class id and class name for the pixel under the cursor.
- Increase `Preview detail` if you want a sharper interactive preview, or lower it if the browser feels heavy.
- Inspect `New Class Distribution` to see pixel percentages for the current remapped taxonomy.
- Use `Distribution: Selected splits` for exact train/val counts, or `First N files` for a faster approximation while iterating.
- Base 64-class pixel counts are cached in the Streamlit session; taxonomy edits only remap cached counts.
- Click `Recount Base Distribution` after changing dataset files or if you want to refresh the cached counts.
- Open `Show Example Containing Class` to jump to a random sample that contains selected classes.
- The example picker works for either the new taxonomy or the original 64-class Goose-EX taxonomy.
- For multiple selected classes, enable `Require all` when the sample must contain every selected class.
- The first class search builds `data/goose_ex/cache/class_presence_base.json`; later searches use this fast class-presence index.
- Click `Rebuild Index` after changing source label files.
- Click `Apply Filter` to constrain the normal preview to matching samples.
- Click `Random` to reroll; it uses the active filter when one is set, otherwise it uses all selected split samples.
- Use `Previous` and `Next` to move through examples you have already viewed.
- Click `Clear` in the class filter controls to return Random to the full selected split pool.
- Click `Save Scheme` when happy.
- Click `Generate Labels` to create the full relabeled dataset.

Saved experiment configs go to:

```text
config/experiments/<scheme_slug>.yaml
```

The UI also keeps an automatic draft at:

```text
config/drafts/autosave.yaml
```

On restart, the app opens this autosave draft by default when it exists. Use `Reset Draft` in the sidebar when you want to discard autosaved progress and return to the base config. Use `Save Scheme` when you want to freeze a version as an immutable experiment config.

Class mappings are stored internally by numeric target-class ID, not by display text. This prevents mapping rows from silently changing if class names are edited or duplicated.

The class-mapping UI is one scrollable table with all Goose-EX source classes. Edits are persisted in Streamlit session state and autosaved to disk whenever the app reruns.

Generated labels and lists go to:

```text
output/goose_ex/<scheme_slug>/
├── labels/<scene>/*_labelids_<scheme_slug>.png
├── lists/<scheme_slug>_train.txt
├── lists/<scheme_slug>_val.txt
├── runtime/
│   ├── experiment_runtime.yaml
│   ├── ros_params.yaml
│   ├── segmentation_palette.yaml
│   ├── projection_params.yaml
│   └── traversability_semantics.yaml
└── manifest.yaml
```

## Runtime Configs

Each saved or generated Goose-EX experiment exports runtime config files under:

```text
output/goose_ex/<scheme_slug>/runtime/
```

Use `ros_params.yaml` when launching the full stack. It contains:

- `inference_node.ros__parameters.class_palette` for Paddle segmentation.
- `trt_inference.ros__parameters.class_palette` for TensorRT segmentation.
- `projector_min.ros__parameters.background_color` for the projection node.
- `dynamic_elevation_grid_map_node.ros__parameters.semantic_class_colors` and `semantic_class_scores` for traversability.

The export also includes aliases for launch files that rename nodes: `projector`, `point_projection_node`, and `terrain_traversability_node`.

The segmentation nodes publish masks using the experiment palette. The projectors pass those RGB colors into `/colored_point_cloud`. The traversability node then converts the same RGB colors into the per-class scores chosen in the UI.

## CLI Workflow

Validate a saved scheme:

```bash
python -m src.relabel_cli validate --config config/experiments/<scheme_slug>.yaml
```

Generate only validation labels:

```bash
python -m src.relabel_cli generate --config config/experiments/<scheme_slug>.yaml --splits val --num-workers 8
```

Generate train and val:

```bash
python -m src.relabel_cli generate --config config/experiments/<scheme_slug>.yaml --splits train,val --num-workers 8
```

Regenerate existing outputs:

```bash
python -m src.relabel_cli generate --config config/experiments/<scheme_slug>.yaml --splits train,val --num-workers 8 --overwrite
```

## Cleanup and Archives

Move generated labels out of the active source tree:

```bash
python -m src.goose_ex_maint --dataset-root data/goose_ex archive-generated
```

Dry-run first if you want to inspect what would move:

```bash
python -m src.goose_ex_maint --dataset-root data/goose_ex archive-generated --dry-run
```

Archived generated source labels are stored under:

```text
data/goose_ex/archive/generated_labels/<version>/labels/
```

Legacy outputs are stored under:

```text
output/archive/legacy_<timestamp>/
```

## Legacy Scripts

The older scripts remain available:

```bash
python src/main.py --config config/rugd_test_1.yaml --temp_visual --sample_image RUGD_annotations/park-2/park-2_01901.png --frame_image RUGD_frames-with-annotations/park-2/park-2_01901.png
python src/view_superclasses.py --input output/rugd/test1_annotations/trail-6/trail-6_02091_test1.png --config config/rugd_test_1.yaml --output output/colorized.png
python src/composite_rellis_labels.py --config config/rellis_test_1.yaml --output_dir output/rellis/composites --num_workers 8
```

Prefer the Streamlit UI plus `src.relabel_cli` for new Goose-EX experiments.

## Tests

From the repo root:

```bash
python3 -m pytest model_training/relabel_framework/tests -q
python3 -m compileall -q model_training/relabel_framework/src model_training/relabel_framework/app.py
```

From inside the venv:

```bash
pytest tests -q
python -m compileall -q src app.py
```

## Troubleshooting

- If the UI shows no pairs, run `verify` and check that labels are unsuffixed `*_labelids.png`.
- If train or val has missing examples, the image or label download is incomplete for those scenes.
- If `streamlit` is not found, activate the venv and reinstall `requirements.txt`.
- If old `test1`/`test2` labels show up as source labels, run `archive-generated`.
- If a downstream training pipeline cannot find files from a list, rebuild lists with `build-lists` and verify paths relative to `data/goose_ex`.
