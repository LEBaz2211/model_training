# PIDNet Training and TensorRT Export

This folder is the non-Docker training and deployment handoff for the TensorRT
semantic segmentation path. The goal is to train with MMSegmentation, export a
plain ONNX graph, and build the TensorRT engine on the target machine.

## Why this path

- MMSegmentation already contains PIDNet configs and training code.
- MMDeploy is useful for many MMSeg models, but PIDNet is not a reliable
  shortcut there, so this uses a direct ONNX export wrapper.
- The exported ONNX keeps preprocessing outside the graph. It expects the same
  normalized `NCHW float32` tensor that `src/semantic_segmentation` already
  prepares at runtime.
- TensorRT engines are hardware and TensorRT-version specific, so build the
  `.trt` file on the machine that will run inference.

## Dataset Layout

The starter config uses `BaseSegDataset`, so image and mask filenames must share
the same stem:

```text
data/terrain_6class/
  images/
    train/
      frame_000001.png
    val/
      frame_000101.png
  masks/
    train/
      frame_000001.png
    val/
      frame_000101.png
```

Masks must be single-channel label-id PNGs with values `0..5`; use `255` for
ignore pixels.

## Environment

Use a normal Python environment on the training machine:

```bash
cd model_training/pidnet_trt
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
mim install "mmcv>=2.0.0,<2.2.0"
```

## Small Experiment

Start with a short run before committing to a long training job:

```bash
python -m mim train mmseg configs/pidnet_s_terrain_1024x544.py \
  --work-dir work_dirs/pidnet_s_smoke \
  --cfg-options train_cfg.max_iters=200 train_cfg.val_interval=100 \
    default_hooks.checkpoint.interval=100 train_dataloader.batch_size=2
```

For a real run, increase `max_iters`, tune `batch_size`, and set
`data_root` to your dataset path:

```bash
python -m mim train mmseg configs/pidnet_s_terrain_1024x544.py \
  --work-dir work_dirs/pidnet_s_terrain \
  --cfg-options data_root=/abs/path/to/data/terrain_6class
```

## Export ONNX

Export the best checkpoint to a static-shape ONNX graph:

```bash
python tools/export_mmseg_pidnet_onnx.py \
  configs/pidnet_s_terrain_1024x544.py \
  work_dirs/pidnet_s_terrain/best_mIoU_iter_*.pth \
  artifacts/pidnet_s_terrain_1024x544.onnx \
  --input-size 544 1024 \
  --verify
```

The ONNX input is named `input` and has shape `1x3x544x1024`. The output is
named `logits` and should have shape `1x6x544x1024`.

## Build TensorRT

Run this on the target inference machine:

```bash
tools/build_trt_engine.sh \
  artifacts/pidnet_s_terrain_1024x544.onnx \
  artifacts/pidnet_s_terrain_1024x544.fp16.trt
```

Optional knobs:

```bash
TRT_WORKSPACE_MB=4096 TRTEXEC_EXTRA_ARGS="--verbose" \
  tools/build_trt_engine.sh model.onnx model.fp16.trt
```

## ROS Contract

The current ROS TensorRT node expects:

- input preprocessing: RGB, ImageNet mean/std, `NCHW float32`
- static dimensions: default `1024x544`
- output: logits with shape `1 x num_classes x H x W`
- current class count: `6`

Keep this contract stable unless the runtime node is updated at the same time.

