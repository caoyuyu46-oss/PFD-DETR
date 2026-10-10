# PFD-DETR

SSWR and RDPF for fruitlet detection, integrated into the official RT-DETR PyTorch implementation. The main model uses ResNet-18, three decoder layers, three decoder scales and 300 queries. The P2 control uses four scales.

Based on [RT-DETR](https://github.com/lyuwenyu/RT-DETR), commit `df2a96c8c3a098567a1149bf0248da6efe6d495d`. Source and checkpoint hashes are in [UPSTREAM.json](UPSTREAM.json). Official Paddle and RT-DETRv2 directories are retained; the commands below use `rtdetr_pytorch`.

## Installation

Use Python 3.11. Create and activate a virtual environment, then install the pinned dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

For CUDA, use matching PyTorch 2.1.2 and torchvision 0.16.2 wheels for your CUDA runtime. CPU operation is supported with `--device cpu`.

## Sample run

The repository includes 200 images: 160 training, 20 validation and 20 test images, with original YOLO labels and derived COCO annotations. See [the data record](data/sample/README.md).

```bash
python tools/train.py --config configs/pfd/sample.yaml --output outputs/sample --seed 0 --device cpu --workers 0 --max-steps 1
python tools/evaluate.py --checkpoint outputs/sample/best.pth --output outputs/sample/evaluation --device cpu --workers 0
python -m pytest -q
```

The short run checks training and evaluation. Its outputs carry a smoke-test marker and are excluded from experiment summaries.

WTConv and FreqFusion controls have additional source dependencies; follow [CONTROLS.md](docs/CONTROLS.md) before running those configurations.

## Dataset preparation

Provide images and YOLO labels in `images/{train,val,test}` and `labels/{train,val,test}`, with one class indexed by zero:

```bash
python tools/prepare_data.py --source /path/to/yolo_dataset --output data/kfpfd --dataset-path data/kfpfd --class-names fruit
```

The converter validates boxes, preserves split membership and creates COCO annotations. Use the original experiment splits for full-data experiments. Dataset paths are defined in `configs/data`. MinneApple-local uses its validation partition for development evaluation. The bundled sample is a separate development subset.

## Training and evaluation

Randomly initialized paired experiments:

```bash
python tools/train.py --config configs/pfd/controlled/baseline.yaml --output outputs/controlled/baseline/seed0 --seed 0 --device cuda:0
python tools/train.py --config configs/pfd/controlled/full.yaml --output outputs/controlled/full/seed0 --seed 0 --device cuda:0
python tools/evaluate.py --checkpoint outputs/controlled/full/seed0/best.pth --output outputs/controlled/full/seed0/evaluation --device cuda:0
```

Run seeds `0 1 2 3 4` in separate output directories. Resume the same configuration and seed with `--resume outputs/.../last.pth`.

COCO-initialized system experiments:

```bash
python tools/download_pretrained.py
python tools/train.py --config configs/pfd/system/kfpfd_baseline.yaml --output outputs/system/baseline/seed0 --seed 0 --device cuda:0
python tools/train.py --config configs/pfd/system/kfpfd_full.yaml --output outputs/system/full/seed0 --seed 0 --device cuda:0
```

Baseline, SSWR, RDPF and full-model configurations within each experiment group share initialization, data and training settings. Common tensors are initialized identically for each paired seed, and their hashes are recorded. The [protocol](docs/PROTOCOL.md) defines experiment mapping and metrics.

Evaluation writes `metrics.json`, validation-selected thresholds and predictions. Aggregate seed records with `tools/summarize.py --metrics <five metrics.json paths> --output outputs/summary.json`. Add `--reference <five baseline metrics.json paths>` for paired differences and confidence intervals.

## Visualization and deployment

```bash
python tools/predict.py --checkpoint outputs/controlled/full/seed0/best.pth --image data/sample/images/test/IMG_20220804_110317_12kv4r1k_11.jpg --output outputs/detections.png --device cuda:0
python tools/heatmap.py --checkpoint outputs/controlled/full/seed0/best.pth --image data/sample/images/test/IMG_20220804_110317_12kv4r1k_11.jpg --output outputs/cam.png --device cuda:0
python tools/export_onnx.py --checkpoint outputs/controlled/full/seed0/best.pth --output outputs/model.onnx
python tools/benchmark.py --checkpoint outputs/controlled/full/seed0/best.pth --output outputs/forward_latency.json --device cuda:0
```

Grad-CAM++ uses the sum of selected class scores, the highest-scoring 2% of queries, a threshold of 0.5 and normalization within predicted boxes followed by image normalization. The default target is `encoder.cam_fusion`, the first top-down concatenation. Raw arrays and method metadata accompany each image. Colors express relative responses within an image.

ONNX export uses fixed batch size 1 and the checkpoint's configured input size (640 by default). The benchmark measures FP32 model forward latency. Upstream TensorRT tools remain in `benchmark`; TensorRT accuracy and image-to-boxes latency require device-specific evaluation.

Measure the CPU-preprocessing-to-CPU-box pipeline with decoded images held in memory:

```bash
python tools/benchmark.py --checkpoint outputs/controlled/full/seed0/best.pth --scope image-to-boxes --images data/sample/images/test/IMG_20220804_110317_12kv4r1k_11.jpg --output outputs/pipeline_latency.json --device cuda:0
python tools/profile_model.py --config configs/pfd/controlled/full.yaml --output outputs/full_profile.json
```

The complexity report counts trainable parameters and executed convolution/linear MACs. Its GFLOPs field uses two FLOPs per MAC and does not include functional attention, interpolation or fixed transforms. The scope is recorded in every report. See [PARAMETER_AUDIT.json](docs/PARAMETER_AUDIT.json).

External detector predictions can be evaluated with the same metrics using `tools/evaluate_predictions.py`. The input is a COCO prediction list in original image coordinates after each detector's native postprocessing. Five-seed component effects, including the eleven Table 5(b) contrasts, are computed by `tools/factorial_effects.py`. Commands and input formats are in [PROTOCOL.md](docs/PROTOCOL.md).

## Release scope

This release contains the official-framework integration, experiment configurations, paired initialization, training, evaluation, frozen interventions, Grad-CAM++, ONNX export and the sample. The experiment matrix and [control record](docs/CONTROLS.md) distinguish executable implementations from external protocol specifications. [RELEASE.md](docs/RELEASE.md) records verification and implementation provenance. Full-data trained PFD-DETR weights and historical per-seed logs are not included. The paper's reported accuracy and latency are not generated from the sample run or the architecture profiles.

New configurations select `model.architecture: paper_v716`, including the corrected RepC3 graph and paper-compatible denoising/loss behavior. Checkpoints from the earlier release without this field continue to load as `legacy_official`; their tensor layout is preserved. Numerical structure checks are recorded in [ARCHITECTURE_VALIDATION.json](docs/ARCHITECTURE_VALIDATION.json).

## License

The integrated program uses [AGPL-3.0](LICENSE), including decoder and loss adaptations from Ultralytics 8.0.201. The original RT-DETR source, other PFD additions and sample assets retain their underlying [Apache-2.0 terms](LICENSES/Apache-2.0.txt). External controls retain their upstream terms. See [NOTICE](NOTICE) for file attribution.

<!-- Modified for the PFD-DETR distribution. -->
