# Pear fruitlet sample

This development sample contains 200 original images and YOLO labels with derived COCO annotations for training and evaluation examples.

| Split | Images | Boxes |
|---|---:|---:|
| Training | 160 | 861 |
| Validation | 20 | 104 |
| Test | 20 | 162 |
| Total | 200 | 1,127 |

The class is `Pear fruitlet`: YOLO index 0 and COCO category 1. Images are 640 x 640. `dataset.yaml` configures the PFD runtime; `data.yaml` provides YOLO-style paths. COCO file names are relative to `images/<split>`.

## Selection

The supplied source contains 1,304 images. Selection preserves original splits and original image and label bytes, covering box count, box area, brightness and filename families. A file-level mapping to the manuscript's 3,200-image KFPFD corpus is not included. This sample is provided for software examples and does not define the manuscript dataset or its evaluation split.

Filename series `12kv1r` and `12kv2r` are assigned to training, `12kv3r` to validation and `12kv4r` to testing. These are filename relations; authenticated orchard, tree or acquisition-group labels are unavailable. No observed family or series crosses sample splits.

Twenty source images containing invalid boxes were excluded. `source_audit.json` records the source counts and box checks. `manifest.csv` and `manifest.json` record source splits, filename groups, geometry and hashes. The sample has no occlusion or leaf-error labels.

## Convert labels

```bash
python tools/prepare_data.py --source data/sample --output data/sample --dataset-path data/sample
```

The converter validates labels and preserves split membership. For another corpus, choose a separate output directory and set `--dataset-path` to its deployment location.

The sample assets use [Apache License 2.0](../../LICENSES/Apache-2.0.txt). See [NOTICE](../../NOTICE).
