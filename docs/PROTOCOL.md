# Experiment protocol

| Group | Configuration directory | Initialization | Evaluation |
|---|---|---|---|
| Component decomposition, Table 5 | `configs/pfd/controlled` | Random | KFPFD test |
| KFPFD system comparison, Table 4 | `configs/pfd/system/kfpfd_*` | Shared COCO checkpoint | KFPFD test |
| Mechanism and target variants, Tables 6 and 8 | `configs/pfd/mechanism` | Random | KFPFD test |
| Sensitivity, Table 9(b) | `configs/pfd/sensitivity` | Random | KFPFD test |
| MangoYOLO, Table 11 | `configs/pfd/system/mango_*` | Shared COCO checkpoint | Official test |
| MinneApple-local, Table 11 | `configs/pfd/system/minneapple_local_*` | Shared COCO checkpoint | Development validation |

`configs/experiment_matrix.yaml` lists executable configurations. Each dataset is trained independently. Names identify experiment groups, not measurements from completed runs.

Table 9(b) uses an independent five-seed training cohort from Table 5. Its three default rows refer to one shared set of runs. The two No support loss entries in Table 8 also use separate cohorts. Their configurations have separate experiment identifiers. A deterministic rerun may yield identical values; software does not add noise or encode historical table values.

Table 9(b) reports test-set comparisons. Defaults are windows `[3, 5, 9]`, RDPF gate floor `0.1` and support-loss weight `0.25`. The release fixes these as the manuscript configuration; it does not reconstruct an undocumented historical parameter-selection procedure.

## Shared training

The base configuration uses 640 x 640 direct resizing, 300 epochs, batch size 8, nominal accumulated batch size 64, AdamW, learning rate `1e-4`, weight decay `1e-4`, betas `(0.937, 0.999)`, cosine decay to 1% of the initial rate, gradient clipping at 10 and EMA. Inputs are RGB in `[0, 1]`. Training is FP32.

The release recipe explicitly uses zero warmup epochs. In the paper-compatible architecture, bias and normalization parameters have zero weight decay; other trainable parameters use the configured decay. The complete trainer is a release implementation; it is not certified as the historical Ultralytics trainer that produced the reported metrics.

For each seed, common baseline tensors are copied identically into paired variants. Added modules have separate random streams; matching added tensors also agree across variants. Augmentation randomness depends on seed, epoch and image identifier. Training records the resolved configuration, initialization hashes, environment, epoch metrics and checkpoints.

The EMA checkpoint with the highest validation AP is retained; ties select the later epoch. Test results are computed after training by the evaluation command.

## Metrics

COCO AP uses IoU thresholds 0.50:0.05:0.95 and at most 100 detections per image. `APS640` evaluates boxes after mapping coordinates to the 640 x 640 input canvas. The detector does not apply NMS.

Operating-point and counting thresholds are selected on validation predictions from `0.05, 0.10, ..., 0.95`. Detection maximizes recall subject to precision at least 0.90; ties select the higher threshold. If no threshold is feasible, the detection operating point is null. Counting minimizes MAE, with the same tie rule. These thresholds are fixed for evaluation.

IoU-0.5 matching assigns score-ordered predictions to unmatched ground-truth boxes before stratifying recall. Counting records TP, FP, FN, MAE, RMSE and signed bias. Severe-occlusion recall requires actual per-instance `occlusion` annotations with value `O2`. Leaf-specific errors require human review of unmatched predictions; the sample contains neither.

## External predictions and reviewed false positives

```bash
python tools/evaluate_predictions.py --annotations data/kfpfd/annotations/test.json --predictions outputs/external/test_predictions.json --validation-annotations data/kfpfd/annotations/val.json --validation-predictions outputs/external/val_predictions.json --configuration external_detector_seed0 --seed 0 --output outputs/external/metrics.json
```

Each prediction contains `image_id`, `category_id`, `bbox` in pixel `xywh` coordinates and `score` in `[0, 1]`. NMS-based external models must apply their native NMS before export; the evaluator does not apply an additional NMS. Predictions below 0.001 are excluded, with at most 100 detections per image. Use `--evaluation-split development` for MinneApple-local.

An optional `--leaf-reviews` JSON file contains the SHA-256 of the original prediction file under `predictions_sha256` and a `reviews` list of `{prediction_id, is_leaf}` records. Identifiers use the explicit `prediction_id` or the zero-based position in the original prediction list. Decisions must cover every retained false positive. The tool counts supplied decisions; it does not infer leaf labels from confidence or geometry.

## Paired component effects

```bash
python tools/factorial_effects.py --baseline outputs/controlled/baseline/seed*/evaluation/metrics.json --sswr outputs/controlled/sswr/seed*/evaluation/metrics.json --support outputs/controlled/support/seed*/evaluation/metrics.json --sswr-support outputs/controlled/sswr_support/seed*/evaluation/metrics.json --rdpf outputs/controlled/rdpf/seed*/evaluation/metrics.json --full outputs/controlled/full/seed*/evaluation/metrics.json --output outputs/component_effects.json
```

Supply five actual metric files per arm, matching seeds 0 through 4. Shell wildcards in the example require a shell that expands them. The output includes all eleven Table 5(b) contrasts. No reported table value is embedded in the calculation.

Seed summaries use sample standard deviation. Paired effects match seed identifiers and report a two-sided Student-t 95% interval over training seeds, without multiple-comparison adjustment.

## Frozen interventions

Use completed full-model checkpoints. SSWR interventions are `structure_zero`, `scale_uniform`, `scale_shuffle`, `scale_shift` and `bypass`. RDPF interventions are `ambiguity_only`, `shallow_support_shuffle`, `deep_support_shuffle`, `independent_both_shuffle`, `gate_mean`, `gate_one` and `bypass`.

```bash
python tools/frozen_interventions.py --checkpoints outputs/controlled/full/seed0/best.pth outputs/controlled/full/seed1/best.pth outputs/controlled/full/seed2/best.pth outputs/controlled/full/seed3/best.pth outputs/controlled/full/seed4/best.pth --module rdpf --intervention independent_both_shuffle --permutations 20 --output outputs/frozen_rdpf.json --device cuda:0
```

Permutation effects are averaged within each checkpoint before aggregation over training seeds. The shift implementation uses a one-cell circular horizontal shift. Shuffle generators are local and do not change the global random stream.
