# Executable controls and evidence boundaries

The four controls below complete previously missing executable variants. They are current release implementations with explicit interfaces. Historical training artifacts were not available to certify that these adapters reproduce the implementations used for the manuscript AP, standard deviations or latency. Static counts below were recomputed from this release; no historical accuracy is encoded in the code.

## Installation and sources

```bash
python -m pip install -r requirements-controls.txt
python tools/download_control_sources.py --control all
python -m pytest tests/test_controls.py -q
```

The downloader fetches only the files in `configs/control_sources.json`, verifies SHA-256 before writing, and stores them under ignored `external_sources/`. Model construction never downloads code. Downloaded source files must be excluded from the distribution archive. Each model construction verifies the pinned source hashes.

| Control | Official source | Locked commit | Distribution |
|---|---|---|---|
| WTConv | [BGU-CS-VIL/WTConv](https://github.com/BGU-CS-VIL/WTConv/tree/4a84da2218924b3973de2235d641f2ecf969c346) | `4a84da2218924b3973de2235d641f2ecf969c346` | MIT license is downloaded and retained alongside the original source |
| FreqFusion | [Linwei-Chen/FreqFusion](https://github.com/Linwei-Chen/FreqFusion/tree/3fb0c70637a3c194fb74294d3ce4681958b26241) | `3fb0c70637a3c194fb74294d3ce4681958b26241` | A root source license was not identified; the source is fetched separately and is not redistributed or relicensed by this release |

All individual file hashes are recorded in `configs/control_sources.json`. The FreqFusion dependency is optional: the rest of the package runs without downloading it. The downloaded original files retain their authors' text and are outside the English release source tree.

## Interfaces and configurations

`configs/pfd/mechanism/wtconv.yaml` replaces SSWR at the 128-channel stride-8 P3 output, retaining RDPF and the three decoder scales. It calls the pinned official `WTConv2d` with kernel 5, one Haar/db1 level, stride 1 and bias enabled, including its base and wavelet scales. There is no extra identity wrapper. Learnable parameters are 16,768: 3,328 base convolution, 128 base scale, 12,800 wavelet convolution and 512 wavelet scale. The 4,096 fixed filter coefficients are stored nontrainable parameters and are excluded from the learnable count.

`configs/pfd/mechanism/freqfusion.yaml` retains SSWR and replaces the RDPF residual path. Raw 64-channel stride-4 P2 is rearranged by pixel unshuffle into 256-channel stride-8 high-resolution input. The 256-channel stride-16 Y4 feature is the low-resolution input. The official module returns refined high-resolution and upsampled low-resolution features; their sum is added to the existing T3 feature, after which the original PAN path is retained. This sum-to-T3 adapter is a release interface choice, not a recovered historical adapter. There are no center-support heads or support loss in this control.

FreqFusion uses compression width 64, low/high-pass kernels 5/3, semi-convolution, high/low-pass filtering, Hamming weighting, four resampling groups and enabled feature resampling, following the [official detection recipe](https://github.com/Linwei-Chen/FreqFusion/blob/3fb0c70637a3c194fb74294d3ce4681958b26241/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_freqfusion.py). The release adapter explicitly selects the PyTorch CARAFE fallback mathematics with reflection padding, independent of whether MMCV is installed. It does not claim numerical identity with MMCV CARAFE. The official module performs the remaining filtering, mask generation and local-similarity sampling. This backend is intended for fresh training, not interchange with unknown historical checkpoints.

`configs/pfd/mechanism/p2_decoding.yaml` adds the raw P2 output to the baseline decoder inputs. Decoder channels are `[64, 256, 256, 256]`, strides are `[4, 8, 16, 32]`, and deformable attention has four levels. The decoder projects P2 once; the encoder does not add a duplicate projection or a new P2 neck. Original P3/P4/P5 decoder projection tensors map from baseline indices 0/1/2 to indices 1/2/3 during shared initialization. Projection and attention tensors whose shapes change are initialized for the new architecture. The 90,912 added learnable parameters comprise 16,896 P2 projection parameters and 74,016 additional sampling-offset/attention parameters across three decoder layers.

`configs/pfd/mechanism/dual_capacity.yaml` adds ordinary residuals at the P3 refinement and T3 fusion locations of the baseline. Each residual uses a 1x1 reduction, depthwise 5x5 filter, 1x1 expansion and a zero-initialized channel-wise residual scale. Hidden widths are 41 and 96, respectively. These natural widths give 11,813 + 52,192 = 64,005 added learnable parameters, compared with 62,207 for SSWR plus RDPF. The second width is a new release choice: the historical 60,992 count did not establish a unique architecture. This control has no support heads or support loss and keeps three decoder scales.

All four configurations inherit the controlled random-initialization training recipe. `mechanism/baseline.yaml` supplies a separately named baseline for the mechanism comparison; `mechanism/default.yaml` supplies its full-model reference. The initializer copies compatible baseline tensors, with the explicit P2 scale mapping described above.

## Static profiles and execution

At input shape 1 x 3 x 640 x 640, evaluation mode, paper-compatible architecture:

| Configuration | Learnable parameters | Frozen stored parameters | GFLOPs, Conv2d + Linear |
|---|---:|---:|---:|
| Baseline | 20,082,772 | 0 | 57.8739712 |
| Full PFD-DETR | 20,144,979 | 0 | 58.6272576 |
| WTConv with RDPF | 20,150,038 | 4,096 | 58.5940480 |
| SSWR with FreqFusion | 20,199,171 | 0 | 58.8300096 |
| P2 decoding | 20,173,684 | 0 | 72.1919488 |
| Dual plain residuals | 20,146,777 | 0 | 58.6813056 |

GFLOPs count executed `nn.Conv2d` and `nn.Linear` hooks at two operations per MAC. This scope excludes functional wavelet transforms, CARAFE, grid sampling, attention matrix multiplication, interpolation, normalization and elementwise work. These omissions matter particularly for the external controls; the values are not comprehensive operation counts or latency estimates.

```bash
python tools/profile_model.py --config configs/pfd/mechanism/wtconv.yaml --output outputs/wtconv_profile.json
python tools/train.py --config configs/pfd/mechanism/wtconv.yaml --seed 0 --output outputs/mechanism/wtconv/seed0 --device cuda:0
```

Replace `wtconv` with `freqfusion`, `p2_decoding` or `dual_capacity` for the other controls. Use actual full-dataset paths in `configs/data/kfpfd.yaml`; the sample dataset is a software example and cannot reproduce full-data measurements.

## Full table coverage

`configs/experiment_matrix.yaml` maps 48 executable experiment configurations to individual table panels. `docs/CONTROL_COVERAGE.json` gives the table-row, frozen-intervention and evaluation mapping. System KFPFD SSWR/RDPF configurations are supplementary configurations: they are not mislabeled as Table 4 rows. `configs/external/protocols.yaml` records the seven external method protocols separately from the executable PFD matrix.

The package provides all six Table 5 arms; Table 6 structural, WTConv, P2 and capacity controls; all Table 8 gating/target/loss arms; seven unique Table 9(b) sensitivity settings; and the four module combinations on each public dataset. Frozen interventions in Tables 7/9(a)/A.2 use completed full-model checkpoints, with 20 permutations for shuffle interventions. Tables 10/12(b)/A.3 use the controlled configurations and common operating-point/counting evaluation. The small/occluded subset and leaf FP analyses require authentic attribute annotations and human leaf-review records; none are fabricated from the sample labels.

Separate experiment names identify cohorts but do not create artificial randomness. Identical configurations and seeds can reproduce identical runs under this deterministic trainer. In particular, the two No support loss configurations are separately named without fabricated seed offsets. Historical independent cohort identities, checkpoint hashes and source logs are required to recover the separate manuscript dispersions. The sensitivity default's three rows refer to one configuration; the code does not reconstruct an undocumented historical default-selection process.

External detector training remains in the corresponding native frameworks, with their manuscript-specific pretrained recipes. `configs/external/protocols.yaml` distinguishes currently verified official source snapshots from unknown historical training commits. It does not represent an external protocol as an implemented PFD model. MRtic-Det's historical implementation and source revision remain unavailable, so no substitute architecture or runnable native command is asserted. External prediction evaluation is available through `tools/evaluate_predictions.py` after exporting native postprocessed predictions in COCO coordinates. See `docs/PROTOCOL.md` for validation threshold selection, paired statistics and manual FP review.

The external protocol file includes native command templates for the six methods with verified official repositories. Clone the listed repository, check out its recorded reference commit, install its own requirements in a separate environment, and create a dataset-adapted native configuration from the locked source. Set the manuscript learning rates, epochs and class count explicitly and supply the intended pretrained checkpoint. Record the resolved recipe and checkpoint hash. These templates are not turnkey historical reproduction commands: historical revisions and several checkpoint identities were not supplied. Native prediction export must map filenames to the exact annotation image IDs and use original-image pixel `xywh` boxes; the common evaluator verifies IDs and coordinates but does not implement each external predictor.

TensorRT export tooling is provided separately. Reproducing the Jetson table additionally requires the stated hardware, engine build and timing conditions; CPU tests and static profiling do not validate those historical measurements.
