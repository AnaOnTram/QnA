# MobileNetV2 → RKNN (RK3588)

Converted on 6 November 2026 (HKT), following the existing
[ResNetV2-50 conversion](../../resnetv2_50/rknn/README.md) and
export script (`Models/resnet50/rknn/NPU/resnet/export_resnet_to_rknn.py`; source workspace only).

## Deliverables

| Artifact | Build precision | Size |
| --- | --- | ---: |
| `mobilenetv2.rknn` | FP16, no quantization | 7.20 MiB |
| `mobilenetv2_int8.rknn` | INT8 W8A8 post-training quantization | 3.92 MiB |
| `mobilenetv2_opset19.onnx` | Compatible FP32 conversion source | 13.54 MiB |

The original `../onnx/mobilenetv2.onnx` and `../trt/mobilenetv2.onnx` are
byte-identical (SHA-256 `9f2d5d94df76f044172dc28345bf975b83e24de1ca33e5892fe074d312d93a77`).
They are untouched. Their opset 20 is rejected by Toolkit2 2.5.0, which accepts
opset ≤19. `prepare_validation.py` uses ONNX's version converter to create
the opset-19 copy and checks bit-identical ONNX Runtime outputs on all 100
validation images. No weights are retrained or substituted.

## Settings inherited from ResNet

- Existing `/home/ross/QA/Models/resnet50/rknn/rknn-venv/bin/python`,
  RKNN-Toolkit2 **2.5.0**, CPU PyTorch **2.4.0**, ONNX **1.18.0**,
  ONNX Runtime **1.26.0**, NumPy **1.26.4**. No environment changes needed.
- Target **RK3588**; source input `input` fixed to **[1, 3, 224, 224]**;
  1,000 output logits.
- Baked RGB mean `[123.675, 116.28, 103.53]` and std
  `[58.395, 57.12, 57.375]`. Pass raw RGB pixels in **[0,255]** through the
  ordinary runtime input interface. Do not normalize a second time.
- FP16 uses `build(do_quantization=False)`. INT8 uses
  `build(do_quantization=True, dataset=dataset.txt)` and the same defaults as
  ResNet: channel quantization, `normal` algorithm, W8A8, optimization level 3.
- `dataset.txt` is an exact copy of the ResNet calibration list: **256
  ImageNet-mini training images**, originally selected with seed 20261104.
  Images are loaded/resized by Toolkit2 as in the original recipe. Calibration
  is not replaced with evaluation crops. There is zero evaluation overlap by
  content SHA-256.

## Validation

The existing MobileNetV2 100-image Imagenette2-320 validation population is
retained, with file hashes checked against its saved manifest. Evaluation
preprocessing follows that model's existing pipeline: RGB → shorter-side
resize 256, PIL bicubic/antialias → center crop 224. The lightweight board
script's crops match torchvision bit-for-bit on all 100 images.

| Execution | Correct / 100 | ONNX top-1 agreement | Logit MAE vs ONNX | Mean cosine |
| --- | ---: | ---: | ---: | ---: |
| Original checkpoint, PyTorch CPU | 83 | 100/100 | — | — |
| Original ONNX, CPU | 83 | 100/100 | 0 | 1 |
| Opset-19 ONNX, CPU | 83 | 100/100 | 0 (bit-exact) | 1 |
| RKNN FP16, PC simulator | 83 | 100/100 | 0.012305 | 0.999983 |
| RKNN INT8, PC simulator | 78 | 90/100 | 0.515319 | 0.974139 |

The original ONNX agrees with the local `mobilenet_v2-b0353104.pth`
checkpoint on every prediction; maximum absolute logit difference is
`2.48e-5`. Both RKNN simulations return finite `[1,1000]` outputs for all inputs.

**INT8 has measurable quantization loss:** five fewer correct predictions and
ten changed predictions relative to ONNX on this sample. Toolkit2 also reports
outlier weights. The initial conservative mean-cosine check of 0.98 failed for
INT8; that diagnostic is explicitly retained as a warning in `results/int8.json`.
It remains a hard check for FP16. The requested, matching ResNet quantization
recipe is retained; no calibration or algorithm tuning was performed to improve
this score. These are pilot results over ten classes, not full ImageNet accuracy.

**Hardware evaluation is complete on the Rock 5C at `192.168.1.39`.** Both
exported artifacts were synced and checked by SHA-256, then executed with RKNN
runtime 2.5.0 / driver 0.9.8. Native input/output are FP16 for the unquantized
model and INT8 for the quantized model. Hardware clean accuracy on the same
100-image cohort is **83/100 FP16** and **80/100 INT8**; these take precedence
over the simulator results above. Median batch-one inference latency is
**5.325 ms FP16** and **2.557 ms INT8**.

The [hardware evaluation and attack report](../evasion_rknn/README.md) includes
all 3,923 ImageNet-mini validation images, same-board CPU/NPU benchmarks,
FGSM/BPDA-PGD, Square, transfer and UINT8-rounding controls. Both models were
fully evaded at 1/255 on their own clean-correct attack subsets, including after
byte rounding. All 6,100 saved candidates were independently replayed exactly;
6,100 additional rounded candidates were scored. The original failed SSH check
at `.30` was a conversion-stage access limitation and is now resolved at `.39`.

## Rebuild and check

Run from `/home/ross/QA`:

```bash
# Uses the existing general environment for torchvision checkpoint validation.
/home/ross/QA/venv/bin/python Models/mobilenetV2/rknn/prepare_validation.py

# Uses the same minimal conversion environment as ResNet.
/home/ross/QA/Models/resnet50/rknn/rknn-venv/bin/python \
  Models/mobilenetV2/rknn/export_mobilenet_to_rknn.py --validate
```

`--precision fp16` and `--precision int8` build just one artifact.
`--validate` checks the complete prepared 100-image sample in the PC simulator.
Rebuilding replaces the corresponding artifacts and reports. As documented in
the ResNet workflow, compiler output is not byte-reproducible, so fresh hashes
can differ while output behavior remains equivalent.

To run an exported file on an RK3588 board with RKNNLite2, NumPy and Pillow
installed, copy `infer_lite.py`, a `.rknn` file, the model's
`imagenet_classes.txt`, and a test image to the board, then run:

```bash
python infer_lite.py mobilenetv2.rknn image.jpg --labels imagenet_classes.txt
python infer_lite.py mobilenetv2_int8.rknn image.jpg --labels imagenet_classes.txt
```

The script feeds a UINT8 RGB NHWC `[1,224,224,3]` crop through the ordinary
runtime input interface and prints the five highest logits/classes. Toolkit2
reports INT8 native input/output for the quantized export; RKNNLite performs
the ordinary input conversion and returns dequantized outputs. Hardware queries
confirmed FP16 native input/output for the unquantized export.

## Evidence

- `results/source_validation.json`: original source/checkpoint parity,
  compatible ONNX hash, and calibration/evaluation separation.
- `results/{fp16,int8}.json`: artifact SHA-256, exact size, build configuration,
  versions, simulator metrics and limitations.
- `results/reference.npz`: raw RGB crops, labels, PyTorch and ONNX logits.
- `results/{fp16,int8}_simulator_logits.npy`: all 100 × 1,000 RKNN logits.
- `results/samples.json`: evaluation paths, labels and source image hashes.
- `SHA256SUMS`: hashes of the two exported binaries and compatible ONNX.
- `conversion.log`, `build_{fp16,int8}.log`: final build/simulation logs.
  `conversion_opset20_failed.log` and `conversion_initial_validation.log`
  preserve the compatibility failure and initial strict cosine diagnostic.
- `check*.onnx`: Toolkit2-generated graph intermediates.
