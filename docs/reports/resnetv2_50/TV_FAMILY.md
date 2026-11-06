# TorchVision-family results — secondary (frozen)

This file is an index, not a new experiment. The study's primary matrix is
built on **one** base model, `timm/resnetv2_50.a1h_in1k`; the TorchVision
ResNet-50 results listed here are the **secondary cross-architecture
replication** and are frozen as of 2026-11-04. Rationale and the measured
confound: [BASE_MODEL_UNIFICATION.md](BASE_MODEL_UNIFICATION.md);
implementation state: [UNIFICATION_STATUS.md](UNIFICATION_STATUS.md); the primary
single-base report: [RESNETV2_50_REPORT.md](RESNETV2_50_REPORT.md).

## Why they are not ranked with the timm rows

The two bases are different networks: zero of the 251 timm initializers match
any of the 320 TorchVision-V2 float tensors (best conv-kernel cosine 0.039),
and the architectures differ (pre-activation vs post-activation, 49 vs 0
BatchNorm nodes). On a matched 2×2 (base × {PyTorch, ORT} fp32, same 75
clean-correct images, same protocol) swapping the base changes 1/255 PGD
evasions by **12–13 images in opposite directions for FGSM vs PGD**, while
swapping the runtime at fp32 changes at most one image. The base-model effect
is therefore as large as the entire runtime/quantizer effect the study reports.
See `BASE_MODEL_UNIFICATION.md` §2.1 for the table.

**Rule:** no table or figure may mix TorchVision rows with timm rows in a
single ranking. Cross-family rows are descriptive only.

## Retained comparison artifacts (frozen)

| Artifact | sha256 |
| --- | --- |
| `evasion/results/comparisons/pytorch.json` | `e62f8d03b510d94b0489213bfbacd7f547804fb7d9a355b3b11900757f8cb61b` |
| `evasion_onnx/results/comparisons/onnx.json` | `a388a3ea2595a3e4c769438224f7f1e41b638b0f26bf3d3d3a6e9e27fe9106b4` |
| `evasion_onnx/results/comparisons/pytorch.json` | `e2f1b52b040eb38648d72a2a267e6d0a65e37a24dc1d04efbd71afe4605be50d` |
| `evasion_onnx/results/comparisons/qnn.json` | `3d4ce4b54accd02dbd6181b7feb717161228e6d031c1a6ebec27583b6bd69edc` |
| `evasion_rknn/results/comparisons/int8_runtime.json` | `530697de7c71b1d83a4e83a0dd84d0929c308af5a07560ae2188c0f0bfb3b7cc` |
| `evasion_rknn/results/comparisons/comparison_validation.json` | `91d6589b12a2cc53d42e342298da3be863af416fa39c6dd6a203a6cdbfc85479` |
| `evasion_rknn/results/comparisons/qnn_alignment_check.json` | `09eb4b2011b81eff8d25b991552a6815ba8d1187b2cc25c61c59662cdce2451f` |

Figures: `evasion/figures/{pytorch,qnn,qnn_v2_matched}.png`,
`evasion_onnx/figures/{onnx,qnn,qnn_robustness}.png`,
`evasion_rknn/figures/{comparison,int8_cross_runtime}.png`.

## The TorchVision deployments (secondary)

| Deployment | Checkpoint / source | Runtime | Retained results |
| --- | --- | --- | --- |
| fp32_v1 / fp32_v2 / fp32_parent | `19c8e357` / `11ad3fa6` / `f46c3f97` | PyTorch CUDA | `evasion/results/{pytorch,parent}/` |
| int8_v2 (FBGEMM) | `resnet50_fbgemm-23753f79.pth` | PyTorch CPU | `evasion/results/pytorch/` |
| fp32_v2.onnx / fp16_v2.onnx | export of `11ad3fa6` | ORT CUDA | `evasion_onnx/results/{fp32_v2,fp16_v2}/` |
| int8_ort_qop_v2 | export of `11ad3fa6` | ORT CPU QOperator | `evasion_onnx/results/int8_ort_qop_v2/` |
| int8_ptq_ort / int8_dq_ort / torchptq_v2_bnfused / torchptq_v2_residual | export of `11ad3fa6` | TensorRT 10.9 INT8 | `evasion_onnx/results/<arm>/`, `evasion_onnx/engines/` |
| QNN HTP V1 (`resnet50_aimet_quantized_6490.bin`) | external ModelScope package, TorchVision-V1 lineage inferred | QNN HTP (QCS6490, QAIRT 2.37) | `evasion/results/qnn/` |

The QNN V1 binary is external and irreproducible (no ONNX/DLC/encodings/
calibration shipped); it is retained only as the cross-family anchor. Its
matched FP32 control is the PyTorch V1 run at resize 232.

## What is still valid from this family

- The **within-family controlled comparisons** remain valid evidence about
  runtime/format/quantization effects *on the TorchVision base*:
  the same-checkpoint ORT-vs-TensorRT subset (80 shared images) in
  `evasion_rknn/INT8_COMPARISON.md`, the historical-parent-vs-FBGEMM pair
  (`evasion/README.md` §"Measured pilot results"), and the TorchVision
  replacement of the base-model sensitivity table.
- The **base-model sensitivity 2×2** is computed from these retained runs and
  is reported as a first-class result, not an appendix.
- Nothing here is used to fill a timm matrix cell, and no timm row is compared
  against these baselines.

## Reproduction

The TorchVision recipes are documented in `quantize/README.md` and
`evasion_onnx/README.md`; `evasion_onnx/scripts/summarize.py` (default
`--source torchvision`) reproduces `evasion_onnx/results/comparisons/onnx.json`
from the retained per-image files (verified: 15,300 rows, 153 comparisons,
56 unions, 80 shared images, byte-identical comparison content).
