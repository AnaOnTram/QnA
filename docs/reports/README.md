# Evasion attack reports

Retained reports for the weights in this repository, with their figures,
comparison tables, methodology and compact validation evidence. These are copies
of existing experiments; packaging did not rerun attacks or change their results.
The two **Core ML** studies are the exception: they are new experiments, run
locally on 10 November 2026 against this repository's Core ML packages.

## Main reports

| Model / deployment | Report | Supporting results |
| --- | --- | --- |
| timm ResNetV2-50, unified nine-deployment comparison | [Unified deployment robustness report](resnetv2_50/RESNETV2_50_REPORT.md) | [Shared-population matrix](resnetv2_50/evasion_onnx/results/comparisons/timm_matrix.md), [figure/table data](resnetv2_50/evasion_onnx/results/comparisons/v2_report_data.json) |
| timm ResNetV2-50, Coral Edge TPU | [Edge TPU attack report](resnetv2_50/TPU/evasion/EVASION_TPU_REPORT.md) | [Cross-runtime comparison](resnetv2_50/TPU/evasion/comparison.json), [attack summary](resnetv2_50/TPU/evasion/results/edgetpu_int8/summary.json) |
| MobileNetV2, TensorRT FP32/FP16/FP8/INT8/INT4 | [TensorRT attack report](mobilenetv2/evasion_trt/README.md) | [Full attack matrix](mobilenetv2/evasion_trt/results/comparisons/matrix.md), [independent verification](mobilenetv2/evasion_trt/results/verification/REPORT.md) |
| MobileNetV2, RKNN FP16/INT8 | [RKNN hardware evaluation and attack report](mobilenetv2/evasion_rknn/README.md) | [Full attack matrix](mobilenetv2/evasion_rknn/results/comparisons/matrix.md), [machine-readable results](mobilenetv2/evasion_rknn/results/comparisons/matrix.json) |
| MobileNetV2, Core ML FP32/FP16/INT8 × CPU / CPU+GPU / CPU+NE (new, 2026-11-10) | [Core ML attack report](mobilenetv2/evasion_coreml/README.md) | [Attack matrix](mobilenetv2/evasion_coreml/results/comparisons/matrix.md), [Square seed check](mobilenetv2/evasion_coreml/results/supplementary/square_seeds.json) |
| timm ResNetV2-50, Core ML FP32/FP16/INT8 × CPU / CPU+GPU / CPU+NE (new, 2026-11-10) | [Core ML attack report](resnetv2_50/evasion_coreml/README.md) | [Attack matrix](resnetv2_50/evasion_coreml/results/comparisons/matrix.md), [Square seed check](resnetv2_50/evasion_coreml/results/supplementary/square_seeds.json) |

The Core ML studies use the same 100-image population and protocol as the
TensorRT and unified ResNetV2-50 studies. Apple's official MobileNetV2 Core ML
models are structurally different. They are kept on the
`reference/apple-mobilenetv2` branch and were not attacked.

The Edge TPU campaign is a separate report; it is not one of the original nine
deployments in the unified matrix. Each report states its own eligible
population, attack budgets and denominators. Preserve those distinctions when
comparing results.

## Methods and runtime detail

- [ResNetV2-50 RKNN deployment and evasion](resnetv2_50/evasion_rknn/README.md).
- [QNN report and matched ResNetV2-source analysis](resnetv2_50/evasion/README.md#matched-comparison-documented-resnetv2-source-versus-the-deployed-graph).
- [ONNX/TensorRT harness and unified timm matrix](resnetv2_50/evasion_onnx/README.md#unified-timm-base-primary-matrix).
- [Unified model provenance and TensorRT replica exception](resnetv2_50/UNIFICATION_STATUS.md).
- [MobileNetV2 TensorRT methodology](mobilenetv2/evasion_trt/METHODOLOGY.md),
  [protocol specification](mobilenetv2/evasion_trt/SPEC.md),
  [differentiable replica details](mobilenetv2/evasion_trt/TYPE1_REPLICA.md),
  and [replica validation](mobilenetv2/evasion_trt/results/_probe/REPORT.md).

Some retained supporting documents contain historical TorchVision ResNet-50
comparisons. They explain the migration and earlier diagnostics; the primary
ResNet report is the unified timm report above. No TorchVision ResNet-50 weights
have been added.

## Archive boundaries and verification

Report bodies, tables and figures retain the source results. Local Markdown
links were adjusted for this archive. References to omitted code or other
source-only files are labeled explicitly and listed in
[SOURCE_REFERENCES.md](SOURCE_REFERENCES.md). Reproduction commands describe the
original experiment workspace; this repository does not contain the attack
implementation, datasets, candidate tensors or full experiment logs.

[manifest.json](manifest.json) records each imported file's original source path,
source SHA-256, packaged SHA-256 and whether its links changed. PNG figures and
JSON/CSV evidence are copied byte-for-byte. Reports, figures and compact evidence
use ordinary Git; the model binaries continue to use Git LFS.

From the repository root, verify the report archive with:

```bash
sha256sum --check docs/reports/SHA256SUMS
```
