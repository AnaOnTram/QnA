# Evasion attack reports

Retained reports for the weights in this repository, with their figures,
comparison tables, methodology and compact validation evidence. These are copies
of existing experiments; packaging did not rerun attacks or change their results.
The two **Core ML** studies were run locally on 10 November 2026 against this
repository's Core ML packages. The MobileNetV2 TPU and QNN studies were also
completed on 10 November and subsequently imported from the source workspace;
archiving them did not rerun their models or attacks.

The expanded ResNetV2-50, consolidated MobileNetV2 and cross-model reports are
newly composed analyses of those retained measurements. Their calculation
script reconciles original summary hashes and counts; it performs no new
inference or attacks. Original experiment reports remain intact.

## Main reports

The current cross-runtime syntheses incorporate the later Core ML, TPU and QNN
studies. The cross-model report compares runtime/quantization patterns across
the two bases, with matched-study views and explicit denominator limits.

| Model / deployment | Report | Supporting results |
| --- | --- | --- |
| timm ResNetV2-50, 19 configurations including Core ML and Coral | [Expanded cross-runtime report](resnetv2_50/RESNETV2_50_CROSS_RUNTIME_REPORT.md) | [Normalized results](cross_runtime/normalized_results.json), [all conditions and intervals](cross_runtime/all_conditions.csv) |
| MobileNetV2, 20 configurations including Core ML, QNN and Coral | [Cross-runtime report](mobilenetv2/MOBILENETV2_REPORT.md) | [Normalized results](cross_runtime/normalized_results.json), [all conditions and intervals](cross_runtime/all_conditions.csv) |
| ResNetV2-50 versus MobileNetV2 | [Model, runtime and quantization comparison](CROSS_MODEL_REPORT.md) | [Matched-study quantization contrasts](cross_runtime/quantization_effects.json), [model rate differences](cross_runtime/cross_model_comparisons.json) |
| timm ResNetV2-50, original nine-deployment comparison | [Historical unified deployment report](resnetv2_50/RESNETV2_50_REPORT.md) | [Shared-population matrix](resnetv2_50/evasion_onnx/results/comparisons/timm_matrix.md), [figure/table data](resnetv2_50/evasion_onnx/results/comparisons/v2_report_data.json) |
| timm ResNetV2-50, Coral Edge TPU | [Edge TPU attack report](resnetv2_50/TPU/evasion/EVASION_TPU_REPORT.md) | [Cross-runtime comparison](resnetv2_50/TPU/evasion/comparison.json), [attack summary](resnetv2_50/TPU/evasion/results/edgetpu_int8/summary.json) |
| MobileNetV2, TensorRT FP32/FP16/FP8/INT8/INT4 | [TensorRT attack report](mobilenetv2/evasion_trt/README.md) | [Full attack matrix](mobilenetv2/evasion_trt/results/comparisons/matrix.md), [independent verification](mobilenetv2/evasion_trt/results/verification/REPORT.md) |
| MobileNetV2, RKNN FP16/INT8 | [RKNN hardware evaluation and attack report](mobilenetv2/evasion_rknn/README.md) | [Full attack matrix](mobilenetv2/evasion_rknn/results/comparisons/matrix.md), [machine-readable results](mobilenetv2/evasion_rknn/results/comparisons/matrix.json) |
| MobileNetV2, TFLite CPU / Coral Edge TPU INT8 | [Edge TPU attack report](mobilenetv2/TPU/evasion/EVASION_TPU_REPORT.md) | [Independent verification and transfer](mobilenetv2/TPU/evasion/results/verification/verification.json), [deployment comparisons](mobilenetv2/TPU/evasion/results/comparisons/all_deployments.json) |
| MobileNetV2, QNN HTP INT8 / FP32 CPU | [Q6A performance, accuracy and attack report](mobilenetv2/evasion_qnn/REPORT.md) | [Performance](mobilenetv2/evasion_qnn/results/performance.json), [attack verification](mobilenetv2/evasion_qnn/results/verification/verification.json), [held-out clean evaluation](mobilenetv2/evasion_qnn/results/full_validation/evaluation.json) |
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

- [Cross-runtime synthesis evidence and reproduction](cross_runtime/README.md), including source hashes, exact denominators, supplementary seeds and reporting-validation limits.
- [MobileNetV2 TPU/QNN archive guide](mobilenetv2/ARCHIVE.md), [TPU conversion and validation](mobilenetv2/TPU/README.md), [QNN conversion and validation](mobilenetv2/qnn/README.md), and [Q6A measurement and evasion methods](mobilenetv2/evasion_qnn/README.md).
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
implementation, datasets, candidate tensors or full experiment logs. Selected
MobileNetV2 conversion/runtime helpers and compact compiler logs are retained;
their original workspace assumptions are explained in the archive guide.

[manifest.json](manifest.json) records each imported file's original source path,
source SHA-256, packaged SHA-256 and whether its links changed. PNG figures and
JSON/CSV evidence and retained helpers are copied byte-for-byte. Reports, figures and compact evidence
use ordinary Git; the model binaries continue to use Git LFS.

From the repository root, verify the report archive with:

```bash
sha256sum --check docs/reports/SHA256SUMS
```
