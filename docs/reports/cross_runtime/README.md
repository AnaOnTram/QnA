# Cross-runtime and cross-model reporting evidence

The current synthesis is in the [ResNetV2-50 report](../resnetv2_50/RESNETV2_50_CROSS_RUNTIME_REPORT.md),
[MobileNetV2 report](../mobilenetv2/MOBILENETV2_REPORT.md) and
[cross-model report](../CROSS_MODEL_REPORT.md). These reports use retained
measurements through 10 November 2026; no new inference or attack was run.

## Reproduce the calculations and figures

From the archive root:

```bash
python docs/reports/cross_runtime/build_reports.py
sha256sum --check SHA256SUMS
sha256sum --check docs/reports/SHA256SUMS
```

The script requires NumPy and Matplotlib (authored with NumPy 2.5.4 and
Matplotlib 3.11.2; the actual versions used for final validation are recorded
in `validation.json`). It reads only archive-local files. Report prose is
maintained in the three Markdown documents; the script regenerates table
fragments, chart assets and JSON/CSV evidence. After changing content or a
renderer version, update the report manifest and checksum ledger as described
in [Git tracking](../../GIT_TRACKING.md). Figure bytes may vary with renderer
versions; numerical tables must still reconcile.

## Evidence files

| File | Purpose |
| --- | --- |
| [normalized_results.json](normalized_results.json) | 39 deployments, 842 conditions, integer numerators/denominators, recomputed rates and Wilson intervals |
| [all_conditions.csv](all_conditions.csv) | Flat complete condition ledger, including controls and transfer conditions |
| [cross_model_comparisons.json](cross_model_comparisons.json) | Own-denominator model contrasts; missing conditions remain null |
| [quantization_effects.json](quantization_effects.json) | FP16-minus-quantized success on source-study common-clean masks |
| [square_seed_results.json](square_seed_results.json) | Three supplementary Square seeds per cell, means recomputed from counts, observed ranges |
| [chart_data.json](chart_data.json) | Exact reviewed inputs for each of the five figures |
| [source_index.json](source_index.json) | SHA-256 of reporting code and measurement inputs, with calculation and interpretation rules |
| [validation.json](validation.json) | Reconciliation coverage and limits; no claim of new candidate replay |

`inputs/resnetv2_50_qnn_summary.json` and `inputs/resnetv2_50_samples.json`
are byte-for-byte copies of the missing matched-QNN summary and the ResNet
cohort ledger from the original source workspace. Their original paths and
hashes are recorded in the archive manifest. The summary retains its historical
FP32 reference as well as the NPU rows; this synthesis selects only `model=npu`
as the QNN deployment. MobileNet's cohort ledger already exists under
`mobilenetv2/TPU/results/samples.json`. Image SHA-256s and labels match across
both ordered 100-image source cohorts, while model-specific preprocessing
differs.

The full matrices use own clean-correct denominators. The shared tables use
only the intersections already available in the corresponding source study:
ResNet original nine-deployment n=71, MobileNet RKNN n=80, MobileNet TensorRT
n=78, Core ML ResNet n=81 and Core ML MobileNet n=82. These are different
populations. There is no new 19-, 20- or 39-way intersection and no cross-model
candidate-transfer evaluation.

Wilson intervals apply to each cell's binomial pilot rate. Supplementary seed
ranges are not confidence intervals. Rate differences and differences of
differences are descriptive; their paired uncertainty cannot be reconstructed
from counts without the relevant per-image discordances. Neither a failed
finite search nor an INT8 label establishes a robustness guarantee.
