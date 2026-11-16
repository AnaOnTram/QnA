# Cross-runtime and cross-model reporting evidence

The current synthesis is in the [ResNetV2-50 report](../resnetv2_50/RESNETV2_50_CROSS_RUNTIME_REPORT.md),
[MobileNetV2 report](../mobilenetv2/MOBILENETV2_REPORT.md) and
[cross-model report](../CROSS_MODEL_REPORT.md). These reports use retained
measurements through 16 November 2026, including the Coral PGD+ supplement and eight newly completed deployment campaigns.
The reporting build itself performs no inference or attacks.

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

## Coral PGD+ update: 16 November 2026

The ResNet Coral condition is now **53/80 (66.25%)**. The stronger check
retains primary PGD successes and adds eight successes from 35 survivors.
The canonical Coral summary preserves its original attack rows and links the
[original snapshot](../resnetv2_50/TPU/evasion/results/edgetpu_int8/summary_before_pgd_plus_20261116.json),
[supplemental summary](../resnetv2_50/TPU/evasion/results/pgd_plus_20261116_01/summary.json),
[200 per-image rows](../resnetv2_50/TPU/evasion/results/pgd_plus_20261116_01/per_image.csv),
and [independent hardware replay](../resnetv2_50/TPU/evasion/results/pgd_plus_20261116_01/validation.json).
The reporting build checks these compact records and primary-success retention;
it does not replay the candidate tensors. The local source workspace retains
those tensors and the runner. The archive continues to exclude them.

## Eight completed campaigns: 16 November 2026

The [completion archive](../missing_attacks_20261116/README.md) adds five
ResNetV2-50 and three MobileNetV2 deployments. Its 168 conditions reconcile
against 16,800 compact per-image rows and successful independent native replay
records. Archived graph/engine hashes match the weight manifest. This build
checks those records without repeating inference or replaying omitted tensors.
The original 39-deployment source registry remains unchanged.

## Evidence files

| File | Purpose |
| --- | --- |
| [normalized_results.json](normalized_results.json) | 47 deployments, 1,011 conditions, integer numerators/denominators, recomputed rates and Wilson intervals |
| [all_conditions.csv](all_conditions.csv) | Flat complete condition ledger, including controls and transfer conditions |
| [cross_model_comparisons.json](cross_model_comparisons.json) | Own-denominator model contrasts; missing conditions remain null |
| [quantization_effects.json](quantization_effects.json) | FP16-minus-quantized success on source-study common-clean masks |
| [square_seed_results.json](square_seed_results.json) | Three supplementary Square seeds per cell, means recomputed from counts, observed ranges |
| [chart_data.json](chart_data.json) | Exact reviewed inputs for each of the five figures |
| [source_index.json](source_index.json) | SHA-256 of reporting code and measurement inputs, with calculation and interpretation rules |
| [validation.json](validation.json) | Reconciliation coverage and limits, including the archived Coral supplemental replay |

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
populations. The new completion campaign has its own three-MobileNet/five-ResNet masks; there is no new 24-, 23- or 47-way intersection and no cross-model
candidate-transfer evaluation.

Wilson intervals apply to each cell's binomial pilot rate. Supplementary seed
ranges are not confidence intervals. Rate differences and differences of
differences are descriptive; their paired uncertainty cannot be reconstructed
from counts without the relevant per-image discordances. Neither a failed
finite search nor an INT8 label establishes a robustness guarantee.
