# MobileNetV2 TPU and QNN archive

Archived on **10 November 2026 (Hong Kong time)** from `QA/Models/mobilenetV2/`.
These are standalone copies of the completed conversions and experiments.
Packaging did not rebuild models, rerun attacks, contact the board or change
the source workspace. Statements about a running Q6A service describe the
recorded experiment, not a live service check.

| Runtime | Final model files | Documentation and evidence |
| --- | --- | --- |
| TFLite CPU INT8 | [CPU model](../../../weights/mobilenetv2/tflite/mobilenetv2_int8_nhwc.tflite) | [Conversion, contract and hardware validation](TPU/README.md), [validation JSON](TPU/results/validation.json) |
| Coral Edge TPU INT8 | [Compiled TPU model](../../../weights/mobilenetv2/edgetpu/mobilenetv2_int8_nhwc_edgetpu.tflite) | [Evasion report](TPU/evasion/EVASION_TPU_REPORT.md), [verification](TPU/evasion/results/verification/verification.json), [comparison data](TPU/evasion/results/comparisons/all_deployments.json) |
| QAIRT / QNN | [FP32 DLC](../../../weights/mobilenetv2/qnn/mobilenetv2_fp32.dlc), [INT8 DLC](../../../weights/mobilenetv2/qnn/mobilenetv2_int8.dlc), [QCS6490 INT8 context](../../../weights/mobilenetv2/qnn/mobilenetv2_quantized.bin) | [Conversion and I/O](qnn/README.md), [Q6A report](evasion_qnn/REPORT.md), [measurement methods](evasion_qnn/README.md), [verification](evasion_qnn/results/verification/verification.json) |

All five model files use Git LFS. Reports, figures, JSON/CSV/TXT/YAML evidence,
requirements, configurations and retained helper source use ordinary Git.
The repository's [weight manifest](../../../manifest.json) records source
paths, exact bytes, SHA-256, recipes and target metadata. The [report manifest](../manifest.json)
records both original and archived hashes, including adjusted Markdown links.

## Retained material

- TPU: conversion and validation notes, compiler mapping and compact compiler
  log, environment and calibration/sample hash ledgers, runtime metadata,
  conversion/validation scripts, requirements and `edgetpu_runtime.py`.
- TPU evasion: report, four figures, both runtime summaries and per-image CSVs,
  preflight checks, candidate digest ledger, independent verification and
  historical comparison/audit JSON and CSV.
- QNN conversion: build configurations and conversion/validation scripts,
  source and deployment digests, DLC graph metadata, native I/O encodings and
  execution metadata.
- QNN evaluation: performance and evasion reports, two figures, raw compact
  timing/profile/memory evidence, service/deployment records, broader clean
  evaluation metadata and sample ledger, attack summaries and independent
  per-image verification. `scripts/runtime.py`, `serve_board.py` and
  `qnn_scorer.c` preserve the deployed interface implementation.

Image datasets, calibration/evaluation tensors, saved logits, attack candidate
tensors, attack implementation, intermediate ONNX/TensorFlow/TFLite builds,
full experiment logs, SDK/runtime libraries, compiled scorer executables and
environments remain in the source workspace. The three native QAIRT benchmark
profile text/CSV exports and compiler logs are retained as compact evidence.

## Using the retained helpers

Conversion scripts and QNN helpers are preserved byte-for-byte as provenance;
their relative/absolute paths, SDK dependencies and service address describe
the original workspace and board deployment. Restore that layout and provide
the referenced dependencies to reproduce those commands. The archive itself
does not install or start a service. Source-only references are listed in
[SOURCE_REFERENCES.md](../SOURCE_REFERENCES.md).

The TPU adapter accepts an explicit model path, so it can use archived weights
directly after installing the recorded runtime requirements. From the archive
root, for example:

```python
import sys
sys.path.insert(0, 'docs/reports/mobilenetv2/TPU')
from edgetpu_runtime import EdgeTpuRuntime

cpu = EdgeTpuRuntime(
    model_path='weights/mobilenetv2/tflite/mobilenetv2_int8_nhwc.tflite',
    backend='cpu',
)
coral = EdgeTpuRuntime(
    model_path='weights/mobilenetv2/edgetpu/mobilenetv2_int8_nhwc_edgetpu.tflite',
    backend='tpu',
)
# runtime.run_uint8(rgb_uint8_nhwc) returns dequantized logits.
```

TPU takes raw UINT8 NHWC RGB with embedded normalization; QNN float-input mode
takes externally ImageNet-normalized NCHW. QNN native UINT8 values encode those
normalized floats, not raw RGB. See [runtime contracts](../../RUNTIMES.md).

QNN's broader clean evaluation excludes the 20 calibration-overlapping images
for its primary held-out result. Its 100-image attack cohort has no calibration
overlap. Preserve these populations and each report's attack denominators when
comparing results. No new accuracy or hardware validation was performed while
archiving.

Verify from the archive root:

```bash
sha256sum --check SHA256SUMS
sha256sum --check docs/reports/SHA256SUMS
```
