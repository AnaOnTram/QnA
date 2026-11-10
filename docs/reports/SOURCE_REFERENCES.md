# References to the original source workspace

The archive contains reports, figures and compact evidence. It excludes attack implementation, datasets, candidate tensors, saved logits and full experiment logs. Selected MobileNetV2 TPU/QNN conversion/runtime helpers and compact compiler logs are retained; see the [archive guide](mobilenetv2/ARCHIVE.md). Links to omitted files were converted to labeled source-workspace references. Reproduction commands in the retained reports refer to the original experiment workspace.

| Source reference | Reports mentioning it |
| --- | --- |
| `/home/ross/.dsh/skills/adversarial-evasion/SKILL.md` | `mobilenetv2/evasion_trt/README.md` |
| `Models/mobilenetV2/evasion_trt/scripts/attacks.py` | `mobilenetv2/evasion_trt/METHODOLOGY.md` |
| `Models/mobilenetV2/evasion_trt/scripts/config.py` | `mobilenetv2/evasion_trt/METHODOLOGY.md` |
| `Models/mobilenetV2/evasion_trt/scripts/probe_replica.py` | `mobilenetv2/evasion_trt/METHODOLOGY.md`, `mobilenetv2/evasion_trt/TYPE1_REPLICA.md` |
| `Models/mobilenetV2/evasion_trt/scripts/replica.py` | `mobilenetv2/evasion_trt/METHODOLOGY.md`, `mobilenetv2/evasion_trt/TYPE1_REPLICA.md` |
| `Models/mobilenetV2/evasion_trt/scripts/run_evasion.py` | `mobilenetv2/evasion_trt/METHODOLOGY.md`, `mobilenetv2/evasion_trt/TYPE1_REPLICA.md` |
| `Models/mobilenetV2/evasion_trt/scripts/runtimes.py` | `mobilenetv2/evasion_trt/METHODOLOGY.md`, `mobilenetv2/evasion_trt/TYPE1_REPLICA.md` |
| `Models/mobilenetV2/evasion_trt/scripts/summarize.py` | `mobilenetv2/evasion_trt/METHODOLOGY.md` |
| `Models/resnet50/evasion/results/qnn_v2_matched` | `resnetv2_50/evasion/README.md` |
| `Models/resnet50/rknn/NPU/resnet/export_resnet_to_rknn.py` | `mobilenetv2/rknn/README.md` |
| `AISR/evasion_coreml/` (harness: `config.py`, `replica.py`, `runtime.py`, `attacks.py`, `run_arm.py`, `summarize.py`, `square_seeds.py`, `figures.py`, `package_reports.py`, `run_all.sh`) | `mobilenetv2/evasion_coreml/README.md`, `resnetv2_50/evasion_coreml/README.md` |

## MobileNetV2 TPU/QNN additions (10 November 2026)

| Source reference | Reports mentioning it |
| --- | --- |
| `/home/ross/qairt/2.42.0.251225/examples/Models/resnet50_qairt_example/QUANTIZATION.md` | `mobilenetv2/qnn/README.md` |
| `Models/mobilenetV2/TPU/results/latencies.npz` | `mobilenetv2/TPU/README.md` |
| `Models/mobilenetV2/TPU/results/logits.npz` | `mobilenetv2/TPU/README.md` |
| `Models/resnet50/evasion/results/qnn_v2_matched/run/summary.json` | `mobilenetv2/TPU/evasion/EVASION_TPU_REPORT.md` |

Inline code paths in the new reports also identify source-only attack runners, datasets, saved logits and candidate tensors; they are retained as historical reproduction instructions. The runtime/conversion helpers listed in the archive guide are the source-code exception.

## Sources added for the cross-runtime synthesis

The matched ResNet QNN summary previously referenced only by its original path
is now retained as [cross_runtime/inputs/resnetv2_50_qnn_summary.json](cross_runtime/inputs/resnetv2_50_qnn_summary.json).
The ResNet cohort ledger is retained as [cross_runtime/inputs/resnetv2_50_samples.json](cross_runtime/inputs/resnetv2_50_samples.json).
Both are byte-for-byte source copies, with original paths and digests in the
report manifest. Historical source references in the copied experiment reports
remain identifiable; the current synthesis links to these archive-local copies.
The new [reporting script](cross_runtime/build_reports.py) is retained to
reproduce derived counts and figures; it is not an attack implementation.
