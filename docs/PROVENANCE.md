# Provenance and packaging scope

The original 6 November 2026 snapshot contains standalone copies of existing weights and their evasion
attack reports, figures and compact result evidence. Packaging did
not train, quantize, convert, rebuild or run the models. Every copied weight was
checked by SHA-256 against its source; previously pinned digests were also checked
where available. Source files were read without changing their contents or mtimes.

On 10 November 2026, four **Core ML** packages were newly converted from the
same pinned timm checkpoint: FP32, FP16, INT8 weight-only and INT4 weight-only.
The same day, four MobileNetV2 Core ML packages with the same precisions were
converted from the repository's `mobilenet_v2-b0353104.pth` checkpoint.
Each addition includes a reproducible conversion script and fresh Core ML
inference checks. See [Core ML provenance and validation](coreml/README.md).
The original snapshot's no-conversion/no-execution statements apply to the
original artifacts; the Core ML exports have their own build and validation
records in `docs/coreml/validation.json` (ResNetV2-50) and
`docs/coreml/validation_mobilenetv2.json` (MobileNetV2).

Also on 10 November 2026, the completed MobileNetV2 **TPU and QNN** conversions,
hardware-validation records and evasion/performance studies were archived from
`QA/Models/mobilenetV2/`. Five final model files were copied: a static NHWC INT8
TFLite/Edge TPU pair and FP32 DLC, INT8 DLC and QCS6490 INT8 QNN context. All five
match the source hashes already pinned in their conversion-validation reports.
Archiving did not convert or execute them and did not access the Q6A board.
The same canonical MobileNet ONNX and TorchVision checkpoint underlie both
conversions. See [the archive guide](reports/mobilenetv2/ARCHIVE.md) for retained
reports, validation evidence and conversion/runtime helpers.

The three cross-runtime/cross-model syntheses under `docs/reports/` were then
newly composed from the retained measurements. The original synthesis reconciled 842 conditions
from 39 hash-pinned deployment summaries before the Coral PGD+ supplement and preserve study-specific common-clean
denominators. Their reporting script, derived tables, figures and source ledger
are under `docs/reports/cross_runtime/`. The missing matched ResNet QNN summary
and ResNet sample ledger were copied from the original workspace to make the
new calculations archive-local. This is reporting and arithmetic validation,
not a new hardware experiment or candidate replay.

## 16 November completed attack campaigns

Eight additional campaigns were completed in the source workspace
`QA/experiments/missing_attacks_20261116/`, filling 40 previously blank cells.
This archive imports their compact records after successful independent native
replay of all 16,800 candidate evaluations. Packaging does not rerun attacks.
The synthesis now reconciles 1,011 conditions across 47 deployments; the original
39-deployment registry and its values remain unchanged.

Seven final model files are added: MobileNet INT8 QOperator, canonical timm
ResNet FP8 and classifier-only INT4 ONNX graphs, and four ResNet TensorRT 11.3
static-batch-1 plans from NV10 / RTX PRO 4000 Blackwell. FP8 uses max calibration
on the existing 512-image array; INT4 uses block-128 symmetric RTN on the
classifier after a verified equivalent Flatten+Gemm rewrite. Other layers are
unchanged FP32. Sources, calibration hashes, recipes, precision evidence and
attack-time graph/engine digests are retained in the manifest and
[completion archive](reports/missing_attacks_20261116/README.md). The source
FP32 graph was protected with a scratch copy during FP8 quantization.

The seven files match the already pinned experiment digests. Native replay was
performed before archiving; the compact archive documents it but omits the raw
candidate tensors and logits required to repeat it.

## Sources

Paths in `manifest.json` are relative to the original user's home directory and
document provenance only. Runtime use does not depend on these original paths.

- **timm ResNetV2-50:** `timm/resnetv2_50.a1h_in1k`, cached snapshot
  `a9035ec8691ab8c6eb6035aa3930c37583bcee5b`; checkpoint SHA-256
  `b809e8c09f4a1cf2168adf089c86c17fe88521b85455db662c0b2d42f489a61b`.
  The canonical ONNX SHA-256 is
  `6416f2e9d2d9c13f3baa60a496d0934e870e19baf8cd2c416979df42c75c13f4`.
  Existing provenance records assert 251/251 initializers match the checkpoint.
- **MobileNetV2:** official `mobilenet_v2-b0353104.pth`, TorchVision
  `MobileNet_V2_Weights.IMAGENET1K_V1`. Existing source validation records ONNX
  checkpoint parity and opset-19 compatibility parity.
- **QNN:** local QAIRT example output under
  `qairt/2.42.0.251225/examples/Models/resnet50_qairt_example/model/output/`.
  The directory name is historical; only `resnetv2_50_*` artifacts were copied.
- **MobileNetV2 QNN and TPU:** `QA/Models/mobilenetV2/qnn/` and `TPU/`, with
  subsequent Q6A studies under `evasion_qnn/` and TPU attacks under `TPU/evasion/`.
- **Other variants:** existing `QA/Models/resnet50/` timm artifacts and
  `QA/Models/mobilenetV2/` artifacts, with their original locations in the manifest.

The local source records used to interpret these files include:

- `QA/Models/resnet50/quantize/sources.json` and `artifacts/timm_rnv2_*.meta.json`.
- `QA/Models/resnet50/evasion_onnx/engines/timm_rnv2_*.json` and runtime records.
- `QA/Models/resnet50/rknn/README.md` and `TPU/EDGETPU_COMPILE_ISSUE.md`.
- `QA/Models/mobilenetV2/rknn/README.md` and `evasion_trt/README.md`.

These source documents remain in place. Relevant runtime and quantization details
are summarized here. [The report archive](reports/README.md) also retains the
evasion reports, figures, comparison tables, methodology and compact validation
evidence. Its own manifest records source and packaged hashes; only local report
links were adjusted. Original source files remain unchanged.

## Included and excluded artifacts

Only timm ResNetV2-50 and MobileNetV2 are included. All TorchVision ResNet-50
checkpoints, exports and engines are excluded, including historical V1/V2,
FBGEMM and TorchVision-derived PTQ variants. MobileNetV2's TorchVision origin
does not make it a TorchVision ResNet-50 artifact.

Duplicate canonical ONNX files and duplicate RKNN binaries are represented by
one copy each. Toolkit-generated `check*.onnx`, scratch builds, reproduced export
intermediates, calibration datasets, adversarial tensors, evaluation samples,
full experiment logs, scripts, SDKs, environments and unrelated vendored model
bundles are excluded. Report figures and compact JSON/CSV result evidence are
included as documentation. Only the final static NHWC TFLite / Edge TPU weight
pair is retained.

The Core ML conversion scripts and selected MobileNetV2 TPU/QNN conversion,
validation and runtime/service helpers are retained as reproducibility exceptions
to the original snapshot's scripts exclusion. Compact compiler logs, DLC tensor
metadata and native benchmark profile exports are also retained as documentation.
Attack implementation, calibration data, saved logits and new photo files are
not included. Source helper paths still describe the original workspace.

MobileNetV2's generic legacy `trt/mobilenetv2.engine` is excluded because its
build identity is not established by the per-precision deployment records. The
documented `trt/engine/*_bs1.engine` / `*_bs8.engine` files are absent locally.
No remote board or server files were fetched. Missing runtime/precision variants
were not fabricated or newly built during the original packaging step.

The newly included QNN FP32 DLC is hashed but has no previously pinned source
digest. The manifest distinguishes previously pinned source hashes from hashes
newly calculated during packaging. Hash verification establishes file identity;
it is not a new accuracy, provenance-chain or hardware compatibility evaluation.

## Upstream references and terms

- [timm ResNetV2-50 model card](https://huggingface.co/timm/resnetv2_50.a1h_in1k).
- [TorchVision MobileNetV2 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html).
- [TorchVision repository license](https://github.com/pytorch/vision/blob/main/LICENSE).
- [Vendored RKNN source repository](https://github.com/harishsdev/NPU).

Preserve the upstream attribution and applicable weight terms when redistributing
these files. This collection does not assert a common license across checkpoints,
converted weights, compiled contexts and SDK components. SDK/runtime libraries
are not included. The Hugging Face model card therefore omits a blanket license
field rather than assigning one to the whole collection.
