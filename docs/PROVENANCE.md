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

The Core ML conversion scripts are retained as a reproducibility exception to the
original snapshot's scripts exclusion. No calibration data or new photo files
are included.

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
