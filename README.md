---
pipeline_tag: image-classification
tags:
  - image-classification
  - imagenet
  - onnx
  - tensorrt
  - rknn
  - qnn
  - tflite
  - quantized
  - coreml
---

# Model weights and evasion attack reports

Pretrained **timm ResNetV2-50 (`resnetv2_50.a1h_in1k`)** and **MobileNetV2
(`IMAGENET1K_V1`)** weights, with the existing converted and quantized runtime
artifacts, plus their evasion attack reports and supporting figures/evidence.
This snapshot was assembled on 6 November 2026 (Hong Kong time).

32 model artifacts represented by 48 files, **1,061,320,331 bytes** (1012.15 MiB),
including eight Core ML packages (four per model) added on 10 November 2026. Binary model data is
tracked with Git LFS. Package JSON manifests, reports, figures, compact result
evidence, checksums and provenance manifests use ordinary Git.

| Runtime / format | timm ResNetV2-50 | MobileNetV2 |
| --- | --- | --- |
| PyTorch | FP32 safetensors checkpoint | FP32 checkpoint |
| ONNX Runtime | FP32, FP16, INT8 QOperator | FP32 |
| TensorRT ONNX graphs | INT8 explicit Q/DQ, compressed INT8 Q/DQ; FP32/FP16 graphs shared from `onnx/` | FP32 shared from `onnx/`; FP16, FP8, INT8, mixed INT4 weight-only |
| TensorRT compiled engines | Two INT8 plans, TensorRT 10.9, batch profile max 10 | Documented deployment engines absent locally; graphs included |
| Apple Core ML | FP32, FP16, INT8 weight-only, experimental INT4 weight-only; iOS 18 / macOS 15+ | FP32, FP16, INT8 weight-only, experimental INT4 weight-only; iOS 18 / macOS 15+ |
| Rockchip RKNN, RK3588 | FP16, INT8 | FP16, INT8; opset-19 conversion source |
| Qualcomm QAIRT / QNN | FP32 DLC, INT8 DLC, QCS6490 INT8 context | No local artifact |
| TensorFlow Lite CPU | Static NHWC INT8, UINT8 I/O | No local artifact |
| Coral Edge TPU | Compiled static NHWC INT8, UINT8 I/O | No local artifact |

TorchVision **ResNet-50** is excluded. MobileNetV2 retains its original
TorchVision checkpoint. ResNetV2-50 and MobileNetV2 are separate model families.

## Contents

- [Evasion attack reports](docs/reports/README.md): unified timm comparison, Edge TPU,
  MobileNetV2 TensorRT and RKNN reports, and new Core ML studies of both models on
  CPU, CPU+GPU and CPU+Neural Engine, with figures, matrices and validation evidence.
- [Core ML conversion and validation](docs/coreml/README.md): four packages per model, reproduction scripts, input contracts and measured quantization drift.
- [Artifact catalog](docs/CATALOG.md): every included file, runtime, precision and size.
- [Runtime and preprocessing notes](docs/RUNTIMES.md): input contracts and deployment limits.
- [Provenance and scope](docs/PROVENANCE.md): sources, quantization recipes and exclusions.
- [Git tracking](docs/GIT_TRACKING.md): LFS setup, verification and update workflow.
- [GitHub and Hugging Face uploads](docs/UPLOAD.md): commands for publishing this repository.
- [manifest.json](manifest.json): exact bytes, SHA-256, source paths and per-file metadata.
- [SHA256SUMS](SHA256SUMS): checksums for all included weights.

## Verify the downloaded weights

Run from the repository root after downloading the LFS objects:

```bash
sha256sum --check SHA256SUMS
sha256sum --check docs/reports/SHA256SUMS
```

This archive contains weights, attack reports, figures and compact result evidence.
It also includes Core ML conversion and validation scripts. It contains no
datasets, calibration images, attack implementation, adversarial candidate
tensors, full experiment logs, virtual environments or SDKs.
Use the appropriate runtime and model architecture; there is no single
`from_pretrained()` entry point for this multi-model collection. See the runtime
notes before passing inputs to the compiled models.

## Attribution

ResNetV2-50 comes from [timm/resnetv2_50.a1h_in1k](https://huggingface.co/timm/resnetv2_50.a1h_in1k).
MobileNetV2 comes from [TorchVision's pretrained MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html).
Upstream weights retain their upstream terms; this archive does not assign a
new blanket license to third-party weights. Source references are recorded in
the provenance documentation.
