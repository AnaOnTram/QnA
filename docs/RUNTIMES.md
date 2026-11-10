# Runtime and input contracts

All models classify into the original 1,000 ImageNet classes. Outputs are logits
or quantized logits, not probabilities. File extensions alone do not establish
runtime compatibility or the precision of every operator.

## Image preparation

| Model | Image preparation used for the retained artifacts |
| --- | --- |
| timm ResNetV2-50 | RGB; resize shorter side to 232 with bilinear interpolation; center crop 224 |
| MobileNetV2 | RGB; resize shorter side to 256 with bicubic interpolation and antialiasing; center crop 224 |

For PyTorch and ONNX graphs, convert pixels to float in `[0,1]`, then normalize
with mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. Use NCHW `[N,3,224,224]` and the graph's declared input
dtype. ResNetV2 FP16 has FP16 inputs and outputs. The timm checkpoint's native
recipe uses mean/std 0.5 and a different resize; the retained converted artifacts
use the shared ImageNet-normalized contract above. Native timm preprocessing
must not be substituted when reproducing these deployments.

## PyTorch and ONNX Runtime

Load `weights/resnetv2_50/pytorch/model.safetensors` into
`timm.create_model('resnetv2_50', pretrained=False)` using
`safetensors.torch.load_file()` and `load_state_dict()`. The checkpoint is
`timm/resnetv2_50.a1h_in1k`, pinned to SHA-256 `b809e8c0…` in the manifest.

The MobileNetV2 `.pth` is a state dictionary for
`torchvision.models.mobilenet_v2(weights=None)`.

For ResNetV2 CPU INT8 inference use `onnx/resnetv2_50_int8_qoperator.onnx`, with
actual integer kernels. It uses per-channel INT8 weights and UINT8 activations,
calibrated on 512 images. FP32 and FP16 graphs are also provided. Select a runtime
provider supporting the graph's operators and dtypes; FP16 is intended for CUDA.

MobileNetV2's FP32 graph is opset 20. Its `rknn/mobilenetv2_opset19.onnx` is a
compatibility conversion for RKNN Toolkit2 2.5.0; the existing validation records
bit-identical outputs on the retained 100-image validation sample.

## Apple Core ML

ResNetV2-50 now includes FP32, FP16, INT8 weight-only (W8A16) and INT4 weight-only
(W4A16) ML Program packages under `weights/resnetv2_50/coreml/`. All target
**iOS 18 / macOS 15** and have FP32 input `image` `[1,3,224,224]` and FP32 output
`logits` `[1,1000]`. Normalization is external: use the archive's ResNetV2 image
preparation and ImageNet normalization above when reproducing archive comparisons.
INT8/INT4 packages use FP16 internal computation; compression does not imply
integer convolution kernels. FP32 runtime parity was checked with `CPU_ONLY`.

MobileNetV2 has the same four precisions under `weights/mobilenetv2/coreml/`,
with the same targets, I/O names, shapes and external normalization. Use the
MobileNetV2 image preparation above.

All eight packages were compiled and executed locally. Both INT4 packages showed
substantial prediction drift and should be treated as experimental; MobileNetV2
INT4 is severely degraded (mean logit cosine 0.66). See the [conversion,
usage and validation notes](coreml/README.md) for exact recipes, sizes,
preprocessing options, checksums and measured parity limits.

The compute-unit request is not a guarantee of placement. On macOS 26.6.2, FP32
packages run entirely on the CPU under `CPU_AND_NE`, because the Neural Engine
has no FP32 path. FP16 and INT8 run on the Neural Engine. The Core ML evasion
studies ([MobileNetV2](reports/mobilenetv2/evasion_coreml/README.md),
[ResNetV2-50](reports/resnetv2_50/evasion_coreml/README.md)) measure how each
request changes clean logits and attack outcomes.

## TensorRT

ResNetV2 INT8 graphs have two storage forms. `*_int8_qdq.onnx` retains float
weights with explicit quantize/dequantize nodes. `*_int8_dq.onnx` stores compressed
INT8 weights. The former is about 98 MiB even though TensorRT executes INT8
convolutions. Both use ModelOpt max-calibration PTQ on 512 images. These graphs
are intended for TensorRT; running them under ONNX Runtime does not guarantee
integer convolution kernels or a speedup.

The two included ResNetV2 `.plan` files were recorded with **TensorRT 10.9.0.34**
on an **NVIDIA RTX 2000 Ada Generation Laptop GPU**, max batch 10, strongly typed,
TF32 disabled, 1 GiB workspace. They are hardware/version-specific deployment
artifacts. Rebuild from the included ONNX graphs for other targets; no portability
guarantee is recorded for these plans. Their metadata and ONNX source hashes are
in the manifest. The shared FP32/FP16 ONNX graphs are under `onnx/`.

MobileNetV2 offers FP16, FP8 E4M3 Q/DQ and INT8 Q/DQ graphs. Its INT4 variant
quantizes **only the classifier Gemm**, weight-only RTN with block size 128;
the convolutions remain floating point. It is a mixed-precision graph, not a
fully INT4 network. The original records describe TensorRT 11.3.0.99 / Blackwell
deployment engines, but those per-precision engines are not present in the local
source tree and are not included. Build compatible engines from the graphs on
the target device.

## RKNN on RK3588

Both models include FP16 and INT8 `.rknn` files for **RK3588**, with
**RKNN-Toolkit2 2.5.0**. INT8 uses 256 ImageNet-mini training images for
calibration. MobileNetV2 uses channel quantization, the `normal` algorithm,
W8A8, and optimization level 3.

Normalization is **baked into the RKNN artifacts**. Feed RGB image values in
`[0,255]` through the ordinary RKNN/RKNNLite input API; do not apply ImageNet
normalization again. The ordinary UINT8 NHWC input convention is
`[1,224,224,3]`; the runtime handles conversion to the native tensor type.
Native I/O is FP16 for the FP16 artifact and INT8 for the quantized artifact.
Native or zero-copy APIs require querying the actual tensor attributes instead
of assuming the ordinary input convention.

RKNN rebuilds can differ byte-for-byte despite equivalent outputs. A new build
needs fresh checksums and runtime validation; keep it separate from this snapshot.

## Qualcomm QAIRT / QNN

The ResNetV2 files are `resnetv2_50_fp32.dlc` (conversion source),
`resnetv2_50_int8.dlc` and `resnetv2_50_quantized.bin` (HTP context). The recorded
deployment is **QCS6490**, **QAIRT 2.42.0.251225**, HTP v68, SoC ID 35,
VTCM 2 MiB. The FP32 DLC is not an INT8 HTP deployment.

The context digest matches the pinned deployment digest; this packaging step
does not independently establish a fresh DLC-to-context build chain. The FP32
DLC had no previously pinned digest and is newly hashed in this snapshot.

Do not send raw RGB bytes directly to a native QNN input buffer. The retained
context uses a native UINT8 encoding of ImageNet-normalized NCHW values, with
scale `0.01865844801068306` and offset `-114`. For normalized float `x`, the
recorded native encoding is `clip(round(x / scale) + 114, 0, 255)`. Query the
context tensor metadata and use the matching QAIRT backend and target libraries.
The original harness used QNN float-input handling to encode normalized floats.

MobileNetV2 now includes `weights/mobilenetv2/qnn/mobilenetv2_fp32.dlc`,
`mobilenetv2_int8.dlc` and `mobilenetv2_quantized.bin`, using the same QAIRT
version and QCS6490/v68/SoC 35/2 MiB VTCM target. The graph name is
`mobilenetv2_fp32` even for the INT8 context. Default `qnn-net-run` float-input
mode takes normalized float32 NCHW `[1,3,224,224]` and returns dequantized
float32 logits `[1,1000]`. Normalization remains external. Native input is UINT8
with scale `0.018658448011`, offset `-114`; native output is UINT8 with scale
`0.149840772152`, offset `-71`, using `real = (code + offset) * scale`.

The retained build uses the original 20 ResNet QNN calibration raws with default
asymmetric min-max W8A8/bias8 settings, per-channel/per-row disabled and the
default dynamic 16-bit weight option enabled. On the 100-image cohort, FP32 DLC
preserves 100/100 ONNX predictions, whereas INT8 HTP is correct on 78/100.
The broader held-out set excludes 20 calibration-overlapping images: FP32
2761/3903 and INT8 HTP 2665/3903. See [conversion and encodings](reports/mobilenetv2/qnn/README.md)
and [Q6A performance, accuracy and evasion](reports/mobilenetv2/evasion_qnn/REPORT.md).

## TensorFlow Lite and Coral Edge TPU

Use `tflite/resnetv2_50_int8_nhwc.tflite` for a CPU TFLite interpreter and
`edgetpu/resnetv2_50_int8_nhwc_edgetpu.tflite` with a Coral Edge TPU delegate.
Both use static UINT8 NHWC `[1,224,224,3]`, input scale `1/255` and zero point 0:
feed the raw RGB crop without a second normalization. ImageNet normalization is
folded into the converted model. Read output scale/zero-point from the runtime
before interpreting logits.

The source documentation records 200 calibration images and Edge TPU compiler
**16.0.384591198**, with 114/114 operations mapped to one TPU subgraph. The old
dynamic NCHW TFLite and intermediate TensorFlow SavedModel are excluded.

MobileNetV2 has its own final pair in `weights/mobilenetv2/tflite/` and
`weights/mobilenetv2/edgetpu/`: `mobilenetv2_int8_nhwc.tflite` and
`mobilenetv2_int8_nhwc_edgetpu.tflite`. Their input is also raw RGB UINT8 NHWC
`[1,224,224,3]`, scale `0.003921568859368563`, zero point 0, with ImageNet
normalization embedded before the padded stem. Use the MobileNet resize/crop
recipe above. Output is UINT8 `[1,1000]`, scale `0.1706031858921051`, zero point
84; dequantize as `(code - 84) * scale`.

The MobileNet conversion used 200 ImageNet-mini training calibration images,
onnx2tf 2.6.9 / TensorFlow 2.21.0, and Edge TPU compiler 16.0.384591198.
73/73 operations map to one TPU subgraph. Recorded TensorFlow CPU 2.19.1 and
Coral hardware validation give 81/100 and 79/100 correct respectively, versus
83/100 for the source ONNX. These are distinct runtime results. See
[conversion and hardware validation](reports/mobilenetv2/TPU/README.md),
[evasion report](reports/mobilenetv2/TPU/evasion/EVASION_TPU_REPORT.md) and
[archive/helper usage](reports/mobilenetv2/ARCHIVE.md).
