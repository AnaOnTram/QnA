# MobileNetV2 TensorFlow Lite / Coral USB conversion

The completed [evasion attack report](evasion/EVASION_TPU_REPORT.md) evaluates
both native CPU and Coral deployments, with comparisons to the previous
ResNetV2-50 and MobileNetV2 studies.

Converted and validated on 10 November 2026 (Hong Kong time), using the successful
ResNetV2-50 static-NHWC / full-INT8 / Edge TPU compiler 16 workflow.
The source is the existing TorchVision MobileNetV2 `IMAGENET1K_V1` checkpoint,
not the Keras MobileNetV2 checkpoint.

## Deliverables

| Artifact | Size | Execution target |
| --- | ---: | --- |
| [mobilenetv2_int8_nhwc.tflite](../../../../weights/mobilenetv2/tflite/mobilenetv2_int8_nhwc.tflite) | 4,011,840 bytes (3.83 MiB) | TFLite CPU |
| [mobilenetv2_int8_nhwc_edgetpu.tflite](../../../../weights/mobilenetv2/edgetpu/mobilenetv2_int8_nhwc_edgetpu.tflite) | 4,231,808 bytes (4.04 MiB) | Coral USB / Edge TPU delegate |

Both use **UINT8 RGB `[1,224,224,3]`**, input scale `0.003921568859368563`
(1/255), zero point 0. Pass ordinary RGB byte values, without ImageNet
normalization. The graph performs `(pixel/255 - mean)/std` with
`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`.

Output is **UINT8 logits `[1,1000]`**, scale `0.1706031858921051`, zero point 84.
Dequantize with `(output.astype(float32) - 84) * 0.1706031858921051`.
These are ImageNet-1k class logits; softmax is optional and external.
Resize and crop are external to the model. The comparison pilot uses RGB,
shorter-side resize to 256 with bicubic antialiasing, then center crop to 224,
matching the existing MobileNet studies.

## Compilation and hardware validation

Compiler **16.0.384591198** mapped **73/73 operations (100%)** to a single TPU
subgraph, including normalization and padding. No CPU operation remains in the
compiled graph. The uncompiled graph has 188 tensors, zero dynamic tensors,
zero floating tensors, zero `TRANSPOSE`, and no Flex/custom operations.
The compiled graph executes as one `edgetpu-custom-op`.

Compiler memory report: 3.89 MiB cached model parameters on chip, 3.80 MiB cache
remaining, 128 bytes uncached streaming. See the [compiler log](logs/compiler.log)
and [operator mapping](mobilenetv2_int8_nhwc_edgetpu.log).

The actual Coral USB device was exercised on this host (`18d1:9302`, USB bus 2,
device path `/sys/bus/usb/devices/2-1`), using `libedgetpu1-std`
`16.0tf2.19.1-1.ubuntu24.04` and TensorFlow CPU 2.19.1.

| Runtime | Top-1 correct / 100 | Top-5 correct / 100 | Top-1 agreement with ONNX FP32 | Logit MAE vs FP32 | Mean logit cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| ONNX FP32 source | 83 | 96 | 100% | 0 | 1 |
| Converted TFLite FP32 | 83 | 96 | 100% | 0.00000334 | 1.000000 |
| TFLite CPU INT8 | 81 | 95 | 86% | 0.65571 | 0.97016 |
| Coral USB INT8 | 79 | 95 | 89% | 0.61917 | 0.97498 |

The prepared raw-RGB ONNX matches the original ONNX **exactly**, including
black/white/random border probes. Converted FP32 TFLite differs by at most
`4.57763671875e-05` across the pilot logits and preserves every top-1 prediction.
Normalization stays before the padded stem convolution; folding it naively into
the padded convolution's weights and bias would alter the border computation.

Quantization changes predictions: the Coral loses four net correct classifications
on this 100-image pilot. CPU INT8 and Coral INT8 also differ (92% top-1 agreement;
logit MAE 0.53301). Use the measured Coral results for accelerator deployment.
These are descriptive measurements, with no user-specified application accuracy
acceptance threshold. A 100-image Imagenette pilot does not establish full
ImageNet accuracy, and no evasion experiment was run for this conversion.

Warm batch-1 `invoke()` latency over 200 calls after ten warmups, same image:

| Runtime | Median | P95 |
| --- | ---: | ---: |
| TFLite CPU INT8, one thread | 5.68 ms | 6.44 ms |
| Coral USB INT8 | **3.97 ms** | 4.24 ms |

Timing excludes preprocessing and tensor copies; CPU and TPU timing were run
sequentially. See [validation.json](results/validation.json),
per-image logits (`Models/mobilenetV2/TPU/results/logits.npz`; source workspace only), latency samples (`Models/mobilenetV2/TPU/results/latencies.npz`; source workspace only),
and [hardware runtime details](results/tpu/runtime.json).

## Calibration and provenance

- Original ONNX: `../onnx/mobilenetv2.onnx`, SHA-256
  `9f2d5d94df76f044172dc28345bf975b83e24de1ca33e5892fe074d312d93a77`.
- Original checkpoint: `../pytorch/mobilenet_v2-b0353104.pth`, SHA-256
  `b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`.
- The original ONNX and checkpoint are unchanged. A separate static batch-1,
  224x224 raw-RGB graph is prepared, then converted by onnx2tf 2.6.9 / TF 2.21.0
  using per-channel INT8 post-training quantization and UINT8 I/O.
- Calibration follows the previous ResNet TPU selection and preprocessing:
  200 ImageNet-mini training images, first image per selected class, bilinear
  resize to 224x224, NHWC float32 `[0,1]`. It has zero image-content overlap
  with the 100 validation images. Images were decoded/resized again with
  Pillow 13.0.0; this reproduces the recipe, not byte-identical old calibration
  pixels. Calibration tensor and image hashes are retained.
- Evaluation reuses the existing validated MobileNet crop tensors and ordered
  sample population. The ONNX reference was recomputed and checked against the
  existing reference before conversion. Validation images were not used to
  select calibration ranges or tune the exported model.

See [source validation](results/source_validation.json),
[calibration ledger](results/calibration_samples.json),
[evaluation ledger](results/samples.json), and [conversion environment](results/conversion_environment.json).
Original source/checkpoint hashes were checked again during final validation.

## Reproduce and run

Existing environments on this host:

```bash
cd /home/ross/QA
bash Models/mobilenetV2/TPU/convert.sh
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 \
  /tmp/tpu_rt/bin/python Models/mobilenetV2/TPU/validate.py
/tmp/tpu_rt/bin/python Models/mobilenetV2/TPU/edgetpu_runtime.py --check
/tmp/tpu_rt/bin/python Models/mobilenetV2/TPU/edgetpu_runtime.py \
  --npz Models/mobilenetV2/TPU/data/evaluation.npz \
  --logits-out /tmp/mobilenetv2_coral_logits.npy
cd Models/mobilenetV2/TPU
sha256sum --check SHA256SUMS
```

`convert.sh` defaults to `/tmp/tf21/bin/python` and the compiler already in
`Models/resnet50/TPU/tools/edgetpu_compiler_16.0/bin/edgetpu_compiler`.
Override `CONVERSION_PYTHON` and `EDGETPU_COMPILER` as needed. For a fresh Python
3.12 environment install [requirements-convert.txt](requirements-convert.txt);
keep runtime scoring in a separate environment using
[requirements-runtime.txt](requirements-runtime.txt) and a compatible installed
`libedgetpu.so.1`. The current onnx2tf invocation also produces intermediate
SavedModel and FP32/FP16/dynamic-range/INT16-activation exports under `build/`;
only the final pair in the table above is intended for INT8 deployment.
`conversion_input/` isolates onnx2tf's in-place ONNX simplification from the
validated prepared graph.

Python adapter (NumPy; Torch pixel tensors also supported):

```python
from edgetpu_runtime import EdgeTpuRuntime
runtime = EdgeTpuRuntime()                  # Coral USB; fails if unavailable
logits = runtime.run_uint8(rgb_nhwc_uint8)   # [N,224,224,3] -> [N,1000]
logits = runtime.run(pixels_nchw_float)      # [N,3,224,224], [0,1]
cpu = EdgeTpuRuntime(backend='cpu')
```

Floating inputs are rounded to 8-bit pixels. Every image is executed with static
batch size 1. The adapter raises for missing TPU/delegate support and supplies
no gradients. It never silently switches a requested TPU run to CPU.

Coral's [model requirements](https://coral.ai/docs/edgetpu/models-intro/)
describe the static-size and 8-bit quantization requirements used here.
