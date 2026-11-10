# Core ML exports

This folder documents Core ML conversions of both archive models:
[ResNetV2-50](#resnetv2-50) and [MobileNetV2](#mobilenetv2). Each model has its own
conversion script and validation report.

## ResNetV2-50

Converted locally on 10 November 2026 (Hong Kong time) in the existing `coreml`
conda environment: Python 3.12.14, PyTorch 2.14.0, timm 1.0.31, coremltools 9.0,
macOS 26.6.2 on Apple Silicon. No environment packages were changed.

Source: `timm/resnetv2_50.a1h_in1k`, Hugging Face snapshot
`a9035ec8691ab8c6eb6035aa3930c37583bcee5b`, SHA-256
`b809e8c09f4a1cf2168adf089c86c17fe88521b85455db662c0b2d42f489a61b`.
The cached checkpoint matched the repository's LFS pointer exactly. All state
dictionary keys loaded strictly into timm `resnetv2_50`; no random or substitute
weights were used. The source checkpoint and existing exports were left intact.

| Package | Weight / compute precision | MiB |
| --- | --- | ---: |
| [resnetv2_50_fp32.mlpackage](../../weights/resnetv2_50/coreml/resnetv2_50_fp32.mlpackage) | FP32 / FP32 | 97.65 |
| [resnetv2_50_fp16.mlpackage](../../weights/resnetv2_50/coreml/resnetv2_50_fp16.mlpackage) | FP16 / FP16 | 48.88 |
| [resnetv2_50_int8.mlpackage](../../weights/resnetv2_50/coreml/resnetv2_50_int8.mlpackage) | INT8 weights / FP16 compute (W8A16) | 24.62 |
| [resnetv2_50_int4.mlpackage](../../weights/resnetv2_50/coreml/resnetv2_50_int4.mlpackage) | INT4 weights / FP16 compute (W4A16) | 13.33 |

All are ML Program packages targeting **iOS 18 / macOS 15 or newer**. Each package
is a directory: distribute its `Manifest.json`, `model.mlmodel` and `weight.bin`
together. Add the package to Xcode or load it with Core ML; an individual
`model.mlmodel` from inside the package is not a standalone deliverable.

INT8 uses symmetric per-output-channel weight quantization. INT4 uses symmetric
blockwise quantization with 32 input channels per block; the RGB stem uses
per-output-channel INT4 because its input has only three channels. All 54
convolution weights, including the classifier implemented as a 1×1 convolution,
are compressed: **25,502,912 weight elements**. Biases and normalization parameters
remain floating point. These are weight-only post-training quantization exports;
no representative calibration dataset or activation quantization was used.
They do not establish integer convolution execution or a device speedup.

### Input and output

- Input `image`: FP32 NCHW `[1,3,224,224]`, externally normalized RGB.
- Output `logits`: FP32 `[1,1000]`, original ImageNet class order, no softmax.
- Batch and spatial dimensions are fixed. FP16 and compressed packages cast
  FP32 I/O to/from FP16 internal computation.

The archive's comparison recipe is RGB, resize shorter side to 232 with bilinear
interpolation, center crop 224, divide pixels by 255, then normalize with
mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`.
Normalization is not embedded in any of these four graphs.

For the checkpoint's native timm recipe, obtain the evaluation transform with
`timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)`.
It uses bicubic resize and mean/std `[0.5,0.5,0.5]` for this checkpoint.
Both recipes can be supplied as tensors; use a consistent recipe when comparing
runtimes. Existing archive experiments use the archive recipe.

```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel(
    "weights/resnetv2_50/coreml/resnetv2_50_fp16.mlpackage",
    compute_units=ct.ComputeUnit.ALL,
)
# image_tensor: RGB already resized, cropped and normalized as above.
logits = model.predict({"image": np.asarray(image_tensor, dtype=np.float32)})["logits"]
predicted_class = int(logits.argmax())
```

FP32 execution was validated with `CPU_ONLY`. Core ML's compute-unit selection
may affect actual execution precision and performance on a deployment device.

### Validation

[validation.json](validation.json) contains file hashes, package sizes, graph
inspection and per-sample results against the original PyTorch FP32 checkpoint.
Every package was saved, reopened from disk, compiled and executed using
`CPU_ONLY`. The ten inputs comprise eight deterministic synthetic images and
Matplotlib's bundled `grace_hopper.jpg`, transformed using each of the two
preprocessing recipes above. The photo is identified by SHA-256 in the report;
it is not copied into this repository.

| Package | Mean logit cosine similarity | Top-1 agreement with PyTorch | Max absolute logit error |
| --- | ---: | ---: | ---: |
| FP32 | 0.999999999995 | 10/10 | 0.000126 |
| FP16 | 0.999973 | 10/10 | 0.248983 |
| INT8 weights | 0.999124 | 9/10 | 1.510082 |
| INT4 weights | 0.979476 | 0/10 | 9.847960 |

All 40 predictions had finite outputs of the expected shape. FP32 passed
`assert_allclose(atol=1e-3, rtol=1e-3)`. Graph inspection confirmed FP32 floating
operations in the FP32 package, FP16 internal floating operations in the other
packages, and the stated INT8/INT4 storage types for all 54 compressed weights.

**INT4 is an experimental compressed export with substantial prediction drift.**
On the bundled photo, both preprocessing recipes changed the top class from
652 to 906. INT8 preserved the photo's top class for both recipes. These small
parity checks measure conversion behavior, not labeled ImageNet accuracy.
No claim is made about production accuracy, ANE coverage, latency or energy use.
Evaluate a representative labeled dataset before selecting a compressed model;
INT4 may require quantization-aware training or a different compression recipe.

### Reproduce

Run from the repository root:

```bash
conda activate coreml
python docs/coreml/convert_resnetv2_50.py --checkpoint /path/to/model.safetensors
# Reopen and validate existing exports without converting again:
python docs/coreml/convert_resnetv2_50.py --checkpoint /path/to/model.safetensors --verify-only
```

The default checkpoint path is `weights/resnetv2_50/pytorch/model.safetensors`;
fetch that LFS payload first if it is still pointer text. The script checks its
SHA-256 before proceeding. A cached copy with the same hash works too. Use
`--validation-image /path/to/photo.jpg` (repeatable) to replace the bundled photo
with your own parity-check images. They are not used to calibrate quantization.

The reproducible procedure is in [convert_resnetv2_50.py](convert_resnetv2_50.py).
PyTorch's newer version generates a coremltools compatibility warning, but all
four exports and runtime checks succeeded in the versions recorded above.

Procedure notes learned during this conversion:

- Convert to FP32 with `compute_precision=ct.precision.FLOAT32`; FP32 I/O alone
  does not establish FP32 internal precision.
- Quantize only convolution/linear weights, leave normalization parameters and
  biases floating point, and handle input channels not divisible by block size.
- Check the serialized package's MIL types: compressed data may appear as inline
  `Argument.value` blob references as well as named constant operations.
- Core ML compilation needs its macOS runtime working directories. The execution
  sandbox blocked those writes here; validation succeeded with the same command
  outside the sandbox. Conversion itself used `skip_model_load=True`.
- The repository's legacy weights are LFS pointers in this checkout. Preserve
  their recorded payload digests when adding new artifacts; do not replace those
  digests with hashes of pointer text.

References: Apple's [weight quantization API](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.quantization.html)
and [optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).

## MobileNetV2

Converted locally on 10 November 2026 (Hong Kong time) in the same `coreml`
conda environment, with torchvision 0.29.0. No environment packages were changed.

Source: TorchVision `MobileNet_V2_Weights.IMAGENET1K_V1`, the repository's
`weights/mobilenetv2/pytorch/mobilenet_v2-b0353104.pth`, SHA-256
`b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`. The payload was
read from the local Git LFS object store because the working-tree file was pointer
text; its hash matched the pointer. All state dictionary keys loaded strictly into
`torchvision.models.mobilenet_v2(weights=None)`.

| Package | Weight / compute precision | MiB |
| --- | --- | ---: |
| [mobilenetv2_fp32.mlpackage](../../weights/mobilenetv2/coreml/mobilenetv2_fp32.mlpackage) | FP32 / FP32 | 13.39 |
| [mobilenetv2_fp16.mlpackage](../../weights/mobilenetv2/coreml/mobilenetv2_fp16.mlpackage) | FP16 / FP16 | 6.74 |
| [mobilenetv2_int8.mlpackage](../../weights/mobilenetv2/coreml/mobilenetv2_int8.mlpackage) | INT8 weights / FP16 compute (W8A16) | 3.48 |
| [mobilenetv2_int4.mlpackage](../../weights/mobilenetv2/coreml/mobilenetv2_int4.mlpackage) | INT4 weights / FP16 compute (W4A16) | 2.01 |

Targets, packaging, I/O names, shapes and external normalization match the
ResNetV2-50 packages above. The quantization recipe is also the same: symmetric
per-output-channel INT8, and symmetric INT4 in blocks of 32 input channels.
INT4 falls back to per-output-channel wherever input channels are not divisible
by 32. That covers the RGB stem, all 17 depthwise convolutions (one input channel
per group) and the four pointwise convolutions with 16, 24 or 144 input channels. `weight_threshold=0`
keeps the small depthwise kernels in scope. All 52 convolutions and the classifier
`linear` are compressed: **3,469,760 weight elements**. ReLU6 is represented as
`clip`.

The archive's comparison recipe is RGB, resize shorter side to 256 with bicubic
interpolation and antialiasing, center crop 224, divide by 255, then normalize with
the ImageNet mean/std. TorchVision's own
`MobileNet_V2_Weights.IMAGENET1K_V1.transforms()` uses bilinear resize with the
same crop and normalization. Both recipes can be supplied as tensors.

### Validation

[validation_mobilenetv2.json](validation_mobilenetv2.json) uses the same eight
synthetic images and the same bundled photo as ResNetV2-50. The photo is
transformed with the archive and native TorchVision recipes above. Every package
was saved, reopened, compiled and executed with `CPU_ONLY`.

| Package | Mean logit cosine similarity | Top-1 agreement with PyTorch | Max absolute logit error |
| --- | ---: | ---: | ---: |
| FP32 | 0.999999999979 | 10/10 | 0.000055 |
| FP16 | 0.999088 | 10/10 | 0.444499 |
| INT8 weights | 0.987313 | 8/10 | 1.670046 |
| INT4 weights | 0.662985 | 3/10 | 9.890896 |

All 40 predictions were finite with the expected shape. FP32 passed
`assert_allclose(atol=1e-3, rtol=1e-3)`. Graph inspection confirmed the precisions
and the INT8/INT4 storage types of all 53 compressed weights.

INT8 changed only the gray and `noise_1` inputs, both of which have near-tied
PyTorch predictions. It preserved the photo's top class for both recipes.
**INT4 is severely degraded.** Its mean cosine of 0.66 means the logits no longer
follow the FP32 model. The archive-recipe photo keeping class 652 should not be
read as preserved accuracy. Depthwise kernels have only nine weights per output
channel, which leaves little averaging to absorb 4-bit rounding error. Do not use
this package where accuracy matters without a different compression recipe or
quantization-aware training. As with ResNetV2-50, these parity checks do not
measure labeled ImageNet accuracy, ANE coverage, latency or energy use.

### Reproduce

Run from the repository root:

```bash
conda activate coreml
python docs/coreml/convert_mobilenetv2.py --checkpoint /path/to/mobilenet_v2-b0353104.pth
# Reopen and validate existing exports without converting again:
python docs/coreml/convert_mobilenetv2.py --checkpoint /path/to/mobilenet_v2-b0353104.pth --verify-only
```

The default checkpoint path is `weights/mobilenetv2/pytorch/mobilenet_v2-b0353104.pth`;
fetch that LFS payload first if it is still pointer text. The script checks its
SHA-256 and imports hashing and graph-inspection helpers from
[convert_resnetv2_50.py](convert_resnetv2_50.py), so keep both scripts together.
The procedure is in [convert_mobilenetv2.py](convert_mobilenetv2.py).
