# Why `resnetv2_50_quant.tflite` won't compile for the Edge TPU

Diagnosis of the `Invalid model / Model not quantized` failure in
`TPU/edgetpu/compiler/x86_64/edgetpu_compiler`, plus the fix and the latest compiler.

---

## TL;DR

**The model *is* fully int8-quantized. The message "Model not quantized" is a false lead.**

There are two independent problems, both introduced *upstream* by the ONNX→TensorFlow
conversion, not by `export_tflight_revised.py`:

1. **The TFLite graph has dynamic-shaped tensors** (294 of 540). The source ONNX declares a
   symbolic batch dimension (`input: ['batch_size', 3, 224, 224]`), `onnx-tf` propagated it
   into the SavedModel as `(None, 3, 224, 224)` → `(None, None)`, and the TFLite converter
   wrote `-1`s into `shape_signature`. The Edge TPU delegate only accepts **static** shapes.
   The old compiler (14.1, July 2020) reports this as the misleading "Model not quantized";
   compiler 16.0 reports it accurately.
2. **The graph is channels-first (NCHW), because `onnx-tf` preserves ONNX layout.** The
   resulting graph contains **100 `TRANSPOSE` ops** (+ `PAD`/`PADV2`) around every conv, so
   even once it compiles, only **2 of 292 ops** map to the TPU and the rest fall back to CPU.

Fix: export with **static shapes and NHWC**. Using `onnx2tf` (not `onnx-tf`) + full-int8
quantization gives **114/114 ops on the Edge TPU** and the whole network as one
`edgetpu-custom-op` (see [Results](#results)).

---

## Evidence

### 1. The .tflite really is quantized

```
$ python -c "tf.lite.Interpreter(model_path='resnetv2_50_quant.tflite') ..."
IN  serving_default_input:0  [1,3,224,224]  uint8  scale=0.0039215689 (=1/255) zp=0
OUT PartitionedCall:0        [1,1000]       uint8  scale=0.10566815         zp=201

tensor dtypes: int8=465, int32=71, int64=2, uint8=2,   FLOAT32 = 0
tensors whose quantization scale == 0 / are unquantized: 0
```

Every activation is int8/uint8, bias int32, **no float32 tensor anywhere** — i.e. textbook
full-integer post-training quantization. So the compiler's "Model not quantized" cannot be
taken literally.

Also note this log line from the converter, which is **not** an error:

```
fully_quantize: 0, inference_type: 6, input_inference_type: UINT8, output_inference_type: UINT8
```

TF 2.10 prints `fully_quantize: 0` for *correct* int8 models too (a freshly-built minimal
2-conv int8 model prints exactly the same line). Ignore it.

### 2. But 294/540 tensors are dynamically shaped

```
tensors with -1 in shape_signature: 294
  serving_default_input:0  [1,3,224,224]      shape_signature [-1,3,224,224]
  PartitionedCall:0        [1,1000]           shape_signature [-1,1000]
```

SavedModel signature (exported by `onnx-tf`):

```
input : TensorSpec(shape=(None, 3, 224, 224), dtype=tf.float32, name='input')
output: TensorSpec(shape=(None, None),        dtype=tf.float32, name='output')
```

Source `/home/ross/QA/Models/resnet50/TPU/resnetv2_50.onnx` (timm `resnetv2_50.a1h_in1k`):

```
IN  input  ['batch_size', 3, 224, 224]
OUT output ['batch_size', 1000]
opset 17, 171 nodes, last nodes: ... GlobalAveragePool -> Conv(fc) -> Flatten
```

The symbolic `batch_size` is the root cause; `onnx-tf` carried it through, and `Flatten` on a
dynamic batch produced the `SHAPE / STRIDED_SLICE / PACK / REDUCE_PROD / RESHAPE` tail.

### 3. Compiler 14.1's message is misleading — proven

Test A — the old compiler **does** support int8 models. Its own `test_data` proves it:

```
$ edgetpu_compiler test_data/efficientnet-edgetpu-S_quant.tflite   # int8: 183, int32: 77, uint8: 2
Model compiled successfully in 370 ms.   (also mobilenet_edgetpu_1.0_224_post_training_quant: int8, OK)
```

Test B — take the *same* model, change nothing but pinning the batch dimension to 1, and the
old compiler accepts it:

```
$ edgetpu_compiler resnetv2_50_static.tflite      # temp copy, batch pinned to 1
Model compiled successfully in 66 ms.  ... Total number of operations: 292
```

So the "Model not quantized" is entirely a consequence of the dynamic shapes.

### 4. The latest compiler says it plainly

```
$ edgetpu_compiler resnetv2_50_quant.tflite          # 16.0.384591198
ERROR: Attempting to use a delegate that only supports static-sized tensors with a graph
       that has dynamic-sized tensors.
Compilation failed: Model failed in Tflite interpreter.
```

### 5. Fixing only the dynamic shapes is not enough — NCHW/transpose soup

That temporary static-shape copy compiles (2/292 ops mapped). Operation log:

```
TRANSPOSE   32  More than one subgraph is not supported
TRANSPOSE   68  Operation is otherwise supported, but not mapped due to some unspecified limitation
CONV_2D     54  More than one subgraph is not supported
PADV2        1  Operation not supported
...
Number of operations that will run on Edge TPU: 2
Number of operations that will run on CPU: 290
```

The Edge TPU compiler emits at most one TPU subgraph, and every `TRANSPOSE` between convs
breaks the chain, so the graph collapses to CPU. Hence a static-shape patch alone is useless
for the TPU — the **layout** has to be fixed too.

Sanity baseline: a stock, NHWC, `tf.keras.applications.ResNet50V2` int8 PTQ gets
**130/130 ops mapped** on the same compiler, so the toolchain is fine.

---

## The latest Edge TPU compiler

The compiler bundled in the repo (`TPU/edgetpu/`, tag `eel2-4`) is **14.1.317412892 from July
2020** — see `edgetpu/CHANGES.md`: "Edge TPU runtime and Edge TPU compiler 14.1 (July 2020
"eel2")". The current release is **16.0.384591198**.

Coral's official install route (apt) **no longer works** — the repo is retired:

```
$ curl -o /dev/null -w '%{http_code}\n' \
    https://packages.cloud.google.com/apt/dists/coral-edgetpu-stable/main/binary-amd64/Packages
403
```

(The docs at coral.ai still tell you to use that apt repo.) The 16.0 package is mirrored
unmodified (this is what ultralytics uses, see PR google-coral/…/ultralytics#26456):

```bash
curl -L -o edgetpu-compiler_16.0_amd64.tar.gz \
  https://github.com/ultralytics/assets/releases/download/v0.0.0/edgetpu-compiler_16.0_amd64.tar.gz
mkdir -p tools && tar xzf edgetpu-compiler_16.0_amd64.tar.gz -C tools
chmod -R +x tools/usr/bin
```

A working copy is installed in this workspace:

```
TPU/tools/edgetpu_compiler_16.0/bin/edgetpu_compiler --version
# Edge TPU Compiler version 16.0.384591198
```

16.0 adds `-d/--search_delegate`, `-k/--delegate_search_step` and `-t/--timeout_sec`
(useful for models that partially fail to delegate). No apt/sudo required — the bundle has
its own `ld-linux-x86-64.so.2`, `libc++`, etc.

---

## Results

| Model | static? | layout | compiler | mapped to TPU |
|---|---|---|---|---|
| `resnetv2_50_quant.tflite` (your original, from `onnx-tf`) | no (294 dyn) | NCHW + 100 TRANSPOSE | 14.1 | **fails**, "Model not quantized" |
| same | no | NCHW | 16.0 | **fails**, "dynamic-sized tensors" |
| (temp) same model with batch pinned to 1 | yes | NCHW + 100 TRANSPOSE | 14.1 / 16.0 | **2 / 292** (0.7 %) |
| (temp) stock `keras.applications.ResNet50V2` int8 baseline | yes | NHWC | 16.0 | **130 / 130** |
| `resnetv2_50_int8_nhwc.tflite` (onnx2tf + ImageNet-norm fold) | **yes (0 dyn)** | **NHWC, 0 TRANSPOSE** | 16.0 | **114 / 114 (100 %)** |

The temporary evidence models (pin-batch-1 patch, `keras` baseline, intermediate onnx2tf
exports, isolated venv) were deleted during cleanup; only the compiler and the final pair of
models are kept. The numbers above are the recorded measurements.

The final model:

```
IN  uint8 [1,224,224,3]  scale=1/255  zp=0        (raw RGB pixel values in)
OUT uint8 [1,1000]       scale=0.10331 zp=203
tensors=261  dynamic=0  float32=0  TRANSPOSE=0  Flex/CUSTOM=0
compiled: one edgetpu-custom-op, single TPU subgraph, 25.31 MiB
on-chip 7.13 MiB, off-chip 17.95 MiB (expected for a 25 MB ResNet-50 on an 8 MB SRAM part)
```

Accuracy on the **project's 100-image Imagenette-320 validation pilot** (10 images x 10
synsets, `Resize(232)+CenterCrop(224)`), i.e. the same protocol as the other runtime arms:

```
TPU int8 clean correct      : 81/100   (other arms: ONNX FP32 81/100, RKNN INT8 78/100)
vs ONNX FP32 (same crops)   : top-1 agreement 82 %, logit MAE 1.02, cosine 0.99901
latency                     : 44.8 ms/inference (USB3, libedgetpu1-std)
```

The ONNX FP32 reference sees full-precision pixels while the TPU arm receives the input
quantized to 8 bits (scale 1/255), so part of the logit MAE is the 8-bit input itself, not
int8 arithmetic. Note also that the model bakes the harness's ImageNet normalization
(0.485/0.229), not the checkpoint's native `mean=std=0.5` recipe: baking 0.5/0.5 scored
86/100 on this pilot, which would have been a preprocessing confound against the other arms.

## How to use / reproduce

```bash
# compile the TF model (done already, kept here for reference)
TPU/tools/edgetpu_compiler_16.0/bin/edgetpu_compiler \
    -o <out_dir> TPU/resnetv2_50_int8_nhwc.tflite
# -> resnetv2_50_int8_nhwc_edgetpu.tflite
```

To regenerate the TF model from `resnetv2_50.onnx` (requires a newer TF than the `qairt`
env; `uv venv --python 3.12` + `uv pip install "onnx2tf[tensorflow]==2.6.9"` works):

```bash
onnx2tf -i resnetv2_50.onnx -o <out_dir> -b 1 -oiqt -tb tf_converter \
        -qt per-channel -iqd uint8 -oqd uint8 \
        -cind input calib_input.npy "[[[[0.0,0.0,0.0]]]]" "[[[[1.0,1.0,1.0]]]]"
# take <out_dir>/resnetv2_50_full_integer_quant.tflite as the deliverable
```

`calib_input.npy` is 200 NHWC RGB images in [0,1] (`numpy: [200,224,224,3]`, float32).

## Evasion experiments (Edge TPU arm)

Results of the full campaign (14 attack configurations, 100 images, 75,977 TPU inferences)
are in [evasion/EVASION_TPU_REPORT.md](evasion/EVASION_TPU_REPORT.md), with the cross-arm
comparison matrix and `evasion/figures/asr_vs_eps.png`.

`edgetpu_runtime.py` exposes the same interface as the runtimes in
`evasion_onnx/scripts/runtimes.py`, so it drops straight into the existing harness:

```python
import sys
sys.path.insert(0, 'evasion_onnx/scripts')
from runtimes import AttackTarget
from edgetpu_runtime import EdgeTpuRuntime          # TPU/edgetpu_runtime.py

runtime = EdgeTpuRuntime('TPU/resnetv2_50_int8_nhwc_edgetpu.tflite')
target  = AttackTarget('edgetpu_int8', runtime, 10, surrogate_model)   # model = timm FP32
clean   = target.predict(x)          # deployed TPU logits for pixel-space [0,1] crops
# PGD/BPDA craft on `surrogate_model`; every score is re-queried on the accelerator
runtime.verify(workdir)              # -> workdir/runtime.json (device, contract, latency)
```

Standalone checks / scoring of saved perturbations:

```bash
python edgetpu_runtime.py --check                      # device + native I/O contract + latency
python edgetpu_runtime.py --pilot --json-out clean.json # 100-image Imagenette pilot + fp32 parity
python edgetpu_runtime.py --npz clean.npz --json-out clean.json
python edgetpu_runtime.py --npz clean.npz --adv-pt pgd_eps8.pt --json-out adv.json   # ASR
python edgetpu_runtime.py --npz x.npz --backend cpu     # XNNPACK parity (no delegation)
```

Properties that matter for the attack analysis:

- **8-bit native input.** `u8 = round(clip(pixel,0,1)*255)`; perturbations below 1/255 are
  quantized away, so treat `eps=1/255` as expected-zero and report the effective quantized
  input (`quantize_pixels()`), not the float PGD tensor.
- **No gradients.** `logits_for_grad` raises by design; use transfer/BPDA via `AttackTarget`
  with a differentiable surrogate, or run black-box attacks against `runtime.predict`.
- **Query cost** ~45 ms/inference on this host (USB3), so a 500-query Square attack is ~23 s
  per image; the JSON output records the measured median latency and inference count.
- Deterministic: same `pixels` -> same uint8 -> same logits (integer arithmetic); no seeding
  needed on the TPU side.

### Environment for the scoring process

`libedgetpu` here is built for **TF 2.19** (`libedgetpu1-std 16.0tf2.19.1`), so the scoring
process needs `tensorflow>=2.19`; the `qairt` env (TF 2.10) cannot even load the delegate.

```bash
/home/ross/miniconda3/envs/jev/bin/python -m venv /tmp/tpu_rt         # Python 3.12
/tmp/tpu_rt/bin/pip install tensorflow-cpu==2.19.1 pillow
# only for in-process AttackTarget / --adv-pt:
/tmp/tpu_rt/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Do **not** install `onnxruntime`/`onnx` into that same env: `onnx` needs `protobuf>=6.31`
while TF 2.19 requires `<6`, and importing the harness's `runtimes.py` there also trips a
duplicate-LLVM-registration abort (triton + TF/MLIR). Two robust options:

1. **Separate scoring process (recommended).** Craft/save attacks in the existing harness
   env, dump `npz` (`images`, `labels`) next to each `*_eps*.pt`, then score with
   `edgetpu_runtime.py --npz ... [--adv-npz|--adv-pt] --logits-out ...` in the TF env and
   merge the resulting `logits.npz`/JSON back into the analysis.
2. **In-process target.** Give the TF env `torch` (CPU wheel) and import only the
   `AttackTarget` class protocol — verified working (torch in -> torch out, 10-image
   batching, `logits_for_grad` through `ExactOutput` for BPDA); avoid importing the whole
   `runtimes` module in that process.

## Files in this workspace (after cleanup)

| Path | What |
|---|---|
| `tools/edgetpu_compiler_16.0/bin/edgetpu_compiler` | latest compiler, 16.0.384591198, self-contained |
| `resnetv2_50_int8_nhwc.tflite` | **the TF model**: static, NHWC, full-int8, raw RGB uint8 in, harness normalization (0.485/0.229) baked into the stem conv |
| `resnetv2_50_int8_nhwc_edgetpu.tflite` | **the final compiled runtime** model (114/114 ops on TPU) |
| `edgetpu_runtime.py` | **evasion runtime/scorer**: `run(pixels) -> logits` adapter + CLI (`--check`, `--pilot`, `--npz`, `--adv-pt`, `--adv-npz`) |
| `EDGETPU_COMPILE_ISSUE.md` | this document |
| `resnetv2_50.onnx`, `resnetv2_50/`, `export_tflight*.py` | your pre-existing sources/scripts (untouched) |
| `resnetv2_50_quant.tflite` | your original broken export (kept for reference, delete when done) |
| `edgetpu/` | your original google-coral/edgetpu clone with the obsolete 14.1 compiler |

## Takeaways / gotchas

- Do **not** use `onnx-tf` to feed a Coral Edge TPU pipeline: it keeps NCHW (transposes
  everywhere) and preserves symbolic batch dims (dynamic shapes). Use `onnx2tf` (or
  `ai-edge-torch` from PyTorch), which emits NHWC with static shapes.
- Make the batch dimension concrete (`-b 1` / `TensorSpec([1,...])`) before converting.
  A dynamic batch alone is enough to make the delegate refuse the whole model.
- Validate **before** compiling: 0 tensors with `-1` in `shape_signature`, 0 float32 tensors,
  0 `TRANSPOSE` ops.
- `fully_quantize: 0` in the TF log and "Model not quantized" from compiler 14.1 are both
  unreliable signals; use compiler 16.0's error text instead.
- The Edge TPU 8 MB SRAM can't hold ResNet-50's 25 MB of parameters; 17.95 MiB streams from
  off-chip. That's normal, but it is why a ResNet-50 on a single Edge TPU is slow
  (use `-n/--num_segments` if you need pipelining across multiple TPUs).
- Input is **raw RGB uint8** (`[1,224,224,3]`, scale 1/255, zp 0) and the harness
  normalization `(pixel - 0.485/0.456/0.406) / (0.229/0.224/0.225)` is folded into the stem
  conv, so callers pass pixel-space `[0,1]` tensors exactly like every other runtime arm.
  The fold is algebraically exact (`w' = w/std`, `b' = -sum(w*mean/std)`); the 1/255 input
  quantization is part of the deployed interface, not a modelling choice.
