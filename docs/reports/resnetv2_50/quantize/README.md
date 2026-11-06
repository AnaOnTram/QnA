# Quantizing ResNet-50 for CPU and GPU

Measured pipelines for fp32 / fp16 / INT8 ResNet-50, and why the existing INT8
artifact was neither smaller nor faster.

All numbers below were produced on this machine and are reproducible with the
scripts in this directory. Where something could not be measured, it says so
instead of estimating.

---

## 1. What was actually wrong

The existing INT8 model (`onnx/model_opt/resnet50_qat_int8.onnx`) is not an
INT8 model. It is an INT8 *recipe*: float weights with Quantize/Dequantize
nodes wrapped around them.

```
$ python quantize/inspect_onnx.py onnx/model_opt/resnet50_qat_int8.onnx
  file size           : 98.00 MiB
  dtypes              : {'FLOAT': 263}
  Q/DQ nodes          : 220 (on initializer 54, on activation 166)
  top ops             : QuantizeLinear 110, DequantizeLinear 110,
                        Conv 53, BatchNormalization 53
  VERDICT             : Q/DQ SIMULATION: weights are stored in float ...
```

Compare the original fp32 model: 97.61 MiB, 108 float tensors, no Q/DQ. The
"INT8" file is 0.4 MiB *larger*.

Four independent causes, each verified:

**a. The weights are float, not int8.** Every conv weight is still a `FLOAT`
initializer, and the graph contains `Q(weight) -> DQ -> Conv`. The weight is
quantized on every forward pass, at run time, and the full-size float tensor
stays in the file. Nothing can be smaller, because nothing was compressed.

**b. The Q/DQ was never meant for ONNX Runtime.** It is a strongly-typed
TensorRT graph. ONNX Runtime's CPU execution provider does not fuse
`DQ -> Conv` into `QLinearConv`; after optimization the graph still holds
float convolutions and separate quantize/dequantize ops:

```
343 nodes: DequantizeLinear 110, QuantizeLinear 56, Conv 53,
           BatchNormalization 53, Relu 49, Add 16 ...
integer kernels in graph: NONE
```

So the runtime does float convolution *plus* 166 separate quantize/dequantize
ops. That is strictly more work than fp32, which is exactly why it was slower.

Reproduce by setting `optimized_model_filepath` on a session with
`ORT_ENABLE_ALL` and reading the op histogram of the written graph;
`quantize/ort_kernels.py` automates the same inspection.

**c. BatchNorm is unfused.** The graph has 53 `BatchNormalization` nodes. A
separate BN after each conv forces the tensor back to fp32 between layers and
prevents any int8 chain from forming.

**d. Wrong runtime for the artifact.** This Q/DQ form only pays off inside
TensorRT, which folds Q/DQ into int8 tactics at build time.

### A separate finding: this model does not classify these images

Running it over the full Imagenette validation split (3,925 images, all 1,000
ImageNet classes retained):

```
n=3925   top-1 9.73%   top-5 31.49%
```

9.73% top-1 is chance for ten classes (1/10 = 10%). The logits are not
degenerate — they are confident (85% peak softmax) and simply wrong. Identical
results at resize 232 and 256, and with pixels fed as raw `[0,1]` instead of
normalized, so this is not a preprocessing mismatch. The artifact is a
different model from the supplied `resnet50_fbgemm-23753f79.pth`, which the
evasion experiment measured at 91% clean accuracy. Its timings are real; its
accuracy story was never checked.

The ONNX model supplied as fp32 is also not bit-identical to any checkpoint
here (93% argmax agreement with `19c8e357`, logit delta 5.7), so every
comparison below uses a freshly exported baseline instead:
`quantize/artifacts/fp32_v2.onnx` from `resnet50-11ad3fa6.pth`, verified to
reproduce PyTorch to `1.9e-06` max logit difference.

---

## 2. Results

Full Imagenette validation split (3,925 images), 1,000-class head, one Intel
Core Ultra 7 165H (avx_vnni), ONNX Runtime 1.24.4 CPU EP, batch 1, median of
3 interleaved sweeps of 50 timed runs.

```
model                         MiB    int8w  top-1    p50 ms   speed   size
fp32 V2 export               97.42    0.00  87.18%   21.50    1.00x   1.00x
fp16 V2 export               48.75    0.00  87.11%   20.77    1.04x   0.50x
int8 QOperator (ORT)         24.48   24.32  85.94%    7.56    2.84x   0.25x
int8 DQ per-channel          30.49   22.39  86.34%   37.49    0.57x   0.31x
torch PTQ + BN + residual Q  24.69   24.32  86.57%   35.77    0.60x   0.25x
supplied QAT int8 QDQ        98.00    0.00   9.73%   38.98    0.55x   1.01x
```

Sources: `quantize/artifacts/benchmark_cpu_b1.json` (latency, size),
`quantize/artifacts/results.json` (storage layout, most accuracies),
`quantize/artifacts/fp16_v2.acc_full.json` (fp16 accuracy).

fp16 speed on this CPU is misleading: the CPU EP widens to fp32 internally, so
the 2x size reduction is real while the speed gain is noise. Its accuracy
(87.11%) shows that fp16 costs essentially nothing here.

Read the speed column with the format in mind:

- **QOperator, 2.84x faster.** Quantized weights are stored *inside* the conv
  node as `QLinearConv`, and the whole graph runs in integer kernels:
  `QLinearConv 53, QLinearAdd 16, QLinearGlobalAveragePool 1, QGemm 1`.
- **Everything else, ~0.6x.** These are Q/DQ graphs. Under the ORT CPU EP they
  keep float convolutions and pay for separate quantize/dequantize ops. They
  are built for TensorRT, not for ORT.

At batch 8 the picture widens: QOperator 52.54 ms vs fp32 169.97 ms (3.23x);
the Q/DQ forms are 0.65x and 0.69x (`benchmark_cpu_b8.json`).

---

## 3. The pipelines

### A. INT8 for ONNX Runtime CPU — the one that works here

```bash
python -m modelopt.onnx.quantization ...        # or onnxruntime.quantization
quantize/ort_qoperator.py \
    --onnx quantize/artifacts/fp32_v2.onnx \
    --calib quantize/calib/calib_512.npy \
    --output quantize/artifacts/int8_ort_qop_v2.onnx \
    --format QOperator
```

24.48 MiB, 85.94%, 2.97x faster. 30 s of calibration on 512 images; no
training. This is the artifact to deploy on CPU.

### B. INT8 with real int8 weights, in ModelOpt's QDQ form

ModelOpt's ONNX PTQ keeps float weights by default — `INT8QuantExporter` is an
unimplemented stub in both the installed 0.47.0 and the checkout at
`/home/ross/QA/Model-Optimizer`:

```python
# modelopt/onnx/export/int8_exporter.py
# TODO: Implement the INT8QuantExporter
class INT8QuantExporter(ONNXQuantExporter):
    @staticmethod
    def compress_weights(onnx_model): return onnx_model   # no-op
```

The weight-compression step that *does* exist is `qdq_to_dq`, exposed as
`dq_only`:

```bash
python quantize/run_onnx_ptq.py \
    --onnx quantize/artifacts/fp32_v2.onnx \
    --calib quantize/calib/calib_512.npy \
    --output quantize/artifacts/int8_dq_ort.onnx --dq-only
```

97.61 -> 30.49 MiB (3.20x), 86.34%. Without `--dq-only` you get back a
97.61 MiB file — the exact failure mode you started with.

### C. INT8 the "QAT" way, done properly

`quantize/ptq_torch.py` shows the full torch-side flow and fixes the two things
that make the example artifacts weak:

1. **Fold BatchNorm before quantizing.** Otherwise 53 BN nodes survive into the
   graph between layers.
2. **Quantize the residual shortcut.** `mtq.INT8_DEFAULT_CFG` never enables a
   shortcut quantizer, so TensorRT cannot keep ResNet's 16 residual adds in
   int8. The quantizers must be installed via `CUSTOM_POST_CONVERSION_PLUGINS`
   — attaching them before `mtq.quantize` fails, because module conversion
   rebuilds the graph and discards them.

```bash
python quantize/ptq_torch.py \
    --checkpoint Pytorch/resnet50-11ad3fa6.pth \
    --calib quantize/calib/calib_512.npy \
    --output quantize/artifacts/torchptq_v2_bnfused.onnx
```

24.69 MiB, 86.57% — the best INT8 accuracy measured here (−0.61 pp vs fp32).
On ORT CPU it runs at 0.60x, because ORT will not fuse it; its target is
TensorRT.

### D. fp16 / bf16

```bash
python quantize/export_onnx.py --checkpoint Pytorch/resnet50-11ad3fa6.pth \
    --dtype fp16 --output quantize/artifacts/fp16_v2.onnx
```

48.75 MiB, true 2x, max logit delta 4.8e-03. For GPU deployment prefer leaving
the ONNX in fp32 and letting TensorRT build with `--fp16`: it keeps sensitive
layers in fp32 and is the more reliable speed win.

**bf16** is refused for quantized graphs — `get_onnx_bytes_and_metadata` raises
`"Converting a quantized ONNX graph to BF16 is not supported yet"` — and on
sm_89 it gives no convolutional throughput over fp16. For a ResNet there is no
case for it.

---

## 4. QAT: what it does and does not fix

In ModelOpt, QAT is `mtq.quantize()` (PTQ + calibration) followed by ordinary
fine-tuning with frozen quantizer scales, saved via `mto.save` and reloaded with
`mto.restore`. It improves **accuracy**. It does not change footprint or kernel
selection: the export artifact is the same Q/DQ graph either way.

NVIDIA's own numbers in `examples/cnn_qat/README.md`: 76.1% fp32, 75.5% PTQ,
75.9% QAT — PTQ costs ~0.6 pp, QAT recovers ~0.4 pp. Here PTQ alone cost
0.61 pp (86.57% vs 87.18%), so QAT would buy back well under a percentage
point. Not worth a training campaign unless you have a hard floor.

`examples/cnn_qat/torchvision_qat.py` expects an ImageNet-style train/val tree
with a 1,000-way head. Fine-tuning that head on Imagenette's ten classes would
damage the model, so the example is not usable with this data as written.

---

## 5. CUDA and TensorRT: installed, but not usable here

TensorRT and CUDA are both present as software; neither can execute. The gap
between "installed" and "usable" is the whole story, so this section states
both. Verify with `python quantize/check_env.py` (exit 1 when no GPU is usable).

### What is present

```
GPU hardware     NVIDIA RTX 2000 Ada Generation Laptop GPU  (sm_89, via /proc)
driver module    NVRM 580.178.04, open kernel module (loaded)
trtexec          /usr/bin/trtexec, TensorRT v11.1.0 (b106)
libnvinfer       libnvinfer.so.11, libnvinfer.so.11.1.0
                 (+ builder resources for sm75/80/86/89/100/120)
cuda libs        libcublasLt.so.12 and .so.13, libcudnn.so.9, libcudart.so.12
torch            2.15.0+cu132 (built for CUDA 13.2)
onnxruntime-gpu  1.24.4, advertises TensorRT + CUDA + CPU providers
```

### What blocks it

**Blocker 1: no GPU device nodes.** `/dev/nvidia*` does not exist — not in
`/dev`, and not in `/proc/1/root/dev` either, so the whole environment was
created without GPU passthrough:

```
device nodes     : 0  (ABSENT - no GPU passthrough into this environment)
  in PID-1 root  : 0
nvidia-smi       : rc=9  NVIDIA-SMI has failed because it couldn't communicate ...
torch            : 2.15.0+cu132 (built cuda 13.2)
  cuda available : False   devices: 0
```

This is not a library or version problem and nothing installed inside can fix
it. `trtexec` fails at the same point: `no CUDA-capable device is detected`.
An escalated probe of the host was refused by the review gate; I did not route
around it.

**Blocker 2: the ORT TensorRT EP cannot load anyway.** Independent of the GPU,
`onnxruntime-gpu` 1.24.4 was built against TensorRT 10, and TensorRT 11.1 is
what is installed:

```
ORT TRT EP wants : ['libnvinfer.so.10']
  MISSING        : ['libnvinfer.so.10']  (installed: ['libnvinfer.so.11', 'libnvinfer.so.11.1.0'])
```

So `TensorrtExecutionProvider` would fail even on a healthy GPU. If you want
TensorRT through ORT you also need `nvidia-modelopt[onnx]`-style TRT 10 wheels,
or a newer onnxruntime built against TRT 11. Use `trtexec` directly meanwhile —
that path is version-correct.

Both providers silently fall back to CPU, which is why a naive benchmark shows
a plausible-looking number with no GPU involved:

```
FAIL CUDAExecutionProvider        silently fell back to CPU
FAIL TensorrtExecutionProvider    silently fell back to CPU
```

`benchmark.py` therefore reports the provider that was actually active per row,
and `check_env.py` fails loudly instead of assuming.

### No TensorRT numbers are claimed

`benchmark_trt.py` refuses to run rather than produce a misleading result:

```
$ python quantize/benchmark_trt.py --onnx quantize/artifacts/torchptq_v2_bnfused.onnx
cannot benchmark: no GPU device nodes (/dev/nvidia*). ...
(exit 2)
```

The commands are ready for a machine with a GPU:

```bash
python quantize/check_env.py            # confirm the GPU is actually reachable

# int8, strongly typed, honours the Q/DQ graph
python quantize/benchmark_trt.py --onnx quantize/artifacts/torchptq_v2_bnfused.onnx \
    --precision stronglyTyped --batch 1 --json quantize/artifacts/trt_int8.json

# fp16
python quantize/benchmark_trt.py --onnx quantize/artifacts/fp32_v2.onnx \
    --precision fp16 --batch 1 --json quantize/artifacts/trt_fp16.json
```

Expectation, not measurement: `torchptq_v2_bnfused.onnx` is the artifact most
likely to beat fp32 on GPU — folded BN, quantized residual shortcuts, int8
weights, which is what TensorRT needs to build int8 conv tactics.

Note that the Python `tensorrt` module is not installed (`MISSING`), so use the
`trtexec` binary or add `pip install tensorrt` for the API.

---

## 6. Reproduce

```bash
cd /home/ross/QA/Models/resnet50

python quantize/check_env.py                    # toolchain + GPU usability verdict
python quantize/build_calib.py --size 512       # calibration, evasion set excluded
python quantize/export_onnx.py --checkpoint Pytorch/resnet50-11ad3fa6.pth \
    --dtype fp32 --output quantize/artifacts/fp32_v2.onnx
python quantize/ort_qoperator.py --onnx quantize/artifacts/fp32_v2.onnx \
    --calib quantize/calib/calib_512.npy \
    --output quantize/artifacts/int8_ort_qop_v2.onnx --format QOperator
python quantize/ptq_torch.py --checkpoint Pytorch/resnet50-11ad3fa6.pth \
    --calib quantize/calib/calib_512.npy \
    --output quantize/artifacts/torchptq_v2_bnfused.onnx

python quantize/inspect_onnx.py quantize/artifacts/*.onnx
python quantize/verify_accuracy.py --onnx quantize/artifacts/int8_ort_qop_v2.onnx \
    --checkpoint Pytorch/resnet50-11ad3fa6.pth --provider cpu --batch 16
python benchmark.py --providers cpu --repeat 3 --runs 50
python quantize/collect_results.py
```

Accuracy protocol: full Imagenette validation split, 3,925 images, predictions
over all 1,000 ImageNet classes, preprocessing unchanged (shorter-side 232,
center crop 224, ImageNet normalization). The 100-image evasion pilot cannot
resolve a sub-percent quantization delta and is kept only for comparability with
`evasion/results/pytorch/`. Calibration uses 512 images disjoint from that pilot.

---

## 7. Known limitations

- **No TensorRT or CUDA numbers.** No GPU device nodes in this environment, and
  the ORT TensorRT EP is version-mismatched (built for TRT 10, installed 11.1).
  Full evidence in §5. `quantize/check_env.py` reports the current state;
  `quantize/benchmark_trt.py` refuses to run rather than mislead.
- **QOperator per-channel hits an ONNX Runtime bug.** Per-channel weight scales
  raise `ValueError: operands could not be broadcast together with shapes
  (16384,) (256,)` inside `quantize_onnx_initializer`, called from
  `_requantize_weight`, which receives `axis=None` for a `[256,64,1,1]` conv
  weight. Per-tensor is used instead, and costs about 0.4 pp against the
  per-channel QDQ model (85.94% vs 86.34%). Worth retrying on a newer ORT.
  If you need that 0.4 pp today, `int8_dq_ort.onnx` is per-channel but runs at
  0.57x on the ORT CPU EP, so it is a speed-for-accuracy trade, not a free win.
- **Latency varies.** Spread across sweeps reached 17% on this laptop. Use
  `--repeat 3` or more for any claim; single sweeps are not reliable here.
- **Not a full ImageNet benchmark.** Imagenette is ten classes out of 1,000.

## 8. Files

```
quantize/
  inspect_onnx.py       storage census + deployable-int8 verdict
  check_env.py          toolchain versions
  build_calib.py        512-image calibration set (evasion samples excluded)
  export_onnx.py        fp32 / fp16 / bf16 export from a hash-checked checkpoint
  run_onnx_ptq.py       ModelOpt ONNX PTQ, with --dq-only weight compression
  ort_qoperator.py      ORT static quantize, QOperator or QDQ
  ptq_torch.py          torch-side PTQ: BN fold, residual quantizers, int8 export
  verify_accuracy.py    full-split top-1/top-5 and PyTorch logit comparison
  benchmark_trt.py      trtexec build + timing
  ort_kernels.py        which kernels actually execute
  sanity_logits.py      degenerate-output check
  identify_source.py    which checkpoint an ONNX graph holds
  collect_results.py    aggregate size/dtype/accuracy into one JSON
  ckpt.py               hash-gated checkpoint loading, BN folding
  _trace_bn.py          node-by-node trace of one residual block
  _ort_opt_ops.py       op histogram of ORT's optimized graph
  calib/                calibration arrays + manifests
  artifacts/            generated models, metadata, results.json,
                        benchmark_cpu_b1.json, benchmark_cpu_b8.json
```

Files beginning with `_` are one-off diagnostic helpers.

## 9. Timm-base artifacts (unified matrix)

The primary robustness matrix uses one canonical base model,
`timm/resnetv2_50.a1h_in1k`, because both NPU deployments already use it and
because mixing it with TorchVision ResNet-50 confounds a ~16 pp base-model shift
with the runtime effect under study (see `../BASE_MODEL_UNIFICATION.md`).
These artifacts extend the recipes above to that base:

| Artifact | Recipe | Clean top-1 /100 | sha256 (prefix) |
| --- | --- | ---: | --- |
| `timm_rnv2_fp32.onnx` | canonical ONNX (QAIRT example export), byte-identical copy | 81 | `6416f2e9…` |
| `timm_rnv2_fp16.onnx` | ModelOpt `convert_to_f16`, fp16 I/O | 82 (99/100 top-1 vs fp32) | `23cc2388…` |
| `timm_rnv2_int8_ort_qop.onnx` | ORT static QOperator, per-channel, 512-image calib | 78 | `ae524e4d…` |
| `timm_rnv2_int8_ptq_ort.onnx` | ModelOpt ONNX PTQ explicit Q/DQ (float weights stored) | 79 | `99755d64…` |
| `timm_rnv2_int8_dq_ort.onnx` | same, `--dq-only` int8 weight compression | 79 | `fd967b8c…` |

The TorchVision INT8 recipe axis (torch PTQ BN-folded / unfused, FBGEMM) is
intentionally not ported: the timm graph is pre-activation, so 17 of its 49
BatchNorms cannot fold (the RKNN optimiser leaves exactly those 17). Porting
those recipes would change the graph structure, not just the quantizer. The TV
QOperator artifact is per-tensor because of an ORT per-channel bug on that graph;
the timm QOperator build uses per-channel successfully.

Reproduce / verify (CPU-only except TensorRT engines and GPU attack runs):

```bash
python quantize/sources.py verify rnv2_50            # 251/251 initializers byte-equal
python quantize/export_timm_onnx.py                  # export == canonical, logit delta 0.0
python quantize/build_timm_artifacts.py --all        # artifacts + gate
python quantize/verify_sources.py --deep --require-all
```

Evidence: accuracy gate `../evasion_onnx/results/validation/timm_artifact_gate.json`,
provenance graph `sources.json`, verifier output
`../evasion_onnx/results/validation/sources_verification.json`.

One deliberate deviation is recorded in the PTQ metas: NVIDIA ModelOpt's
`load_onnx_model` runs `infer_shapes_path` in place on its input, which would
rewrite the pinned fp32 graph, so the PTQ steps run on a byte-identical scratch
copy and the canonical `timm_rnv2_fp32.onnx` is kept byte-identical to
`onnx/resnetv2_50.onnx`.
