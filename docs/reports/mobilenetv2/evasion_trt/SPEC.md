# MobileNetV2 TRT evasion study — shared spec & interface contract

Status: **authoritative for all agents**. Read this before writing code. The Lead
owns this file; do not edit it — propose changes in a message to the Lead.

## 1. Objective

Replicate the ResNet-50 adversarial-evasion methodology
(`/home/ross/QA/Models/resnet50/evasion_onnx/`) against the MobileNetV2
TensorRT deployment, for **two attack families**:

* **Type 1 — white-box gradient attacks.** FGSM and PGD (adaptive BPDA for the
  quantized arms), with an *autograd bridge*: every forward score is the real
  TensorRT engine logit (bit-identical), every backward derivative comes from a
  differentiable FP32 replica of the arm's own ONNX graph.
* **Type 2 — black-box score-only attack.** Square attack, 500 engine queries
  per clean-correct image, no gradients and no transfer warm start.

Plus supporting controls from the ResNet-50 protocol: random-noise control,
1/255 stronger-survivor check (100 steps × 5 restarts), and FP32→arm transfer.

## 2. Where things run

* Code is edited **only** in the local checkout
  `/home/ross/QA/Models/mobilenetV2/evasion_trt/`.
* Heavy execution happens on **pro4000** (`ssh pro4000`, 192.168.1.32,
  RTX PRO 4000 Blackwell, TensorRT 11.3.0.99, CUDA 13.2, torch 2.15.0+cu132).
  Use `evasion_trt/sync.sh` (push / pull / run) — never edit on pro4000.
* Remote interpreter: `/home/ross/QA/venv/bin/python`.
* The 10 prebuilt engines are at `~/QA/Models/mobilenetV2/trt/engine/`:
  `<arm>_{bs1,bs8}.engine`. They are sm_120/TensorRT-11.3 specific.

Example:
```bash
./evasion_trt/sync.sh run evasion_trt/scripts/run_evasion.py --arm fp32 --smoke
```

## 3. Verified environment facts (recon already done by Lead — trust these)

* Docker/venv has `onnx` 1.21, `onnxruntime` 1.24.4, `torch` 2.15, `tensorrt` 11.3.
* Engines have ONE static profile: bs1 or bs8. All I/O is **FLOAT**:
  input `[B,3,224,224]`, output `[B,1000]`, name `input`/`logits`.
* **The bs1 engine is the authoritative attack target.** bs1 and bs8 engines are
  not bit-identical (max logit diff 0.0092 on the 100 samples), so all numbers
  must be scored on the *bs1* engine only. `predict()` loops bs1 runs.
* Preprocessing (from `TensorRT-Model-Optimizer/onnx_ptq/evaluate.py`):
  `Resize(256, interpolation=BICUBIC, antialias=True)` → `CenterCrop(224)` →
  `ToTensor` → normalize mean `(0.485,0.456,0.406)` std `(0.229,0.224,0.225)`.
  **NOTE: resize 256, not 232** (232 was the timm ResNetV2 setting).
* Sample population: Imagenette2-320 `val`, identical on both hosts, 10 images ×
  10 classes, `numpy.random.default_rng(20261030)` `choice(..., replace=False)`,
  files sorted. This reproduces the ResNet-50 100-image pilot.
* Clean top-1 on the 100 samples (bs1 engines), already measured:
  fp32 83, fp16 83, fp8 84, int8 79, int4 83; 78 correct in all five.
* fp32 engine vs torchvision PyTorch: 100% top-1 agreement, max abs logit diff
  0.042, mean 0.0044. fp32 engine likely uses TF32 in some conv tactics — this is
  the expected surrogate gap, same situation as the ResNet-50 study.
* ONNX graph op set (`trt/mobilenetv2*.onnx`): Conv, Clip (ReLU6, min/max as
  constant inputs), Add, ReduceMean (axes as input `[-1,-2]`, keepdims=1),
  Reshape (shape input `[-1,1280]`, allowzero=1), Gemm (transB=1), Cast
  (identity for the replica), QuantizeLinear/DequantizeLinear.
  * fp8: 64 Q/DQ pairs, weights per-channel `axis=0`, activations per-tensor,
    all zero_points = 0, zero-point dtype `float8_e4m3fn`.
  * int8: 104 Q/DQ pairs, weights `axis=0`, activations per-tensor, zp int8 = 0.
  * int4: ONE `DequantizeLinear` on `classifier.1.weight_i4` shape `(1000,1280)`,
    scale `(1000,10)` float32, `axis=0`, `block_size=128`. Verified dequant is
    `W_dq[o,i] = w4[o,i] * scale[o, i//128]` (mean abs err vs fp32 weight 0.0046;
    this is interpretation "A", the only sane one). The graph has no other INT4.
* Every arm's engine is FP32 I/O. `/home/ross/QA/datasets/imagenette2-320/val`
  exists on both hosts with identical file counts.

## 4. Directory layout (under `evasion_trt/`)

```
scripts/config.py        # Lead-owned: paths, arm registry, protocol constants
scripts/replica.py       # replica-dev: GraphReplica (differentiable ONNX replica)
scripts/probe_replica.py # replica-dev: replica-vs-engine validation probe
scripts/runtimes.py      # harness-dev: EngineRuntime + AttackTarget (autograd bridge)
scripts/attacks.py       # harness-dev: load_samples, pgd, square, project, validate_bounds
scripts/run_evasion.py   # harness-dev: full per-arm protocol runner
scripts/summarize.py     # harness-dev: reconcile arms -> comparisons/matrix
results/<arm>/           # Lead-owned run outputs
results/_probe/          # replica-dev-owned probe outputs
results/_smoke/          # harness-dev-owned smoke outputs
results/comparisons/     # summarize.py output
figures/                 # verifier-owned
logs/                    # redirected run logs
SPEC.md  sync.sh  README.md
```

`config.py` is the single source of truth for paths/arms/protocol. Import it;
do not hardcode paths or epsilons elsewhere.

## 5. Interface contract (must match exactly so the pieces compose)

### `scripts/replica.py`
```python
class GraphReplica(torch.nn.Module):
    def __init__(self, onnx_path): ...
    # pixels: float32 [B,3,224,224] in [0,1]; module normalizes internally
    def forward(self, pixels) -> Tensor[B,1000]
    op_counts: dict            # Counter of executed dynamic ops, for summary.json
```
* Runs in float32 even for the fp16/fp8/int8 graph (Cast is identity).
* Quantized ops use straight-through estimators on the graph's own recorded
  scale/zero-point grids. int4 blocked dequant happens while folding constants.
* Must raise `NotImplementedError(op)` for an unsupported op rather than silently
  returning wrong values.

### `scripts/runtimes.py`
```python
class EngineRuntime:
    def __init__(self, engine_path): ...
    def run(self, pixels) -> Tensor[B,1000]     # B must equal the engine's static batch
    def predict(self, pixels) -> Tensor[B,1000] # arbitrary B, loops run()
    info: dict                                  # runtime.json payload (hash, TRT version, IO)
    def verify(self, workdir): ...              # writes runtime.json

class AttackTarget:
    def __init__(self, name, runtime, batch_size, surrogate): ...
    @torch.no_grad()
    def predict(self, pixels) -> Tensor         # actual engine logits
    def logits_for_grad(self, pixels) -> Tensor # exact forward, surrogate backward
```
`logits_for_grad` must be bit-identical in forward to `predict` (assert it).

### `scripts/attacks.py`
```python
def load_samples(per_class=config.PER_CLASS, seed=config.SAMPLE_SEED) -> (Tensor[N,3,224,224], LongTensor[N], list[Path])
def project(x, clean, eps); def validate_bounds(x, clean, eps)
def pgd(target, clean, labels, eps, steps, restarts, seed, fgsm=False)
def margin(logits, labels)
def square(target, clean, labels, eps, max_queries, seed, clean_logits) -> (adv, queries, trace)
def log(msg); def sha256(path)
```
Attack math is copied from
`resnet50/evasion/scripts/run_evasion.py::pgd/square/margin/project/validate_bounds`
and `resnet50/evasion_onnx/scripts/run_model.py::record` — same semantics:
CE ascent, per-step best-success retention, restart selection, eps/4 step,
L∞ projection then [0,1] clip after every step.

## 6. Protocol to run per arm (mirror of ResNet-50)

1. clean: engine logits on all 100 images; save `clean.npz`, `samples.json`.
2. `random` control at eps ∈ {1,2,4,8}/255.
3. `fgsm` at the same eps.
4. `pgd` 20 steps × 2 restarts at the same eps (step eps/4).
5. `pgd_stronger` at 1/255: survivors get 100 steps × 5 restarts; baseline
   successes retained; reported separately.
6. `square` at 4/255 and 8/255, 500 queries, stripe init, margin objective,
   decreasing patch schedule; stop on success; independent of white-box.
7. For every non-fp32 arm: `transfer_fp32_<fgsm|pgd>` for each eps from the fp32
   arm's saved candidates (no adaptation).
8. Replica gate: replica vs engine clean top-1 agreement must be measured and
   reported; a run with agreement < 90% must STOP and report, not silently
   produce a "robustness" number (same rule as the ResNet-50 runner). The Lead
   may accept a lower floor only with a recorded reason.

Outputs per arm: `summary.json`, `per_image.csv`, `clean.npz`, `samples.json`,
`<attack>_eps<e>.pt` candidates, `runtime.json`.

`summary.json` must contain at minimum: arm, onnx/engine hashes, runtime info,
environment, config, clean counts, gradient/replica diagnostics, per-attack
`n/clean_correct/fooled/attack_success_pct/robust_accuracy_pct/max_linf_255/
seconds` (+ query stats for Square), and `complete`.

## 7. Acceptance criteria

* Every reported adversarial sample passes `validate_bounds` (finite, [0,1],
  L∞ ≤ eps) and is re-scored on the bs1 engine.
* `summarize.py` independently reconciles every `per_image.csv` row, checks
  hashes and query budgets, and emits `results/comparisons/matrix.{json,md}`
  with per-arm success on (a) own clean-correct set, (b) shared 78-image set.
* No model weights, ONNX or engines are modified (source and engine hashes
  unchanged at end of run).
* Reproducibility: fixed seeds recorded; re-running an arm overwrites only its
  own `results/<arm>/`.

## 8. Team rules

* Disjoint write scopes as in §4. If you must touch another owner's file, message
  the Lead first.
* Edit locally, run remotely through `sync.sh`, pull with `sync.sh pull`.
* Long GPU jobs: redirect to `logs/<name>.log` and report the job id/command.
  Do not run two full attack matrices concurrently.
* Report back with: files written, exact commands run, measured numbers,
  anomalies, and anything the Lead must re-verify.
