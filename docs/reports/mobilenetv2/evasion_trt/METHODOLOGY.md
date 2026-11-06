# MobileNetV2 TensorRT evasion study — methodology memory

Project memory for the MobileNetV2 input-evasion study in this directory. It
records the protocol, the model/precision matrix, the environment, the exact
reproduction commands, the harness schema, and the results table. The
authoritative pre-run contract is [`SPEC.md`](SPEC.md); this file is the durable
summary that survives the session (and is reusable for later MobileNet/ResNet/
YOLO ports).

**Status (final):** **the full study is complete and reconciled.** All seven
scripts are built — `config.py` (`Models/mobilenetV2/evasion_trt/scripts/config.py`; source workspace only),
`runtimes.py` (`Models/mobilenetV2/evasion_trt/scripts/runtimes.py`; source workspace only), `replica.py` (`Models/mobilenetV2/evasion_trt/scripts/replica.py`; source workspace only),
`attacks.py` (`Models/mobilenetV2/evasion_trt/scripts/attacks.py`; source workspace only), `run_evasion.py` (`Models/mobilenetV2/evasion_trt/scripts/run_evasion.py`; source workspace only),
`summarize.py` (`Models/mobilenetV2/evasion_trt/scripts/summarize.py`; source workspace only),
`probe_replica.py` (`Models/mobilenetV2/evasion_trt/scripts/probe_replica.py`; source workspace only). The replica probe PASSED all five
arms (agreement 100/100/99/95/100; INT4 dequant PASS; directional probe
informative), and the **full 5-arm matrix ran on pro4000 (2026-11-06, ~5 min GPU)
and was reconciled**: 107 conditions / 10,700 per-image rows, shared
clean-correct set **n=78**, **all validation gates passed**. Headline: **PGD/BPDA
evaded 100% of clean-correct images on every arm at every budget (1, 2, 4 and
8/255)**; FGSM was 82.3–91.6% at 1/255 and 93.7–97.5% across 4–8/255; Square
(score-only, 500 queries) was 19.0–48.2% at 4/255 and 49.4–85.5% at 8/255. See
§7.2, `results/comparisons/matrix.{json,md}` and `figures/`.

---

## 1. Objective

Replicate the ResNet-50 adversarial-evasion methodology
(`/home/ross/QA/Models/resnet50/evasion_onnx/`) against the **deployed
MobileNetV2 TensorRT engines**, for two attack families:

- **Type 1 — white-box gradient attacks.** FGSM and PGD (adaptive BPDA for the
  quantized arms), using an **exact-runtime-forward / replica-backward**
  autograd bridge: every forward score is the real TensorRT engine logit
  (bit-identical); every backward derivative comes from a differentiable FP32
  replica of that arm's own ONNX graph.
- **Type 2 — black-box score-only Square attack.** 500 engine queries per
  clean-correct image, no gradients and no transfer warm start.

Supporting controls: random-noise control, 1/255 stronger-survivor check
(100 steps × 5 restarts), and FP32→arm transfer with no adaptation.

The scientific question is a **deployed-configuration** comparison: how much does
the precision/quantization of the deployment change empirical evasion at fixed
images, preprocessing and protocol — not which runtime is "secure".

---

## 2. Model / precision matrix

One MobileNetV2 base model, exported by TensorRT-Model-Optimizer with five
precisions. All engines have one static batch profile (bs1 or bs8) and **FP32
I/O**: input `[B,3,224,224]`, output `[B,1000]`, tensor names `input`/`logits`.
The **bs1 engine is the authoritative attack target**.

| Arm | ONNX artifact | Engine files | Precision | Notes |
|---|---|---|---|---|
| `fp32` | `trt/mobilenetv2.onnx` | `trt/engine/mobilenetv2_fp32_{bs1,bs8}.engine` | TensorRT FP32 | transfer reference arm |
| `fp16` | `trt/mobilenetv2_fp16.onnx` | `…_fp16_{bs1,bs8}.engine` | TensorRT FP16 | |
| `fp8` | `trt/mobilenetv2_fp8.onnx` | `…_fp8_{bs1,bs8}.engine` | FP8 E4M3, ModelOpt Q/DQ | 64 Q/DQ pairs; weights per-channel `axis=0`, activations per-tensor, all zero_points 0, zp dtype `float8_e4m3fn` |
| `int8` | `trt/mobilenetv2_int8.onnx` | `…_int8_{bs1,bs8}.engine` | INT8, ModelOpt Q/DQ | 104 Q/DQ pairs; weights `axis=0`, activations per-tensor, zp int8 = 0 |
| `int4` | `trt/mobilenetv2_int4.onnx` | `…_int4_{bs1,bs8}.engine` | INT4 weight-only | ONE `DequantizeLinear` on `classifier.1.weight_i4` `(1000,1280)`, scale `(1000,10)` f32, `axis=0`, `block_size=128`; `W_dq[o,i]=w4[o,i]*scale[o, i//128]` (interpretation A; mean abs err vs FP32 weight 0.0046) |

Config source of truth: `scripts/config.py` (`Models/mobilenetV2/evasion_trt/scripts/config.py`; source workspace only) (`ARMS`,
`ARM_ORDER`, `REFERENCE_ARM = 'fp32'`, protocol constants). Paths resolve
relative to the file, so the same tree works locally and on pro4000.

**Engine hashes** are not hardcoded: they are recorded per run in each arm's
`runtime.json` (`engine_sha256`) and copied into `summary.json.engine_sha256`,
together with TensorRT version, GPU, I/O shapes/dtypes, static batch, and the
engine inspector's layer JSON (`tensorrt_layers.json`). The runner asserts the
ONNX and engine hashes are unchanged at the end
(`onnx_hash_unchanged`, `engine_hash_unchanged`, `source_hash_unchanged`).

Engine SHA-256 measured by the replica probe (TensorRT 11.3.0.99, bs1, FLOAT
I/O; recorded in `results/_probe/replica_validation.json`):

| arm | engine | sha256 |
|---|---|---|
| fp32 | `mobilenetv2_fp32_bs1.engine` | `de6c44806e79ad8e3d79be14b3f38e2fabff68cf8ddc6cc6974c2a3e89e63a86` |
| fp16 | `mobilenetv2_fp16_bs1.engine` | `7c9a2d74b1097cb13badf12f551754d17837f5395b22736213f2cb46eacbf3bb` |
| fp8 | `mobilenetv2_fp8_bs1.engine` | `16638a12ea1864530f5b2d8afc03eb245cd28c4bd05434ab8bd59a5b3c58b799` |
| int8 | `mobilenetv2_int8_bs1.engine` | `7636d85e9e859fc4fed928317e42c95a8ad153da1b5859d04c9b4770ff2461bf` |
| int4 | `mobilenetv2_int4_bs1.engine` | `11a44751c11d23ed87238445241a2c8e7b6480d4d281cc578f9e076b87e31810` |

---

## 3. Verified pre-run facts (from `SPEC.md` §3 — trust these)

- Environment: `onnx` 1.21, `onnxruntime` 1.24.4, `torch` 2.15,
  `tensorrt` 11.3.0.99, CUDA 13.2, on pro4000 (RTX PRO 4000 Blackwell, sm_120).
- **Clean top-1 / 100 on the bs1 engines** (already measured):

  | arm | fp32 | fp16 | fp8 | int8 | int4 |
  |---|---:|---:|---:|---:|---:|
  | clean correct | 83 | 83 | 84 | 79 | 83 |

  **78** images are correct in **all five** arms → the shared-population
  denominator for the matrix. `summarize.py` hard-codes these as
  `EXPECTED_FULL_CLEAN` and `EXPECTED_SHARED_ALL_FIVE = 78` and asserts them on
  a full (non-smoke) run.
- **FP32 engine vs torchvision PyTorch**: 100% top-1 agreement, max abs logit
  difference **0.042**, mean **0.0044**. The FP32 engine likely uses TF32 in some
  conv tactics — the expected surrogate gap, same situation as ResNet-50.
- **bs1 vs bs8 engines are not bit-identical**: max logit diff **0.0092** on the
  100 samples. Therefore all numbers are scored on the **bs1** engine only;
  `predict()` loops bs1 runs; `summarize.py` asserts `runtime.static_batch == 1`.
- **Preprocessing** (from TensorRT-Model-Optimizer `onnx_ptq/evaluate.py`):
  `Resize(256, interpolation=BICUBIC, antialias=True)` → `CenterCrop(224)` →
  `ToTensor` → normalize ImageNet mean `(0.485,0.456,0.406)` / std
  `(0.229,0.224,0.225)`. **Resize 256, not 232** (232 was the timm ResNetV2
  setting). Implemented in `attacks.py::load_samples` from `config.RESIZE`.
- **Population**: Imagenette2-320 `val`, identical on both hosts, 10 images ×
  10 classes, `numpy.random.default_rng(20261030).choice(..., replace=False)`,
  files sorted; 1000-class head; labels from folder synsets
  `{0,217,482,491,497,566,569,571,574,701}`.
- **ONNX op set** (`trt/mobilenetv2*.onnx`): Conv, Clip (ReLU6, min/max as
  constant tensor inputs), Add, ReduceMean (axes input `[-1,-2]`, keepdims=1),
  Reshape (shape input `[-1,1280]`, allowzero=1), Gemm (transB=1), Cast
  (identity for the replica), QuantizeLinear/DequantizeLinear. `replica.py`
  supports exactly this set (`_SUPPORTED`) and raises `NotImplementedError` for
  anything else.

---

## 4. Threat model and protocol

- Untargeted digital input evasion; pixel-space L∞ budgets **1, 2, 4, 8 / 255
  applied before normalization**, after resize/crop; pixels in `[0,1]`.
- Attack success is counted only on images the arm classifies correctly clean;
  already-wrong images never count as robust.
- Conditions per arm (constants in `config.py`; runner order is random → fgsm →
  pgd per epsilon, then the strong check, then Square):

| Condition | Budgets | Settings |
|---|---|---|
| `clean` | — | engine logits for all 100; `clean.npz`, `samples.json` |
| `random` | 1, 2, 4, 8 | uniform `[-eps,eps]`, clamp |
| `fgsm` | 1, 2, 4, 8 | 1 CE-ascent step of size `eps` |
| `pgd` | 1, 2, 4, 8 | 20 steps × 2 restarts, step `eps/4`, best-success retention |
| `pgd_stronger` | 1 | survivors re-attacked 100 steps × 5 restarts, `eps/4`, seed `STRONG_SEED=20262730` |
| `square` | 4, 8 | 500 queries/image, stripe init, margin objective, p=.05 schedule |
| `transfer_fp32_{fgsm,pgd}` | 1, 2, 4, 8 | non-FP32 arms score fp32 candidates, no adaptation |

- Seeds: sample `20261030`; per-eps attack seed `20261030 + round(eps255*100)`
  (same start for a given eps across arms); strong check seed `20262730`.
- Replica gate: replica-vs-engine clean top-1 agreement must be **≥ 90%**
  (`REPLICA_GATE_PCT`) for every arm. A full run below the gate **stops** and
  records `replica_gate.note = 'run STOPPED: agreement below gate (not
  robustness)'`; the Lead may accept a lower floor only with
  `--allow-replica-agreement PCT --override-reason TEXT`, which is recorded as
  `replica_gate.overridden / accepted_floor_pct / reason`. `--smoke` records a
  below-gate result without aborting (smoke is a pipeline test, not a result).
- No artifact mutation: ONNX/engine hashes unchanged at end of run.

### 4.1 The autograd bridge (why the result is defensible)

TensorRT is forward-only. `scripts/runtimes.py` implements:

```python
class ExactOutput(torch.autograd.Function):
    forward(ctx, approximate, actual) -> actual.clone()      # exact engine logits
    backward(ctx, gradient) -> (gradient, None)              # replica gradient

class AttackTarget:
    @torch.no_grad()
    def predict(self, pixels) -> Tensor                      # real engine logits
    def logits_for_grad(self, pixels) -> Tensor:
        approximate = self.surrogate(pixels)                 # differentiable replica
        actual = self.predict(pixels.detach().cpu()).to(pixels.device)
        return ExactOutput.apply(approximate, actual)
```

`run_evasion.py` asserts the bridged forward is **bit-identical** to the engine
before attacking (`exact_runtime_forward_verified`), and that the replica
produces a finite non-zero input gradient. The replica
(`scripts/replica.py` (`Models/mobilenetV2/evasion_trt/scripts/replica.py`; source workspace only)) executes the arm's own ONNX graph
in float32 (Cast=identity, weights/Clip bounds/ReduceMean axes/Reshape shape
folded at build time), uses STE on the recorded Q/DQ grids for int8/fp8, folds
the INT4 blocked dequant as a constant, and raises on unknown ops. Quantized arms
are therefore **adaptive BPDA**, not native autograd;
`gradient.native_runtime_autograd` is `false` in every summary, and
`gradient.method` says which convention was used.

The runner also computes `gradient.sign_direction_probe`: one `eps=8/255`
sign(gradient) step vs one random sign step, both scored on the engine. A working
gradient flips many engine predictions; a random direction flips few. This is
diagnostic evidence for the replica gate, not an attack condition.

---

## 5. Environment and workflow

| Item | Value |
|---|---|
| Local checkout (source of truth) | `/home/ross/QA/Models/mobilenetV2/evasion_trt/` |
| Remote host / execution | `pro4000` (`ssh pro4000`), `/home/ross/QA/Models/mobilenetV2` |
| Remote interpreter | `/home/ross/QA/venv/bin/python` |
| GPU | RTX PRO 4000 Blackwell, sm_120 |
| TensorRT / CUDA / torch | 11.3.0.99 / 13.2 / 2.15.0+cu132 |
| Sync tool | `./evasion_trt/sync.sh push\|push-all\|pull\|run` |
| Dataset | `/home/ross/QA/datasets/imagenette2-320/val` (identical on both hosts) |

Rules: edit files **only** in the local checkout; `sync.sh run` rsyncs code then
executes remotely; `sync.sh pull` brings back `results/`, `figures/`, `logs/`.
Never edit on pro4000 (the next push would overwrite it). Long GPU jobs redirect
to `logs/<name>.log`; never run two full matrices concurrently.

### 5.1 Harness files (as built)

| File | Role |
|---|---|
| `config.py` (`Models/mobilenetV2/evasion_trt/scripts/config.py`; source workspace only) | paths, five-arm registry, protocol constants, preprocessing |
| `runtimes.py` (`Models/mobilenetV2/evasion_trt/scripts/runtimes.py`; source workspace only) | `EngineRuntime` (static-batch TRT engine, `predict` loops bs1) + `AttackTarget` + `ExactOutput` bridge |
| `replica.py` (`Models/mobilenetV2/evasion_trt/scripts/replica.py`; source workspace only) | `GraphReplica`: FP32 differentiable replica of the arm's ONNX graph; STE quantizers; INT4 blocked dequant; `op_counts`; `NotImplementedError` on unknown ops |
| `attacks.py` (`Models/mobilenetV2/evasion_trt/scripts/attacks.py`; source workspace only) | `load_samples` (resize 256 bicubic antialias), `project`, `validate_bounds`, `pgd` (FGSM/PGD/BPDA), `margin`, `square`, `log`, `sha256` |
| `run_evasion.py` (`Models/mobilenetV2/evasion_trt/scripts/run_evasion.py`; source workspace only) | per-arm protocol: clean → gate → random/fgsm/pgd → strong 1/255 → square → transfer; writes `summary.json`, `per_image.csv`, `clean.npz`, `samples.json`, `<attack>_eps<e>.pt`, `runtime.json` |
| `summarize.py` (`Models/mobilenetV2/evasion_trt/scripts/summarize.py`; source workspace only) | independent reconciler → `<out>/comparisons/matrix.{json,md}`; re-derives every count, gate, bound, query and hash |
| `probe_replica.py` (`Models/mobilenetV2/evasion_trt/scripts/probe_replica.py`; source workspace only) | all-arm replica gate: clean top-1 agreement ≥90% (ideal ≥97% for fp32/fp16/int4), logit diff/correlation, finite non-zero gradient, one-step directional probe at 8/255 (gradient sign vs negated vs random, scored on the engine), INT4 folded-weight check (<0.01 mean abs err); writes `results/_probe/replica_validation.json` + `REPORT.md`, exits non-zero if the gate fails |

Ownership per `SPEC.md`: `config.py` Lead; `replica.py`/`probe_replica.py`
replica-dev; `runtimes.py`/`attacks.py`/`run_evasion.py`/`summarize.py`
harness-dev; `results/` + report Lead; this file + the reusable skill
skill-author.

### 5.2 Per-arm output schema (`summary.json`)

Important top-level keys: `arm`, `description`, `onnx`, `onnx_sha256`, `engine`,
`engine_sha256`, `runtime` (TensorRT info + hash), `environment`, `config`
(seed, n, resize/crop, eps sets, steps/restarts, square budget, seeds),
`clean` (`correct/n/accuracy_pct`), `gradient`
(`native_runtime_autograd`, `method`, `replica_clean_agreement_pct`,
`mean/max_absolute_logit_difference`, `exact_runtime_forward_verified`,
`probe_zero_gradient_images`, `replica_dynamic_ops`, `sign_direction_probe`),
`replica_gate` (`default_gate_pct`, `observed_pct`, `passed`, `overridden`,
`accepted_floor_pct`, `reason`, `note`), `attacks[]` (one entry per condition
with `n/clean_correct/fooled/robust_correct/attack_success_pct/
robust_accuracy_pct/max_linf_255/seconds`, plus query stats for Square), `complete`,
`engine_hash_unchanged`, `onnx_hash_unchanged`, `source_hash_unchanged`,
`elapsed_seconds`.

`per_image.csv` columns: `index, model, attack, eps_255, label,
clean_prediction, adv_prediction, clean_correct, adv_correct, linf_255, queries`
(`queries` blank for non-Square conditions). Candidate `.pt` files hold
`{adversarial, indices, labels}`; the FP32 reference keeps its full pool, other
arms one representative per class.

---

## 6. Reproduction commands

```bash
cd /home/ross/QA/Models/mobilenetV2

# --- 0. replica-vs-engine preflight for all five arms (gate; results/_probe/) ---
./evasion_trt/sync.sh run evasion_trt/scripts/probe_replica.py --device cuda
./evasion_trt/sync.sh pull

# --- 1. inspect one arm: clean parity, replica agreement, sign probe, runtime.json ---
./evasion_trt/sync.sh run evasion_trt/scripts/run_evasion.py --arm fp32 --inspect-only

# --- 2. smoke the pipeline (1 img/class, tiny budgets; writes results/_smoke) ---
./evasion_trt/sync.sh run evasion_trt/scripts/run_evasion.py --arm fp32 --smoke \
    --out evasion_trt/results/_smoke
./evasion_trt/sync.sh run evasion_trt/scripts/run_evasion.py --arm int8 --smoke \
    --out evasion_trt/results/_smoke
./evasion_trt/sync.sh run evasion_trt/scripts/summarize.py --out evasion_trt/results/_smoke
./evasion_trt/sync.sh pull

# --- 3. full per-arm protocol, FP32 first (transfer reference) ---
./evasion_trt/sync.sh run evasion_trt/scripts/run_evasion.py --arm fp32 --eps 1 2 4 8 --square-eps 4 8
for arm in fp16 fp8 int8 int4; do
  ./evasion_trt/sync.sh run evasion_trt/scripts/run_evasion.py --arm "$arm"
done

# --- 4. independent reconciliation + pull ---
./evasion_trt/sync.sh run evasion_trt/scripts/summarize.py
./evasion_trt/sync.sh pull
```

Argument names (`--arm`, `--eps`, `--square-eps`, `--device`, `--batch-size`,
`--inspect-only`, `--smoke`, `--out`, `--reference`, `--reference-dir`,
`--allow-replica-agreement`, `--override-reason`) are implemented in
`run_evasion.py`. Smoke defaults: `per_class=1`, eps `[1, 8]`, square `[8]`,
2 steps × 1 restart, strong 2×1, 10 queries. Full defaults come from
`config.py` (20×2, 100×5, 500 queries). If a run must be backgrounded, redirect
remotely:

```bash
ssh pro4000 'cd /home/ross/QA/Models/mobilenetV2 && \
  /home/ross/QA/venv/bin/python -u evasion_trt/scripts/run_evasion.py --arm fp32 \
  > evasion_trt/logs/fp32.log 2>&1'
./evasion_trt/sync.sh pull
```

Outputs per arm (`results/<arm>/`): `summary.json`, `per_image.csv`,
`clean.npz`, `samples.json`, `<attack>_eps<e>.pt` candidates,
`runtime.json` (+ `tensorrt_layers.json`). Reconciliation:
`results/comparisons/matrix.{json,md}` (or `<out>/comparisons/` for
`--out`). Figures: `figures/`.

---

## 7. Results

**Measured and reconciled** — `results/comparisons/matrix.md` (human-readable,
with 95% Wilson intervals for every cell) and `matrix.json` (structured), from
`summarize.py` at 2026-11-06T21:23. Validation: 5 arms reconciled, 0 pending,
10,700 per-image rows, 107 conditions, shared clean-correct n=78 (all five arms
present), model/engine/sample hashes unchanged, L∞ bounds valid, Square query
budgets valid, summary counts reconciled, transfer candidates verified,
PGD→stronger retention verified.

`own` = the arm's own clean-correct set; `shared` = the 78 images clean-correct in
all five arms. The two are never mixed in the tables below.

### 7.1 Clean

| arm | clean correct /100 | shared all-five-correct |
|---|---:|---:|
| fp32 | 83 | 78 |
| fp16 | 83 | 78 |
| fp8 | 84 | 78 |
| int8 | 79 | 78 |
| int4 | 83 | 78 |

### 7.2 Attack matrix (measured, own-denominator %)

`own n` = clean-correct images for the arm. FGSM = 1 step; PGD = 20 steps × 2
restarts at eps/4; `PGD 1 +strong` = 100 steps × 5 restarts on 1/255 survivors
(none remained, so it equals PGD). All percentages are `fooled / own n`.

| arm | own n | FGSM 1 | FGSM 2 | FGSM 4 | FGSM 8 | PGD 1 | PGD 2 | PGD 4 | PGD 8 | PGD 1 +strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 83 | 91.6 | 92.8 | 96.4 | 95.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| fp16 | 83 | 90.4 | 92.8 | 96.4 | 95.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| fp8 | 84 | 85.7 | 91.7 | 96.4 | 94.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| int8 | 79 | 82.3 | 96.2 | 97.5 | 93.7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| int4 | 83 | 90.4 | 94.0 | 96.4 | 95.2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

**Shared-78 companion** for the discriminative (non-saturated) conditions —
PGD is 100.0% on the shared set at every budget for every arm:

| arm | FGSM 1 | FGSM 2 | FGSM 4 | FGSM 8 | Square 4 | Square 8 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 91.0 | 92.3 | 96.2 | 94.9 | 44.9 | 84.6 |
| fp16 | 89.7 | 92.3 | 96.2 | 94.9 | 25.6 | 74.4 |
| fp8 | 84.6 | 92.3 | 96.2 | 93.6 | 20.5 | 51.3 |
| int8 | 82.1 | 96.2 | 97.4 | 93.6 | 17.9 | 48.7 |
| int4 | 89.7 | 93.6 | 96.2 | 94.9 | 41.0 | 84.6 |

### 7.3 Controls, Square, transfer (measured, own-denominator %)

| arm | random 1/2/4/8 | Square 4 | Square 8 | transfer PGD 1/2/4/8 | transfer FGSM 1 | transfer FGSM 8 |
|---|---|---|---|---|---|---|
| fp32 | 1.2 / 0.0 / 1.2 / 1.2 | 48.2 | 85.5 | — (reference) | — | — |
| fp16 | 0.0 / 0.0 / 0.0 / 1.2 | 30.1 | 75.9 | 100 / 100 / 100 / 100 | 91.6 | 95.2 |
| fp8 | 2.4 / 1.2 / 1.2 / 6.0 | 25.0 | 54.8 | 100 / 100 / 100 / 100 | 86.9 | 94.0 |
| int8 | 0.0 / 1.3 / 1.3 / 5.1 | 19.0 | 49.4 | 100 / 100 / 100 / 100 | 79.7 | 97.5 |
| int4 | 0.0 / 0.0 / 0.0 / 2.4 | 44.6 | 85.5 | 100 / 100 / 100 / 100 | 90.4 | 95.2 |

Square (500-query, score-only) query traces show the quantized arms plateauing:
fp32 4/255 went 18→23→31→40 at q=50/100/250/500 and 8/255 47→54→65→71; int8
4/255 went 13→14→15→15 and 8/255 35→37→38→39; fp8 4/255 17→17→20→21. int4
behaves float-like (4/255 18→22→32→37), consistent with weight-only
quantization leaving activations unchanged.

### 7.4 Findings

1. **Adaptive gradient attacks fully saturate every deployed precision.**
   PGD/BPDA-PGD evaded **100.0%** of clean-correct images on **all five arms at
   all four budgets**, and the 100×5 survivor check found no additional
   survivors (there were none to re-attack). MobileNetV2 is markedly more
   vulnerable than the ResNet-50 family at 1/255, where ResNet-50 float PGD was
   partial (76–80%) and INT8 resisted more (58–67%).
2. **FGSM is strong here** (82.3–91.6% at 1/255, rising to 96.4–97.5% at 4/255
   and 93.7–95.2% at 8/255), unlike the ResNet-50 family where FGSM at 1/255 was
   ~44%. The quantized arms are slightly more resistant to the one-step attack
   at 1/255 (int8 82.3%, fp8 85.7%) but not at 4/255.
3. **Quantization only matters for weak/finite attacks.** The float-like arms
   (fp32, fp16, int4) reach 75.9–85.5% Square success at 8/255, while fp8 is
   54.8% and int8 49.4%; the same ordering holds at 4/255 (17.9–48.2%). No
   precision survives PGD.
4. **INT4 ≈ FP32 under attack** (Square 44.6/85.5 vs fp32 48.2/85.5; FGSM
   90.4–96.4), because it is weight-only: activations remain full precision.
   **FP8 sits between float and INT8** (Square 25.0/54.8), despite having the
   best clean accuracy of the five (84/100) — clean accuracy is not a robustness
   proxy.
5. **Transfer is near-perfect between siblings**: FP32 PGD candidates transfer at
   100.0% on every arm at every budget with no adaptation; FP32 FGSM candidates
   transfer at 79.7–91.6% (1/255) up to 94.0–97.5% (8/255) own-denominator
   (93.6–97.4% on the shared 78). Per-arm adaptation gains int8/fp8 little.
6. **Random-noise control is flat**: 0.0–6.0% overall (fp8 worst at 6.0% at
   8/255; int8 5.1%; others ≤2.4%), so the attack numbers are far above chance.
7. **All acceptance gates passed**, so the numbers are reportable: hashes
   unchanged, bounds valid, query budgets respected, counts reconciled,
   replica gates 5/5, transfer candidates verified.

### 7.5 Replica / gradient diagnostics (probe PASSED; caveats)

From `results/_probe/replica_validation.json` / `REPORT.md` (probe run
2026-11-06T21:18, `--device cuda --ort-check`, 100 images, eps 8/255 directional
probe). All five arms passed the 90% gate; fp32/fp16/fp8/int4 passed the 97%
ideal. Engine clean counts matched SPEC §3 exactly (83/83/84/79/83). Every arm
has a finite, non-zero replica gradient on all 100 images, and the directional
probe is informative for all five (gradient-sign flips 90–92/100 vs random
12–15; negated-sign flips 20–22, margin moves up).

| arm | engine clean | replica clean | top-1 agree | mean abs dlogit | max abs dlogit | global r | dynamic ops | sign flips grad/rand/neg | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| fp32 | 83 | 83 | 100/100 | 0.004406 | 0.041681 | 1.00000 | 100 | 92/12/20 | PASS (ideal) |
| fp16 | 83 | 83 | 100/100 | 0.006361 | 0.070450 | 1.00000 | 102 | 90/12/20 | PASS (ideal) |
| fp8 | 84 | 84 | 99/100 | 0.202224 | 1.399876 | 0.99623 | 166 | 92/14/20 | PASS (ideal) |
| int8 | 79 | 81 | 95/100 | 0.249506 | 2.678207 | 0.99391 | 206 | 92/15/22 | PASS (not ideal) |
| int4 | 83 | 83 | 100/100 | 0.004334 | 0.037517 | 1.00000 | 100 | 92/12/22 | PASS (ideal) |

Replica composition (executed dynamic ops): fp32/int4 = 52 Conv + 35 Clip + 10
Add + 1 ReduceMean + 1 Reshape + 1 Gemm; fp16 adds 2 Cast (identity); fp8 adds
32 Q/DQ pairs; int8 adds 52 Q/DQ pairs. Folded tensors: fp32/fp16 110, fp8 498,
int8 698, int4 180.

INT4 folded classifier weight vs the torchvision FP32 checkpoint: mean abs error
**0.004592**, max **0.101982** → PASS (<0.01 check). fp16/fp8/int8 folded
weights differ from the FP32 checkpoint by mean 7.9e-06 (max 1.2e-04).

**Why int8 is 95% and fp8 is 99% — not a replica bug.** The probe's cross-arm
table shows each engine against the FP32 engine: fp8 94.0%, **int8 92.0%**, int4
99.0% top-1 agreement (mean |Δlogit| 0.32–0.38 for fp8/int8). The replica is
*closer to its own engine* than that engine is to the FP32 engine (int8 replica
vs own engine 95% > engine vs fp32 92%), and the directional probe is strongly
informative — so the gap is TensorRT-engine vs ONNX-Q/DQ-graph difference, not
replica error. Both still clear the 90% gate with no override.

**Saturated-gradient image.** One image (index 88) has a near-zero replica CE
gradient in every arm (L1 < 1e-6) because it is saturated-but-correct (replica
top-2 margin ~24–25 logit units). `sign()` still gives a valid FGSM/PGD
direction; this is a numerical property of a confident prediction, not a
zero-gradient defect. Zero-gradient images: 0/100 in every arm.

**Authoritative engine.** All numbers are on the **bs1** engine; bs1 vs bs8
engines are not bit-identical (max logit difference 0.0092 on the 100 samples),
so batch size never enters the metric (`summarize.py` asserts static batch 1).

Re-run the probe with `./evasion_trt/sync.sh run
evasion_trt/scripts/probe_replica.py --device cuda --ort-check` (outputs under
`results/_probe/`; exits non-zero if any arm is below gate).

### 7.6 Acceptance checklist (all passed for the full run)

These are enforced by `probe_replica.py` + `summarize.py` (plus
`run_evasion.py` at run time):

- [x] `probe_replica.py` gate: agreement ≥90% for all five arms (100/100/99/95/
      100) and ideal ≥97% for fp32/fp16/int4; finite non-zero gradient on all
      100 images; directional probe informative; INT4 dequant <0.01. **PASS.**
- [x] Every adversarial sample finite, in `[0,1]`, `L∞ ≤ eps`
      (`LINF_TOLERANCE_255 = 3e-4` re-check from `per_image.csv`). **PASS.**
- [x] Square queries ≤ 500 per image; per-condition query totals reconcile;
      ineligible images have `queries == 0`, `linf == 0`, adv == clean pred.
      **PASS.**
- [x] ONNX + engine SHA-256 unchanged at end (`onnx_hash_unchanged`,
      `engine_hash_unchanged`, `source_hash_unchanged`); runtime evidence
      present; `static_batch == 1`; `cpu_fallback == false`; native TensorRT.
      **PASS.**
- [x] Sample population identity + per-file SHA-256 re-loaded from the dataset;
      clean top-1 equals `EXPECTED_FULL_CLEAN`; shared set equals
      `EXPECTED_SHARED_ALL_FIVE = 78`. **PASS.**
- [x] `summarize.py` reconciles every row: 10,700 rows / 107 conditions, shared
      78, replica gates 5/5, bounds valid, query budgets valid, transfer
      candidates verified, PGD→stronger retention verified; `matrix.{json,md}`
      emitted; own and shared denominators both reported with 95% Wilson
      intervals; never mixed. **PASS.**
- [x] `summary.json` has `gradient.native_runtime_autograd: false`, the
      approximate-gradient `method` string, and the replica gate fields. **PASS.**
- [x] Report states the limitations (§9); no claim of certified robustness.

### 7.7 Figures

Generated from the reconciled artifacts (no re-inference):

| figure | content |
|---|---|
| [`figures/evasion_vs_eps.png`](figures/evasion_vs_eps.png) | PGD/FGSM success vs L∞ budget, per arm |
| [`figures/arm_comparison.png`](figures/arm_comparison.png) | per-arm bars for the key conditions |
| [`figures/square_query_budget.png`](figures/square_query_budget.png) | Square success vs query budget (plateau behaviour) |
| [`figures/transfer_vs_whitebox.png`](figures/transfer_vs_whitebox.png) | FP32→arm transfer vs per-arm white-box attack |
| [`figures/replica_validation.png`](figures/replica_validation.png) | replica-vs-engine agreement and direction-probe evidence |

---

## 8. Known pitfalls for this tree (carried from ResNet-50)

1. **Mismatched surrogate destroys the result.** Attack with the arm's own ONNX
   graph; never reuse another architecture's replica. Precedent: cosine 0.0068,
   13/79 vs 89/89 transfer.
2. **Do not swap the base model mid-matrix.** The base-model confound is ~16 pp
   at 1/255 — as large as the runtime effect.
3. **Replica op coverage**: MobileNetV2 needs Clip/min-as-inputs, ReduceMean with
   axes input, Reshape with shape input, Cast=identity. Unknown ops must raise.
4. **FP8 zero-point dtype is `float8_e4m3fn`** — never `np.iinfo` on it; round
   the quantized values through `torch.float8_e4m3fn` (clamp to
   `torch.finfo(...).max`), then STE.
5. **INT4 is weight-only, blocked** (block 128, axis 0): fold the dequant when
   building the replica; there is no int4 activation grid. `replica.py` infers
   the block axis from the recorded scale shape and raises if it is ambiguous.
6. **bs1 is authoritative**: bs1 vs bs8 differ (max 0.0092); never mix batches;
   `summarize.py` asserts static batch 1.
7. **TF32**: FP32 engine likely uses TF32 tactics; disable TF32 in torch for
   replicas and record the engine-vs-PyTorch logit gap (0.042 max / 0.0044 mean)
   as expected, not a defect.
8. **Preprocessing**: resize **256 bicubic antialias** for MobileNetV2 — do not
   reuse the ResNet-50 232 recipe.
9. **ORT INT8 is broken for Q/DQ graphs**: evaluate INT8 Q/DQ on native
   TensorRT (the five engines here are all TRT, so no ORT path is needed).
10. **Smoke is not a result**: `--smoke` uses 1 img/class and tiny budgets and
    records below-gate replica agreement without aborting; only the full run
    determines the gate.
11. **Finite attacks are not certificates**: 100 images, one seed, approximate
    BPDA; report Wilson intervals and limitations.

---

## 9. What this study establishes and does not

**Establishes** (one MobileNetV2 base, one 100-image pilot, one attack seed,
deployed engines): the empirical evasion rates of five deployed precisions under
the fixed protocol; that an adaptive PGD/BPDA attack evades all five at every
tested budget (1–8/255) with no survivors of the strong 1/255 check; that the
precision effect appears only for weak/finite attacks (FGSM at 1/255 and Square),
where FP8/INT8 resist more while INT4 behaves like FP32; that FP32→arm transfer
is near-perfect under PGD (100%) and high under FGSM; and that the score-only
attack reproduces the expected plateau behaviour for quantized output grids.

**Does not establish**: certified robustness or a radius; worst-case adversarial
risk; a causal separation of quantization vs calibration vs graph/runtime (the
artifacts were produced by different recipes by design, and TRT-vs-ONNX-Q/DQ
differences contribute to the fp8/int8 replica gap); physical-world robustness.
Adjacent arms with overlapping Wilson intervals are not ranked. The correct
summary sentence is: "empirical evasion under the attempted finite attacks;
approximate BPDA gradients; 100-image pilot, one seed; not a robustness
certificate."

---

## 10. Reusable output

The method is packaged as a user-level skill so later model experiments do not
re-derive it:

- `/home/ross/.dsh/skills/adversarial-evasion/SKILL.md` — when to use, the two
  attack families, protocol, gates, porting steps, pitfalls, commands.
- `/home/ross/.dsh/skills/adversarial-evasion/references/methodology.md` —
  detailed protocol, rationale, exact reference semantics.
- `/home/ross/.dsh/skills/adversarial-evasion/references/case-studies.md` —
  ResNet-50 findings (base-model confound, bad-surrogate incident, measured
  matrices) and the completed MobileNetV2 study with its measured attack matrix,
  replica validation, cross-model contrast and figures.
