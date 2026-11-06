# Base-model unification for the cross-runtime robustness study

**Question:** the crossing matrix mixes two base networks — TorchVision ResNet-50
(IMAGENET1K_V2 / V1 weights, torchvision architecture) on PyTorch, ONNX Runtime,
TensorRT and FBGEMM, and `timm` ResNetV2-50 (BiT pre-activation, `a1h_in1k`) on
Qualcomm QNN and Rockchip RKNN. Should everything be unified on the TIMM model?
If yes, how?

**Decision: yes — unify on `timm/resnetv2_50.a1h_in1k`, not on `timm/resnet50`.**
The base-model swap moves the headline evasion number by ~16 percentage points —
as large as the entire runtime/quantizer effect this study exists to measure — so
the confound is not cosmetic. `resnetv2_50` is the correct target because both NPU
deployments already use it and are byte-verified against one ONNX source, so
unification costs **zero NPU conversions** and puts the rework on the cheap,
scripted GPU/CPU side. The alternative (re-base the two NPUs onto TorchVision V2)
is a similar amount of engineering with much higher board/toolchain risk and
destroys the only fully verifiable provenance chain.

Everything below is measured from retained artifacts unless marked estimate.

**Implementation status:** complete. See [UNIFICATION_STATUS.md](UNIFICATION_STATUS.md)
for the built artifacts, verified provenance chain and run log, and
[RESNETV2_50_REPORT.md](RESNETV2_50_REPORT.md) for the standalone report on the
unified base (all nine deployments, six figures).

---

## 1. What actually differs today

Current matrix (clean top-1 on the shared 100-image Imagenette pilot, resize 232 /
crop 224; TV = TorchVision ResNet-50, timm = ResNetV2-50):

| Base | Deployments (runtime · format) | Clean /100 |
|---|---|---|
| **TorchVision** (V1 / V2 / V2-parent) | PyTorch fp32_v1, fp32_v2, fp32_parent · PyTorch CPU FBGEMM int8 · ORT CUDA fp32, fp16 · ORT CPU int8 QOperator · TensorRT int8 Q/DQ, int8 compressed Q/DQ, int8 BN-folded, int8 unfused-BN · QNN HTP V1 int8 | 83–93 |
| **timm ResNetV2-50** | QNN HTP int8 · RKNN FP16 · RKNN INT8 · PyTorch/ORT FP32 controls | 78–82 |

The two networks are not the same model under a different name:

| Check | Result | Evidence |
|---|---|---|
| Weight identity | **0 of 251** timm initializers byte-equal any of the 320 float tensors in `resnet50-11ad3fa6.pth`; best conv-kernel cosine 0.0395 | in-memory comparison of `onnx/resnetv2_50.onnx` vs the checkpoint |
| Graph structure | timm: 171 nodes, 49 `BatchNormalization` in pre-activation order (`BN→ReLU→Conv`); TV: 122 nodes, **0** BN (folded at export) | ONNX graph inspection |
| Gradient geometry | cosine **0.0068** between an `int8_v2` (TV) input gradient and the timm graph's own source | `evasion/results/qnn_v2_matched/gradient_probe.json`; `evasion/README.md:606-608` |
| Clean accuracy | timm 81–82 vs TV V2 85 on identical crops | `evasion/README.md:636-642` |

Provenance chains (relevant to the choice):

| Base | Chain | Verdict |
|---|---|---|
| timm | HF checkpoint `timm/resnetv2_50.a1h_in1k` `b809e8c0…` → ONNX `6416f2e9…` (4 byte-identical copies) → QNN context `928508ae…`, RKNN `e997bb14…`/`f71ab0e1…`; **251/251 initializers byte-equal** the checkpoint | strong, reproducible |
| TV V2 | checkpoint `11ad3fa6…` → `quantize/artifacts/fp32_v2.onnx` `bbe0d27d…` → ORT/TRT artifacts, all hash-pinned | strong |
| TV V1 (QNN) | external ModelScope binary `65f62829…`, **no ONNX/DLC/calibration/encodings**; lineage inferred only from 100/100 clean parity | weak, **irreproducible** |

Missing cells that unification must fill: **quantized timm on ORT/TensorRT/FBGEMM**;
There is also no TV deployment on RKNN. (TV V2 exists on TensorRT/ORT only; TV V1
exists on PyTorch and the external QNN binary only.)

---

## 2. Why it matters: the base-model effect is as large as the signal

### 2.1 A clean 2×2 factorial already exists in the retained results

Two bases × two fp32 runtimes (PyTorch CUDA, ORT CUDA), same 100 images, same
protocol (FGSM, PGD 20×2, Square 500 queries; eps in pixel/255), restricted to the
**75 images all four models classify correctly cleanly**. Cells are successful
evasions / 75. Sources: `evasion/results/pytorch/` (TV fp32_v2),
`evasion_onnx/results/fp32_v2/` (TV ORT), `evasion_rknn/results/run/pytorch_fp32/`
and `.../onnx_fp32/` (timm fp32 controls).

| Attack | TV · PyTorch | TV · ORT | timm · PyTorch | timm · ORT | **Base effect** | Runtime effect |
|---|---:|---:|---:|---:|---:|---:|
| FGSM 1/255 | 22 | 22 | 35 | 35 | **+13 (timm easier)** | 0 |
| PGD 20×2 1/255 | 71 | 71 | 59 | 59 | **−12 (timm harder)** | 0 |
| PGD 20×2 2/255 | 74 | 75 | 67 | 67 | −7 | ≤1 |
| PGD 20×2 4/255 | 75 | 74 | 72 | 73 | −2…−3 | ≤1 |
| PGD 20×2 8/255 | 75 | 75 | 75 | 75 | 0 (saturated) | 0 |
| Square 4/255 | 18 | 18 | 30 | 31 | **+12 (timm easier)** | ≤1 |
| Square 8/255 | 39 | 39 | 43 | 43 | +4 | 0 |

Reading: at fp32 the runtime axis is a **null control** — PyTorch and ORT execute
the same exported graph, and they change **at most one image of 75**. At fixed
runtime, swapping the base changes **12–13 images at 1/255**, in *opposite
directions* for FGSM vs PGD, and 12 images for Square at 4/255. The base-model
effect is not a constant offset that could be subtracted out; it interacts with
the attack. (The runtime axis is not generally null — see the INT8 row below —
but the point stands: a model swap at fixed everything else is worth 10+ images,
so a matrix that swaps both at once is unattributable.)

### 2.2 It is comparable to the effects the study reports

| Contrast | Delta at 1/255 | Same base? |
|---|---:|---|
| TV V2 checkpoint: ORT CPU int8 vs TensorRT int8 (80 shared images) | 63 vs 51–52 → **~14–15 pp** | yes (`INT8_COMPARISON.md:62-73`) |
| timm: QNN int8 vs RKNN int8 (66 shared images) | 36 vs 40 → **~6 pp** | yes |
| **Base swap at fixed fp32 runtime (75 images)** | 71 vs 59 → **~16 pp** | **no — this is the confound** |
| Same device (QNN HTP), TV V1 vs timm ResNetV2 int8 | 100% vs 62% evaded at 1/255 → **~38 pp** | no (also different QAIRT versions) |

So a cross-runtime ranking built on mixed bases is dominated by a factor larger
than the runtime effect it is meant to isolate. The existing reports already say
this in prose ("architecture, checkpoint, calibration … prevent a single causal
ranking", `evasion_rknn/INT8_COMPARISON.md:3-6`); unification removes the caveat
instead of documenting it.

### 2.3 Secondary observations that affect the plan

- **Pre-activation BN cannot be fully folded.** 17 of 49 BNs in the timm graph
  receive an `Add`/`MaxPool` tensor (`check3_fuse_ops.onnx` keeps exactly those 17).
  This is why timm INT8 loses ~2–3 clean images on both NPUs (QNN 79 vs 82 float,
  RKNN 78 vs 81 FP16). GPU INT8 arms will show the same structural property; it is
  a property of the deployed base, not a bug, but it must be a gated acceptance
  criterion rather than a surprise.
- **Preprocessing is TV-native, not timm-native.** The shared protocol is
  resize 232 / crop 224 / ImageNet mean·std. timm `a1h_in1k` expects bicubic
  resize ~236 and mean=std=0.5. A CPU ONNX probe on the same 100 crops (same crop
  code, only normalization/resize changed): protocol **83/100**, 232+0.5·0.5
  **87/100**, native 236 bicubic+0.5·0.5 **86/100**. So the current timm arms run
  ~3–4 points below their native recipe. This is constant across runtimes and does
  not bias the comparison, but it should be stated — and it is another reason not
  to churn the NPU artifacts (RKNN bakes mean/std; QNN input quantizer encodes the
  ImageNet domain).

---

## 3. Decision: which base

**Unify on `timm/resnetv2_50.a1h_in1k`** (checkpoint `b809e8c0…`, canonical ONNX
`6416f2e9…`, preprocessing unchanged).

Why not the alternatives:

| Candidate | Why not |
|---|---|
| **`timm/resnet50.*` (v1.5 arch)** | This is *not* the model the NPUs run. `timm/resnet50.a1_in1k` is the torchvision-style v1.5 block structure with rsb weights — it would need rebuilding **all GPU artifacts *and* both NPU artifacts** (nothing is reusable except scripts), to obtain an architecture you already have as TorchVision V2. Highest cost, no scientific gain, and it would invalidate the 251-initializer provenance chain. |
| **TorchVision V2** | Requires 3 NPU conversions (QNN context, RKNN FP16, RKNN INT8) + board redeploy + 3 re-attack suites + fresh matched surrogates; ~2–4 days and medium–high board risk (2.37 vs 2.42 context coupling, CDSP hangs, HTP is INT8-only). It also retires the only TV-family QNN deployment (V1 is an irreproducible external binary). Legitimate only if torchvision-native preprocessing/architecture is a hard requirement. |
| **Keep both, document** | Renounces the stated aim: no cross-runtime causal claim is defensible while bases differ, and §2.1 shows the confound is not small. Keep as fallback (§5). |

Why `resnetv2_50` is the right common base:

1. Both NPU deployments — the least flexible, most expensive-to-rebuild deployments
   — already use it, built and re-verified locally, with 251/251 initializer equality.
2. The timm fp32 PyTorch **and** ORT CUDA arms already exist and were already run
   through the full attack protocol (`evasion_rknn/results/run/{pytorch_fp32,onnx_fp32}`,
   81/100). Only fp16, ORT-CPU int8 and TensorRT int8 are missing on the GPU side.
3. The surrogate/BPDA layer already supports it: `GraphReplica` parses the timm
   graph with no unsupported ops; `matched_source.py` rebuilds the model from ONNX
   with no download; `run_npu_evasion.py` refuses an unmatched surrogate.
4. Offline reproducible end-to-end: HF checkpoint cached, ONNX hash-pinned,
   vendored QAIRT build recipe identical to the deployed context.
5. The base-model sensitivity experiment (§2.1) remains available as an appendix,
   using retained TV results — nothing is thrown away.

---

## 4. Cost of the three options

Measured compute anchors: full 7-model ONNX suite = **4,432 s ≈ 1.23 GPU-h**
(15,300 reconciled rows); QNN matched suite = 940 s; RKNN suite ≈ 2,200 s on board.
TRT 10.9 per-engine build time was never logged individually, but the four retained
engines landed within ~10 minutes total.

| Option | New artifacts | Compute | Engineering | Board risk |
|---|---|---|---|---|
| **A. Unify on timm `resnetv2_50` (recommended, scoped)** | timm fp16, ORT QOperator int8, TRT int8 Q/DQ ×2 (explicit + compressed); optionally port torch-PTQ recipes later | ~1.5–2.5 GPU-h + ~26 min CPU (ORT CPU arm) | **16–24 h (2–3 d)** | **none** (NPU artifacts untouched) |
| A+. Full timm rebuild incl. torch-PTQ ×2 + FBGEMM equivalent | 9–10 artifacts, 4 engines | 2.5–4 GPU-h | 32–48 h (4–6 d) | none |
| **B. Unify on TorchVision V2** | 3 NPU conversions (QNN context, RKNN FP16, RKNN INT8) + full board revalidation | 1.5–2 board-h + ~0 GPU | 2–4 d | medium–high |
| **C. No unification** | — | 0 | 1–2 d (reporting only) | none — but aim unmet |

Option A is recommended: it preserves every board artifact and puts the work into
scripted, reversible GPU/CPU steps. Option A+ is not needed for the stated aim —
the quantization-recipe axis (torch-PTQ BN-folded/unfused, FBGEMM) can stay as a
TorchVision-only appendix; only port it if a reviewer asks for recipe parity on
the unified base.

---

## 5. Plan

Assumptions: option A; keep the shared 232/224/ImageNet protocol; keep the
100-image pilot for continuity and add statistical power in Phase 5.

### Phase 0 — freeze and declare (0.5 d, no compute)

1. Add `sources/` manifest (or `quantize/sources.json`) pinning both bases:
   TV V2: `11ad3fa6…`; timm: `b809e8c0…` → `6416f2e9…`.
2. Snapshot the current cross-family comparison JSON/figures under a `tv_family/`
   naming (`results/comparisons/tv_family.json`, `figures/tv_family.png`) and mark
   them "secondary / cross-family anchor" in the READMEs — do not delete
   (precedent: the superseded QNN ResNetV2 run is retained).
3. Fix the stale provenance notes while here: `evasion/README.md:592` still says
   the graph spec sha is `PENDING` although `configs/resnetv2_50_graph_spec.json:3`
   carries `928508ae…`; record that the QNN V1 binary is external/irreproducible.
4. Record the decision and the §2 numbers in this file; link from all three READMEs.

### Phase 1 — one source registry + reproducible timm export (1–1.5 d, CPU only)

1. `quantize/sources.py`: a `SourceSpec` (id, model builder, checkpoint path+sha256,
   ONNX path+sha256, resize/crop/mean/std) with two entries — `tv_v2` and
   `rnv2_50`. Add the timm checkpoint hash to `quantize/ckpt.py` `KNOWN_SHA256`
   and keep the hash gate.
2. `quantize/export_onnx.py --source rnv2_50`: `timm.create_model('resnetv2_50')`
   + `safetensors` load → `torch.onnx.export` (opset 17, dynamic batch, input name
   `input`) → assert graph equality with the canonical `6416f2e9…` ONNX
   (251 initializers equal; logits max|diff| ≤ 1e-4 on 10 crops). This closes the
   current "vendored ONNX is not regenerable in-repo" gap.
3. Canonicalize paths: treat `onnx/resnetv2_50.onnx` (sha `6416f2e9…`) as the single
   source for NPU and GPU arms; never re-export over it, add `onnx/timm/rnv2_50.onnx`
   if a fresh export is wanted for comparison.
4. Extend `evasion_onnx/scripts/check_provenance.py` to walk the timm chain
   (checkpoint → ONNX → each artifact meta → engine).

### Phase 2 — build the missing timm artifacts (1–1.5 d + a GPU window)

CPU/builder work, all arch-agnostic ONNX transforms (no model surgery):

| Artifact | Recipe | Entry point | Cost |
|---|---|---|---|
| `timm_fp16.onnx` | ModelOpt `convert_to_f16` on timm fp32 | `quantize/export_onnx.py` / small cast script | seconds |
| `timm_int8_ort_qop.onnx` | ORT `quantize_static` QOperator (per-channel — fix the current per-tensor default noted in `ort_qoperator.py`), 512-image `calib_512.npy` | `quantize/ort_qoperator.py --source` | ~30 s |
| `timm_int8_ptq.trt.onnx` | ModelOpt ONNX PTQ explicit Q/DQ (fp32 high precision) | `quantize/run_onnx_ptq.py --source` | ~45 s |
| `timm_int8_dq.trt.onnx` | same + `--dq-only` weight compression | `quantize/run_onnx_ptq.py --source --dq-only` | ~50 s |
| TRT engines | build with the isolated TensorRT **10.9** `dlopen` path (`runtimes.py:79-90`); system 11.1 fails | `evasion_onnx/scripts/run_gpu_variants.py --inspect-only` first | tens of s each |
| *(optional)* torch-PTQ BN-folded / unfolded | port `ptq_torch.py` to timm (pre-activation: fold only the 32 foldable BN, keep the 17) | new `--source rnv2_50` branch | 1–2 d |
| *(optional)* FBGEMM-equivalent int8 | torch.ao eager on a custom module; **no torchvision prebuilt checkpoint exists** | new script | 1–2 d |

Gates before any attack (extend `run_model.py` preflight):

- clean top-1 on the 100-image pilot: fp32 81–82, fp16 81 (must match fp32 top-1
  exactly, as RKNN FP16 does), int8 **≥ 77** (i.e. ≤ 4-image loss vs fp32, matching
  the NPU INT8 behaviour 79/82 and 78/81);
- replica clean agreement ≥ 90 % (existing hard abort);
- artifact hashes + calibration hash + source hash in each `*.meta.json`;
- note the 17 unfoldable BNs and the resulting fp32 segments in the graph report.

**Stop gate:** if ORT/TensorRT INT8 clean drops more than ~5/100, do not attack the
artifact as-is — adjust calibration (entropy vs max, 99.99 percentile), or accept
and document; do not tune the attack protocol to compensate.

### Phase 3 — re-run attacks (0.5 d compute)

Only the new arms need runs; reuse the existing timm fp32 PyTorch/ORT arms
(protocol-identical, same seed/images — re-verify hashes in the reconciler):

```bash
# new arms, same protocol: eps 1 2 4 8, FGSM, PGD 20x2, stronger 100x5 at 1/255,
# Square 500q at 4,8, transfer from fp32
uv run --no-project --python ./venv/bin/python Models/resnet50/evasion_onnx/scripts/run_gpu_variants.py \
  --source rnv2_50 --models timm_fp16 timm_int8_ort_qop timm_int8_ptq timm_int8_dq
```

Expected: ~1.5 GPU-h (fp16/TRT ~10 min each, CPU ORT QOperator ~25 min).
Every model must go through the existing surrogate preflight; no cross-base
surrogate may be used (the guard in `run_npu_evasion.py` already enforces the
analogous rule for the board).

### Phase 4 — one reconciled matrix, one narrative (2–3 d)

1. New `evasion_onnx/scripts/compare_timm.py` (or a `--source` branch of
   `summarize.py`) emitting one JSON + one figure with:
   - the **timm-only matrix**: PyTorch fp32, ORT CUDA fp32/fp16, ORT CPU int8,
     TensorRT int8 ×2, QNN HTP int8, RKNN FP16/INT8 (8–9 deployments);
   - shared clean-correct intersection (report n; expect ~70–75) **and**
     per-deployment own denominators;
   - paired quantized-vs-its-own-fp32 rows (the only controlled quantization claim);
   - bootstrap/Wilson intervals over images — at n≈75 a 4-image gap is ~5 pp and
     overlaps; do not rank on point estimates alone.
2. Move the TV results to `tv_family_appendix` tables/figures; state they are a
   cross-architecture replication, not part of the primary ranking.
3. Re-point all three READMEs: primary matrix = timm; add a "one base model"
   methodology section with the provenance table, the preprocessing deviation
   (§2.3), and the 17-unfoldable-BN note.
4. Keep the §2.1 2×2 table as a first-class result: it is the evidence that the old
   cross-family matrix could not have isolated runtime effects.

### Phase 5 — provenance, power, and optional fidelity (1 d + optional compute)

1. `check_provenance.py` covers timm artifacts; add `scripts/verify_sources.py`
   that re-derives every hash in the manifest and fails closed.
2. Statistical power: either add a second attack seed or expand to 300 images
   (attack cost scales linearly; ~3.6 GPU-h for the GPU arms, ~1 h CPU, board
   suites are the expensive part — prefer extra seeds on the GPU arms and keep the
   board at 100).
3. Optional, GPU-only sensitivity row: run ORT fp32 with **native** timm
   preprocessing (236 bicubic / 0.5·0.5) and compare the attack counts with the
   protocol-preprocessing arm. If ordering is unchanged, the preprocessing
   deviation is empirically benign and can be documented as such. (Do not rebuild
   the NPUs to change preprocessing.)
4. Lock artifact hashes before attacks; if a QNN recalibration is ever done
   (19-image calibration is thin and per-tensor), treat it as a new artifact with
   its own matched control and re-attack.

### Acceptance criteria for "unified"

- Every primary row derives, by hash, from `b809e8c0…` (checkpoint) / `6416f2e9…`
  (ONNX); the manifest re-derives and passes.
- No primary table mixes bases; TV rows appear only in the appendix.
- All 100-image attack rows reconciled (conditions × 100), bounds/query caps
  verified, replicate agreement ≥ 90 %, runtime-provider evidence retained.
- INT8 clean ≥ 77/100 and ≥ 3/4 of the fp32 clean set; FP16 top-1 identical to fp32.
- Every claimed runtime/format/quantization difference is reported with an
  interval and a paired denominator; the 2×2 base-model sensitivity table is cited
  as the reason for the single-base design.

---

## 6. What not to do

- **Don't** unify on `timm/resnet50.*` (v1.5) — that is a different model from
  `timm/resnetv2_50`; it rebuilds everything and reuses nothing.
- **Don't** rebuild the NPU artifacts on TorchVision unless torchvision-native
  preprocessing is a hard requirement (option B).
- **Don't** keep the TV rows in the same ranking as the timm rows "for context" —
  that is the confound, and it is ~16 pp at 1/255.
- **Don't** reuse a TV fp32 model as the reference/denominator or surrogate for a
  timm target; the 0.0068-cosine incident is the precedent.
- **Don't** change resize/crop/normalization mid-unification: the 100-image crops
  are hashed, the NPU artifacts encode the current domain, and all board numbers
  are tied to them.
- **Don't** attack an artifact whose hash/output quantizer was not locked: the
  QNN Square counts depend on the output grid (0.0828 here vs 0.1678 for V1), so
  old tensors cannot be re-scored onto a rebuilt artifact.
- **Don't** present the current INT8 table as a runtime ranking; it is a
  deployment survey across two models.

---

## 7. Decisions needed from the project owner

1. **Scope** — is the deliverable a single-base deployment matrix (option A,
   recommended) or a set of within-family controlled comparisons (option C)? The
   plan above assumes A.
2. **Recipe parity** — is the torch-PTQ/FBGEMM recipe axis needed on the unified
   base (option A+) or can it stay TorchVision-only (appendix)? Recommendation:
   appendix.
3. **Power** — is 100 images / one seed sufficient for the final claims, or should
   the unified matrix expand to 300 images / multiple seeds (≈ +2–4 GPU-h)?
4. **Preprocessing** — accept the shared 232/ImageNet protocol with the timm base
   running ~3–4 points off its native recipe (recommended, no NPU rebuild), or
   require timm-native preprocessing (implies rebuilding both NPU artifacts)?
5. **QNN V1** — retain as an appendix cross-family anchor (recommended) or drop
   from the repository entirely?

---

## Appendix A — verified provenance hashes

| Object | sha256 |
|---|---|
| TV V1 checkpoint | `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097` |
| TV V2 checkpoint | `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| TV V2 historical parent | `f46c3f97e1aaad63bcc2032f0a6e7065be4611298bdda9d87541cc0805734f7b` |
| TV FBGEMM int8 | `23753f790ed53f950d72eac5aee7046ebdfbb6db96c141e3d6ff2f65c529f7ca` |
| TV V2 fp32 ONNX (`fp32_v2.onnx`) | `bbe0d27db128eda60ab63db83f2e2cd5557620bee8f6123e9a6912971d3a66bc` |
| **timm checkpoint** `resnetv2_50.a1h_in1k` | `b809e8c09f4a1cf2168adf089c86c17fe88521b85455db662c0b2d42f489a61b` |
| **timm ONNX** (canonical) | `6416f2e9d2d9c13f3baa60a496d0934e870e19baf8cd2c416979df42c75c13f4` |
| QNN ResNetV2 context | `928508ae776e277d81451176d71eea80ec24679f1cf9b038f08758979f47cc9a` |
| QNN ResNetV2 INT8 DLC | `548940e753b83ec5c83284619fc3003c3f68726bec53a4e88653d6ac18d956ef` |
| QNN V1 context (external) | `65f6282958ab7df304261ae2fe5532cdae668ff4940d4dd7cef8e24dfa887cfd` |
| RKNN FP16 | `e997bb14f08434d1d193f65e44a77d4d67241fcff11e1d8429b7e6f0ff08ef1f` |
| RKNN INT8 | `f71ab0e12fcba8d2d6f5eccfaaa11280e165965ca2c31332a483093de759bf6a` |

## Appendix B — current matrix cells (runtime × base)

| Runtime | TorchVision | timm ResNetV2-50 |
|---|---|---|
| PyTorch CUDA | fp32 V1 / V2 / parent; int8 FBGEMM (CPU) | fp32 (controls) |
| ONNX Runtime | CUDA fp32, fp16; CPU int8 QOperator | CUDA fp32 (control) |
| TensorRT 10.9 | int8 ×4 recipes | **absent** |
| QNN HTP (QCS6490) | V1 int8 (external) | int8 |
| RKNN (RK3588) | **absent** | FP16, INT8 |

## Appendix C — evidence pointers

| Claim | File |
|---|---|
| Base confound acknowledged; QNN V2 surrogate error and matched rerun | `evasion/README.md:15-57`, `:521-744` |
| TV weight-version confound (fp32_parent vs int8_v2) | `evasion/README.md:80-88`, `evasion/results/pytorch/historical_weight_alignment.json` |
| RKNN timm provenance / 251 initializers | `evasion_rknn/README.md:16-27`, `evasion_rknn/results/review/review.json` |
| Same-checkpoint TV subset (ORT + TRT) | `evasion_rknn/INT8_COMPARISON.md:60-73` |
| Nine-deployment INT8 table with source families | `evasion_rknn/INT8_COMPARISON.md:12-58` |
| per-image CSVs used for the §2.1 2×2 | `evasion/results/pytorch/per_image.csv`, `evasion_onnx/results/fp32_v2/per_image.csv`, `evasion_rknn/results/run/{pytorch_fp32,onnx_fp32}/per_image.csv` |
| QNN V1 external package, no encodings | `evasion/results/validation/qnn_deployment_package.json` |
| QNN V2 build recipe (QAIRT example) | `/home/ross/qairt/2.42.0.251225/examples/Models/resnet50_qairt_example/QUANTIZATION.md` |
| RKNN conversion scripts and int8 build | `rknn/NPU/resnet/export_resnet_to_rknn.py`, `make_quant_dataset.py`, `rknn/NPU/resnet/resnetv2_50_int8.rknn` |
| TRT 10.9 isolation / 11.1 failure | `evasion_onnx/README.md:56-76`, `evasion_onnx/logs/trt_build_*.log` |
| Calibration set (512, zero overlap) | `quantize/calib/calib_512.npy`, `evasion_onnx/results/validation/provenance_check.json` |
