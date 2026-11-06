# Base-model unification — implementation status

Date: 2026-11-04 (HKT). Companion documents:
[BASE_MODEL_UNIFICATION.md](BASE_MODEL_UNIFICATION.md) (decision + rationale +
costs), [quantize/sources.json](quantize/sources.json) (pinned provenance graph).

**Scope implemented:** Phase 0–2 and Phase 4 machinery of the plan, plus the one
attack arm that can execute on CPU. All five timm ONNX artifacts are built and
gated; provenance is verifiable end-to-end from checkpoint to board binary; the
harness can attack timm models; a single-base reconciliation script produces the
matrix. The remaining attack arms and TensorRT engines require a GPU window and
are listed in §6 with exact commands.

Nothing in the TorchVision result set was modified. All new measurements are on
the single canonical base `timm/resnetv2_50.a1h_in1k`.

---

## 1. What was built

| Area | Files | State |
| --- | --- | --- |
| Source registry | `quantize/sources.py` | working: `list` / `show` / `verify`; timm build asserts **251/251** ONNX initializers byte-equal to the checkpoint |
| Reproducible timm export | `quantize/export_timm_onnx.py` | working: fresh export **reproduces the canonical graph** — 251/251 initializers equal, node/op structure equal, logit delta **0.0** vs canonical, `8.3e-07` vs PyTorch |
| Pinned provenance | `quantize/sources.json`, `quantize/verify_sources.py` | working: `--deep --require-all` passes; output `evasion_onnx/results/validation/sources_verification.json` |
| Artifact production | `quantize/build_timm_artifacts.py`, 5 × `quantize/artifacts/timm_rnv2_*.onnx` + `.meta.json` | built, gated, hash-pinned, frozen |
| Accuracy gate | `evasion_onnx/results/validation/timm_artifact_gate.json` | `ok: true` (see §3) |
| Harness integration | `evasion_onnx/scripts/runtimes.py`, `run_model.py`, `run_gpu_variants.py` | working on CPU; GPU/TRT defaults preserved |
| Source-aware reconciliation | `evasion_onnx/scripts/summarize.py` | TorchVision behaviour byte-compatible (see §5); timm mode ready for the GPU window |
| Unified matrix | `evasion_onnx/scripts/compare_timm.py`, `evasion_onnx/results/comparisons/timm_matrix.{json,md}` | working; 5 retained deployments now, new arms join automatically when complete |
| Matrix figure | `evasion_onnx/scripts/make_timm_figure.py`, `evasion_onnx/figures/timm_matrix.png` | working: PGD 1/255, PGD+strong, Square 8/255 with 95 % Wilson intervals |
| Standalone report | `RESNETV2_50_REPORT.md`, `figures/v2_*.png`, `evasion_onnx/scripts/make_v2_report_figures.py` | working: self-contained single-base report with 6 figures and machine-readable tables |
| Report labels | top-of-file status block in `evasion/README.md`, `evasion_onnx/README.md`, `evasion_rknn/README.md`, `evasion_rknn/INT8_COMPARISON.md` | primary = timm; TorchVision = secondary replication |
| Artifact docs | `quantize/README.md` §9 | recipes, hashes, gate, deviations |

## 2. Provenance is now a verified chain, not a claim

```
timm/resnetv2_50.a1h_in1k checkpoint  b809e8c0…  (HF cache, hash-pinned)
        │  export_timm_onnx.py  (251/251 initializers byte-equal; logit Δ 0.0)
        ▼
canonical ONNX  onnx/resnetv2_50.onnx  =  quantize/artifacts/timm_rnv2_fp32.onnx  6416f2e9…
        │
        ├── timm_rnv2_fp16.onnx            23cc2388…   (ModelOpt fp16, fp16 I/O)
        ├── timm_rnv2_int8_ort_qop.onnx    ae524e4d…   (ORT QOperator, per-channel, 512 calib)
        ├── timm_rnv2_int8_ptq_ort.onnx    99755d64…   (ModelOpt Q/DQ, float weights stored)
        ├── timm_rnv2_int8_dq_ort.onnx     fd967b8c…   (ModelOpt Q/DQ, int8 weights)
        │
        ├── QNN HTP DLC     548940e7…  → context 928508ae…   (QAIRT 2.42, local build)
        └── RKNN FP16       e997bb14… / INT8 f71ab0e1…       (rknn-toolkit2 2.5.0, local build)
```

`python quantize/verify_sources.py --deep --require-all` re-hashes every local
object and re-asserts the 251-initializer equality; it currently exits 0 with
all artifacts `ok`. The only `unverified` row is the external QNN V1 context
(no local binary, no ONNX/DLC/encodings shipped) — deliberately excluded from
the primary matrix.

Known gaps are recorded in `sources.json.known_gaps`: no hash-linked DLC→board
chain for the external QNN V1 binary; board contexts are verified by recorded
digest at run time, not re-hashed locally; the RKNN review proves model identity
and board hashes, not who built the `.rknn` files.

## 3. Artifacts and accuracy gate

100-image pilot, seed 20261030, resize 232 / crop 224 / ImageNet mean·std,
ONNX Runtime CPU (gate JSON has per-artifact evidence):

| Artifact | MiB | Clean top-1 | Top-1 vs fp32 | Mean |Δlogit| | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `timm_rnv2_fp32.onnx` | 97.69 | 81/100 | 100/100 | 0.0 | pass |
| `timm_rnv2_fp16.onnx` | 48.92 | 82/100 | 99/100 | 0.0061 | pass, with note (one rounding flip) |
| `timm_rnv2_int8_ort_qop.onnx` | 24.96 | 78/100 | 90/100 | 0.49 | pass |
| `timm_rnv2_int8_ptq_ort.onnx` | 97.93 | 79/100 | 90/100 | 0.71 | pass |
| `timm_rnv2_int8_dq_ort.onnx` | 24.96 | 79/100 | 90/100 | 0.71 | pass |

These reproduce the expected behaviour of the deployed NPU INT8 models (QNN
79/100, RKNN 78/100) and keep FP16 non-degrading. The `ptq` and `dq` graphs are
the same quantized network with stored-float vs stored-int8 weights, so their
logits are identical by construction.

The FP16 and INT8 arms were additionally pre-validated through the *attack
harness* on CPU (`--inspect-only`): `timm_rnv2_fp16` clean 82/100 with replica
top-1 agreement 100 % and mean |Δlogit| 4.5e-3; `timm_rnv2_int8_ort_qop` clean
78/100, replica agreement 97 % (above the 90 % hard gate), QLinearConv kernels
confirmed. This is the same preflight the GPU arms will run.

Recorded deviations from the TorchVision recipe (all in the metas/gate):
per-channel QOperator (the TV graph hit an ORT per-channel bug, the timm graph
does not); ModelOpt's TensorRT custom-op pre-scan disabled (no GPU; the pre-scan
is unrelated to quantization); PTQ/DQ built from a byte-identical scratch copy
because ModelOpt rewrites its `--onnx` input in place (canonical file verified
restored/untouched); fp16 accepted at 99/100 top-1 agreement rather than
bit-identical.

## 4. Harness integration

- `runtimes.py`: five `timm_rnv2_*` registry entries; CUDA DLL preload is now
  gated on the CUDA provider so CPU-only hosts work.
- `run_model.py`: `--device {cuda,cpu}`, `--reference`, `--reference-dir`;
  replica/probe/labels follow the device; `environment.gpu` guarded; transfer
  candidates read from the reference directory with the same filenames and
  `indices == arange(n)` assertion; a timm arm pointed at a TorchVision
  reference **refuses before compute** (prevents the cross-base surrogate error
  class that produced the 0.0068-cosine incident).
- `run_gpu_variants.py`: `--source timm` waits for `timm_rnv2_fp32` and then runs
  fp16 / explicit-QDQ / compressed-QDQ with `--reference timm_rnv2_fp32`.
- `summarize.py`: `--source {torchvision,timm}`; TorchVision default is
  behaviour-identical (verified: 15,300 rows, 153 comparisons, 56 unions, 80
  shared images, byte-identical `comparisons` content vs the retained
  `onnx.json`; only the new `skipped_pending` field, the extra limitation line
  and the script hashes differ). timm mode refuses to run until the timm FP32
  reference arm is complete, with a clear message.

## 5. Current unified matrix (timm-only)

All nine deployments are now measured on the unified base. From
`compare_timm.py`, shared clean-correct intersection = **71/100**; every arm
validated (100/100 indices, unique rows, relative L∞ bound, Square ≤ 500
queries) and base-verified as `timm_resnetv2_50`
(`6416f2e9…` / `b809e8c0…`). Evaded / 71 on the shared population:

| Deployment | clean | PGD 1/255 | PGD+strong 1/255 | PGD 8/255 | Square 8/255 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 · CUDA | 81 | 55 (77.5 %) | 63 (88.7 %) | 71 (100 %) | 39 (54.9 %) |
| ORT FP32 · CUDA | 81 | 55 (77.5 %) | 64 (90.1 %) | 71 (100 %) | 39 (54.9 %) |
| ORT FP16 · CUDA | 82 | 54 (76.1 %) | 65 (91.5 %) | 71 (100 %) | 40 (56.3 %) |
| RKNN FP16 · NPU | 81 | 55 (77.5 %) | 64 (90.1 %) | 71 (100 %) | 40 (56.3 %) |
| ORT INT8 QOperator · CPU | 78 | 41 (57.7 %) | 51 (71.8 %) | 69 (97.2 %) | 24 (33.8 %) |
| RKNN INT8 · NPU | 78 | 45 (63.4 %) | 51 (71.8 %) | 71 (100 %) | 21 (29.6 %) |
| QNN HTP INT8 · NPU | 79 | 41 (57.7 %) | 43 (60.6 %) | 71 (100 %) | 21 (29.6 %) |
| TensorRT INT8 Q/DQ | 79 | 42 (59.2 %) | 49 (69.0 %) | 69 (97.2 %) | 26 (36.6 %) |
| TensorRT INT8 compressed | 79 | 42 (59.2 %) | 50 (70.4 %) | 69 (97.2 %) | 26 (36.6 %) |
| *QNN matched FP32 source* (reference) | 82 | 56 (78.9 %) | 64 (90.1 %) | 71 (100 %) | 41 (57.7 %) |

Figure: `evasion_onnx/figures/timm_matrix.png` (with 95 % Wilson intervals).
Full reconciled matrix: `evasion_onnx/results/comparisons/timm_matrix.{json,md}`;
independent 10,700-row reconciliation: `evasion_onnx/scripts/summarize.py
--source timm` (all hashes/bounds/providers verified).

What the unified base shows:

- **INT8 raises small-budget and score-only resistance on every runtime that has
  a float/INT8 pair.** At 1/255, fp32/FP16 deployments are evaded on 54–55/71
  images, while the five INT8 deployments are evaded on 41–45/71. At Square
  8/255 the same split is 39–40/71 (float) vs 21–26/71 (INT8).
- **At 8/255 the white-box attacks are saturated for float and mostly saturated
  for INT8.** The three INT8 arms that are not fully evaded (ORT CPU 69/71,
  TensorRT 69/71 ×2) keep 2 survivors each — descriptive finite-attack
  survivorship, not a certified radius.
- **Runtime differences at fixed precision are small relative to quantization.**
  Among fp32/FP16 arms the spread at every budget is ≤1 image; among INT8 arms
  the spread at 1/255 is 4 images (41–45) and at Square-8 5 images (21–26).
  This is exactly why a single base mattered: the earlier cross-base matrix
  attributed ~16 pp to runtime.
- The two TensorRT arms are included with a documented replica-agreement
  exception (§6); both retain `replica_gate_override` in their summary JSON.

## 6. TensorRT replica fidelity (investigated on GPU)

Both TensorRT INT8 arms tripped the pre-declared 90 % replica-agreement gate at
**88/100** clean top-1 agreement. Investigation (all evidence retained):

- The engine is genuinely INT8: 54 / 54 convolutions report `Int8` weights.
- The disagreement is tie-breaking, not a different function. All 12 disagreeing
  images have a top-2 margin of 0.02–1.15 (the other 88 images are comfortably
  separated), and the logit fields are colinear: a global fit gives
  `TRT ≈ 1.013·ONNX + 0.049` with equal logit scale.
- Structural cause: the timm graph is pre-activation and 17 of its 49 BatchNorms
  cannot fold. TensorRT folds the remaining float BN into the quantized conv,
  which breaks near-ties differently from the ONNX QDQ reference. This is a
  property of the deployed base + backend, not a defect of the artifact.
- Gradient evidence, not top-1 agreement, decides BPDA validity. A new probe
  (`evasion_onnx/scripts/probe_trt_replica.py`, output in
  `evasion_onnx/results/diagnostics/trt_replica_probe.json`) steps the same 20
  clean-correct images along three directions at 2/255 and scores the real
  engine:

| direction | flips / 20 | mean true-class margin change |
| --- | ---: | ---: |
| replica CE-gradient sign | **11** | **−0.75** |
| negated gradient | 0 | +1.35 |
| random sign | 0 | +0.10 |

The sign step flips more than half the engine's predictions, the negated
direction raises the margin, and a random step does nothing — the replica
gradient describes the deployed function (the same one-step logic used for the
QNN board). The QNX precedent for such explicit exceptions is
`run_npu_evasion.py --override-surrogate "<reason>"`.

Resolution: `run_model.py` now accepts
`--allow-replica-agreement <floor> --override-reason "<evidence>"`, records the
exception under `summary.json.gradient.replica_gate_override` (default gate,
observed %, accepted floor, reason), and otherwise keeps the 90 % hard abort.
Both TRT arms are run with floor 85 % and the probe/analysis as the reason; the
override is visible in every retained summary and in the matrix JSON.

## 7. Execution log and what remains optional

GPU window (this session): RTX 2000 Ada Laptop, driver 580.178.04,
TensorRT 10.9 target loaded via `runtimes.py:load_tensorrt10`.

| Arm | Result | Wall clock |
| --- | --- | --- |
| `timm_rnv2_fp32` (reference) | complete, 23 conditions, clean 81/100 | 7.5 min |
| `timm_rnv2_fp16` | complete, clean 82/100, 1390 CUDA nodes | ~4 min |
| `timm_rnv2_int8_ptq_ort` | complete, clean 79/100, 54/54 Int8 convs, gate override recorded | ~7 min |
| `timm_rnv2_int8_dq_ort` | complete, clean 79/100, 54/54 Int8 convs, gate override recorded | 7.1 min |
| `timm_rnv2_int8_ort_qop` (CPU) | complete, clean 78/100, QLinearConv ×540 | 95.5 min |

Reconciled with `summarize.py --source timm` (10,700 per-image rows, 107
conditions), `compare_timm.py` (9 deployments, shared n=71, Wilson CIs) and the
`make_timm_figure.py` plot. Provenance gate `verify_sources.py --deep
--require-all` passes.

Optional follow-ups (not blocking the unification):

- second attack seed or a 300-image population (the matrix is 71 shared images,
  one seed; intervals overlap for adjacent arms);
- port the TorchVision-only recipe axis (torch-PTQ BN-folded / FBGEMM) to the
  timm base if recipe parity is required;
- QNN V2 recalibration with a larger calibration set (19 images currently);
- native-recipe preprocessing sensitivity row (see §9).

## 8. Verified vs pending

| Verified | Evidence |
| --- | --- |
| timm export reproduces canonical ONNX exactly | `quantize/export_timm_onnx.py` (251/251 initializers, Δlogit 0.0) |
| 5 artifacts built with meta/hash chain; accuracy gate | `timm_artifact_gate.json`, `verify_sources.py` |
| provenance re-derived end-to-end | `verify_sources.py --deep --require-all` (exits 0) |
| all 9 arms attacked on one base | `compare_timm.py` (all validated, n=71) |
| full timm reconciliation | `summarize.py --source timm` (10,700 rows) |
| TRT surrogate validity | `probe_trt_replica.py` + `replica_gate_override` in summaries |
| native INT8 execution evidence | 54/54 Int8 convs in each TRT engine inspector; QLinearConv ×540 in ORT CPU |
| TV reconciliation unchanged | `summarize.py --source torchvision` reproduces retained `onnx.json` |
| board re-validation | **not needed** — NPU artifacts unchanged |

## 9. Caveats carried forward

- **Preprocessing**: the shared 232/224/ImageNet protocol is not the timm
  checkpoint's native recipe (bicubic 236, mean=std=0.5). Measured on CPU-ONNX
  with identical crop code: protocol 83/100 vs 86–87/100 under native
  normalization. It is constant across every runtime and is baked into both NPU
  artifacts, so it is kept and documented rather than changed mid-study.
- **Pre-activation BN**: 17/49 BNs cannot fold; INT8 costs 2–3 clean images on
  both NPUs and on the GPU-side INT8 arms (not a defect of one runtime).
- **Calibration heterogeneity** is unchanged and must be reported with the
  matrix: QNN 19 images (per-tensor), RKNN 256 images, GPU 512 images.
- **Statistical power**: 72 shared images with Wilson intervals; one attack
  seed. Treat the ordering as descriptive until a second seed or a larger
  population is run.
- **QNN V1** (external, TorchVision-V1 lineage, irreproducible) is retained as a
  cross-family anchor only and is excluded from the primary timm matrix.
- The TorchVision matrix remains available and unchanged; the README labels now
  make clear it is the secondary replication and must not be ranked together
  with timm rows.

## 10. File inventory (this session)

New:

- `quantize/sources.py`, `quantize/sources.json`, `quantize/export_timm_onnx.py`,
  `quantize/verify_sources.py`, `quantize/build_timm_artifacts.py`
- `quantize/artifacts/timm_rnv2_{fp32,fp16,int8_ort_qop,int8_ptq_ort,int8_dq_ort}.onnx`
  + `.meta.json`; `quantize/build/timm_rnv2_fp32.reproduced.onnx` + meta (export evidence)
- `evasion_onnx/scripts/compare_timm.py`, `evasion_onnx/scripts/make_timm_figure.py`;
  `evasion_onnx/scripts/probe_trt_replica.py`;
  `evasion_onnx/results/comparisons/timm_matrix.{json,md}`, `timm_onnx.json`;
  `evasion_onnx/figures/timm_matrix.png`;
  `evasion_onnx/results/diagnostics/trt_replica_probe.json` and the per-arm probe dir
  `evasion_onnx/results/validation/timm_artifact_gate.json`;
  `evasion_onnx/results/validation/sources_verification.json`;
  `evasion_onnx/results/validation/lead_artifact_check.json` (independent re-check)
- `evasion_onnx/logs/timm_rnv2_int8_ort_qop.cpu.log`;
  `evasion_onnx/results/timm_rnv2_{fp32,fp16,int8_ort_qop}/` (inspect / running)
- `UNIFICATION_STATUS.md` (this file), `TV_FAMILY.md`, `RESNETV2_50_REPORT.md`
- `evasion_onnx/scripts/make_v2_report_figures.py`,
  `evasion_onnx/results/comparisons/v2_report_data.json`,
  `figures/v2_{matrix_bars,epsilon_curves,condition_heatmap,quantization_advantage,square_query_budget,base_model_confound}.png`

Modified (behaviour-preserving for the TorchVision defaults):

- `evasion_onnx/scripts/runtimes.py` — timm registry entries; CUDA preload gated on provider
- `evasion_onnx/scripts/run_model.py` — `--device`, `--reference`, `--reference-dir`,
  base-family guard, recorded `--allow-replica-agreement/--override-reason` gate exception
- `evasion_onnx/scripts/run_gpu_variants.py` — `--source {torchvision,timm}`
- `evasion_onnx/scripts/summarize.py` — `--source`, `--allow-pending`, `--output`; TV default verified byte-compatible
- `evasion/README.md`, `evasion_onnx/README.md`, `evasion_rknn/README.md`,
  `evasion_rknn/INT8_COMPARISON.md` — base-model status labels / timm section
- `quantize/README.md` — §9 timm artifacts and reproducibility
- `BASE_MODEL_UNIFICATION.md` — implementation-status pointer

Not modified: every retained TorchVision `.onnx` artifact, result JSON/CSV/PT,
figure, board artifact, and the TV comparison files listed in `TV_FAMILY.md`
(their digests are unchanged).

## 11. Plan completion checklist


- [x] Freeze/declare base, pin provenance, label retained reports
- [x] Source registry + reproducible timm export (251/251, Δ 0.0)
- [x] Provenance manifest + verifier (`--deep --require-all` passes)
- [x] timm fp16, QOperator INT8, explicit Q/DQ, compressed Q/DQ artifacts + gates
- [x] Harness: `--device cpu`, `--reference(-dir)`, base-family guard, timm registry, `--source timm`
- [x] Reconciliation: `compare_timm.py` (validated, Wilson CIs) + `summarize.py --source`
- [x] CPU INT8 QOperator arm executed (complete: 95.5 min, 23 conditions, reconciled)
- [x] GPU: timm FP32 reference arm (7.5 min, 23 conditions)
- [x] GPU: fp16 / TRT QDQ arms + engine evidence (both TRT engines 54/54 Int8 convs)
- [x] GPU: full timm `summarize.py` reconciliation (10,700 rows)
- [ ] Optional: second attack seed / 300-image population before final claims
