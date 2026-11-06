# Independent verification of the MobileNetV2 TensorRT evasion matrix

Generated `2026-11-06T21:35:47` by `scripts/verify_independent.py` on `NV10` with Python `/home/ross/QA/venv/bin/python`, PyTorch `2.15.0+cu132`, TensorRT `11.3.0.99`.

This report is an **independent reconstruction**: the population loader, TensorRT scorer, per-image CSV parser, L-infinity bound check, Wilson intervals and matrix.md parser were re-implemented for this check and do not call `attacks.py`, `run_evasion.py`, `runtimes.py` or `summarize.py`. Only saved artifacts and forward engine inference were used; no attack was re-run.

**Overall verdict: PASS**

| # | check | result |
|---|---|---|
| 1 | 1. Clean reproduction | **PASS** |
| 2 | 2. Candidate re-scoring | **PASS** |
| 3 | 3. Transfer provenance | **PASS** |
| 4 | 4. Aggregate recomputation | **PASS** |
| 5 | 5. Artifact integrity | **PASS** |
| 6 | 6. Headline-claim sanity | **PASS** |

## Discrepancies

**None.** Every required equality/bound held exactly.

## Check 1 — Clean reproduction: PASS

Population re-selected independently from `/home/ross/QA/datasets/imagenette2-320/val`: 10 images x 10 synsets, `numpy.random.default_rng(20261030).choice(..., replace=False)`, sorted files; Resize 256 BICUBIC antialias -> CenterCrop 224 -> ToTensor. All 100 selected files matched the `samples.json` SHA-256 manifest for every arm, with 0 path/label mismatches.

| arm | fresh clean top-1 | clean.npz top-1 | expected | max abs logit diff vs clean.npz | top-1 agreement | samples.json clean_pred mismatches |
|---|---|---|---|---|---|---|
| fp32 | 83/100 | 83/100 | 83/100 | 0 | 100/100 | 0 |
| fp16 | 83/100 | 83/100 | 83/100 | 0 | 100/100 | 0 |
| fp8 | 84/100 | 84/100 | 84/100 | 0 | 100/100 | 0 |
| int8 | 79/100 | 79/100 | 79/100 | 0 | 100/100 | 0 |
| int4 | 83/100 | 83/100 | 83/100 | 0 | 100/100 | 0 |

Per-image `clean_prediction` mismatches across all 10700 rows: 0; label mismatches: 0.

## Check 2 — Candidate re-scoring: PASS

Every saved `.pt` adversarial tensor (87 files, 1860 candidates) was re-scored on its arm's bs1 engine; 1860 `per_image.csv` rows were reconciled directly.  Per-file counts of prediction / `adv_correct` / L-inf mismatches and over-epsilon or clean-retained candidates:

| arm | files | candidates | pred mism | adv_correct mism | max |linf_255 diff| | over-eps | clean-retained |
|---|---|---|---|---|---|---|---|
| fp32 | 11 | 1100 | 0 | 0 | 0.00e+00 | 0 | 37 |
| fp16 | 19 | 190 | 0 | 0 | 0.00e+00 | 0 | 0 |
| fp8 | 19 | 190 | 0 | 0 | 0.00e+00 | 0 | 1 |
| int8 | 19 | 190 | 0 | 0 | 0.00e+00 | 0 | 0 |
| int4 | 19 | 190 | 0 | 0 | 0.00e+00 | 0 | 0 |

FGSM single-step structural provenance (`cand_e == clip(clean + e/255*sign(g))`, common sign vector across budgets): 0 mismatched pixels.  Random control reconstructed from the recorded seeds with 0 prediction mismatches (20 conditions x 100 images).
- FGSM structure fp32: images=100, pairs assessed=397, pixels checked=59034191, mismatches=0, clean-retained pairs=3.
- FGSM structure fp16: images=10, pairs assessed=40, pixels checked=6008876, mismatches=0, clean-retained pairs=0.
- FGSM structure fp8: images=9, pairs assessed=36, pixels checked=5374544, mismatches=0, clean-retained pairs=0.
- FGSM structure int8: images=9, pairs assessed=36, pixels checked=5397900, mismatches=0, clean-retained pairs=0.
- FGSM structure int4: images=10, pairs assessed=40, pixels checked=6008776, mismatches=0, clean-retained pairs=0.

Clean-retained candidates are all explained: `Square` leaves the 17 clean-incorrect fp32 images (and one per-class fallback on fp8) untouched by design, and fp32 FGSM at 1-4/255 legitimately retained the clean image for index 88 because the step did not increase the engine loss. In every case the stored row is `adv_correct == clean_correct`; **zero** clean-retained candidates are claimed as successes.

## Check 3 — Transfer provenance: PASS

The `fp32` reference candidate tensors were scored on all four non-fp32 bs1 engines; all 3200 `transfer_*` rows matched the destination-engine re-score exactly. The `transfer_*.pt` copies stored in each destination arm are bit-identical to the fp32 source candidates at their stored indices. Counts per cell:

| arm | cell | dst pred mism | dst adv_correct mism | copy == fp32 | copy pred mism | rows dst-score != fp32-score |
|---|---|---|---|---|---|---|
| fp16 | fgsm_eps1 | 0 | 0 | True | 0 | 0 |
| fp16 | pgd_eps1 | 0 | 0 | True | 0 | 0 |
| fp16 | fgsm_eps2 | 0 | 0 | True | 0 | 0 |
| fp16 | pgd_eps2 | 0 | 0 | True | 0 | 0 |
| fp16 | fgsm_eps4 | 0 | 0 | True | 0 | 3 |
| fp16 | pgd_eps4 | 0 | 0 | True | 0 | 0 |
| fp16 | fgsm_eps8 | 0 | 0 | True | 0 | 2 |
| fp16 | pgd_eps8 | 0 | 0 | True | 0 | 0 |
| fp8 | fgsm_eps1 | 0 | 0 | True | 0 | 17 |
| fp8 | pgd_eps1 | 0 | 0 | True | 0 | 0 |
| fp8 | fgsm_eps2 | 0 | 0 | True | 0 | 14 |
| fp8 | pgd_eps2 | 0 | 0 | True | 0 | 1 |
| fp8 | fgsm_eps4 | 0 | 0 | True | 0 | 18 |
| fp8 | pgd_eps4 | 0 | 0 | True | 0 | 0 |
| fp8 | fgsm_eps8 | 0 | 0 | True | 0 | 18 |
| fp8 | pgd_eps8 | 0 | 0 | True | 0 | 0 |
| int8 | fgsm_eps1 | 0 | 0 | True | 0 | 16 |
| int8 | pgd_eps1 | 0 | 0 | True | 0 | 0 |
| int8 | fgsm_eps2 | 0 | 0 | True | 0 | 12 |
| int8 | pgd_eps2 | 0 | 0 | True | 0 | 0 |
| int8 | fgsm_eps4 | 0 | 0 | True | 0 | 19 |
| int8 | pgd_eps4 | 0 | 0 | True | 0 | 0 |
| int8 | fgsm_eps8 | 0 | 0 | True | 0 | 18 |
| int8 | pgd_eps8 | 0 | 0 | True | 0 | 0 |
| int4 | fgsm_eps1 | 0 | 0 | True | 0 | 7 |
| int4 | pgd_eps1 | 0 | 0 | True | 0 | 0 |
| int4 | fgsm_eps2 | 0 | 0 | True | 0 | 5 |
| int4 | pgd_eps2 | 0 | 0 | True | 0 | 0 |
| int4 | fgsm_eps4 | 0 | 0 | True | 0 | 4 |
| int4 | pgd_eps4 | 0 | 0 | True | 0 | 1 |
| int4 | fgsm_eps8 | 0 | 0 | True | 0 | 6 |
| int4 | pgd_eps8 | 0 | 0 | True | 0 | 0 |

## Check 4 — Aggregate recomputation: PASS

- `per_image.csv`: 10700 rows, 107 conditions, no duplicate `(arm, attack, eps, index)` keys, no unexpected/missing conditions (row-level inconsistencies: `{"model_column_mismatches": 0, "clean_correct_inconsistent": 0, "adv_correct_inconsistent": 0, "linf_over_budget": 0}`).
- Shared all-five clean-correct set recomputed as **n=78**; matrix.json indices match: True.
- Rebuilt every `summary.json` aggregate: 5 arms, `attack_success_pct` from its own clean-correct denominator, `robust_accuracy_pct`, `max_linf_255`, Square query counts/means/medians; 0 mismatches.
- Rebuilt `matrix.json`: 107 per-arm cells (own + shared + Wilson 95% CI), 32 transfer cells, 40 union cells: 0 mismatches.
- Rebuilt `matrix.md` tables from `per_image.csv`: 107 per-arm rows, 32 transfer rows, 40 union rows, 0 clean-table and 0 gate mismatches.

## Check 5 — Artifact integrity: PASS

| arm | ONNX sha256 (still matches summary+runtime+matrix) | engine sha256 |
|---|---|---|
| fp32 | `9f2d5d94df76f044172dc28345bf975b83e24de1ca33e5892fe074d312d93a77` | `de6c44806e79ad8e3d79be14b3f38e2fabff68cf8ddc6cc6974c2a3e89e63a86` |
| fp16 | `cf8a3b902a5f182c930f691a2216bc739912d224150b7498a4df8b4e5532b5d5` | `7c9a2d74b1097cb13badf12f551754d17837f5395b22736213f2cb46eacbf3bb` |
| fp8 | `38be5a8ede5edba98ef5f6dfab0baacc326589e340b2a8f444c68b3713034712` | `16638a12ea1864530f5b2d8afc03eb245cd28c4bd05434ab8bd59a5b3c58b799` |
| int8 | `f8990e426d29cb7221950a09e3a4f7cac221c43ed2aeaaa4d8e13057146870be` | `7636d85e9e859fc4fed928317e42c95a8ad153da1b5859d04c9b4770ff2461bf` |
| int4 | `1b04e8a57b23dfac4a23a047c82bf92ce317db1762df49e88f3664e614d03260` | `11a44751c11d23ed87238445241a2c8e7b6480d4d281cc578f9e076b87e31810` |

Snapshot of every file under `results/<arm>/`: 117 files hashed before and after the verification; added=0, removed=0, changed=0.

Informational: matrix-recorded script hashes that differ from the current checkout: `make_figures.py`.

## Check 6 — Headline-claim sanity: PASS

| arm | PGD 1/255 | PGD 2/255 | PGD 4/255 | PGD 8/255 | PGD+ 1/255 | Square 4/255 | Square 8/255 |
|---|---|---|---|---|---|---|---|
| fp32 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 48.2% | 85.5% |
| fp16 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 30.1% | 75.9% |
| fp8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 25.0% | 54.8% |
| int8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 19.0% | 49.4% |
| int4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 44.6% | 85.5% |

- PGD/BPDA-PGD: **100.0% of the arm's clean-correct images at 1, 2, 4 and 8/255 on all five arms** (own denominators 83/83/84/79/83).
- `pgd_stronger` at 1/255: 100.0% on every arm, but the 100x5 survivor check was **vacuous** — 0 survivors existed on every arm, so the rows are identical to `pgd` 1/255.
- Square (500 queries) is far below PGD: 4/255 = 48.2%, 30.1%, 25.0%, 19.0%, 44.6%; 8/255 = 85.5%, 75.9%, 54.8%, 49.4%, 85.5% (minimum PGD-minus-Square gap 14.5pp).
- Random-noise control: max success **6.0%** (worst cell fp8 at 8/255: 5/84).
- Failure-mode hunt: {"fooled_exceeds_clean_correct": 0, "robust_plus_fooled_ne_clean_correct": 0, "clean_retained_claimed_fooled": 0, "over_epsilon_rows": 0, "mislabeled_transfer_rows": 0, "nonfinite_candidates": 0}.

## Observations (not defects)

- Measured effect of the saved fp32 FGSM 1/255 candidates on the deployed engine: mean max |delta logit| = 9.84, mean L2 |delta logit| = 45.44, top-1 flip = 83.0%, mean margin drop = 9.97 on clean-correct images (mean clean margin 5.19). The 1/255 budget is nominal only: this model is brittle enough that a single-step sign perturbation genuinely moves the engine logits by several units, so the 91.6% FGSM success is not a counting/scoring artifact.
  - `{"evidence": {"mean_max_abs_logit_shift": 9.8404541015625, "max_abs_logit_shift": 19.91797637939453, "mean_l2_logit_shift": 45.438507080078125, "top1_flip_pct": 82.99999833106995, "mean_margin_drop_clean_correct": 9.97005558013916, "mean_clean_margin_clean_correct": 5.194793224334717}}`
- matrix.json source_hashes do not match the current scripts; the scripts changed after reconciliation (informational only)
  - `{"mismatched": {"make_figures.py": {"recorded": "cb40d7f276aec9d6e7157dde66bf8778f4bcbbb5efd95835237cf98761211120", "current": "16801233770664138c09bf9f292b444b10c396f2ec2d157394561aa1a85c0ef7"}}}`
- the 100-step x 5-restart "stronger survivor" check was vacuous on every arm: PGD already reached 100% at 1/255, so there were zero survivors to re-attack (additional_survivor_n == 0 everywhere, and the pgd_stronger rows/candidates are identical to pgd 1/255). This is a protocol-coverage limitation, not a numerical defect.
- random-noise control is near zero overall but reaches 6.0% on fp8 at 8/255 (quantized arms are more noise-sensitive; still far below the 99-100% gradient-attack success).
  - `{"worst": ["fp8", 8, 5, 84]}`
- FGSM success is not strictly monotone in eps for some arms (expected with per-step best retention and [0,1] clipping); this is reported as an observation, not a defect.
  - `{"cells": [["fp32", 4, 8, 96.3855421686747, 95.18072289156626], ["fp16", 4, 8, 96.3855421686747, 95.18072289156626], ["fp8", 4, 8, 96.42857142857143, 94.04761904761905], ["int8", 4, 8, 97.46835443037975, 93.67088607594937], ["int4", 4, 8, 96.3855421686747, 95.18072289156626]]}`

## Limitations of this verification

- Only the saved candidates can be re-scored: all 100 are kept on `fp32` and all 100 `transfer_*` rows are covered, but non-fp32 own-attack pools keep one representative per class (10), so the other 90 rows per cell are verified only through aggregate/consistency checks, not direct re-scoring.
- The attacks themselves were not re-run (per the task constraints); FGSM candidates are structurally proven to be single-step sign-gradient examples, and the random control is reconstructed exactly, but the PGD optimisation trajectory and the replica gradients are taken as recorded.
- Replica-vs-engine agreement was re-derived from the stored `clean.npz` logits, not by rebuilding the replica.

## Reproduction

```bash
cd /home/ross/QA/Models/mobilenetV2
./evasion_trt/sync.sh run evasion_trt/scripts/verify_independent.py
./evasion_trt/sync.sh pull
```
