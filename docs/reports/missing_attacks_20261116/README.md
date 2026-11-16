# Completed missing attack cells — 16 November 2026

Eight new deployment campaigns fill all **40 previously blank screenshot cells**.
The full fixed protocol contains **168 conditions and 16,800 per-image rows**;
every saved candidate was independently replayed on the native target before
archiving. The archive now covers **47 deployments: 24 ResNetV2-50 and 23
MobileNetV2**. Existing measurements are preserved.

- [Completed cell counts and rates](missing_cells.csv): exactly the 40 screenshot cells.
- [Full completion table, intervals and study-specific shared masks](results.json).
- [Completion and native replay coverage](completion.json).
- [Artifact validation](artifact_validation.json): canonical source hashes and INT4 classifier rewrite checks.
- [Archive graph/engine mapping](archive_map.json): hashes of the packaged weight artifacts.
- [Current complete matrices and figures](../cross_runtime/README.md).

## Scope and protocol

The same 100 stratified Imagenette validation images, seed 20261030, and model-specific
preprocessing are retained. Untargeted pixel-space FGSM/PGD budgets are 1, 2, 4
and 8/255; PGD uses 20 steps and two starts. PGD+ re-attacks 1/255 survivors
with 100 steps and five starts while retaining primary successes. Independent
score-only Square uses 500 queries at 4 and 8/255. Random controls and matched
FP32 transfer are included in the full per-image evidence.

NV10 (`192.168.1.32`, RTX PRO 4000 Blackwell) executed the intensive GPU work.
TFLite INT8 CPU executed locally on an Intel Core Ultra 7 165H, using independent
native interpreters whose output agreement with the serial scorer was checked.
ONNX Runtime scoring batches remain 10; the new TensorRT engines use static
batch 1. Native engine scores are authoritative; matched graph FP32 derivatives
and STE quantizers supply approximate backward gradients. All replica gates
passed at the predeclared 90% floor. INT4 quantizes only the classifier weights;
its preceding layers remain FP32. It does not imply INT4 activation/compute.

Each `results/<arm>/` retains the source summary, 100-image source ledger,
per-image CSV, runtime/provider evidence, independent native-replay validation
and, for TensorRT, engine inspector/precision evidence. Imported evidence is
byte-for-byte identical to the source workspace. The graph/engine mapping
checks the packaged weight files against attack-time hashes.

Packaging performs no inference or attacks. The reporting builder independently
reconciles compact per-image rows, bounds, query limits, success retention,
sample identities and archived native-replay records. Raw candidate tensors,
saved logits, full logs, datasets, calibration data and attack implementation
remain in the source workspace `QA/experiments/missing_attacks_20261116/` and
are omitted here. Consequently, downloading this archive alone does not enable
repeating the native candidate replay. The archived validation records document
the completed replay; the reporting build does not recreate it.

Rates use each deployment's own clean-correct denominator. Shared masks in this
campaign cover only its three MobileNet or five ResNet deployments, not every
archived runtime. No new 47-way intersection or cross-model candidate transfer
is claimed. These are 100-image, one-seed finite-attack results, not certificates.

See [source-only references](../SOURCE_REFERENCES.md), [weight provenance](../../PROVENANCE.md)
and [runtime compatibility](../../RUNTIMES.md). Verify from the repository root
with `sha256sum --check SHA256SUMS` and `sha256sum --check docs/reports/SHA256SUMS`.
