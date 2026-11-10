# MobileNetV2 on Radxa Q6A: CPU/NPU performance and evasion

[Measured results and figures](REPORT.md).

Board: **Radxa Dragon Q6A / QCS6490**, `192.168.1.40`. All performance
measurements run on this board. Model and runtime deployment is
`/home/ross/mobilenetv2_qairt/6490_v242`. The dedicated CPU Python environment
is `/home/ross/mobilenetv2_qairt/venv` (ONNX Runtime 1.31.1, NumPy 2.5.4).

The [conversion](../qnn/README.md) supplies the original FP32 ONNX, FP32 DLC,
INT8 DLC and INT8 HTP context. All are synced together with QAIRT **2.42.0.251225**
CPU/HTP libraries, the v68 DSP skeleton and matching SDK headers. The full
deployment manifest is [results/deployment.json](results/deployment.json).

## Performance method

The primary pipeline comparison is batch one, pre-cropped float32 RGB CHW in
`[0,1]` to 1,000 dequantized float32 logits. It includes ImageNet normalization,
the synchronous runtime call, and NPU native input quantization/output decoding.
The NPU pipeline includes board-local pipes to the resident C scorer; the CPU
pipeline is in-process ONNX Runtime. Image decode, resize/crop, network traffic
and model startup are excluded. These boundaries are intentionally recorded
separately from graph-only profiling.

CPU uses only `CPUExecutionProvider`, with one or four intra-op threads and one
inter-op thread. Processes inherit affinity to CPUs 4–7 (the performance cores).
The governor is `schedutil`; clocks and thermal state are saved around each run.
There is no clock locking, power-mode tuning or measured energy claim.

Each phase has 20 warmups and 200 timed inferences, with two measurement passes
(400 timing samples per reported median). The first six-case sweep runs CPU1,
CPU4, NPU, then reverses the order. The final two NPU measurements use the
updated scorer that reads the actual version-3 context metadata. Earlier scorer
measurements remain in the evidence directory but are excluded from the summary.
CPU4 has higher tail latency in the measured samples; raw samples are retained.

RAM is sampled from `/proc/<pid>/smaps_rollup` every 5 ms. RSS includes resident
shared pages; PSS apportions shared pages. The NPU pipeline sums the Python
driver and resident C scorer, whereas CPU is one Python runtime process. Summed
RSS can double-count shared pages; use PSS for the apportioned comparison. These
are **host-visible process memory figures**, not total board or accelerator RAM.
DSP firmware, accelerator-private allocations and the configured 2 MiB VTCM are
not accounted for by Linux process RSS/PSS. Native C graph benchmarking also
records process peak RSS using `getrusage`; that separate native C run uses the
board's default CPU affinity.

Additional QAIRT measurements run `qnn-net-run` on FP32 CPU DLC, INT8 CPU DLC
and INT8 HTP context. Each has 220 executions, with the first 20 discarded from
the backend and NETRUN timing summaries. Their peak RAM is measured across the
whole child-process lifetime, including startup. These are distinct from the
Python pipeline results. Profiling distinguishes HTP DSP execution from the QNN
host/RPC call; neither should be presented as full image-to-result latency.

CPU source quality is checked on the same 100-image cohort: both ONNX Runtime
thread counts preserve all ONNX top-1 predictions (83/100 correct). The served
NPU native input path matches all prior `qnn-net-run` float-input logits exactly
on the cohort (78/100 correct), so the service does not introduce a new
preprocessing convention.

Measured tables and figures are generated from
[results/performance.json](results/performance.json) and `results/benchmark/`.

## Persistent service

The user systemd service **`mobilenetv2-qnn`** keeps the context loaded once and
serializes requests through one FastRPC session. It is a running transient user
service; it is not installed as an enabled boot service. Health and native tensor
metadata are available at:

```text
http://192.168.1.40:8088/health
```

`POST /predict` accepts an octet-stream payload: little-endian uint32 batch
count, then that many native UINT8 NCHW `[1,3,224,224]` tensors. Batch limit is
100. The response contains the same count, followed by 1,000 native UINT8 logits
per image. Tensor encodings and context hash are returned by `/health`.
The host [runtime adapter](scripts/runtime.py) handles normalization,
quantization, transport and dequantization. It refuses a context hash mismatch.

```bash
q6a 'systemctl --user status mobilenetv2-qnn --no-pager'
q6a 'journalctl --user -u mobilenetv2-qnn -n 30 --no-pager'
q6a 'systemctl --user stop mobilenetv2-qnn'
```

To create the service again after it has been removed or the board rebooted:

```bash
q6a 'systemd-run --user --unit=mobilenetv2-qnn \
  --property=WorkingDirectory=/home/ross/mobilenetv2_qairt/6490_v242 \
  --property=Restart=on-failure \
  /usr/bin/python3 /home/ross/mobilenetv2_qairt/6490_v242/serve_board.py'
```

For a stopped unit that still exists, use `systemctl --user start mobilenetv2-qnn`.

## Evasion method

The original 100-image stratified Imagenette cohort and MobileNet preprocessing
are retained. Image hashes are rechecked. Attacks operate on RGB pixels in
`[0,1]`, with L∞ budgets **1, 2, 4, 8 /255** and the original 1,000-class labels.
Success rates use each target's own clean-correct subset: 83 for FP32 and 78
for HTP. Clean-misclassified inputs are excluded from success denominators.

- Random uniform perturbations serve as a control.
- FGSM and PGD use cross-entropy ascent. PGD has 20 steps, two random starts,
  step size ε/4, per-step projection/clipping and per-image success retention.
- At 1/255, surviving PGD examples receive an additional 100-step, five-start
  check. Already-evaded examples retain their primary PGD candidates.
- Square is score-only, without a derivative or transfer initialization, and
  uses at most 500 attack queries per eligible image. HTP is tested at all four
  budgets; the FP32 control uses 4/255 and 8/255. Query counts exclude clean
  baselines and final confirmation/replay calls.
- HTP also receives the saved FP32 FGSM/PGD candidates as transfer attacks.
- Any 1/255 survivor of the primary HTP attack union receives a separate Square
  search with up to 5,000 additional queries, recorded independently.

All HTP scores and decisions come from the actual board service. BPDA supplies
derivatives from the differentiable FP32 replica of the **same source ONNX**.
The replica is numerically checked against ONNX; it is not a native gradient of
the INT8 implementation and does not reproduce every internal quantization
grid. Its clean top-1 agreement with HTP is 89/100. Exact-forward and finite,
nonzero-gradient checks run before attacks. The provenance chain identifies the
source, DLC/context build and deployed context, so a different architecture is
never substituted as a surrogate to improve agreement.

Every candidate is scored both as float pixels and after rounding RGB to UINT8
(`round(x*255)/255`). Bounds are checked again after rounding. HTP records how
many input codes actually change. Saved candidates and rounded candidates are
independently replayed in a fresh evaluation pass, with bit-exact saved-logit
checks and per-image results. The cumulative attack union counts failure under
any tested attack at an equal or smaller budget. Random controls are excluded
from that union. A surviving example is not a robustness certificate.

The larger clean evaluation covers all **3,923 local ImageNet-mini validation
images**, using 1,000-class labels. This is distinct from the 100-image attack
cohort and from full ImageNet validation. Calibration overlap is checked by
original JPEG hash; held-out accuracy is reported separately if overlap exists.

## Reproduce and inspect

Run from `/home/ross/QA`, using the existing host `venv/bin/python`:

```bash
q6a 'systemctl --user stop mobilenetv2-qnn'
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/sync_board.py
# Run board benchmarks while the inference service is stopped.
q6a 'cd /home/ross/mobilenetv2_qairt/6490_v242 && \
  /home/ross/mobilenetv2_qairt/venv/bin/python benchmark_board.py ort1 --tag 1'
# Repeat ort1/ort4/htp in forward/reverse order; see saved benchmark logs.

venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/run_suite.py --target pytorch_fp32
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/run_suite.py --target qnn_htp
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/check_survivors.py
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/verify_results.py
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/full_validation.py
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/evaluate_service.py
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/summarize.py
venv/bin/python Models/mobilenetV2/evasion_qnn/scripts/write_report.py
```

The attack runner checkpoints completed attack/budget cells and checks model,
cohort and configuration identity before resuming. All saved tensors are in
`results/run/`; replay results and a per-image CSV are in `results/verification/`.
The broader clean evaluation is in `results/full_validation/`. Logs and native
profiling exports are retained alongside the summary figures.
