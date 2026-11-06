# MobileNetV2 replica-vs-engine validation

Generated: 2026-11-06T21:18:21  
Command: `evasion_trt/scripts/probe_replica.py --device cuda --ort-check`  
Host: Linux-7.0.0-38-generic-x86_64-with-glibc2.43  
Python: `/home/ross/QA/venv/bin/python`  torch 2.15.0+cu132  numpy 2.5.4  onnx 1.21.0  
GPU: NVIDIA RTX PRO 4000 Blackwell  
Population: 100 images, 10 per class x 10 classes, seed 20261030

Preprocessing: Resize(256, BICUBIC, antialias=True) -> CenterCrop(224) -> ToTensor (pixels in [0,1]; mean/std applied inside `GraphReplica`/`EngineRuntime`).

## Gate: replica vs bs1 TensorRT engine, clean top-1 agreement

| arm | engine clean | replica clean | top-1 agree | mean \|dlogit\| | max \|dlogit\| | global r | gate >=90% | ideal >=97% |
|---|---|---|---|---|---|---|---|---|
| fp32 | 83/100 | 83/100 | 100/100 (100.0%) | 0.004406 | 0.04168 | 1.00000 | PASS | PASS |
| fp16 | 83/100 | 83/100 | 100/100 (100.0%) | 0.006361 | 0.07045 | 1.00000 | PASS | PASS |
| fp8 | 84/100 | 84/100 | 99/100 (99.0%) | 0.2022 | 1.4 | 0.99623 | PASS | PASS |
| int8 | 79/100 | 81/100 | 95/100 (95.0%) | 0.2495 | 2.678 | 0.99391 | PASS | - |
| int4 | 83/100 | 83/100 | 100/100 (100.0%) | 0.004334 | 0.03752 | 1.00000 | PASS | PASS |

Overall gate (>= 90% on all five arms): **PASS**  
All ideal (>= 97% fp32/fp16/int4 and >= 90% fp8/int8): **PASS**

Engine clean counts use the population the Lead measured in SPEC section 3 (fp32 83, fp16 83, fp8 84, int8 79, int4 83); a mismatch would mean this probe is not scoring the agreed images/preprocessing.

## Gradient and one-step directional probe

One FGSM-style sign step at eps = 8.0/255, candidates scored on the real bs1 engine. Flips = engine top-1 changes relative to the clean engine prediction.

| arm | grad finite | zero-grad imgs | near-zero imgs (saturated CE) | grad L1 mean | grad sign flips | random sign flips | negated sign flips | grad margin change | random margin change | informative |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | True | 0 | 1 | 88.87 | 92 (92%) | 12 (12%) | 20 | -3.032 | -0.8088 | True |
| fp16 | True | 0 | 1 | 88.67 | 90 (90%) | 12 (12%) | 20 | -3.025 | -0.8117 | True |
| fp8 | True | 0 | 1 | 89.68 | 92 (92%) | 14 (14%) | 20 | -3.174 | -0.8418 | True |
| int8 | True | 0 | 1 | 91.88 | 92 (92%) | 15 (15%) | 22 | -3.113 | -0.8207 | True |
| int4 | True | 0 | 1 | 88.47 | 92 (92%) | 12 (12%) | 22 | -2.947 | -0.8173 | True |

## Folded classifier weight (INT4 blocked dequant)

| arm | folded mean abs err vs FP32 checkpoint | max abs err | int4 < 0.01 check |
|---|---|---|---|
| fp32 | 0 | 0 | n/a |
| fp16 | 7.90908e-06 | 0.000121951 | n/a |
| fp8 | 7.90908e-06 | 0.000121951 | n/a |
| int8 | 7.90908e-06 | 0.000121951 | n/a |
| int4 | 0.00459241 | 0.101982 | PASS |

## Cross-arm context (reference: fp32)

Deployment gap = how far each engine is from the FP32 engine; surrogate gap = how far each replica is from its own engine. A replica is only the limiting factor when its surrogate gap is much larger than the deployment gap.

| arm | engine vs fp32 engine | engine mean \|dlogit\| | replica vs fp32 replica | replica mean \|dlogit\| | replica vs own engine |
|---|---|---|---|---|---|
| fp32 | 100.0% | 0 | 100.0% | 0 | 100.0% |
| fp16 | 100.0% | 0.01372 | 100.0% | 0.01256 | 100.0% |
| fp8 | 94.0% | 0.3788 | 93.0% | 0.3809 | 99.0% |
| int8 | 92.0% | 0.3234 | 92.0% | 0.3288 | 95.0% |
| int4 | 99.0% | 0.09401 | 99.0% | 0.09401 | 100.0% |

## ORT cross-check (diagnostic, NOT a gate)

The same ONNX graph scored by onnxruntime CPU (`config.normalized(pixels)` input). ORT is authoritative ONNX-semantics evidence for fp32/fp16; for the Q/DQ arms it is not (SPEC section 3: ORT INT8 Q/DQ execution of these ModelOpt->TRT graphs is unreliable and ORT rejects the blocked INT4 DequantizeLinear), which is why the deployed TRT engine is the reference.

| arm | ORT ran | authoritative | ORT vs engine | replica vs ORT | replica vs ORT mean \|dlogit\| | note |
|---|---|---|---|---|---|---|
| fp32 | yes | True | 100.0% | 100.0% | 2.163e-06 | matches ONNX semantics |
| fp16 | yes | True | 100.0% | 100.0% | 2.075e-06 | matches ONNX semantics |
| fp8 | yes | False | 99.0% | 100.0% | 0.1839 | not authoritative |
| int8 | yes | False | 3.0% | 3.0% | 2.23 | not authoritative |
| int4 | no | False | | | | RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running DequantizeLinear node. Name:'classifier.1.weight_DequantizeLinear' Status Message: /onnxruntime_src/onnxruntime/core/providers/cpu/quantization/quantize_linear.cc:97 void onnxruntime::PrepareForQDQ(const TensorShape&, const Tensor&, const Tensor*, int64_t, int64_t, int64_t&, int64_t&, int64_t&) scale.Shape()[i] == (input_shape[i] + quant_block_size - 1) / quant_block_size was false. x_scale must be ceil(Di/block_size) on the quantize axis i for blocked quantization
 |

## Replica composition (executed dynamic ops)

| arm | dynamic ops | constants folded (registered) | op counts |
|---|---|---|---|
| fp32 | 100 | 110 (110) | Addx10, Clipx35, Convx52, Gemmx1, ReduceMeanx1, Reshapex1 |
| fp16 | 102 | 110 (110) | Addx10, Castx2, Clipx35, Convx52, Gemmx1, ReduceMeanx1, Reshapex1 |
| fp8 | 166 | 498 (306) | Addx10, Castx2, Clipx35, Convx52, DequantizeLinearx32, Gemmx1, QuantizeLinearx32, ReduceMeanx1, Reshapex1 |
| int8 | 206 | 698 (386) | Addx10, Castx2, Clipx35, Convx52, DequantizeLinearx52, Gemmx1, QuantizeLinearx52, ReduceMeanx1, Reshapex1 |
| int4 | 100 | 180 (178) | Addx10, Clipx35, Convx52, Gemmx1, ReduceMeanx1, Reshapex1 |

## Notes

- Scores are the real bs1 TensorRT engine logits; the replica supplies derivatives only (no native runtime autograd).
- The replica runs the arm ONNX graph in float32 (Cast is identity), folds weight Q/DQ chains onto the recorded grids and uses straight-through estimators for dynamic Q/DQ.
- INT4 is weight-only: the single blocked DequantizeLinear (block_size=128, axis=0, scale shape (1000,10)) is folded as W_dq[o,i] = w4[o,i] * scale[o, i//128].
- Clean top-1 agreement is a blunt proxy for gradient fidelity, hence the separate one-step directional probe on the real engine.
- Near-zero CE gradients are a numerical property of saturated-but-correct predictions (huge top-2 margin), not a replica defect: sign() still defines a valid FGSM/PGD direction, but the magnitude carries no signal for those images.
- The cross-arm table compares each arm with the FP32 reference. The surrogate gap (replica vs own engine) should be read against the deployment gap (engine vs FP32 engine): a replica much further from its engine than the engines are from each other is the only worrying configuration.

## Per-arm logs

- **fp32**: engine 83/100, agreement 100/100 (100.0%), mean|dlogit| 0.004406, max|dlogit| 0.04168, disagreement engine-margin mean n/a (none), near-zero gradients 1 (replica top-2 margin mean 25.51774787902832), 0.04s engine clean.
- **fp16**: engine 83/100, agreement 100/100 (100.0%), mean|dlogit| 0.006361, max|dlogit| 0.07045, disagreement engine-margin mean n/a (none), near-zero gradients 1 (replica top-2 margin mean 25.523569107055664), 0.03s engine clean.
- **fp8**: engine 84/100, agreement 99/100 (99.0%), mean|dlogit| 0.2022, max|dlogit| 1.4, disagreement engine-margin mean 0.140625, near-zero gradients 1 (replica top-2 margin mean 24.80852508544922), 0.03s engine clean.
- **int8**: engine 79/100, agreement 95/100 (95.0%), mean|dlogit| 0.2495, max|dlogit| 2.678, disagreement engine-margin mean 0.40234375, near-zero gradients 1 (replica top-2 margin mean 24.349225997924805), 0.03s engine clean.
- **int4**: engine 83/100, agreement 100/100 (100.0%), mean|dlogit| 0.004334, max|dlogit| 0.03752, disagreement engine-margin mean n/a (none), near-zero gradients 1 (replica top-2 margin mean 25.575197219848633), 0.04s engine clean.
