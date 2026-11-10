#!/usr/bin/env python3
"""Validate static INT8 graph, float conversion fidelity, and real Coral inference."""
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from edgetpu_runtime import CPU_MODEL, TPU_MODEL, EdgeTpuRuntime, sha256

HERE = Path(__file__).resolve().parent


def comparison(logits, reference, labels):
    assert logits.shape == reference.shape and np.isfinite(logits).all()
    cosine = (logits * reference).sum(1) / (np.linalg.norm(logits, axis=1) * np.linalg.norm(reference, axis=1))
    return dict(n=len(logits), correct=int((logits.argmax(1) == labels).sum()),
                top5_correct=int((np.argsort(logits, axis=1)[:, -5:] == labels[:, None]).any(1).sum()),
                top1_agreement=int((logits.argmax(1) == reference.argmax(1)).sum()),
                logit_mae=float(np.abs(logits - reference).mean()), logit_max_abs=float(np.abs(logits - reference).max()),
                mean_cosine=float(cosine.mean()), min_cosine=float(cosine.min()))


def main():
    data = np.load(HERE / 'data/evaluation.npz')
    rgb, labels, reference = data['rgb'], data['labels'], data['onnx_logits']
    source = json.loads((HERE / 'results/source_validation.json').read_text())
    assert sha256(HERE / 'data/evaluation.npz') == source['evaluation_sha256']
    assert sha256(HERE.parent / 'onnx/mobilenetv2.onnx') == source['source_sha256']
    assert sha256(HERE.parent / 'pytorch/mobilenet_v2-b0353104.pth') == source['checkpoint_sha256']
    assert sha256(HERE / 'mobilenetv2_raw_rgb_static.onnx') == source['prepared_onnx_sha256']

    # No delegates while inspecting graph structure.
    inspector = tf.lite.Interpreter(model_path=str(CPU_MODEL), experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    inspector.allocate_tensors()
    tensors = inspector.get_tensor_details()
    ops = [op['op_name'] for op in inspector._get_ops_details()]
    structure = dict(tensors=len(tensors), dynamic_tensors=sum(bool(np.any(t['shape_signature'] < 0)) for t in tensors),
                     floating_tensors=sum(bool(np.issubdtype(t['dtype'], np.floating)) for t in tensors),
                     transpose_ops=ops.count('TRANSPOSE'), operations={name: ops.count(name) for name in sorted(set(ops))})
    assert structure['dynamic_tensors'] == structure['floating_tensors'] == structure['transpose_ops'] == 0
    assert not any(op.startswith(('Flex', 'CUSTOM')) for op in ops)

    # Check the unquantized onnx2tf graph so padding/layout errors cannot hide
    # behind the expected INT8 quantization drift.
    fp = tf.lite.Interpreter(model_path=str(HERE / 'build/mobilenetv2_raw_rgb_static_float32.tflite'), num_threads=1)
    fp.allocate_tensors()
    inp, out = fp.get_input_details()[0], fp.get_output_details()[0]
    fp_logits = []
    for image in rgb:
        fp.set_tensor(inp['index'], image[None].astype(np.float32) / np.float32(255))
        fp.invoke()
        fp_logits.append(fp.get_tensor(out['index'])[0])
    fp_logits = np.stack(fp_logits)
    np.testing.assert_allclose(fp_logits, reference, atol=2e-4, rtol=2e-4)
    fp_check = comparison(fp_logits, reference, labels)
    assert fp_check['top1_agreement'] == len(rgb)

    cpu, tpu = EdgeTpuRuntime(backend='cpu'), EdgeTpuRuntime(backend='tpu')
    cpu_logits, tpu_logits = cpu.run_uint8(rgb), tpu.run_uint8(rgb)
    cpu_check, tpu_check = comparison(cpu_logits, reference, labels), comparison(tpu_logits, reference, labels)
    hardware_parity = comparison(tpu_logits, cpu_logits, labels)
    # Quantization and accelerator arithmetic change predictions. Record their
    # measured effect; no application-specific accuracy target was supplied.
    # Exact fidelity is required above for the FP32 conversion, not INT8.
    assert tpu.info['operations'] == ['edgetpu-custom-op', 'DELEGATE'], tpu.info['operations']
    # Public floating-pixel adapter must produce exactly the same native inputs.
    np.testing.assert_array_equal(tpu.run(rgb[:3].transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255)), tpu_logits[:3])
    for runtime in (cpu, tpu):
        for _ in range(10):
            runtime.run_uint8(rgb[:1])
        runtime.latencies.clear()
        for _ in range(200):
            runtime.run_uint8(rgb[:1])
    cpu_runtime = cpu.verify(HERE / 'results/cpu')
    tpu_runtime = tpu.verify(HERE / 'results/tpu')
    np.savez(HERE / 'results/logits.npz', labels=labels, onnx=reference, float32=fp_logits, cpu_int8=cpu_logits, edgetpu_int8=tpu_logits)
    np.savez(HERE / 'results/latencies.npz', cpu=cpu.latencies, edgetpu=tpu.latencies)
    result = dict(source=source, structure=structure, fp32_conversion=fp_check, cpu_int8=cpu_check, edgetpu_int8=tpu_check,
                  tpu_vs_cpu_int8=hardware_parity, cpu_runtime=cpu_runtime, tpu_runtime=tpu_runtime,
                  benchmark='200 warm batch-1 invoke() calls after 10 warmups, same RGB image; excludes preprocessing and tensor copies; CPU 1 thread',
                  cpu_latency_ms_p95=float(np.percentile(cpu.latencies, 95)),
                  tpu_latency_ms_p95=float(np.percentile(tpu.latencies, 95)),
                  artifacts={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in (CPU_MODEL, TPU_MODEL)},
                  conversion_optimized_onnx_sha256=sha256(HERE / 'conversion_input/mobilenetv2_raw_rgb_static.onnx'),
                  quantization_notes=['INT8 accuracy and agreement are descriptive pilot measurements, not application acceptance thresholds.',
                                      'CPU INT8 and Edge TPU INT8 are compared separately; hardware results are authoritative for Coral deployment.'],
                  versions={name: importlib.metadata.version(name) for name in ('tensorflow-cpu', 'numpy', 'pillow')})
    (HERE / 'results/validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
