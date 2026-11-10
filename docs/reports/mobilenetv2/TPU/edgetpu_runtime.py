#!/usr/bin/env python3
"""MobileNetV2 Coral USB adapter: pixel-space RGB in, dequantized logits out."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CPU_MODEL = HERE / 'mobilenetv2_int8_nhwc.tflite'
TPU_MODEL = HERE / 'mobilenetv2_int8_nhwc_edgetpu.tflite'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def device_info(lib):
    library = ctypes.CDLL(lib)
    library.edgetpu_version.restype = ctypes.c_char_p

    class Device(ctypes.Structure):
        _fields_ = [('type', ctypes.c_int), ('path', ctypes.c_char_p)]

    library.edgetpu_list_devices.restype = ctypes.POINTER(Device)
    library.edgetpu_list_devices.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    library.edgetpu_free_devices.argtypes = [ctypes.POINTER(Device)]
    count = ctypes.c_size_t()
    devices = library.edgetpu_list_devices(ctypes.byref(count))
    try:
        return dict(version=library.edgetpu_version().decode(), devices=[dict(type=int(devices[i].type), path=devices[i].path.decode()) for i in range(count.value)])
    finally:
        library.edgetpu_free_devices(devices)


class EdgeTpuRuntime:
    has_gradients = False

    def __init__(self, model_path=None, backend='tpu', lib='libedgetpu.so.1', device=None, num_threads=1):
        import tensorflow as tf
        if backend not in ('cpu', 'tpu'):
            raise ValueError(f'Unknown backend: {backend}')
        self.backend = backend
        self.model_path = Path(model_path or (TPU_MODEL if backend == 'tpu' else CPU_MODEL)).resolve()
        self._delegate = None
        if backend == 'tpu':
            options = {'device': device} if device else {}
            for attempt in range(3):
                try:
                    self._delegate = tf.lite.experimental.load_delegate(lib, options)
                    break
                except (ValueError, OSError):
                    if attempt == 2:
                        raise
                    time.sleep(2)  # First open uploads firmware and re-enumerates USB.
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path), num_threads=num_threads,
            experimental_delegates=[self._delegate] if self._delegate is not None else [])
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        assert self.input['shape'].tolist() == [1, 224, 224, 3]
        assert self.output['shape'].tolist() == [1, 1000]
        assert self.input['dtype'] == np.uint8 and self.output['dtype'] == np.uint8
        assert np.isclose(self.input['quantization'][0], 1 / 255) and self.input['quantization'][1] == 0
        self.latencies = []
        self.info = dict(model=str(self.model_path), sha256=sha256(self.model_path), bytes=self.model_path.stat().st_size,
                         backend=backend, tensorflow=tf.__version__, num_threads=num_threads,
                         input=self._contract(self.input), output=self._contract(self.output),
                         preprocessing='Raw RGB uint8 NHWC; ImageNet normalization is inside graph before padded stem',
                         operations=[op['op_name'] for op in self.interpreter._get_ops_details()])
        if backend == 'tpu':
            self.info['edgetpu'] = device_info(lib)
            assert 'edgetpu-custom-op' in self.info['operations'], 'No Edge TPU subgraph found'

    @staticmethod
    def _contract(detail):
        return dict(shape=detail['shape'].tolist(), dtype=detail['dtype'].__name__,
                    scale=float(detail['quantization'][0]), zero_point=int(detail['quantization'][1]))

    def run_uint8(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (224, 224, 3):
            raise ValueError('Expected uint8 RGB [N,224,224,3]')
        result = np.empty((len(rgb), 1000), np.float32)
        scale, zero = self.output['quantization']
        for i, image in enumerate(rgb):
            self.interpreter.set_tensor(self.input['index'], np.ascontiguousarray(image[None]))
            start = time.perf_counter()
            self.interpreter.invoke()
            self.latencies.append((time.perf_counter() - start) * 1000)
            result[i] = (self.interpreter.get_tensor(self.output['index'])[0].astype(np.float32) - zero) * scale
        return result

    def run(self, pixels):
        # Avoid importing Torch into a TensorFlow-only scoring process.
        is_torch = hasattr(pixels, 'detach')
        array = pixels.detach().cpu().numpy() if is_torch else np.asarray(pixels)
        if array.ndim == 4 and array.shape[1] == 3:
            array = array.transpose(0, 2, 3, 1)
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError('Expected finite floating RGB pixels in [0,1]; use run_uint8 for uint8')
        output = self.run_uint8(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
        if is_torch:
            import torch
            return torch.from_numpy(output)
        return output

    predict = run
    __call__ = run

    def logits_for_grad(self, pixels):
        raise NotImplementedError('Edge TPU has no input gradients; use a differentiable surrogate')

    def verify(self, workdir):
        path = Path(workdir)
        path.mkdir(parents=True, exist_ok=True)
        result = dict(self.info, n_inferences=len(self.latencies), latency_ms_median=float(np.median(self.latencies)) if self.latencies else None)
        (path / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'tpu'], default='tpu')
    parser.add_argument('--model')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--npz', help='NPZ containing uint8 rgb or floating [0,1] images, optional labels')
    parser.add_argument('--logits-out')
    parser.add_argument('--json-out')
    args = parser.parse_args()
    runtime = EdgeTpuRuntime(args.model, args.backend)
    result = dict(runtime.info)
    if args.check:
        image = np.zeros((1, 224, 224, 3), np.uint8)
        for _ in range(5):
            runtime.run_uint8(image)
        runtime.latencies.clear()
        for _ in range(args.repeat):
            runtime.run_uint8(image)
    if args.npz:
        data = np.load(args.npz)
        logits = runtime.run_uint8(data['rgb']) if 'rgb' in data else runtime.run(data['images'])
        result['n'] = len(logits)
        if 'labels' in data:
            result['correct'] = int((logits.argmax(1) == data['labels']).sum())
        if args.logits_out:
            np.save(args.logits_out, logits)
    result['latency_ms_median'] = float(np.median(runtime.latencies)) if runtime.latencies else None
    result['n_inferences'] = len(runtime.latencies)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
