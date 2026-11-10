#!/usr/bin/env python3
"""Prepare a static raw-RGB MobileNetV2 ONNX and the ResNet TPU calibration set."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
QA = MODEL.parents[1]
MEAN = np.array([.485, .456, .406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([.229, .224, .225], np.float32).reshape(1, 3, 1, 1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def main():
    out = HERE / 'results'
    data = HERE / 'data'
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)
    source = MODEL / 'onnx/mobilenetv2.onnx'
    previous = json.loads((MODEL / 'rknn/results/source_validation.json').read_text())
    assert sha256(source) == previous['source_sha256']
    checkpoint = MODEL / 'pytorch/mobilenet_v2-b0353104.pth'
    assert sha256(checkpoint) == previous['checkpoint_sha256']

    model = onnx.load(source)
    # Keep normalization before the padded stem convolution. A naive weight/bias
    # fold changes border pixels because zero padding follows normalization.
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == 'input':
                node.input[i] = 'normalized_rgb'
    model.graph.node.insert(0, helper.make_node('Div', ['centered_rgb', 'rgb_std'], ['normalized_rgb'], name='normalize_std'))
    model.graph.node.insert(0, helper.make_node('Sub', ['input', 'rgb_mean'], ['centered_rgb'], name='normalize_mean'))
    model.graph.initializer.extend([numpy_helper.from_array(MEAN, 'rgb_mean'), numpy_helper.from_array(STD, 'rgb_std')])
    for value, shape in ((model.graph.input[0], [1, 3, 224, 224]), (model.graph.output[0], [1, 1000])):
        value.type.tensor_type.shape.CopyFrom(helper.make_tensor_value_info(value.name, onnx.TensorProto.FLOAT, shape).type.tensor_type.shape)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    prepared = HERE / 'mobilenetv2_raw_rgb_static.onnx'
    onnx.save(model, prepared)

    # Exact image-selection and resize recipe used for the successful ResNet TPU
    # conversion: first image from each of the first 200 nonempty training classes.
    train = QA / 'datasets/imagenetmini-1000/imagenet-mini/train'
    paths = []
    for folder in sorted(p for p in train.iterdir() if p.is_dir())[:220]:
        files = sorted(folder.glob('*.JPEG'))
        if files:
            paths.append(files[0])
    paths = paths[:200]
    assert len(paths) == 200
    calibration = []
    for path in paths:
        with Image.open(path) as image:
            calibration.append(np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), np.float32) / np.float32(255))
    calibration = np.stack(calibration)
    np.save(data / 'calibration.npy', calibration)
    samples = json.loads((MODEL / 'rknn/results/samples.json').read_text())
    assert all(sha256(row['path']) == row['sha256'] for row in samples)
    calibration_rows = [dict(path=str(p), sha256=sha256(p)) for p in paths]
    assert not {row['sha256'] for row in calibration_rows} & {row['sha256'] for row in samples}
    (out / 'calibration_samples.json').write_text(json.dumps(calibration_rows, indent=2) + '\n')
    (out / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    reference = np.load(MODEL / 'rknn/results/reference.npz')
    rgb, labels = reference['rgb'], reference['labels']
    pixels = rgb.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255)
    original, wrapped = session(source), session(prepared)
    source_logits = np.concatenate([original.run(None, {'input': x[None]})[0] for x in (pixels - MEAN) / STD])
    logits = np.concatenate([wrapped.run(None, {'input': x[None]})[0] for x in pixels])
    np.testing.assert_allclose(source_logits, reference['onnx_logits'], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(logits, source_logits, atol=1e-4, rtol=1e-4)
    rng = np.random.default_rng(20261110)
    edge_cases = [np.zeros((1, 3, 224, 224), np.float32), np.ones((1, 3, 224, 224), np.float32), rng.random((1, 3, 224, 224), dtype=np.float32)]
    border_max_abs = 0.0
    for x in edge_cases:
        a = original.run(None, {'input': (x - MEAN) / STD})[0]
        b = wrapped.run(None, {'input': x})[0]
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4)
        border_max_abs = max(border_max_abs, float(np.abs(a - b).max()))
    np.savez(data / 'evaluation.npz', rgb=rgb, labels=labels, onnx_logits=source_logits, pytorch_logits=reference['pytorch_logits'])
    result = dict(source=str(source), source_sha256=sha256(source), checkpoint_sha256=sha256(checkpoint),
                  prepared_onnx_sha256=sha256(prepared), calibration_n=len(paths),
                  calibration_sha256=sha256(data / 'calibration.npy'), calibration_content_overlap=0,
                  calibration_recipe='Same 200 ImageNet-mini train images and bilinear 224x224 resize as ResNet TPU',
                  evaluation_n=len(rgb), evaluation_preprocessing=previous['preprocessing'],
                  evaluation_sha256=sha256(data / 'evaluation.npz'),
                  onnx_correct=int((source_logits.argmax(1) == labels).sum()),
                  wrapped_top1_agreement=int((logits.argmax(1) == source_logits.argmax(1)).sum()),
                  wrapped_logit_max_abs=float(np.abs(logits - source_logits).max()),
                  black_white_random_max_abs=border_max_abs,
                  normalization='Explicit (pixel - ImageNet mean) / std before padded stem; raw RGB float [0,1] input')
    (out / 'source_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
