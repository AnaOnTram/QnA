#!/usr/bin/env python3
"""Pin the existing MobileNet source, ResNet QNN calibration and evaluation inputs."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent
PREVIOUS = Path('/home/ross/qairt/2.42.0.251225/examples/Models/resnet50_qairt_example')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for name in ('data/calibration', 'data/evaluation', 'results', 'logs'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    source = MODEL / 'onnx/mobilenetv2.onnx'
    old = json.loads((MODEL / 'rknn/results/source_validation.json').read_text())
    assert digest(source) == old['source_sha256']
    assert digest(MODEL / 'pytorch/mobilenet_v2-b0353104.pth') == old['checkpoint_sha256']
    calibration = []
    for line in (PREVIOUS / 'data/calib_list.txt').read_text().splitlines():
        src = Path(line.strip())
        assert src.stat().st_size == 3 * 224 * 224 * 4
        dst = ROOT / 'data/calibration' / src.name
        shutil.copyfile(src, dst)
        calibration.append(dict(source=str(src), path=str(dst), sha256=digest(dst)))
    assert len(calibration) == 20
    (ROOT / 'data/calib_list.txt').write_text(''.join(row['path'] + '\n' for row in calibration))
    ref = np.load(MODEL / 'rknn/results/reference.npz')
    samples = json.loads((MODEL / 'rknn/results/samples.json').read_text())
    for sample in samples:
        assert digest(sample['path']) == sample['sha256']
    # Same normalized float NCHW domain as the source ONNX and ResNet QNN.
    x = ref['rgb'].transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255)
    x = (x - np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]) / np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(source), sess_options=options, providers=['CPUExecutionProvider'])
    logits = np.concatenate([session.run(None, {'input': b})[0] for b in np.array_split(x, 10)])
    np.testing.assert_array_equal(logits, ref['onnx_logits'])
    inputs = []
    for i, item in enumerate(x):
        p = ROOT / f'data/evaluation/{i:03d}.raw'
        item.tofile(p)
        inputs.append(str(p))
    (ROOT / 'data/eval_list.txt').write_text('\n'.join(inputs) + '\n')
    np.savez(ROOT / 'results/reference.npz', labels=ref['labels'], onnx_logits=logits)
    (ROOT / 'results/samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    # Check image-content overlap, including the original calibration JPEGs.
    calib_images = [PREVIOUS / 'data/calibration' / (Path(r['source']).stem + '.JPEG') for r in calibration]
    overlap = {digest(p) for p in calib_images} & {r['sha256'] for r in samples}
    assert not overlap
    report = dict(source=str(source), source_sha256=digest(source),
                  checkpoint_sha256=old['checkpoint_sha256'],
                  calibration=calibration, calibration_count=len(calibration),
                  calibration_list_source_sha256=digest(PREVIOUS / 'data/calib_list.txt'),
                  calibration_recipe='Exact 20 ResNet QNN float32 NCHW raw files, same order; square center crop, bilinear 224, saved/reloaded JPEG, ImageNet normalization',
                  evaluation_preprocessing=old['preprocessing'], evaluation_count=len(samples),
                  evaluation_calibration_content_overlap=len(overlap),
                  onnx_correct=int((logits.argmax(1) == ref['labels']).sum()),
                  previous_reference_bit_exact=True)
    (ROOT / 'results/source_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Prepared {len(calibration)} exact ResNet calibration raws and {len(samples)} evaluation inputs; ONNX correct {report["onnx_correct"]}/100')


if __name__ == '__main__':
    main()
