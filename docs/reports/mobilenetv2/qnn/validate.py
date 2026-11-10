#!/usr/bin/env python3
"""Compare DLC/HTP outputs to the pinned ONNX cohort; optionally deploy to Q6A."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
SDK = Path('/home/ross/qairt/2.42.0.251225')
REMOTE = '/home/ross/mobilenetv2_qairt/6490_v242'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deploy():
    # Dedicated model directory. Each run checks the deployed artifact hashes.
    files = {
        'qnn-net-run': SDK / 'bin/aarch64-ubuntu-gcc9.4/qnn-net-run',
        **{n: SDK / 'lib/aarch64-ubuntu-gcc9.4' / n for n in
           ('libQnnHtp.so', 'libQnnHtpV68Stub.so', 'libQnnSystem.so')},
        'libQnnHtpV68Skel.so': SDK / 'lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so',
        'mobilenetv2_quantized.bin': ROOT / 'mobilenetv2_quantized.bin',
        **{f'data/{p.name}': p for p in sorted((ROOT / 'data/evaluation').glob('*.raw'))},
    }
    hashes = {name: digest(path) for name, path in files.items()}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / 'eval_list.txt').write_text(''.join(f'data/{i:03d}.raw\n' for i in range(100)))
        (td / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in hashes.items()))
        files.update({n: td / n for n in ('eval_list.txt', 'SHA256SUMS')})
        bundle = td / 'deploy.tar'
        with tarfile.open(bundle, 'w') as tar:
            for name, path in files.items():
                tar.add(path, arcname=name)
        with bundle.open('rb') as stream:
            subprocess.run(['q6a', f'mkdir -p {REMOTE} && tar xf - -C {REMOTE}'], stdin=stream, check=True, timeout=120)
    command = (f'cd {REMOTE} && sha256sum -c SHA256SUMS && '
               'export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}" ADSP_LIBRARY_PATH="$PWD" && '
               './qnn-net-run --version && '
               './qnn-net-run --backend ./libQnnHtp.so --retrieve_context ./mobilenetv2_quantized.bin '
               '--input_list eval_list.txt --output_dir output_htp')
    run = subprocess.run(['q6a', command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    (ROOT / 'logs/board.log').write_bytes(run.stdout)
    run.check_returncode()
    result = ROOT / 'results/board_htp'
    result.mkdir(exist_ok=True)
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['q6a', f'cd {REMOTE}/output_htp && tar cf - .'], stdout=stream, check=True, timeout=120)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as tar:
            tar.extractall(result, filter='data')
    # Query device-side native output encoding rather than guessing it from ResNet.
    native_cmd = (f'cd {REMOTE} && export LD_LIBRARY_PATH="$PWD:${{LD_LIBRARY_PATH:-}}" ADSP_LIBRARY_PATH="$PWD" && '
                  "head -1 eval_list.txt > smoke_list.txt && "
                  './qnn-net-run --backend ./libQnnHtp.so --retrieve_context ./mobilenetv2_quantized.bin '
                  '--input_list smoke_list.txt --output_dir output_native --use_native_output_files')
    native = subprocess.run(['q6a', native_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (ROOT / 'logs/board_native.log').write_bytes(native.stdout)
    native.check_returncode()
    native_dir = ROOT / 'results/board_native'
    native_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['q6a', f'cd {REMOTE}/output_native && tar cf - .'], stdout=stream, check=True, timeout=60)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as tar:
            tar.extractall(native_dir, filter='data')
    (ROOT / 'results/deployment.json').write_text(json.dumps(dict(
        remote=REMOTE, files_sha256=hashes, all_remote_hashes_verified=True,
        sdk='2.42.0.251225', dsp_arch='v68', soc_id=35, vtcm_mb=2), indent=2) + '\n')


def load_outputs(folder, n):
    rows = []
    for i in range(n):
        row = np.fromfile(folder / f'Result_{i}/logits.raw', dtype=np.float32)
        assert row.shape == (1000,) and np.isfinite(row).all(), (folder, i, row.shape)
        rows.append(row)
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board', action='store_true', help='Deploy and validate on the existing Q6A SSH route')
    parser.add_argument('--local-only', action='store_true', help='Validate the host DLCs only')
    args = parser.parse_args()
    if args.board:
        deploy()
    ref = np.load(ROOT / 'results/reference.npz')
    baseline = ref['onnx_logits'].astype(np.float64)
    report = dict(n=len(baseline), onnx_correct=int((baseline.argmax(1) == ref['labels']).sum()), runtimes={})
    for name in ('fp32_cpu', 'int8_cpu', 'board_htp'):
        if name == 'board_htp' and args.local_only:
            continue
        path = ROOT / 'results' / name
        if not path.exists():
            continue
        if name == 'board_htp':
            deployment = json.loads((ROOT / 'results/deployment.json').read_text())
            assert deployment['files_sha256']['mobilenetv2_quantized.bin'] == digest(ROOT / 'mobilenetv2_quantized.bin'), 'Board evidence is stale; rerun with --board'
            metadata = (path / 'execution_metadata.yaml').read_text()
            assert 'inferences_completed: 100' in metadata
            assert 'graph_name: mobilenetv2_fp32' in metadata
            assert 'dimensions: [1,3,224,224]' in metadata
        outputs = load_outputs(path, len(baseline))
        np.save(ROOT / 'results' / f'{name}_logits.npy', outputs)
        x = outputs.astype(np.float64)
        cosine = (x * baseline).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(baseline, axis=1))
        report['runtimes'][name] = dict(correct=int((x.argmax(1) == ref['labels']).sum()),
            onnx_top1_agreement=int((x.argmax(1) == baseline.argmax(1)).sum()),
            logit_mae=float(np.abs(x-baseline).mean()), logit_max_abs=float(np.abs(x-baseline).max()),
            mean_cosine=float(cosine.mean()), all_finite=True)
        if name == 'fp32_cpu':
            assert report['runtimes'][name]['onnx_top1_agreement'] == len(baseline)
            np.testing.assert_allclose(x, baseline, atol=2e-4, rtol=2e-4)
    report['artifacts'] = {n: dict(sha256=digest(ROOT / n), bytes=(ROOT / n).stat().st_size)
                           for n in ('mobilenetv2_fp32.dlc', 'mobilenetv2_int8.dlc', 'mobilenetv2_quantized.bin')}
    if 'board_htp' in report['runtimes']:
        a = np.load(ROOT / 'results/board_htp_logits.npy')
        b = np.load(ROOT / 'results/int8_cpu_logits.npy')
        report['htp_vs_int8_cpu'] = dict(top1_agreement=int((a.argmax(1)==b.argmax(1)).sum()), logit_mae=float(np.abs(a-b).mean()))
        info = (ROOT / 'results/int8_dlc_info.txt').read_text()
        encodings = {}
        for name in ('input', 'logits'):
            row = [line for line in info.splitlines() if re.match(r'\|\s*' + name + r'\s*\|', line)][-1]
            scale, offset = re.search(r'scale ([\d.e+-]+), offset ([\d.e+-]+)', row).groups()
            encodings[name] = dict(scale=float(scale), offset=int(float(offset)))
        native = np.fromfile(ROOT / 'results/board_native/Result_0/logits_native.raw', dtype=np.uint8)
        assert native.shape == (1000,)
        encoding = encodings['logits']
        decoded = (native.astype(np.float32) + encoding['offset']) * np.float32(encoding['scale'])
        np.testing.assert_array_equal(decoded, a[0])
        report['io_encodings'] = encodings
        report['native_output_smoke_matches_float_output_exactly'] = True
    (ROOT / 'results/validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
