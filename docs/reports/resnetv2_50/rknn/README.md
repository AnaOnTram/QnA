# `rknn/` — local RKNN toolchain and vendored NPU source

| Item | Purpose |
| --- | --- |
| `rknn-venv/` | Minimal local venv for building `.rknn` models with rknn-toolkit2 2.5.0. **Not** a tracked artifact; ~680 MB. |
| `NPU/` | Vendored clone of <https://github.com/harishsdev/NPU> whose `resnet/` holds the conversion source (`resnetv2_50.onnx`, `dataset.txt`, `export_resnet_to_rknn.py`) and the pinned `.rknn` binaries referenced by `evasion_rknn/scripts/{review,validate}.py`. |
| `resnetv2_50.onnx` | Copy of the canonical timm ONNX (same bytes as `../onnx/resnetv2_50.onnx`). |

## Rebuilding the minimal venv

The previous environment was 5.7 GB because it carried a full CUDA stack
(torch 2.4.0+cu121, torchvision, transformers, and CUDA 12 **and** 13 wheels).
rknn-toolkit2 needs torch only for `RKNN.load_pytorch`; this project converts
from ONNX, so a CPU-only torch is sufficient. Rebuild with:

```bash
uv venv --python 3.12 rknn/rknn-venv
uv pip install --python rknn/rknn-venv/bin/python --no-deps rknn-toolkit2==2.5.0
uv pip install --python rknn/rknn-venv/bin/python \
    numpy==1.26.4 protobuf==4.25.4 psutil==7.2.2 ruamel.yaml==0.19.1 \
    scipy==1.17.1 tqdm==4.70.1 opencv-python==4.11.0.86 fast-histogram==0.14 \
    onnx==1.18.0 onnxruntime==1.26.0 setuptools==80.10.2
uv pip install --python rknn/rknn-venv/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match torch==2.4.0+cpu
```

Constraints worth remembering:

- `rknn-toolkit2` imports torch unconditionally, so torch cannot be omitted.
- `setuptools` must stay `<81`: rknn-toolkit2 imports `pkg_resources`, which
  setuptools 81+ removed.
- Do **not** install `torchvision`, `transformers`, `cuda-toolkit` or any
  `nvidia-*` CUDA wheels here; nothing in the conversion path uses them.

## Builds are not byte-reproducible

rknn-toolkit2 2.5.0 produces a deterministic-size but **not byte-identical**
`.rknn` file on every build: ~800 scattered bytes differ while the compiler
header, graph/attribute JSON and quantization tables are identical. Verified by
rebuilding the INT8 model twice with the same environment:

| Build | sha256 |
| --- | --- |
| pinned artifact (`NPU/resnet/resnetv2_50_int8.rknn`, 2026-11-04 11:20) | `f71ab0e1…` |
| rebuild, previous 5.7 GB env | `9ebf9fcd…` |
| rebuild, minimal env | `62e1625a…` |

All three are 26,731,147 bytes and share an identical header. Consequence:
rebuilt models will not match the hashes pinned in
[`../BASE_MODEL_UNIFICATION.md`](../BASE_MODEL_UNIFICATION.md); the retained
binaries remain the provenance reference, and a rebuild must be validated by
output equivalence (or a fresh board score), not by file hash.
