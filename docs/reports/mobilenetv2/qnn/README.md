# MobileNetV2 → QAIRT / QNN HTP

The subsequent [Q6A CPU/NPU performance, serving and evasion report](../evasion_qnn/REPORT.md)
contains board-local timing/RAM measurements, the running service and attack evaluations.

Converted and hardware validated on 10 November 2026 (HKT), following the
existing ResNet QAIRT workflow (`/home/ross/qairt/2.42.0.251225/examples/Models/resnet50_qairt_example/QUANTIZATION.md`; source workspace only).

| Artifact | Purpose | Size |
| --- | --- | ---: |
| `mobilenetv2_fp32.dlc` | FP32 conversion source; CPU validation | 13.42 MiB |
| `mobilenetv2_int8.dlc` | INT8 post-training quantized DLC | 3.44 MiB |
| `mobilenetv2_quantized.bin` | Compiled QCS6490 HTP context | 5.25 MiB |

The source is the existing `../onnx/mobilenetv2.onnx`, SHA-256
`9f2d5d94df76f044172dc28345bf975b83e24de1ca33e5892fe074d312d93a77`,
matching the existing TorchVision MobileNetV2 V1 checkpoint validation.
The supplied opset-20 ONNX converted directly. Its dynamic input is fixed to
`input [1,3,224,224]` for this deployment; output is `logits [1,1000]`.
The original ONNX and checkpoint are unchanged.

## Matching ResNet settings

- Existing conda `qairt` environment, Python 3.10; QAIRT **2.42.0.251225**.
- `qairt-converter` → FP32 DLC → `qairt-quantizer` → INT8 DLC →
  `qnn-context-binary-generator` with `libQnnModelDlc.so` and `libQnnHtp.so`.
- QCS6490 / Dragon Q6A, Hexagon **v68**, **SoC ID 35**, **2 MiB VTCM**.
  The backend graph name is `mobilenetv2_fp32`, including in the INT8 context.
- Exact same **20 ResNet QNN calibration raw files in the same order**,
  copied locally and checked by SHA-256. These are the original square center
  crops, bilinear resize 224, JPEG save/reload, normalized float32 NCHW.
- Default ResNet PTQ settings: 8-bit weights, activations and biases,
  asymmetric min-max encoding; per-channel/per-row quantization disabled.
  QAIRT's default dynamic 16-bit weight option remains enabled, as in ResNet.
  No AIMET overrides, tuning or alternative calibration set was introduced.

## Validation

The retained MobileNet 100-image Imagenette validation cohort uses RGB,
shorter-side resize 256 with bicubic/antialias, center crop 224, then ImageNet
normalization. Input image hashes are rechecked, and fresh ONNX logits match
the earlier reference bit-for-bit. There is no calibration/evaluation overlap
by source JPEG content hash.

| Runtime | Correct / 100 | ONNX top-1 agreement | Logit MAE | Mean cosine |
| --- | ---: | ---: | ---: | ---: |
| Original ONNX CPU | 83 | 100 | 0 | 1 |
| FP32 DLC, QNN CPU | 83 | 100 | 0.00000194 | 1.000000 |
| INT8 DLC, QNN CPU | 75 | 86 | 0.554362 | 0.971634 |
| INT8 context, Q6A HTP | **78** | **89** | 0.618993 | 0.964191 |

All outputs are finite `[1,1000]`. FP32 DLC maximum absolute logit difference
from ONNX is `1.91e-5`. The converter emitted operator-version warnings for
Clip and Reshape; successful FP32 numerical validation covers their behavior
on this cohort. Full logs are retained.

**The matching INT8 recipe loses five correct predictions relative to FP32 on
this sample.** HTP and INT8 CPU predictions agree on 90/100 images, so hardware
results take precedence. This is a ten-class pilot sample, not a full ImageNet
accuracy measurement. The original 20-image calibration recipe is retained;
accuracy optimization would require a separate build and comparison.

The context and all deployed runtime/input files were verified by SHA-256 on
the Q6A through the existing `q6a` SSH alias. Deployment is in
`/home/ross/mobilenetv2_qairt/6490_v242` on the board and uses matching QAIRT
2.42 runtime libraries. Board metadata confirms 100 completed inferences,
graph `mobilenetv2_fp32`, native UINT8 NCHW input and UINT8 output. A separate
native-output smoke check dequantizes bit-exactly to its float-mode result.

## Inputs and outputs

Default `qnn-net-run` consumes **ImageNet-normalized float32 NCHW** input:
RGB / 255, subtract `[0.485,0.456,0.406]`, divide by `[0.229,0.224,0.225]`.
It handles graph quantization and emits dequantized float32 logits.

For native I/O only, the DLC reports `real = (code + offset) * scale`:

| Tensor | Native type | Shape | Scale | Offset |
| --- | --- | --- | ---: | ---: |
| `input` | UINT8 | `[1,3,224,224]` | 0.018658448011 | -114 |
| `logits` | UINT8 | `[1,1000]` | 0.149840772152 | -71 |

Native input codes represent normalized values; raw RGB bytes are not valid
native input. The saved device output sidecar independently confirms its
UINT8 dtype, shape, scale (rounded to six decimals by the tool), and offset.

## Rebuild and deploy

From `/home/ross/QA`:

```bash
bash Models/mobilenetV2/qnn/convert.sh
/home/ross/QA/venv/bin/python Models/mobilenetV2/qnn/validate.py --board
```

The first command rebuilds all three artifacts and validates both DLCs on the
host. The second copies the context, matching runtime and evaluation raws to
the dedicated board directory, checks hashes and validates hardware outputs.
Run `validate.py` without flags to recheck saved host and hardware outputs.
Rebuilding updates artifacts/checksums; rerun board validation after a rebuild.

For inference on the board from its deployment directory:

```bash
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD"
./qnn-net-run --backend ./libQnnHtp.so \
  --retrieve_context ./mobilenetv2_quantized.bin \
  --input_list eval_list.txt --output_dir output_htp
```

## Evidence

- `SHA256SUMS`: the three build artifact digests.
- `convert.sh`, `config_*.json`: exact build commands and target settings.
- `results/source_validation.json`: source identity, calibration raw hashes,
  cohort identity and overlap check.
- `results/validation.json`: all accuracy/parity metrics and artifact hashes.
- `results/deployment.json`: deployed context, runtime and input file hashes.
- `results/{fp32,int8}_dlc_info.txt`: graph metadata, quantizer settings, I/O encodings.
- `results/{fp32_cpu,int8_cpu,board_htp}_logits.npy`: complete saved logits.
- `results/board_htp/execution_metadata.yaml`: device graph and execution metadata.
- `results/board_native/`: native output and device encoding sidecar.
- `logs/`: converter, quantizer, context build, host and device execution logs.
