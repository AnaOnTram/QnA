#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/ross/miniconda3/etc/profile.d/conda.sh
conda activate qairt
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /home/ross/qairt/2.42.0.251225/bin/envsetup.sh
cd "$ROOT"
mkdir -p logs results
/home/ross/QA/venv/bin/python prepare.py | tee logs/prepare.log
qairt-converter --input_network ../onnx/mobilenetv2.onnx \
  --source_model_input_shape input 1,3,224,224 \
  --output_path mobilenetv2_fp32.dlc 2>&1 | tee logs/converter.log
qairt-quantizer --input_dlc mobilenetv2_fp32.dlc \
  --input_list data/calib_list.txt --output_dlc mobilenetv2_int8.dlc \
  2>&1 | tee logs/quantizer.log
qairt-dlc-info --input_dlc mobilenetv2_fp32.dlc > results/fp32_dlc_info.txt
qairt-dlc-info --input_dlc mobilenetv2_int8.dlc > results/int8_dlc_info.txt
qnn-context-binary-generator --model libQnnModelDlc.so --backend libQnnHtp.so \
  --dlc_path mobilenetv2_int8.dlc --output_dir . \
  --binary_file mobilenetv2_quantized --config_file config_file.json \
  2>&1 | tee logs/context_generator.log
qnn-net-run --model libQnnModelDlc.so --backend libQnnCpu.so \
  --dlc_path mobilenetv2_fp32.dlc --input_list data/eval_list.txt \
  --output_dir results/fp32_cpu 2>&1 | tee logs/fp32_cpu.log
qnn-net-run --model libQnnModelDlc.so --backend libQnnCpu.so \
  --dlc_path mobilenetv2_int8.dlc --input_list data/eval_list.txt \
  --output_dir results/int8_cpu 2>&1 | tee logs/int8_cpu.log
sha256sum mobilenetv2_fp32.dlc mobilenetv2_int8.dlc mobilenetv2_quantized.bin > SHA256SUMS
/home/ross/QA/venv/bin/python validate.py --local-only | tee logs/host_validation.log
