#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONVERSION_PYTHON="${CONVERSION_PYTHON:-/tmp/tf21/bin/python}"
EDGETPU_COMPILER="${EDGETPU_COMPILER:-$TASK_DIR/../../resnet50/TPU/tools/edgetpu_compiler_16.0/bin/edgetpu_compiler}"
export PATH="$(dirname -- "$CONVERSION_PYTHON"):$PATH"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
mkdir -p "$TASK_DIR/logs"
"$CONVERSION_PYTHON" "$TASK_DIR/prepare.py" > "$TASK_DIR/logs/prepare.log" 2>&1
mkdir -p "$TASK_DIR/conversion_input"
# onnx2tf runs onnxsim in place; preserve the validated prepared source.
cp "$TASK_DIR/mobilenetv2_raw_rgb_static.onnx" "$TASK_DIR/conversion_input/mobilenetv2_raw_rgb_static.onnx"
"$CONVERSION_PYTHON" -m onnx2tf \
    -i "$TASK_DIR/conversion_input/mobilenetv2_raw_rgb_static.onnx" \
    -o "$TASK_DIR/build" -b 1 -oiqt -tb tf_converter \
    -qt per-channel -iqd uint8 -oqd uint8 \
    -cind input "$TASK_DIR/data/calibration.npy" '[[[[0.0,0.0,0.0]]]]' '[[[[1.0,1.0,1.0]]]]' \
    > "$TASK_DIR/logs/conversion.log" 2>&1
cp "$TASK_DIR/build/mobilenetv2_raw_rgb_static_full_integer_quant.tflite" "$TASK_DIR/mobilenetv2_int8_nhwc.tflite"
"$EDGETPU_COMPILER" -s -o "$TASK_DIR" "$TASK_DIR/mobilenetv2_int8_nhwc.tflite" \
    > "$TASK_DIR/logs/compiler.log" 2>&1
cat "$TASK_DIR/logs/compiler.log"
