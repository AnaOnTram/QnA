# Artifact catalog

Each file is a standalone copy. SHA-256 digests, exact sizes, original locations and quantization details are in [manifest.json](../manifest.json). Sizes below use MiB (1,048,576 bytes).

| File | Runtime | Precision | MiB |
| --- | --- | --- | ---: |
| [weights/mobilenetv2/onnx/mobilenetv2_fp32.onnx](../weights/mobilenetv2/onnx/mobilenetv2_fp32.onnx) | ONNX Runtime / TensorRT | FP32 | 13.55 |
| [weights/mobilenetv2/pytorch/mobilenet_v2-b0353104.pth](../weights/mobilenetv2/pytorch/mobilenet_v2-b0353104.pth) | PyTorch | FP32 | 13.55 |
| [weights/mobilenetv2/rknn/mobilenetv2.rknn](../weights/mobilenetv2/rknn/mobilenetv2.rknn) | Rockchip RKNN Toolkit2 2.5.0 / RK3588 | FP16 | 7.20 |
| [weights/mobilenetv2/rknn/mobilenetv2_int8.rknn](../weights/mobilenetv2/rknn/mobilenetv2_int8.rknn) | Rockchip RKNN Toolkit2 2.5.0 / RK3588 | INT8 | 3.92 |
| [weights/mobilenetv2/rknn/mobilenetv2_opset19.onnx](../weights/mobilenetv2/rknn/mobilenetv2_opset19.onnx) | RKNN conversion source / ONNX Runtime | FP32 | 13.33 |
| [weights/mobilenetv2/tensorrt/mobilenetv2_fp16.onnx](../weights/mobilenetv2/tensorrt/mobilenetv2_fp16.onnx) | TensorRT | FP16 | 6.89 |
| [weights/mobilenetv2/tensorrt/mobilenetv2_fp8.onnx](../weights/mobilenetv2/tensorrt/mobilenetv2_fp8.onnx) | TensorRT | FP8 E4M3 Q/DQ | 6.80 |
| [weights/mobilenetv2/tensorrt/mobilenetv2_int4.onnx](../weights/mobilenetv2/tensorrt/mobilenetv2_int4.onnx) | TensorRT | INT4 weight-only, mixed precision | 9.10 |
| [weights/mobilenetv2/tensorrt/mobilenetv2_int8.onnx](../weights/mobilenetv2/tensorrt/mobilenetv2_int8.onnx) | TensorRT | INT8 Q/DQ | 6.88 |
| [weights/resnetv2_50/edgetpu/resnetv2_50_int8_nhwc_edgetpu.tflite](../weights/resnetv2_50/edgetpu/resnetv2_50_int8_nhwc_edgetpu.tflite) | Coral Edge TPU | INT8 with UINT8 I/O | 25.31 |
| [weights/resnetv2_50/onnx/resnetv2_50_fp16.onnx](../weights/resnetv2_50/onnx/resnetv2_50_fp16.onnx) | ONNX Runtime CUDA | FP16 | 48.92 |
| [weights/resnetv2_50/onnx/resnetv2_50_fp32.onnx](../weights/resnetv2_50/onnx/resnetv2_50_fp32.onnx) | ONNX Runtime | FP32 | 97.69 |
| [weights/resnetv2_50/onnx/resnetv2_50_int8_qoperator.onnx](../weights/resnetv2_50/onnx/resnetv2_50_int8_qoperator.onnx) | ONNX Runtime CPU | INT8 | 24.96 |
| [weights/resnetv2_50/pytorch/model.safetensors](../weights/resnetv2_50/pytorch/model.safetensors) | PyTorch/timm | FP32 | 97.67 |
| [weights/resnetv2_50/qnn/resnetv2_50_fp32.dlc](../weights/resnetv2_50/qnn/resnetv2_50_fp32.dlc) | Qualcomm QAIRT 2.42 | FP32 | 97.67 |
| [weights/resnetv2_50/qnn/resnetv2_50_int8.dlc](../weights/resnetv2_50/qnn/resnetv2_50_int8.dlc) | Qualcomm QNN HTP (QCS6490), QAIRT 2.42.0.251225 | INT8 | 24.55 |
| [weights/resnetv2_50/qnn/resnetv2_50_quantized.bin](../weights/resnetv2_50/qnn/resnetv2_50_quantized.bin) | Qualcomm QNN HTP (QCS6490), QAIRT 2.42.0.251225 | INT8 | 24.80 |
| [weights/resnetv2_50/rknn/resnetv2_50.rknn](../weights/resnetv2_50/rknn/resnetv2_50.rknn) | Rockchip RKNN NPU (RK3588), rknn-toolkit2 2.5.0 | FP16 | 50.07 |
| [weights/resnetv2_50/rknn/resnetv2_50_int8.rknn](../weights/resnetv2_50/rknn/resnetv2_50_int8.rknn) | Rockchip RKNN NPU (RK3588), rknn-toolkit2 2.5.0 | INT8 | 25.49 |
| [weights/resnetv2_50/tensorrt/resnetv2_50_int8_dq.onnx](../weights/resnetv2_50/tensorrt/resnetv2_50_int8_dq.onnx) | TensorRT | INT8 compressed Q/DQ | 24.96 |
| [weights/resnetv2_50/tensorrt/resnetv2_50_int8_dq.trt109.b10.plan](../weights/resnetv2_50/tensorrt/resnetv2_50_int8_dq.trt109.b10.plan) | TensorRT 10.9.0.34 | INT8 | 27.90 |
| [weights/resnetv2_50/tensorrt/resnetv2_50_int8_qdq.onnx](../weights/resnetv2_50/tensorrt/resnetv2_50_int8_qdq.onnx) | TensorRT | INT8 Q/DQ | 97.93 |
| [weights/resnetv2_50/tensorrt/resnetv2_50_int8_qdq.trt109.b10.plan](../weights/resnetv2_50/tensorrt/resnetv2_50_int8_qdq.trt109.b10.plan) | TensorRT 10.9.0.34 | INT8 | 27.85 |
| [weights/resnetv2_50/tflite/resnetv2_50_int8_nhwc.tflite](../weights/resnetv2_50/tflite/resnetv2_50_int8_nhwc.tflite) | TensorFlow Lite CPU | INT8 with UINT8 I/O | 25.07 |
