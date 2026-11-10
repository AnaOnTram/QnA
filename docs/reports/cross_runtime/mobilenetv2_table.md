| Deployment | Clean /100 | FGSM 1 | PGD 1 | PGD+ 1 | PGD 8 | Square 4 | Square 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [PyTorch FP32](../mobilenetv2/evasion_rknn/results/run/pytorch_fp32/summary.json) | 83 | 76/83 (91.6%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 38/83 (45.8%) | 71/83 (85.5%) |
| [TensorRT FP32](../mobilenetv2/evasion_trt/results/fp32/summary.json) | 83 | 76/83 (91.6%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 40/83 (48.2%) | 71/83 (85.5%) |
| [TensorRT FP16](../mobilenetv2/evasion_trt/results/fp16/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 25/83 (30.1%) | 63/83 (75.9%) |
| [TensorRT FP8](../mobilenetv2/evasion_trt/results/fp8/summary.json) | 84 | 72/84 (85.7%) | 84/84 (100.0%) | 84/84 (100.0%) | 84/84 (100.0%) | 21/84 (25.0%) | 46/84 (54.8%) |
| [TensorRT INT8](../mobilenetv2/evasion_trt/results/int8/summary.json) | 79 | 65/79 (82.3%) | 79/79 (100.0%) | 79/79 (100.0%) | 79/79 (100.0%) | 15/79 (19.0%) | 39/79 (49.4%) |
| [TensorRT INT4](../mobilenetv2/evasion_trt/results/int4/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 37/83 (44.6%) | 71/83 (85.5%) |
| [RKNN FP16](../mobilenetv2/evasion_rknn/results/run/rknn_fp16/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 31/83 (37.3%) | 70/83 (84.3%) |
| [RKNN INT8](../mobilenetv2/evasion_rknn/results/run/rknn_int8/summary.json) | 80 | 65/80 (81.2%) | 80/80 (100.0%) | 80/80 (100.0%) | 80/80 (100.0%) | 20/80 (25.0%) | 46/80 (57.5%) |
| [QNN HTP INT8](../mobilenetv2/evasion_qnn/results/run/qnn_htp/summary.json) | 78 | 60/78 (76.9%) | 77/78 (98.7%) | 77/78 (98.7%) | 78/78 (100.0%) | 22/78 (28.2%) | 42/78 (53.8%) |
| [Core ML FP32 CPU](../mobilenetv2/evasion_coreml/results/fp32_cpu_only/summary.json) | 83 | 76/83 (91.6%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 37/83 (44.6%) | 70/83 (84.3%) |
| [Core ML FP32 CPU+GPU](../mobilenetv2/evasion_coreml/results/fp32_cpu_and_gpu/summary.json) | 83 | 76/83 (91.6%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 37/83 (44.6%) | 70/83 (84.3%) |
| [Core ML FP32 CPU+NE](../mobilenetv2/evasion_coreml/results/fp32_cpu_and_ne/summary.json) | 83 | 76/83 (91.6%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 37/83 (44.6%) | 70/83 (84.3%) |
| [Core ML FP16 CPU](../mobilenetv2/evasion_coreml/results/fp16_cpu_only/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 22/83 (26.5%) | 52/83 (62.7%) |
| [Core ML FP16 CPU+GPU](../mobilenetv2/evasion_coreml/results/fp16_cpu_and_gpu/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 38/83 (45.8%) | 71/83 (85.5%) |
| [Core ML FP16 CPU+NE](../mobilenetv2/evasion_coreml/results/fp16_cpu_and_ne/summary.json) | 83 | 74/83 (89.2%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 33/83 (39.8%) | 61/83 (73.5%) |
| [Core ML INT8 CPU](../mobilenetv2/evasion_coreml/results/int8_cpu_only/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 23/83 (27.7%) | 54/83 (65.1%) |
| [Core ML INT8 CPU+GPU](../mobilenetv2/evasion_coreml/results/int8_cpu_and_gpu/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 36/83 (43.4%) | 70/83 (84.3%) |
| [Core ML INT8 CPU+NE](../mobilenetv2/evasion_coreml/results/int8_cpu_and_ne/summary.json) | 83 | 75/83 (90.4%) | 83/83 (100.0%) | 83/83 (100.0%) | 83/83 (100.0%) | 31/83 (37.3%) | 63/83 (75.9%) |
| [TFLite CPU INT8](../mobilenetv2/TPU/evasion/results/run/tflite_int8/summary.json) | 81 | 64/81 (79.0%) | 81/81 (100.0%) | 81/81 (100.0%) | 81/81 (100.0%) | 28/81 (34.6%) | 50/81 (61.7%) |
| [Coral USB INT8](../mobilenetv2/TPU/evasion/results/run/edgetpu_int8/summary.json) | 79 | 61/79 (77.2%) | 79/79 (100.0%) | 79/79 (100.0%) | 79/79 (100.0%) | 26/79 (32.9%) | 49/79 (62.0%) |
