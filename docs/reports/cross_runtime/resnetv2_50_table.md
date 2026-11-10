| Deployment | Clean /100 | FGSM 1 | PGD 1 | PGD+ 1 | PGD 8 | Square 4 | Square 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [PyTorch FP32](../resnetv2_50/evasion_rknn/results/run/pytorch_fp32/summary.json) | 81 | 41/81 (50.6%) | 65/81 (80.2%) | 73/81 (90.1%) | 81/81 (100.0%) | 36/81 (44.4%) | 49/81 (60.5%) |
| [ORT CUDA FP32](../resnetv2_50/evasion_rknn/results/run/onnx_fp32/summary.json) | 81 | 41/81 (50.6%) | 65/81 (80.2%) | 74/81 (91.4%) | 81/81 (100.0%) | 37/81 (45.7%) | 49/81 (60.5%) |
| [RKNN FP16](../resnetv2_50/evasion_rknn/results/run/rknn_fp16/summary.json) | 81 | 41/81 (50.6%) | 65/81 (80.2%) | 74/81 (91.4%) | 81/81 (100.0%) | 32/81 (39.5%) | 50/81 (61.7%) |
| [RKNN INT8](../resnetv2_50/evasion_rknn/results/run/rknn_int8/summary.json) | 78 | 35/78 (44.9%) | 52/78 (66.7%) | 58/78 (74.4%) | 78/78 (100.0%) | 15/78 (19.2%) | 28/78 (35.9%) |
| [ORT CUDA FP16](../resnetv2_50/evasion_onnx/results/timm_rnv2_fp16/summary.json) | 82 | 42/82 (51.2%) | 65/82 (79.3%) | 76/82 (92.7%) | 82/82 (100.0%) | 31/82 (37.8%) | 51/82 (62.2%) |
| [ORT CPU INT8 QOperator](../resnetv2_50/evasion_onnx/results/timm_rnv2_int8_ort_qop/summary.json) | 78 | 27/78 (34.6%) | 48/78 (61.5%) | 58/78 (74.4%) | 76/78 (97.4%) | 18/78 (23.1%) | 31/78 (39.7%) |
| [TensorRT INT8 Q/DQ](../resnetv2_50/evasion_onnx/results/timm_rnv2_int8_ptq_ort/summary.json) | 79 | 32/79 (40.5%) | 50/79 (63.3%) | 57/79 (72.2%) | 77/79 (97.5%) | 22/79 (27.8%) | 34/79 (43.0%) |
| [TensorRT INT8 compressed](../resnetv2_50/evasion_onnx/results/timm_rnv2_int8_dq_ort/summary.json) | 79 | 32/79 (40.5%) | 50/79 (63.3%) | 58/79 (73.4%) | 77/79 (97.5%) | 22/79 (27.8%) | 34/79 (43.0%) |
| [QNN HTP INT8](inputs/resnetv2_50_qnn_summary.json) | 79 | 32/79 (40.5%) | 49/79 (62.0%) | 51/79 (64.6%) | 79/79 (100.0%) | 18/79 (22.8%) | 29/79 (36.7%) |
| [Coral USB INT8](../resnetv2_50/TPU/evasion/results/edgetpu_int8/summary.json) | 80 | 28/80 (35.0%) | 45/80 (56.2%) | Not run | 79/80 (98.8%) | 21/80 (26.2%) | 31/80 (38.8%) |
| [Core ML FP32 CPU](../resnetv2_50/evasion_coreml/results/fp32_cpu_only/summary.json) | 81 | 41/81 (50.6%) | 64/81 (79.0%) | 73/81 (90.1%) | 81/81 (100.0%) | 34/81 (42.0%) | 51/81 (63.0%) |
| [Core ML FP32 CPU+GPU](../resnetv2_50/evasion_coreml/results/fp32_cpu_and_gpu/summary.json) | 81 | 41/81 (50.6%) | 66/81 (81.5%) | 73/81 (90.1%) | 81/81 (100.0%) | 34/81 (42.0%) | 51/81 (63.0%) |
| [Core ML FP32 CPU+NE](../resnetv2_50/evasion_coreml/results/fp32_cpu_and_ne/summary.json) | 81 | 41/81 (50.6%) | 64/81 (79.0%) | 73/81 (90.1%) | 81/81 (100.0%) | 34/81 (42.0%) | 51/81 (63.0%) |
| [Core ML FP16 CPU](../resnetv2_50/evasion_coreml/results/fp16_cpu_only/summary.json) | 82 | 41/82 (50.0%) | 66/82 (80.5%) | 76/82 (92.7%) | 82/82 (100.0%) | 23/82 (28.0%) | 43/82 (52.4%) |
| [Core ML FP16 CPU+GPU](../resnetv2_50/evasion_coreml/results/fp16_cpu_and_gpu/summary.json) | 82 | 41/82 (50.0%) | 67/82 (81.7%) | 75/82 (91.5%) | 82/82 (100.0%) | 34/82 (41.5%) | 49/82 (59.8%) |
| [Core ML FP16 CPU+NE](../resnetv2_50/evasion_coreml/results/fp16_cpu_and_ne/summary.json) | 82 | 41/82 (50.0%) | 67/82 (81.7%) | 76/82 (92.7%) | 82/82 (100.0%) | 33/82 (40.2%) | 49/82 (59.8%) |
| [Core ML INT8 CPU](../resnetv2_50/evasion_coreml/results/int8_cpu_only/summary.json) | 81 | 37/81 (45.7%) | 63/81 (77.8%) | 75/81 (92.6%) | 80/81 (98.8%) | 24/81 (29.6%) | 43/81 (53.1%) |
| [Core ML INT8 CPU+GPU](../resnetv2_50/evasion_coreml/results/int8_cpu_and_gpu/summary.json) | 81 | 38/81 (46.9%) | 67/81 (82.7%) | 75/81 (92.6%) | 80/81 (98.8%) | 31/81 (38.3%) | 46/81 (56.8%) |
| [Core ML INT8 CPU+NE](../resnetv2_50/evasion_coreml/results/int8_cpu_and_ne/summary.json) | 81 | 38/81 (46.9%) | 65/81 (80.2%) | 74/81 (91.4%) | 81/81 (100.0%) | 28/81 (34.6%) | 49/81 (60.5%) |
