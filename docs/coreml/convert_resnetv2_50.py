#!/usr/bin/env python3
"""Export the pinned timm checkpoint and validate saved Core ML packages.

Run with the coreml conda environment on macOS; see README.md in this folder.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import coremltools as ct
import numpy as np
from safetensors.torch import load_file
import timm
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "b809e8c09f4a1cf2168adf089c86c17fe88521b85455db662c0b2d42f489a61b"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(checkpoint):
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint does not match the repository's pinned pretrained weights")
    model = timm.create_model("resnetv2_50", pretrained=False).eval()
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    return model


def annotate(model, precision):
    model.short_description = f"timm resnetv2_50.a1h_in1k; {precision}; 1000 ImageNet logits"
    model.input_description["image"] = "Externally normalized RGB NCHW [1,3,224,224], float32"
    model.output_description["logits"] = "Unnormalized ImageNet-1k class logits, float32 [1,1000]"
    model.user_defined_metadata.update({
        "architecture": "timm/resnetv2_50.a1h_in1k",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "precision": precision,
        "preprocessing": "External. Archive contract: RGB resize short side 232 bilinear, center crop 224, /255, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].",
        "native_timm_preprocessing": "Also accepted as normalized tensors: bicubic resize/crop using timm config, mean=std=[0.5,0.5,0.5]. Choose the recipe appropriate for your experiment; normalization is not embedded.",
        "minimum_os": "iOS 18 / macOS 15",
        "coremltools": ct.__version__,
        "timm": timm.__version__,
        "torch": torch.__version__,
    })


def export(model, output):
    output.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, check_trace=True)
    base_fp16 = None
    for name, precision in (("fp32", ct.precision.FLOAT32), ("fp16", ct.precision.FLOAT16)):
        print(f"Converting {name}", flush=True)
        converted = ct.convert(
            traced,
            source="pytorch",
            convert_to="mlprogram",
            inputs=[ct.TensorType(name="image", shape=example.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name="logits", dtype=np.float32)],
            compute_precision=precision,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
        )
        annotate(converted, name.upper())
        converted.save(str(output / f"resnetv2_50_{name}.mlpackage"))
        if name == "fp16":
            base_fp16 = converted

    for dtype in ("int8", "int4"):
        print(f"Compressing weights to {dtype}", flush=True)
        # Only convolution/linear weights are quantized. Norm parameters/biases
        # stay FP16. The RGB stem has 3 input channels, so use per-channel INT4.
        # Other conv input channel counts are divisible by block size 32.
        config = ct.optimize.coreml.OptimizationConfig(
            op_type_configs={
                op_type: ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype=dtype,
                    granularity="per_channel" if dtype == "int8" else "per_block",
                    block_size=32, weight_threshold=0,
                ) for op_type in ("conv", "linear")
            }
        )
        if dtype == "int4":
            for function in base_fp16._mil_program.functions.values():
                for op in function.operations:
                    if op.op_type in ("conv", "linear") and op.weight.shape[1] % 32:
                        config.set_op_name(op.name, ct.optimize.coreml.OpLinearQuantizerConfig(
                            dtype=dtype, granularity="per_channel", weight_threshold=0,
                        ))
        quantized = ct.optimize.coreml.linear_quantize_weights(base_fp16, config)
        annotate(quantized, f"{dtype.upper()} weights / FP16 compute")
        quantized.save(str(output / f"resnetv2_50_{dtype}.mlpackage"))


def samples(photo_paths):
    rng = np.random.default_rng(20261110)
    y, x = np.mgrid[0:224, 0:224].astype(np.float32) / 223
    pixels = [
        ("black", np.zeros((3, 224, 224), dtype=np.float32)),
        ("gray", np.full((3, 224, 224), 0.5, dtype=np.float32)),
        ("white", np.ones((3, 224, 224), dtype=np.float32)),
        ("gradient", np.stack((x, y, (x + y) / 2))),
        ("checkerboard", np.tile(((np.indices((224, 224)).sum(axis=0) // 16) % 2)[None], (3, 1, 1)).astype(np.float32)),
    ]
    pixels.extend((f"noise_{i}", rng.random((3, 224, 224), dtype=np.float32)) for i in range(3))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    inputs = [(name, np.ascontiguousarray(((rgb - mean) / std)[None])) for name, rgb in pixels]
    sources = []
    archive_transform = transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean.flatten().tolist(), std.flatten().tolist()),
    ])
    native_cfg = timm.models.get_pretrained_cfg("resnetv2_50").to_dict()
    native_transform = timm.data.create_transform(**timm.data.resolve_data_config(native_cfg), is_training=False)
    for i, path in enumerate(photo_paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            for recipe, transform in (("archive", archive_transform), ("native_timm", native_transform)):
                inputs.append((f"photo_{i}_{recipe}", transform(image).unsqueeze(0).numpy()))
        sources.append({"sample_prefix": f"photo_{i}", "filename": path.name, "sha256": sha256(path)})
    return inputs, sources


def structure(spec):
    ops = [op for fn in spec.mlProgram.functions.values()
           for block in fn.block_specializations.values() for op in block.operations]
    dtype_counts = Counter()
    float_output_dtypes = Counter()
    compressed_elements = 0
    for op in ops:
        for out in op.outputs:
            dtype = ct.proto.MIL_pb2.DataType.Name(out.type.tensorType.dataType)
            if dtype.startswith("FLOAT"):
                float_output_dtypes[dtype] += 1
        if op.type == "constexpr_blockwise_shift_scale":
            argument = op.inputs["data"].arguments[0]
            if argument.HasField("value"):
                tensor = argument.value.type.tensorType
            else:
                const = next(item for item in ops if any(out.name == argument.name for out in item.outputs))
                tensor = const.attributes["val"].type.tensorType
            dtype_counts[ct.proto.MIL_pb2.DataType.Name(tensor.dataType)] += 1
            compressed_elements += int(np.prod([d.constant.size for d in tensor.dimensions]))
    return {
        "operations": dict(Counter(op.type for op in ops)),
        "floating_operation_output_dtypes": dict(float_output_dtypes),
        "compressed_weight_tensor_dtypes": dict(dtype_counts),
        "compressed_weight_elements": compressed_elements,
        "specification_version": spec.specificationVersion,
    }


def validate(model, output, report_path, photo_paths):
    inputs, photo_sources = samples(photo_paths)
    with torch.no_grad():
        expected = np.concatenate([model(torch.from_numpy(value)).numpy() for _, value in inputs]).astype(np.float64)
    report = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "timm": timm.__version__, "coremltools": ct.__version__},
        "input": {"name": "image", "shape": [1, 3, 224, 224], "dtype": "float32", "normalization_embedded": False},
        "output": {"name": "logits", "shape": [1, 1000], "dtype": "float32"},
        "validation": {"compute_units": "CPU_ONLY", "samples": [name for name, _ in inputs],
                       "photo_sources": photo_sources,
                       "limitation": "Smoke/parity tests on synthetic inputs and bundled photos; no labeled ImageNet accuracy or hardware performance measurement."},
        "artifacts": {},
    }
    for precision in ("fp32", "fp16", "int8", "int4"):
        package = output / f"resnetv2_50_{precision}.mlpackage"
        print(f"Loading and predicting saved {package.name}", flush=True)
        mlmodel = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
        predictions = []
        for _, value in inputs:
            actual = mlmodel.predict({"image": value})["logits"]
            assert actual.shape == (1, 1000), actual.shape
            assert np.isfinite(actual).all(), f"Non-finite {precision} output"
            predictions.append(actual)
        actual = np.concatenate(predictions).astype(np.float64)
        difference = actual - expected
        cosine = np.sum(actual * expected, axis=1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
        graph = structure(mlmodel.get_spec())
        if precision in ("int8", "int4"):
            assert graph["compressed_weight_elements"] > 25_000_000, graph
            assert set(graph["compressed_weight_tensor_dtypes"]) == {precision.upper()}, graph
            assert graph["operations"]["constexpr_blockwise_shift_scale"] == graph["operations"].get("conv", 0) + graph["operations"].get("linear", 0), graph
        if precision == "fp32":
            assert set(graph["floating_operation_output_dtypes"]) == {"FLOAT32"}, graph
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
        metrics = {
            "max_absolute_logit_error": float(np.abs(difference).max()),
            "mean_absolute_logit_error": float(np.abs(difference).mean()),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "mean_cosine_similarity": float(cosine.mean()),
            "top1_agreement": float(np.mean(actual.argmax(axis=1) == expected.argmax(axis=1))),
        }
        per_sample = [{"name": name, "pytorch_top1": int(expected[i].argmax()),
                       "coreml_top1": int(actual[i].argmax()),
                       "cosine_similarity": float(cosine[i]),
                       "max_absolute_logit_error": float(np.abs(difference[i]).max())}
                      for i, (name, _) in enumerate(inputs)]
        files = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                  "sha256": sha256(path)} for path in sorted(package.rglob("*")) if path.is_file()]
        report["artifacts"][precision] = {"path": str(package.relative_to(ROOT)),
            "bytes": sum(item["bytes"] for item in files), "metrics_vs_pytorch_fp32": metrics,
            "per_sample": per_sample, "graph": graph, "files": files}
        print(json.dumps({"precision": precision, **metrics, "bytes": report["artifacts"][precision]["bytes"]}), flush=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: four saved Core ML packages, {len(inputs)} predictions each; report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/resnetv2_50/pytorch/model.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/resnetv2_50/coreml")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/coreml/validation.json")
    parser.add_argument("--verify-only", action="store_true")
    bundled_photo = Path(importlib.util.find_spec("matplotlib").origin).parent / "mpl-data/sample_data/grace_hopper.jpg"
    parser.add_argument("--validation-image", type=Path, action="append",
                        default=None, help="Repeat for additional parity-check photos; defaults to Matplotlib's bundled grace_hopper.jpg")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_model(args.checkpoint)
    if not args.verify_only:
        export(model, args.output)
    photos = args.validation_image if args.validation_image is not None else ([bundled_photo] if bundled_photo.exists() else [])
    validate(model, args.output, args.report, photos)


if __name__ == "__main__":
    main()
