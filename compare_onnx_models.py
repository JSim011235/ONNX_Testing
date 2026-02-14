import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import onnxruntime as ort

try:
	from torch.utils.data import DataLoader
	from torchvision import datasets, transforms
	except_import_error = None
except Exception as exc:  # pragma: no cover - only when torch/torchvision are missing
	except_import_error = exc
	DataLoader = None
	datasets = None
	transforms = None


@dataclass
class ModelResult:
	name: str
	size_mb: float
	accuracy: Optional[float]
	avg_ms: Optional[float]
	p95_ms: Optional[float]
	error: Optional[str]


def _get_model_size_mb(path: str) -> float:
	return os.path.getsize(path) / (1024 ** 2)


def _get_input_dtype(input_type: str) -> np.dtype:
	if "float16" in input_type:
		return np.float16
	if "float" in input_type:
		return np.float32
	if "int64" in input_type:
		return np.int64
	if "int32" in input_type:
		return np.int32
	if "int8" in input_type:
		return np.int8
	if "uint8" in input_type:
		return np.uint8
	return np.float32


def _load_test_data(
	data_dir: str,
	limit: Optional[int],
	batch_size: int,
	normalize: bool,
) -> List:
	if except_import_error is not None:
		raise RuntimeError(
			"torch/torchvision are required for accuracy evaluation. "
			"Install with: pip install torch torchvision"
		) from except_import_error

	steps = [transforms.ToTensor()]
	if normalize:
		steps.append(transforms.Normalize((0.5,), (0.5,)))
	transform = transforms.Compose(steps)
	test_data = datasets.FashionMNIST(
		root=data_dir,
		train=False,
		download=True,
		transform=transform,
	)
	loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
	if limit is None:
		return list(loader)

	limited = []
	for idx, batch in enumerate(loader):
		if idx >= limit:
			break
		limited.append(batch)
	return limited


def evaluate_model(
	model_path: str,
	model_label: str,
	data_dir: str,
	limit: Optional[int],
	batch_size: int,
	num_threads: Optional[int],
	use_cpu: bool,
) -> ModelResult:
	name = model_label
	size_mb = _get_model_size_mb(model_path)
	if not hasattr(ort, "SessionOptions") or not hasattr(ort, "InferenceSession"):
		return ModelResult(
			name=name,
			size_mb=size_mb,
			accuracy=None,
			avg_ms=None,
			p95_ms=None,
			error=(
				"onnxruntime is missing core APIs (SessionOptions/InferenceSession). "
				"Install onnxruntime or onnxruntime-gpu in the active environment."
			),
		)
	session_options = ort.SessionOptions()
	if num_threads:
		session_options.intra_op_num_threads = num_threads
		session_options.inter_op_num_threads = num_threads

	available = ort.get_available_providers()
	if use_cpu or "CUDAExecutionProvider" not in available:
		providers = ["CPUExecutionProvider"]
	else:
		providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

	try:
		session = ort.InferenceSession(
			model_path,
			sess_options=session_options,
			providers=providers,
		)
		provider_list = session.get_providers()
		provider_used = provider_list[0] if provider_list else "(unknown)"
		print(f"Using provider for {model_label}: {provider_used}")
		input_name = session.get_inputs()[0].name
		input_type = session.get_inputs()[0].type
		input_shape = session.get_inputs()[0].shape
		normalize = not (input_shape and len(input_shape) == 2)
		dtype = _get_input_dtype(input_type)
		batches = _load_test_data(data_dir, limit, batch_size, normalize)

		correct = 0
		total = 0
		timings_ms: List[float] = []

		for images, targets in batches:
			inputs = images.numpy().astype(dtype, copy=False)
			if input_shape and len(input_shape) == 2 and inputs.ndim > 2:
				inputs = inputs.reshape(inputs.shape[0], -1)
			start = time.perf_counter()
			outputs = session.run(None, {input_name: inputs})
			logits = outputs[0]
			for output in outputs:
				if hasattr(output, "ndim") and output.ndim >= 2:
					logits = output
					break
			elapsed_ms = (time.perf_counter() - start) * 1000.0
			timings_ms.append(elapsed_ms)
			if logits.ndim == 1:
				preds = logits.astype(np.int64, copy=False)
			else:
				preds = np.argmax(logits, axis=1)
			true = targets.numpy()
			correct += int((preds == true).sum())
			total += int(true.shape[0])

		accuracy = correct / total if total else None
		timings = np.array(timings_ms, dtype=np.float64)
		avg_ms = float(timings.mean()) if timings.size else None
		p95_ms = float(np.percentile(timings, 95)) if timings.size else None

		return ModelResult(
			name=name,
			size_mb=size_mb,
			accuracy=accuracy,
			avg_ms=avg_ms,
			p95_ms=p95_ms,
			error=None,
		)
	except Exception as exc:
		return ModelResult(
			name=name,
			size_mb=size_mb,
			accuracy=None,
			avg_ms=None,
			p95_ms=None,
			error=f"{type(exc).__name__}: {str(exc)[:120]}",
		)


def _print_results(results: List[ModelResult]) -> None:
	print("\n=== Model Comparison ===")
	name_width = max(32, max((len(result.name) for result in results), default=0))
	header = (
		f"{'Model':{name_width}} | {'Size(MB)':>8} | {'Acc':>7} | {'Avg(ms)':>8} | "
		f"{'P95(ms)':>8} | Status"
	)
	print(header)
	print("-" * len(header))
	for result in results:
		acc = f"{result.accuracy:.4f}" if result.accuracy is not None else "-"
		avg = f"{result.avg_ms:.3f}" if result.avg_ms is not None else "-"
		p95 = f"{result.p95_ms:.3f}" if result.p95_ms is not None else "-"
		status = "ok" if result.error is None else result.error
		print(
			f"{result.name:{name_width}} | "
			f"{result.size_mb:8.2f} | "
			f"{acc:>7} | "
			f"{avg:>8} | "
			f"{p95:>8} | "
			f"{status}"
		)


def _write_csv(results: List[ModelResult], output_path: str) -> None:
	with open(output_path, "w", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(["model", "size_mb", "accuracy", "avg_ms", "p95_ms", "status"])
		for result in results:
			writer.writerow(
				[
					result.name,
					f"{result.size_mb:.6f}",
					"" if result.accuracy is None else f"{result.accuracy:.6f}",
					"" if result.avg_ms is None else f"{result.avg_ms:.6f}",
					"" if result.p95_ms is None else f"{result.p95_ms:.6f}",
					"ok" if result.error is None else result.error,
				]
			)


def _should_skip_dir(dir_name: str, ignore_old_folders: bool) -> bool:
	if not ignore_old_folders:
		return False
	return dir_name.lower().startswith("old_")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compare speed, size, and accuracy for all ONNX models."
	)
	parser.add_argument(
		"--onnx-dir",
		type=str,
		default=None,
		help="Path to ONNX folder (default: <script>/onnx)",
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default=None,
		help="Path to dataset folder (default: <script>/data)",
	)
	parser.add_argument(
		"--threads",
		type=int,
		default=None,
		help="Number of CPU threads to use (default: auto)",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit number of samples (default: full test set)",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=64,
		help="Batch size for evaluation (default: 64)",
	)
	parser.add_argument(
		"--normalize",
		action="store_true",
		help="Apply mean/std normalization (0.5, 0.5)",
	)
	parser.add_argument(
		"--output-csv",
		type=str,
		default=None,
		help="Write results to CSV file",
	)
	parser.add_argument(
		"--use-cpu",
		action="store_true",
		default=False,
		help="Force CPU execution instead of GPU",
	)
	parser.add_argument(
		"--include-old-folders",
		action="store_true",
		help="Include folders under onnx/ starting with 'old_'",
	)
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.abspath(__file__))
	onnx_dir = args.onnx_dir or os.path.join(base_dir, "onnx")
	data_dir = args.data_dir or os.path.join(base_dir, "data")

	if not os.path.isdir(onnx_dir):
		raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

	model_paths: List[str] = []
	ignore_old_folders = not args.include_old_folders
	for root, dirs, files in os.walk(onnx_dir):
		if ignore_old_folders:
			dirs[:] = [d for d in dirs if not _should_skip_dir(d, True)]
		for name in files:
			if name.lower().endswith(".onnx"):
				model_paths.append(os.path.join(root, name))
	model_paths.sort()
	if not model_paths:
		raise FileNotFoundError(f"No .onnx files found in: {onnx_dir}")

	results: List[ModelResult] = []
	for model_path in model_paths:
		relative_label = os.path.relpath(model_path, onnx_dir)
		results.append(
			evaluate_model(
				model_path,
				relative_label,
				data_dir,
				args.limit,
				args.batch_size,
				args.threads,
				args.use_cpu,
			)
		)

	_print_results(results)
	if args.output_csv:
		_write_csv(results, args.output_csv)
		print(f"\nSaved CSV to: {args.output_csv}")


if __name__ == "__main__":
	main()
