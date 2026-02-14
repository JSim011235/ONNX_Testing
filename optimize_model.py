import argparse
import os
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


class OnnxOptimizer:
	@staticmethod
	def onnx_load(path):
		return onnx.load(path)

	@staticmethod
	def onnx_save(model, path):
		onnx.save(model, path)

	@staticmethod
	def get_model_size_mb(path):
		"""Get file size in MB."""
		return os.path.getsize(path) / (1024 ** 2)

	@staticmethod
	def is_tree_model(path: str) -> bool:
		try:
			model = onnx.load(path)
			expected = {"TreeEnsembleClassifier", "TreeEnsembleRegressor"}
			return any(node.op_type in expected for node in model.graph.node)
		except Exception:
			return False

	@classmethod
	def optimize_with_constant_folding(cls, model_path, output_path):
		"""Optimize ONNX model with constant folding and operator fusion."""
		print(f"\n=== ONNX Constant Folding & Fusion ===")
		print(f"Input:  {model_path} ({cls.get_model_size_mb(model_path):.2f} MB)")

		try:
			opt_model = cls.onnx_load(model_path)
			# Enable constant folding
			opt_passes = [
				"eliminate_nop_transpose",
				"eliminate_unused_initializer",
				"fuse_bn_into_conv",
				"fuse_add_bias_into_conv",
				"fuse_consecutive_squeezes",
				"fuse_consecutive_unsqueezes",
				"fuse_slice_into_constant_conv",
			]

			try:
				from onnx.optimizer import optimize

				opt_model = optimize(opt_model, opt_passes)
				print(f"Applied {len(opt_passes)} optimization passes")
			except ImportError:
				print("ONNX optimizer not available, using basic optimization")
				# Fallback: just save as-is (will help with quantization)
				pass

			optimized_model_path = output_path + "_optimized.onnx"
			cls.onnx_save(opt_model, optimized_model_path)

			size_after = cls.get_model_size_mb(optimized_model_path)
			ratio = (1 - size_after / cls.get_model_size_mb(model_path)) * 100
			print(f"Output: {optimized_model_path} ({size_after:.2f} MB)")
			if ratio > 0:
				print(f"Size reduction: {ratio:.1f}%")
			return optimized_model_path
		except Exception as e:
			print(f"Error: {e}")
			return None

	@classmethod
	def quantize_model_dynamic(cls, model_path, output_path):
		"""Quantize model to int8 (dynamic) for faster inference and smaller size."""
		print(f"\n=== Dynamic Int8 Quantization ===")
		print(f"Input:  {model_path} ({cls.get_model_size_mb(model_path):.2f} MB)")

		try:
			quantized_path = output_path + "_quantized_int8.onnx"
			quantize_dynamic(
				model_input=model_path,
				model_output=quantized_path,
				per_channel=True,
				weight_type=QuantType.QInt8,
			)
			size_before = cls.get_model_size_mb(model_path)
			size_after = cls.get_model_size_mb(quantized_path)
			ratio = (1 - size_after / size_before) * 100
			print(f"Output: {quantized_path} ({size_after:.2f} MB)")
			print(f"Size reduction: {ratio:.1f}%")
			return quantized_path
		except Exception as e:
			print(f"Error: {e}")
			return None

	@classmethod
	def prune_model(cls, model_path, output_path):
		"""Remove unused initializers and simplify graph."""
		print(f"\n=== Model Pruning ===")
		print(f"Input:  {model_path} ({cls.get_model_size_mb(model_path):.2f} MB)")

		try:
			model = cls.onnx_load(model_path)
			pruned_path = output_path + "_pruned.onnx"

			# Get all node inputs
			used_names = set()
			for node in model.graph.node:
				used_names.update(node.input)
				used_names.update(node.output)

			# Mark graph inputs/outputs as used
			for inp in model.graph.input:
				used_names.add(inp.name)
			for out in model.graph.output:
				used_names.add(out.name)

			# Remove unused initializers
			initializers_to_remove = []
			for init in model.graph.initializer:
				if init.name not in used_names:
					initializers_to_remove.append(init.name)

			for name in initializers_to_remove:
				for i, init in enumerate(model.graph.initializer):
					if init.name == name:
						del model.graph.initializer[i]
						break

			cls.onnx_save(model, pruned_path)
			size_before = cls.get_model_size_mb(model_path)
			size_after = cls.get_model_size_mb(pruned_path)
			ratio = (1 - size_after / size_before) * 100
			print(f"Output: {pruned_path} ({size_after:.2f} MB)")
			if ratio > 0.01:
				print(f"Size reduction: {ratio:.1f}%")
			else:
				print("No unused initializers found")
			return pruned_path
		except Exception as e:
			print(f"Error: {e}")
			return None

	@classmethod
	def convert_to_float16(cls, model_path, output_path):
		"""Convert ONNX model weights to float16 with Cast nodes for I/O (no PyTorch)."""
		print(f"\n=== Float16 Conversion ===")
		print(f"Input:  {model_path} ({cls.get_model_size_mb(model_path):.2f} MB)")

		try:
			model = cls.onnx_load(model_path)
			if len(model.graph.output) != 1 or not model.graph.node:
				print("Skipping FP16 conversion: unsupported graph outputs")
				return None
			model_output_name = model.graph.output[0].name
			last_node = model.graph.node[-1]
			if not last_node.output or model_output_name not in last_node.output:
				print("Skipping FP16 conversion: output is not last node output")
				return None
			fp16_path = output_path + "_fp16.onnx"

			# Step 1: Convert FLOAT initializers to FLOAT16
			converted_count = 0
			for i, initializer in enumerate(model.graph.initializer):
				if initializer.data_type == TensorProto.FLOAT:
					arr = numpy_helper.to_array(initializer).astype(np.float16)
					new_init = numpy_helper.from_array(arr, initializer.name)
					model.graph.initializer[i].CopyFrom(new_init)
					converted_count += 1
			if converted_count == 0:
				print("Skipping FP16 conversion: no float initializers found")
				return None

			# Step 2: Add Cast node for input: float32 -> float16 (first input only)
			if model.graph.input:
				model_input_name = model.graph.input[0].name
				cast_input_name = model_input_name + "_fp16"

				cast_input_node = helper.make_node(
					"Cast",
					inputs=[model_input_name],
					outputs=[cast_input_name],
					to=TensorProto.FLOAT16,
					name="cast_input_to_fp16",
				)

				# Redirect first node input to casted tensor
				if model.graph.node and model.graph.node[0].input:
					model.graph.node[0].input[0] = cast_input_name

				model.graph.node.insert(0, cast_input_node)

			# Step 3: Add Cast node for output: float16 -> float32 (first output only)
			if model.graph.output and model.graph.node:
				last_output_name = last_node.output[0]
				cast_output_node = helper.make_node(
					"Cast",
					inputs=[last_output_name],
					outputs=[model_output_name],
					to=TensorProto.FLOAT,
					name="cast_output_to_fp32",
				)
				# Ensure last node outputs the fp16 value
				last_node.output[0] = model_output_name + "_fp16"
				cast_output_node.input[0] = last_node.output[0]
				model.graph.node.append(cast_output_node)

			cls.onnx_save(model, fp16_path)

			size_after = cls.get_model_size_mb(fp16_path)
			ratio = (1 - size_after / cls.get_model_size_mb(model_path)) * 100
			print(f"Output: {fp16_path} ({size_after:.2f} MB)")
			print(f"Converted initializers: {converted_count}")
			if ratio > 0:
				print(f"Size reduction: {ratio:.1f}%")

			return fp16_path
		except Exception as e:
			print(f"Error: {e}")
			return None

	@staticmethod
	def create_optimized_session(model_path):
		"""Create ONNX Runtime session with graph optimization enabled."""
		print(f"\n=== ONNX Runtime Graph Optimization (Session-level) ===")
		session_options = ort.SessionOptions()
		session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
		session_options.log_verbosity_level = 2

		session = ort.InferenceSession(model_path, sess_options=session_options)
		print("Session created with full graph optimization enabled")
		return session

	@staticmethod
	def _make_dummy_input(inp):
		"""Create a dummy input tensor based on ONNX Runtime input metadata."""
		shape = []
		for d in inp.shape:
			if isinstance(d, int) and d > 0:
				shape.append(d)
			else:
				shape.append(1)

		onnx_type = inp.type
		if "float16" in onnx_type:
			dtype = np.float16
		elif "float" in onnx_type:
			dtype = np.float32
		elif "int64" in onnx_type:
			dtype = np.int64
		elif "int32" in onnx_type:
			dtype = np.int32
		elif "int8" in onnx_type:
			dtype = np.int8
		elif "uint8" in onnx_type:
			dtype = np.uint8
		else:
			# Default safe fallback
			dtype = np.float32

		if np.issubdtype(dtype, np.floating):
			return np.random.randn(*shape).astype(dtype)
		return np.random.randint(0, 2, size=shape, dtype=dtype)

	@classmethod
	def benchmark_model(cls, model_path, num_runs=100):
		"""Benchmark model inference speed."""
		import time

		try:
			session = ort.InferenceSession(model_path)

			# Build inputs based on model metadata (supports FP16)
			feed = {}
			for inp in session.get_inputs():
				feed[inp.name] = cls._make_dummy_input(inp)

			# Warmup
			for _ in range(10):
				session.run(None, feed)

			# Benchmark
			times = []
			for _ in range(num_runs):
				start = time.perf_counter()
				session.run(None, feed)
				times.append((time.perf_counter() - start) * 1000)

			times = np.array(times)
			return {
				"min_ms": times.min(),
				"avg_ms": times.mean(),
				"max_ms": times.max(),
				"p95_ms": np.percentile(times, 95),
			}
		except Exception as e:
			print(f"  Benchmark failed: {type(e).__name__}: {str(e)[:50]}")
			return None

	@classmethod
	def compare_models(cls, original_path, optimized_paths):
		"""Compare original vs optimized models."""
		print(f"\n=== Model Comparison ===")
		print(f"Original: {original_path} ({cls.get_model_size_mb(original_path):.2f} MB)")

		results = {}
		for opt_path in optimized_paths:
			if not os.path.exists(opt_path):
				continue

			name = os.path.basename(opt_path).replace(".onnx", "")
			size = cls.get_model_size_mb(opt_path)
			bench = cls.benchmark_model(opt_path, num_runs=50)

			results[name] = {
				"size_mb": size,
				"benchmark": bench,
			}

			print(f"\n{name}:")
			print(f"  Size: {size:.2f} MB")
			if bench:
				print(f"  Latency (ms): avg={bench['avg_ms']:.3f}, p95={bench['p95_ms']:.3f}")
			else:
				print("  Latency: Could not benchmark (model may be incompatible)")

	@classmethod
	def main(cls) -> None:
		print("=" * 60)
		print("ONNX Model Optimization Tool")
		print("=" * 60)

		parser = argparse.ArgumentParser(description="Optimize an ONNX model.")
		parser.add_argument(
			"--model",
			type=str,
			default=None,
			help="Path to the FP32 ONNX model (default: <script>/onnx/fashion_mnist_cnn.onnx)",
		)
		parser.add_argument(
			"--enable-int8",
			action="store_true",
			help="Enable INT8 quantization and INT8 variants",
		)
		args = parser.parse_args()

		base_dir = os.path.dirname(os.path.abspath(__file__))
		default_model = os.path.join(base_dir, "onnx", "fashion_mnist_cnn.onnx")
		fp32_model_path = args.model or default_model
		if not os.path.isabs(fp32_model_path):
			fp32_model_path = os.path.join(base_dir, fp32_model_path)
		fp32_model_path = os.path.abspath(fp32_model_path)
		output_dir = os.path.dirname(fp32_model_path)

		if not os.path.exists(fp32_model_path):
			print(f"Error: FP32 model not found at {fp32_model_path}")
			return

		os.makedirs(output_dir, exist_ok=True)
		optimized_models = []
		is_tree = cls.is_tree_model(fp32_model_path)
		if is_tree:
			opt1_fp32 = cls.optimize_with_constant_folding(
				fp32_model_path, os.path.join(output_dir, "cf_fp32")
			)
			if opt1_fp32:
				optimized_models.append(opt1_fp32)

			opt3_fp32 = cls.prune_model(fp32_model_path, os.path.join(output_dir, "p_fp32"))
			if opt3_fp32:
				optimized_models.append(opt3_fp32)

			cls.create_optimized_session(fp32_model_path)
			if optimized_models:
				cls.compare_models(fp32_model_path, [fp32_model_path] + optimized_models)
			print("\n" + "=" * 60)
			print("Optimization complete!")
			print(f"Files saved to: {output_dir}")
			print("=" * 60)
			return

		# Method 0: Direct FP16 conversion (no PyTorch)
		fp16_model_path = cls.convert_to_float16(
			fp32_model_path, os.path.join(output_dir, "converted")
		)
		if not fp16_model_path:
			fp16_model_path = fp32_model_path
		model_path = fp16_model_path

		# Method 1: Constant folding & fusion (FP16)
		opt1_fp16 = cls.optimize_with_constant_folding(
			fp16_model_path, os.path.join(output_dir, "cf_fp16")
		)
		if opt1_fp16:
			optimized_models.append(opt1_fp16)

		opt2_int8 = None
		if args.enable_int8:
			# Method 2: Dynamic Int8 Quantization (use FP32 to avoid invalid INT8 graphs)
			opt2_int8 = cls.quantize_model_dynamic(
				fp32_model_path,
				os.path.join(output_dir, "q"),
			)
			if opt2_int8:
				optimized_models.append(opt2_int8)

		# Method 3: Model Pruning (FP16)
		opt3_fp16 = cls.prune_model(fp16_model_path, os.path.join(output_dir, "p_fp16"))
		if opt3_fp16:
			optimized_models.append(opt3_fp16)

		# Method 4: Constant folding & fusion (INT8)
		if opt2_int8:
			opt4_int8 = cls.optimize_with_constant_folding(
				opt2_int8, os.path.join(output_dir, "cf_int8")
			)
			if opt4_int8:
				optimized_models.append(opt4_int8)

		# Method 5: Model Pruning (INT8)
		if opt2_int8:
			opt5_int8 = cls.prune_model(opt2_int8, os.path.join(output_dir, "p_int8"))
			if opt5_int8:
				optimized_models.append(opt5_int8)

		# Method 6: Session-level optimization (stored as original, but optimized at runtime)
		cls.create_optimized_session(model_path)

		# Compare all models
		if optimized_models:
			cls.compare_models(model_path, [model_path] + optimized_models)

		print("\n" + "=" * 60)
		print("Optimization complete!")
		print(f"Files saved to: {output_dir}")
		print("=" * 60)


def onnx_load(path):
	return OnnxOptimizer.onnx_load(path)


def onnx_save(model, path):
	return OnnxOptimizer.onnx_save(model, path)


def get_model_size_mb(path):
	return OnnxOptimizer.get_model_size_mb(path)


def optimize_with_constant_folding(model_path, output_path):
	return OnnxOptimizer.optimize_with_constant_folding(model_path, output_path)


def quantize_model_dynamic(model_path, output_path):
	return OnnxOptimizer.quantize_model_dynamic(model_path, output_path)


def prune_model(model_path, output_path):
	return OnnxOptimizer.prune_model(model_path, output_path)


def convert_to_float16(model_path, output_path):
	return OnnxOptimizer.convert_to_float16(model_path, output_path)


def create_optimized_session(model_path):
	return OnnxOptimizer.create_optimized_session(model_path)


def benchmark_model(model_path, num_runs=100):
	return OnnxOptimizer.benchmark_model(model_path, num_runs=num_runs)


def compare_models(original_path, optimized_paths):
	return OnnxOptimizer.compare_models(original_path, optimized_paths)


def main():
	OnnxOptimizer.main()


if __name__ == "__main__":
	main()
