import argparse
import importlib.util
import os
from typing import Any

from optimize_model import OnnxOptimizer


class TrainAndOptimizeRunner:
	@staticmethod
	def load_model_script(script_path: str) -> Any:
		spec = importlib.util.spec_from_file_location("user_model", script_path)
		if spec is None or spec.loader is None:
			raise RuntimeError(f"Unable to load model script: {script_path}")
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		if not hasattr(module, "train_and_export"):
			raise AttributeError("Model script must define train_and_export()")
		return module

	@classmethod
	def optimize_models(cls, fp32_model_path: str) -> None:
		output_dir = os.path.dirname(fp32_model_path)
		optimized_models = []
		opt1_fp32 = OnnxOptimizer.optimize_with_constant_folding(
			fp32_model_path, os.path.join(output_dir, "cf_fp32")
		)
		if opt1_fp32:
			optimized_models.append(opt1_fp32)

		opt3_fp32 = OnnxOptimizer.prune_model(
			fp32_model_path, os.path.join(output_dir, "p_fp32")
		)
		if opt3_fp32:
			optimized_models.append(opt3_fp32)

		is_tree = OnnxOptimizer.is_tree_model(fp32_model_path)
		if is_tree:

			# OnnxOptimizer.create_optimized_session(fp32_model_path)
			print("\nOptimization complete.")
			print(f"Outputs saved to: {output_dir}")
			if optimized_models:
				print("Generated models:")
				for path in optimized_models:
					print(f" - {path}")
			return

		fp16_model_path = OnnxOptimizer.convert_to_float16(
			fp32_model_path, os.path.join(output_dir, "converted")
		)
		if not fp16_model_path:
			fp16_model_path = fp32_model_path

		opt1_fp16 = OnnxOptimizer.optimize_with_constant_folding(
			fp16_model_path, os.path.join(output_dir, "cf_fp16")
		)
		if opt1_fp16:
			optimized_models.append(opt1_fp16)

		# opt2_int8 = OnnxOptimizer.quantize_model_dynamic(
		# 	fp32_model_path, os.path.join(output_dir, "q")
		# )
		# if opt2_int8:
		# 	optimized_models.append(opt2_int8)

		opt3_fp16 = OnnxOptimizer.prune_model(
			fp16_model_path, os.path.join(output_dir, "p_fp16")
		)
		if opt3_fp16:
			optimized_models.append(opt3_fp16)

		# if opt2_int8:
		# 	opt4_int8 = OnnxOptimizer.optimize_with_constant_folding(
		# 		opt2_int8, os.path.join(output_dir, "cf_int8")
		# 	)
		# 	if opt4_int8:
		# 		optimized_models.append(opt4_int8)

		# if opt2_int8:
		# 	opt5_int8 = OnnxOptimizer.prune_model(opt2_int8, os.path.join(output_dir, "p_int8"))
		# 	if opt5_int8:
		# 		optimized_models.append(opt5_int8)

		# OnnxOptimizer.create_optimized_session(fp16_model_path)

		print("\nOptimization complete.")
		print(f"Outputs saved to: {output_dir}")
		if optimized_models:
			print("Generated models:")
			for path in optimized_models:
				print(f" - {path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Train a model by name and export ONNX to a folder under onnx/."
	)
	parser.add_argument(
		"model_name",
		type=str,
		help="Model script name under models/ (without .py)",
	)
	parser.add_argument(
		"output_folder",
		type=str,
		help="Folder name under onnx/ to store outputs",
	)
	parser.add_argument(
		"--optimize",
		action="store_true",
		help="Run ONNX optimizations after export",
	)
	args = parser.parse_args()

	base_dir = os.path.dirname(os.path.abspath(__file__))
	model_script = os.path.join(base_dir, "models", f"{args.model_name}.py")
	if not os.path.exists(model_script):
		raise FileNotFoundError(f"Model script not found: {model_script}")

	onnx_root = os.path.join(base_dir, "onnx")
	model_dir = os.path.join(onnx_root, args.output_folder)
	os.makedirs(model_dir, exist_ok=True)

	model_module = TrainAndOptimizeRunner.load_model_script(model_script)
	models_dir = os.path.join(base_dir, "models", "_temp")
	os.makedirs(models_dir, exist_ok=True)

	fp32_model_path = model_module.train_and_export(
		output_dir=model_dir,
		temp_models_dir=models_dir,
	)
	print(f"Exported ONNX: {fp32_model_path}")

	if args.optimize:
		TrainAndOptimizeRunner.optimize_models(fp32_model_path)


if __name__ == "__main__":
	main()
