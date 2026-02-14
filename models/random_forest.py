import os
import shutil

import numpy as np

# Note: Random Forest training on GPU is not practical on Windows
# sklearn's CPU implementation with n_jobs=-1 uses all cores efficiently

try:
	from sklearn.ensemble import RandomForestClassifier
	from skl2onnx import convert_sklearn
	from skl2onnx.common.data_types import FloatTensorType
	from torchvision import datasets
	except_import_error = None
except Exception as exc:  # pragma: no cover - only when deps are missing
	except_import_error = exc
	RandomForestClassifier = None
	convert_sklearn = None
	FloatTensorType = None
	datasets = None


# Training settings live in the model file.
N_ESTIMATORS = 100  # More trees = better accuracy
MAX_DEPTH = 20  # No limit = full tree growth
MIN_SAMPLES_SPLIT = 2  # Control overfitting
MIN_SAMPLES_LEAF = 1  # Smaller leaves = more detailed trees
TRAIN_SAMPLES = 60000  # Full dataset
TEST_SAMPLES = 2000
RANDOM_STATE = 42

# Training loop configurations
TRAINING_ITERATIONS = 1  # Reduced since CPU training is deterministic
PROGRESSIVE_TRAINING = False  # If True, trains with increasing data
HYPERPARAMETER_SEARCH = False  # If True, tries different hyperparameters


def _get_data_dir() -> str:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	return os.path.abspath(os.path.join(base_dir, "..", "data"))


def _load_mnist(limit: int) -> tuple[np.ndarray, np.ndarray]:
	if except_import_error is not None:
		raise RuntimeError(
			"scikit-learn, skl2onnx, torch, and torchvision are required. "
			"Install with: pip install scikit-learn skl2onnx torch torchvision"
		) from except_import_error

	data_dir = _get_data_dir()
	train_data = datasets.FashionMNIST(root=data_dir, train=True, download=True)
	images = train_data.data
	labels = train_data.targets
	if hasattr(images, "numpy"):
		images = images.numpy()
	if hasattr(labels, "numpy"):
		labels = labels.numpy()

	if limit is not None:
		images = images[:limit]
		labels = labels[:limit]

	features = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
	return features, labels.astype(np.int64)


def _load_mnist_test(limit: int) -> tuple[np.ndarray, np.ndarray]:
	if except_import_error is not None:
		raise RuntimeError(
			"scikit-learn, skl2onnx, torch, and torchvision are required. "
			"Install with: pip install scikit-learn skl2onnx torch torchvision"
		) from except_import_error

	data_dir = _get_data_dir()
	test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True)
	images = test_data.data
	labels = test_data.targets
	if hasattr(images, "numpy"):
		images = images.numpy()
	if hasattr(labels, "numpy"):
		labels = labels.numpy()

	if limit is not None:
		images = images[:limit]
		labels = labels[:limit]

	features = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
	return features, labels.astype(np.int64)



def _train_single_model(
	features: np.ndarray,
	labels: np.ndarray,
	test_features: np.ndarray,
	test_labels: np.ndarray,
	n_estimators: int,
	max_depth: int,
	iteration: int,
) -> tuple[RandomForestClassifier, float]:
	"""Train a single random forest model and return it with accuracy."""
	# sklearn RandomForest with n_jobs=-1 uses all CPU cores efficiently
	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		min_samples_split=MIN_SAMPLES_SPLIT,
		min_samples_leaf=MIN_SAMPLES_LEAF,
		n_jobs=-1,  # Use all CPU cores
		random_state=RANDOM_STATE + iteration,
		verbose=0,
	)
	
	if iteration == 0:
		print(f"    [Training with sklearn RandomForest: {n_estimators} trees, max_depth={max_depth}, using all CPU cores]")
	
	model.fit(features, labels)
	accuracy = model.score(test_features, test_labels)
	return model, accuracy


def train_and_export(
	output_dir: str,
	temp_models_dir: str,
) -> str:
	# Load full dataset once
	full_features, full_labels = _load_mnist(TRAIN_SAMPLES)
	test_features, test_labels = _load_mnist_test(TEST_SAMPLES)
	
	best_model = None
	best_accuracy = 0.0
	
	print(f"\n{'='*60}")
	print(f"Training Mode: CPU (sklearn RandomForest with all cores)")
	print(f"Training with {TRAINING_ITERATIONS} iterations")
	print(f"Dataset: {len(full_features):,} training samples, {len(test_features):,} test samples")
	print(f"{'='*60}\n")
	
	if PROGRESSIVE_TRAINING:
		# Train with increasing amounts of data
		print("Progressive Training: Using increasing data amounts\n")
		for i in range(TRAINING_ITERATIONS):
			# Use progressively more data: 20%, 40%, 60%, 80%, 100%
			data_fraction = (i + 1) / TRAINING_ITERATIONS
			n_samples = int(len(full_features) * data_fraction)
			features = full_features[:n_samples]
			labels = full_labels[:n_samples]
			
			print(f"Iteration {i+1}/{TRAINING_ITERATIONS} - Training with {n_samples:,} samples ({data_fraction*100:.0f}%)")
			model, accuracy = _train_single_model(
				features, labels, test_features, test_labels,
				N_ESTIMATORS, MAX_DEPTH, i
			)
			print(f"  → Accuracy: {accuracy:.4f}\n")
			
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_model = model
				
	elif HYPERPARAMETER_SEARCH:
		# Train with different hyperparameter combinations
		print("Hyperparameter Search: Testing different configurations\n")
		estimator_options = [50, 100, 150, 200, 250]
		depth_options = [5, 10, 15, 20, None]
		
		for i in range(min(TRAINING_ITERATIONS, len(estimator_options))):
			n_est = estimator_options[i % len(estimator_options)]
			depth = depth_options[i % len(depth_options)]
			
			print(f"Iteration {i+1}/{TRAINING_ITERATIONS} - n_estimators={n_est}, max_depth={depth}")
			model, accuracy = _train_single_model(
				full_features, full_labels, test_features, test_labels,
				n_est, depth, i
			)
			print(f"  → Accuracy: {accuracy:.4f}\n")
			
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_model = model
	else:
		# Train multiple models with same hyperparameters (different seeds)
		print("Multiple Training Runs: Same hyperparameters, different seeds\n")
		for i in range(TRAINING_ITERATIONS):
			print(f"Iteration {i+1}/{TRAINING_ITERATIONS} - n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
			model, accuracy = _train_single_model(
				full_features, full_labels, test_features, test_labels,
				N_ESTIMATORS, MAX_DEPTH, i
			)
			print(f"  → Accuracy: {accuracy:.4f}\n")
			
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_model = model
	
	print(f"{'='*60}")
	print(f"Best Test Accuracy: {best_accuracy:.4f}")
	print(f"{'='*60}\n")
	
	# Export the best model
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(temp_models_dir, exist_ok=True)
	onnx_path = os.path.join(output_dir, "model.onnx")
	temp_name = f"{os.path.basename(output_dir)}_temp.onnx"
	temp_onnx_path = os.path.join(temp_models_dir, temp_name)

	initial_type = [("input", FloatTensorType([None, full_features.shape[1]]))]
	onnx_model = convert_sklearn(best_model, initial_types=initial_type, target_opset=13)
	with open(temp_onnx_path, "wb") as handle:
		handle.write(onnx_model.SerializeToString())
	shutil.copy2(temp_onnx_path, onnx_path)
	return onnx_path


def main() -> None:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	onnx_dir = os.path.join(base_dir, "..", "onnx", "v3_rf")
	temp_dir = os.path.join(base_dir, "_temp")
	train_and_export(os.path.abspath(onnx_dir), os.path.abspath(temp_dir))
	print(f"Exported ONNX: {os.path.abspath(os.path.join(onnx_dir, 'model.onnx'))}")


if __name__ == "__main__":
	main()
