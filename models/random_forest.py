import os
import shutil

import numpy as np

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
N_ESTIMATORS = 200
MAX_DEPTH = None
TRAIN_SAMPLES = 20000
TEST_SAMPLES = 2000
RANDOM_STATE = 42


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



def train_and_export(
	output_dir: str,
	temp_models_dir: str,
) -> str:
	features, labels = _load_mnist(TRAIN_SAMPLES)
	model = RandomForestClassifier(
		n_estimators=N_ESTIMATORS,
		max_depth=MAX_DEPTH,
		n_jobs=-1,
		random_state=RANDOM_STATE,
	)
	model.fit(features, labels)

	if TEST_SAMPLES:
		test_features, test_labels = _load_mnist_test(TEST_SAMPLES)
		accuracy = model.score(test_features, test_labels)
		print(f"Test accuracy: {accuracy:.4f}")

	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(temp_models_dir, exist_ok=True)
	onnx_path = os.path.join(output_dir, "model.onnx")
	temp_name = f"{os.path.basename(output_dir)}_temp.onnx"
	temp_onnx_path = os.path.join(temp_models_dir, temp_name)

	initial_type = [("input", FloatTensorType([None, features.shape[1]]))]
	onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=13)
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
