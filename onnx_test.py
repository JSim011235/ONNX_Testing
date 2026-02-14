import time
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
	model_path = "onnx/fashion_mnist_cnn.onnx"
	providers = ["CPUExecutionProvider"]
	session = ort.InferenceSession(model_path, providers=providers)

	input_name = session.get_inputs()[0].name

	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,)),
		]
	)
	# Use real Fashion-MNIST test images.
	test_data = datasets.FashionMNIST(
		root="./data",
		train=False,
		download=True,
		transform=transform,
	)
	test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

	labels = [
		"T-shirt/top",
		"Trouser",
		"Pullover",
		"Dress",
		"Coat",
		"Sandal",
		"Shirt",
		"Sneaker",
		"Bag",
		"Ankle boot",
	]

	correct = 0
	total = 0

	timings_ms = []
	for images, targets in test_loader:
		inputs = images.numpy().astype(np.float32)
		start = time.perf_counter()
		logits = session.run(None, {input_name: inputs})[0]
		elapsed_ms = (time.perf_counter() - start) * 1000.0
		timings_ms.append(elapsed_ms)
		pred = int(np.argmax(logits, axis=1)[0])
		true = int(targets.item())
		correct += int(pred == true)
		total += 1

	acc = correct / total if total else 0.0
	print(f"Test accuracy: {acc:.4f} ({correct}/{total})")

	timings = np.array(timings_ms, dtype=np.float64)
	if timings.size:
		p50 = np.percentile(timings, 50)
		p95 = np.percentile(timings, 95)
		print(
			"Timing (ms) | "
			f"min {timings.min():.3f} | "
			f"p50 {p50:.3f} | "
			f"p95 {p95:.3f} | "
			f"max {timings.max():.3f} | "
			f"avg {timings.mean():.3f}"
		)


if __name__ == "__main__":
	main()
