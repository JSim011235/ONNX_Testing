import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_model():
	return nn.Sequential(
		nn.Conv2d(1, 32, kernel_size=3, padding=1),
		nn.ReLU(),
		nn.MaxPool2d(2),
		nn.Conv2d(32, 64, kernel_size=3, padding=1),
		nn.ReLU(),
		nn.MaxPool2d(2),
		nn.Flatten(),
		nn.Linear(64 * 7 * 7, 128),
		nn.ReLU(),
		nn.Dropout(0.3),
		nn.Linear(128, 10),
	)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
	model.train()
	total_loss = 0.0
	correct = 0
	total = 0
	for images, labels in loader:
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)

		optimizer.zero_grad()
		logits = model(images)
		loss = loss_fn(logits, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * labels.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

	avg_loss = total_loss / total
	acc = correct / total
	return avg_loss, acc


def evaluate(model, loader, loss_fn, device):
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			logits = model(images)
			loss = loss_fn(logits, labels)

			total_loss += loss.item() * labels.size(0)
			preds = logits.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)

	avg_loss = total_loss / total
	acc = correct / total
	return avg_loss, acc


def main():
	print(f"CUDA Available: {torch.cuda.is_available()}")
	print(f"GPU Name: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU detected")
	if not torch.cuda.is_available():
		raise RuntimeError(
			"CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA drivers."
		)
	device = torch.device("cuda:0")
	use_cuda = True
	if use_cuda:
		torch.backends.cudnn.benchmark = True

	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,)),
		]
	)

	train_data = datasets.FashionMNIST(
		root="./data",
		train=True,
		download=True,
		transform=transform,
	)
	test_data = datasets.FashionMNIST(
		root="./data",
		train=False,
		download=True,
		transform=transform,
	)

	train_loader = DataLoader(
		train_data,
		batch_size=128,
		shuffle=True,
		num_workers=2 if use_cuda else 0,
		pin_memory=use_cuda,
		persistent_workers=use_cuda,
	)
	test_loader = DataLoader(
		test_data,
		batch_size=512,
		num_workers=2 if use_cuda else 0,
		pin_memory=use_cuda,
		persistent_workers=use_cuda,
	)

	model = build_model().to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	epochs = 5
	for epoch in range(1, epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model, train_loader, optimizer, loss_fn, device
		)
		test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

		print(
			f"Epoch {epoch}/{epochs} | "
			f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
			f"test loss {test_loss:.4f} acc {test_acc:.4f}"
		)

	os.makedirs("onnx", exist_ok=True)
	model.eval()
	dummy_input = torch.randn(1, 1, 28, 28, device=device)
	# Export a portable model with a dynamic batch dimension.
	torch.onnx.export(
		model,
		dummy_input,
		"onnx/fashion_mnist_cnn.onnx",
		export_params=True,
		opset_version=13,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["logits"],
		dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
	)


if __name__ == "__main__":
	main()
