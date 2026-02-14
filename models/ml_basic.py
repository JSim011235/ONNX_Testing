import os
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MlBasicTrainer:
	EPOCHS = 5
	BATCH_SIZE = 128
	DEVICE = "auto"

	@staticmethod
	def get_device(requested: str) -> torch.device:
		if requested == "cuda" and torch.cuda.is_available():
			return torch.device("cuda:0")
		if requested == "cpu":
			return torch.device("cpu")
		return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

	@staticmethod
	def build_model() -> nn.Module:
		return nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Flatten(),
			nn.Linear(128 * 7 * 7, 256),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(256, 10),
		)

	@staticmethod
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

	@staticmethod
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

	@classmethod
	def main(cls) -> None:
		device = cls.get_device(cls.DEVICE)
		use_cuda = device.type == "cuda"
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
			batch_size=cls.BATCH_SIZE,
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

		model = cls.build_model().to(device)
		loss_fn = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

		epochs = cls.EPOCHS
		for epoch in range(1, epochs + 1):
			train_loss, train_acc = cls.train_one_epoch(
				model, train_loader, optimizer, loss_fn, device
			)
			test_loss, test_acc = cls.evaluate(model, test_loader, loss_fn, device)

			print(
				f"Epoch {epoch}/{epochs} | "
				f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
				f"test loss {test_loss:.4f} acc {test_acc:.4f}"
			)

		output_dir = os.path.join(os.getcwd(), "onnx")
		temp_dir = os.path.join(os.getcwd(), "models", "_temp")
		train_and_export(
			output_dir=output_dir,
			temp_models_dir=temp_dir,
		)


def build_model():
	return MlBasicTrainer.build_model()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
	return MlBasicTrainer.train_one_epoch(model, loader, optimizer, loss_fn, device)


def evaluate(model, loader, loss_fn, device):
	return MlBasicTrainer.evaluate(model, loader, loss_fn, device)


def main():
	MlBasicTrainer.main()


if __name__ == "__main__":
	main()


def train_and_export(
	output_dir: str,
	temp_models_dir: str,
) -> str:
	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,)),
		]
	)

	base_dir = os.path.dirname(os.path.abspath(__file__))
	data_dir = os.path.join(base_dir, "..", "data")

	train_data = datasets.FashionMNIST(
		root=data_dir,
		train=True,
		download=True,
		transform=transform,
	)
	test_data = datasets.FashionMNIST(
		root=data_dir,
		train=False,
		download=True,
		transform=transform,
	)

	device_obj = MlBasicTrainer.get_device(MlBasicTrainer.DEVICE)
	use_cuda = device_obj.type == "cuda"
	train_loader = DataLoader(
		train_data,
		batch_size=MlBasicTrainer.BATCH_SIZE,
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

	model = MlBasicTrainer.build_model().to(device_obj)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	for epoch in range(1, MlBasicTrainer.EPOCHS + 1):
		train_loss, train_acc = MlBasicTrainer.train_one_epoch(
			model, train_loader, optimizer, loss_fn, device_obj
		)
		test_loss, test_acc = MlBasicTrainer.evaluate(
			model, test_loader, loss_fn, device_obj
		)
		print(
			f"Epoch {epoch}/{MlBasicTrainer.EPOCHS} | "
			f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
			f"test loss {test_loss:.4f} acc {test_acc:.4f}"
		)

	os.makedirs(output_dir, exist_ok=True)
	model.eval()

	weights_path = os.path.join(output_dir, "fashion_mnist_cnn.pth")
	torch.save(model.state_dict(), weights_path)

	model_cpu = model.to("cpu")
	dummy_input = torch.randn(1, 1, 28, 28)
	onnx_path = os.path.join(output_dir, "fashion_mnist_cnn.onnx")
	os.makedirs(temp_models_dir, exist_ok=True)
	temp_name = f"{os.path.basename(output_dir)}_temp.onnx"
	temp_onnx_path = os.path.join(temp_models_dir, temp_name)
	torch.onnx.export(
		model_cpu,
		dummy_input,
		temp_onnx_path,
		export_params=True,
		opset_version=13,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["logits"],
		dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
	)
	shutil.copy2(temp_onnx_path, onnx_path)
	return onnx_path
