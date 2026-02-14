import os
import shutil

import torch
from torch import nn


# Define your model here.
def build_model() -> nn.Module:
	return nn.Sequential(
		nn.Flatten(),
		nn.Linear(28 * 28, 128),
		nn.ReLU(),
		nn.Linear(128, 10),
	)


# Provide a sample input matching your model's expected input shape.
def get_dummy_input() -> torch.Tensor:
	return torch.randn(1, 1, 28, 28)


def _get_device(requested: str) -> torch.device:
	if requested == "cuda" and torch.cuda.is_available():
		return torch.device("cuda:0")
	if requested == "cpu":
		return torch.device("cpu")
	return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# Training settings live in the model file.
EPOCHS = 1
BATCH_SIZE = 32
DEVICE = "auto"


def train_and_export(
	output_dir: str,
	temp_models_dir: str,
) -> str:
	# main.py calls this function; keep training logic inside models/.
	device_obj = _get_device(DEVICE)
	model = build_model().to(device_obj)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	# Example training loop on synthetic data.
	for epoch in range(1, EPOCHS + 1):
		inputs = torch.randn(BATCH_SIZE, 1, 28, 28, device=device_obj)
		labels = torch.randint(0, 10, (BATCH_SIZE,), device=device_obj)
		optimizer.zero_grad()
		logits = model(inputs)
		loss = loss_fn(logits, labels)
		loss.backward()
		optimizer.step()
		print(f"Epoch {epoch}/{EPOCHS} | loss {loss.item():.4f}")

	os.makedirs(output_dir, exist_ok=True)
	model.eval()
	model_cpu = model.to("cpu")
	dummy_input = get_dummy_input().to("cpu")
	onnx_path = os.path.join(output_dir, "model.onnx")
	os.makedirs(temp_models_dir, exist_ok=True)
	temp_name = f"{os.path.basename(output_dir)}_temp.onnx"
	temp_onnx_path = os.path.join(temp_models_dir, temp_name)
	# Export to a temp ONNX and then copy to the final output folder.
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
