# ONNX_Testing
ONNX Testing for OLSN

## Layout
- models/: Python model scripts used for training
- models/_temp/: Temporary ONNX exports created during training
- onnx/: Output folders for trained and exported models

## Model Script Template
Each model script must define `train_and_export()` and keep all training
settings (epochs, batch size, device) inside the model file. Use this signature:

```python
def train_and_export(
    output_dir: str,
    temp_models_dir: str,
) -> str:
    ...
```

See [models/template_model.py](models/template_model.py) for a starting point.

## Train + Export
Train using a model script and export ONNX to a folder under onnx/:

```bash
python main.py template_model v1_run
```

With optimizations:

```bash
python main.py template_model v1_run --optimize
```

Flags (main.py):
- model_name: Model script name under models/ (positional)
- output_folder: Folder name under onnx/ for outputs (positional)
- --optimize: Run ONNX optimizations after export

## Test Model
Run a simple accuracy + latency test:

```bash
python onnx_test.py
python onnx_test.py --threads 4
```

Flags (onnx_test.py):
- --threads N: Number of CPU threads to use (default: auto/all available)
- --latency-multiplier F: Multiply measured latency by this factor

## Compare Models
Compare all ONNX models under onnx/ (old_ folders are ignored by default):

```bash
python compare_onnx_models.py
python compare_onnx_models.py --include-old-folders
```

Flags (compare_onnx_models.py):
- --onnx-dir PATH: Path to ONNX folder (default: script dir + /onnx)
- --data-dir PATH: Path to dataset folder (default: script dir + /data)
- --threads N: Number of CPU threads to use
- --limit N: Limit number of samples
- --batch-size N: Batch size for evaluation (default: 64)
- --output-csv PATH: Write results to CSV file
- --use-cpu: Force CPU execution instead of GPU
- --include-old-folders: Include folders under onnx/ starting with "old_"

## Optimize Model
Generate optimized variants for a single model:

```bash
python optimize_model.py
python optimize_model.py --model onnx/fashion_mnist_cnn.onnx
```

Flags (optimize_model.py):
- --model PATH: Path to the FP32 ONNX model (default: script dir + /onnx/fashion_mnist_cnn.onnx)
- --enable-int8: Enable INT8 quantization and INT8 variants

## ONNX to ORT
To convert ONNX to ORT:

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort PATH_TO_MODEL.onnx
```

## GPU Setup
Install the GPU build and verify CUDA provider:

```bash
pip install onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

On Windows, ensure these are on your PATH and then reopen your terminal:
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
- C:\Program Files\NVIDIA\CUDNN\v9.19\bin

Quick DLL checks:

```bash
where cublasLt64_12.dll
where cudnn64_9.dll
```

