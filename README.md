# 📘 README — SQL Report Finetuning with Qwen

This project allows you to **fine-tune Qwen2.5-Coder** (or similar LLMs) on your company’s SQL/business-report dataset, and export the trained model to **GGUF** format for efficient inference with [llama.cpp](https://github.com/ggerganov/llama.cpp).

The pipeline is fully Dockerized, GPU-accelerated, and runs in **one command**.

---

## 📂 Project Structure

```
sql-finetune/
│── Dockerfile
│── requirements.txt
│── configs/
│    └── training.json       # Training configuration
│── scripts/
│    ├── train.py            # LoRA training
│    ├── merge_and_convert.py# Merge LoRA → HF → GGUF + quantization
│    └── run_all.sh          # Runs the entire pipeline
│── data/
│    └── dataset.jsonl       # Training data
│── models/                  # Final GGUF + quantized models
│── output/                  # Intermediate LoRA adapter weights
```

---

## ⚙️ Configuration

### 1. Training config (`configs/training.json`)

```json
{
  "base_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
  "output_dir": "/workspace/output",
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-4,
  "num_train_epochs": 3,
  "logging_steps": 20,
  "save_steps": 500,
  "max_seq_length": 2048
}
```

* `base_model`: Any Hugging Face model (e.g. Qwen/Qwen2.5-Coder-3B-Instruct).
* `num_train_epochs`: Increase if you have small data.
* `max_seq_length`: Should cover your longest query.

---

### 2. Dataset format (`data/dataset.jsonl`)

Each line is a JSON object with a **prompt** and a **completion**:

```json
{"prompt": "User: Show me the total delivered orders per month this year", "completion": "SELECT ..."}
{"prompt": "User: Average start and end time per employee", "completion": "WITH sub AS (..."}
```

* Use multiple examples per report (with different filters).
* `prompt` can be in multiple languages (English, Arabic, French).
* `completion` should be the **final SQL query**.

---

## 🚀 Usage

### 1. Build the Docker image

```bash
docker build -t sql-finetune .
```

### 2. Run full training pipeline

Mount your **dataset** and an output folder for the final model:

```bash
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  sql-finetune bash scripts/run_all.sh
```

This will:

1. Train a LoRA adapter (`output/`)
2. Merge LoRA into base model (`models/merged/`)
3. Convert merged model → GGUF (`models/qwen2.5-coder-3b-sql.gguf`)
4. Quantize GGUF (`models/qwen2.5-coder-3b-sql.Q4_K_M.gguf`)

---

## 🎯 Outputs

After running, you will have:

```
models/
│── merged/                               # Hugging Face merged model
│── qwen2.5-coder-3b-sql.gguf             # Full-precision GGUF
│── qwen2.5-coder-3b-sql.Q4_K_M.gguf      # Quantized GGUF
```

---

## 🧪 Inference

You can now run inference with `llama.cpp` or any GGUF-compatible runtime:

```bash
./main -m models/qwen2.5-coder-3b-sql.Q4_K_M.gguf -p "User: Show me average start time per employee"
```

The model will generate SQL queries aligned with your training reports.

---

## 🔧 Customization

* **Different models**: Update `base_model` in `configs/training.json`.
* **Training size**: Adjust `num_train_epochs`, `batch_size`, etc.
* **Dataset**: Expand `dataset.jsonl` with multiple report classes and filters.
* **Quantization**: Change `Q4_K_M` to other quantization modes (e.g. `Q5_K_S`).

---

## ⚡ Tips

* For 8GB VRAM, 3B models (like `Qwen2.5-Coder-3B-Instruct`) are manageable.
* Use **LoRA** instead of full finetuning to save memory.
* Keep prompts diverse (different languages, phrasings).
* Validate generated SQL against your DB schema before execution.

---

✅ That’s the complete training → GGUF → quantization pipeline in one Docker project.

Do you also want me to extend the README with a **section on generating dataset automatically from your Laravel reports** (so your team knows how to produce `dataset.jsonl` directly from PHP)?
