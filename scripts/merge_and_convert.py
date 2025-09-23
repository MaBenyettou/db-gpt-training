import os
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_DIR = "/workspace/output"
MERGED_DIR = "/workspace/models/merged"
GGUF_OUT = "/workspace/models/qwen2.5-coder-3b-sql.gguf"

def merge_lora():
    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")

    print("🔄 Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_DIR)

    print("🔄 Merging LoRA into base...")
    merged_model = model.merge_and_unload()

    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)

def convert_and_quantize():
    os.makedirs("/workspace/models", exist_ok=True)

    print("🔄 Converting merged model to GGUF...")
    os.system(
        f"python3 /workspace/llama.cpp/convert-hf-to-gguf.py "
        f"--model {MERGED_DIR} --outfile {GGUF_OUT}"
    )

    print("🔄 Quantizing GGUF...")
    os.system(
        f"/workspace/llama.cpp/quantize {GGUF_OUT} "
        f"/workspace/models/qwen2.5-coder-3b-sql.Q4_K_M.gguf Q4_K_M"
    )

    print("✅ GGUF + quantized saved in /workspace/models/")

if __name__ == "__main__":
    merge_lora()
    convert_and_quantize()
