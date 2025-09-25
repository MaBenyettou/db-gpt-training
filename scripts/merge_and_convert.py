import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training


cfg = json.load(open("/workspace/configs/training.json"))


BASE_MODEL = cfg["base_model"]
LORA_DIR = cfg["output_dir"]
MERGED_DIR = "/workspace/output/merged"
GGUF_OUT = "/workspace/output/qwen2.5-coder-3b-sql.gguf"

def merge_lora():

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

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
        f"python3 /workspace/llama.cpp/convert_hf_to_gguf.py "
        f"--model {MERGED_DIR} --outfile {GGUF_OUT}"
    )

    print("🔄 Quantizing GGUF...")
    os.system(
        f"/workspace/llama.cpp/quantize {GGUF_OUT} "
        f"/workspace/output/qwen2.5-coder-3b-sql.Q4_K_M.gguf Q4_K_M"
    )


    print("✅ GGUF + quantized saved in /workspace/models/")

if __name__ == "__main__":
    merge_lora()
    convert_and_quantize()
