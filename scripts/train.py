import json
import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load config
cfg = json.load(open("/workspace/configs/training.json"))

print("🔄 Loading base model & tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    cfg["base_model"],
    use_fast=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("🔄 Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    cfg["base_model"],
    quantization_config=bnb_config,
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

print("🔄 Preparing LoRA...")
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

print("🔄 Loading and processing dataset...")
dataset = load_dataset("json", data_files="/workspace/data/dataset.jsonl", split="train")

def preprocess_function(examples):
    texts = []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        text = prompt + completion + tokenizer.eos_token
        texts.append(text)
    
    model_inputs = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=cfg["max_seq_length"],
        return_tensors=None
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

args = TrainingArguments(
    output_dir=cfg["output_dir"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    logging_steps=cfg["logging_steps"],
    save_steps=cfg["save_steps"],
    fp16=True,
    optim="paged_adamw_8bit",
    save_total_limit=2,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    warmup_steps=100,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("🔄 Starting training...")
trainer.train()

print("🔄 Saving model...")

# Get timestamp for unique naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Save to main output directory (mounted to host)
main_output = cfg["output_dir"]
model.save_pretrained(main_output)
tokenizer.save_pretrained(main_output)

# 2. Save timestamped copy inside container
container_backup = f"/workspace/model_backups/{timestamp}"
os.makedirs(container_backup, exist_ok=True)
model.save_pretrained(container_backup)
tokenizer.save_pretrained(container_backup)

# 3. Save timestamped copy to host (if output is mounted)
host_timestamped = f"{main_output}/checkpoints/{timestamp}"
os.makedirs(host_timestamped, exist_ok=True)
model.save_pretrained(host_timestamped)
tokenizer.save_pretrained(host_timestamped)

# 4. Save training config for reference
config_copy = {
    "training_config": cfg,
    "model_info": {
        "base_model": cfg["base_model"],
        "timestamp": timestamp,
        "quantization": "4bit_nf4",
        "lora_r": peft_cfg.r,
        "lora_alpha": peft_cfg.lora_alpha,
    }
}

with open(f"{main_output}/training_info.json", "w") as f:
    json.dump(config_copy, f, indent=2)

with open(f"{container_backup}/training_info.json", "w") as f:
    json.dump(config_copy, f, indent=2)

print(f"✅ Training complete!")
print(f"📁 Main model saved to: {main_output} (accessible on host)")
print(f"📁 Container backup: {container_backup}")
print(f"📁 Timestamped checkpoint: {host_timestamped}")
print(f"🏷️  Training timestamp: {timestamp}")