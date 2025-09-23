import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

cfg = json.load(open("/workspace/configs/training.json"))

print("🔄 Loading base model & tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
model = AutoModelForCausalLM.from_pretrained(
    cfg["base_model"], device_map="auto"
)

print("🔄 Preparing LoRA...")
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

print("🔄 Loading dataset...")
dataset = load_dataset("json", data_files="/workspace/data/dataset.jsonl")

def tokenize(batch):
    return tokenizer(batch["prompt"], text_target=batch["completion"],
                     truncation=True, max_length=cfg["max_seq_length"])

tokenized = dataset.map(tokenize, batched=True, remove_columns=["prompt","completion"])

args = TrainingArguments(
    output_dir=cfg["output_dir"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["num_train_epochs"],
    logging_steps=cfg["logging_steps"],
    save_steps=cfg["save_steps"],
    bf16=True,
    optim="adamw_torch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized["train"],
    args=args,
    tokenizer=tokenizer,
    dataset_text_field=None
)

trainer.train()
model.save_pretrained(cfg["output_dir"])
print("✅ Training complete! LoRA adapters saved to", cfg["output_dir"])
