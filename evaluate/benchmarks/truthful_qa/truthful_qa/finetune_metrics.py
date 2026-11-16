"""Fine-tune Mistral 7B to provide the info and judge metrics for TruthfulQA"""
import os
import json
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset
from transformers import TrainingArguments
from datetime import datetime
from examples.honesty.rep_manager_utils import load_model

# Change to "judge" or "info" to fine-tune the model for the corresponding task
timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M")

for task in ["judge", "info"]:
    DATASET_DIR = "./evaluate/benchmarks/truthful_qa/data/"
    FILE_NAME = "finetune_truth.jsonl" if task == "judge" else "finetune_info.jsonl"
    DATASET_PATH = os.path.join(DATASET_DIR, FILE_NAME)

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.2", compile=False, fast_tokenizer=False)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        #modules_to_save=["lm_head", "embed_token"],
        task_type="CAUSAL_LM"
    )

    train_args = SFTConfig(
        # torch_compile=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        # warmup_ratio=0.1,
        num_train_epochs=5,
        bf16=True,
        logging_steps=30,
        output_dir=f"TruthfulQA-{task}",
        optim="adamw_bnb_8bit",
        eval_steps=30,
        save_steps=60,
        save_total_limit=5,
        max_seq_length=256,
        seed=0
    )

    trainer = SFTTrainer(
        model,
        args=train_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config
    )
    trainer.train()

    ADAPTER_PATH = f"./models/mistral/{timestamp}/{task}"
    os.makedirs(ADAPTER_PATH, exist_ok=True)
    trainer.save_model(ADAPTER_PATH)