#!/usr/bin/env python3
"""
🦙 Universal Llama finetuning launcher
-------------------------------------
• Q-only, LoRA-only or full QLoRA
• YAML-driven config
• Emoji logs
• Early VRAM check to save time
"""
import argparse, os, yaml, torch, random, numpy as np
from utils.logging_utils import setup_logger
from utils.resource_estimator import check_device_fits
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

log = setup_logger()

# ---------- helpers ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------- main ----------
def main(cfg_path):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["general"]["seed"])

    # ① Resource sanity check
    if cfg["resources"]["auto_check"]:
        ok, want, free = check_device_fits(cfg, cfg["resources"]["vram_margin_mb"])
        if not ok:
            log.error(f"Need ≈{want//1024} GB, only {free//1024} GB free ‼️  – consider ↓batch or ↑quant")
            return
        log.info(f"Resource check ✅  (need {want//1024} GB, have {free//1024} GB)")

    # ② Data
    data_cfg = cfg["dataset"]
    if os.path.isfile(data_cfg["name_or_path"]) and data_cfg["name_or_path"].endswith(".csv"):
        ds = load_dataset("csv", data_files=data_cfg["name_or_path"])
    else:
        ds = load_dataset(data_cfg["name_or_path"])
    ds = ds["train"].train_test_split(test_size=data_cfg["validation_split_percentage"]/100.0)
    log.info(f"Dataset loaded with {len(ds['train'])} train / {len(ds['test'])} val samples")

    # ③ Tokeniser & Model
    quant = cfg["model"]["load_in_4bit"]
    lora  = cfg["model"]["use_lora"]
    model_kwargs = {}
    if quant:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": getattr(torch, cfg["model"]["compute_dtype"]),
            "bnb_4bit_quant_type": cfg["model"]["quant_type"],
        })
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model"], **model_kwargs)

    # ③-bis LoRA
    if lora:
        model = prepare_model_for_kbit_training(model) if quant else model
        lora_cfg = LoraConfig(
            r=cfg["model"]["lora_r"],
            lora_alpha=cfg["model"]["lora_alpha"],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=cfg["model"]["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        log.info("LoRA adapters injected ✨")

    # ④ Training arguments
    targs = TrainingArguments(
        output_dir=cfg["general"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        max_steps=cfg["training"]["max_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=cfg["training"]["warmup_steps"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        report_to=["wandb"] if cfg["wandb"]["use_wandb"] else [],
        run_name=cfg["general"]["project_name"],
        push_to_hub=cfg["general"]["push_to_hub"],
        hub_model_id=cfg["general"]["hub_model_id"],
    )

    # ⑤ Data collator (simple, causal LM)
    def collate_fn(batch):
        toks = tokenizer([b[data_cfg["text_column"]] for b in batch],
                         truncation=True, padding="max_length",
                         max_length=tokenizer.model_max_length, return_tensors="pt")
        toks["labels"] = toks["input_ids"].clone()
        return toks

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate_fn,
    )

    # ⑥ GO!
    log.info("🚀 Starting training loop …")
    trainer.train()
    log.info("✅ Training complete")

    # ⑦ Push / deploy
    if cfg["general"]["push_to_hub"]:
        log.info("📤 Pushing adapter to 🤗 Hub …")
        trainer.push_to_hub()

    if cfg["general"]["deploy_after_training"]:
        log.info("⚡ Spinning up text-generation-inference … (stub)")
        #  -> call `tii` docker / hf-inference endpoints here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
