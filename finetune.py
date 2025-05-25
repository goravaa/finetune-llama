#!/usr/bin/env python3
"""
🦙Llama finetuning launcher
-------------------------------------
• Q-only, LoRA-only, full QLoRA, or vanilla FP16/32
• YAML-driven config
• Early VRAM check via dummy profiling to avoid mid-run OOM
• Hugging Face authentication & model availability checks
"""
import argparse
import os
import sys
import yaml
import torch
import random
import numpy as np
from utils.logging_utils import setup_logger
from utils.resource_estimator import check_device_fits
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset import prepare_dataset
from inference import run_inference
from huggingface_hub import HfApi, HfFolder

log = setup_logger()

# ---------- helpers ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_hf_login():
    token = HfFolder.get_token()
    if not token:
        log.error(
            "🔒 No Hugging Face token found. "
            "Run `huggingface-cli login` or set HUGGINGFACE_TOKEN."
        )
        sys.exit(1)
    try:
        HfApi().whoami(token=token)
        log.info("✅ Hugging Face authentication OK")
    except Exception as e:
        log.error(f"🔒 Hugging Face auth failed: {e}")
        sys.exit(1)


def verify_model_access(repo_id: str):
    try:
        HfApi().model_info(repo_id)
        log.info(f"✅ Access confirmed for model {repo_id}")
    except Exception:
        log.error(
            f"❌ Cannot load model '{repo_id}'. "
            "Check name/access rights or HF_TOKEN scopes."
        )
        sys.exit(1)

# ---------- main ----------
def main(cfg_path):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["general"]["seed"])

    # 0️⃣ HF login & model access
    base = cfg["model"]["base_model"]
    if cfg["general"].get("push_to_hub", False) or "huggingface" in base:
        check_hf_login()
        verify_model_access(base)

    # ① Dataset preparation
    ds = prepare_dataset(cfg["dataset"])
    log.info(
        f"Dataset prepared: {len(ds['train'])} train / {len(ds['test'])} val samples"
    )

    # ② Tokenizer & Model loading
    quant = cfg["model"].get("load_in_4bit", False)
    eight_bit = cfg["model"].get("load_in_8bit", False)
    use_lora = cfg["model"].get("use_lora", False)

    if not any([quant, eight_bit, use_lora]):
        log.info(
            "🛣️ Running vanilla full-precision fine-tuning (no quant, no LoRA)"
        )

    model_kwargs = {}
    if quant or eight_bit:
        if quant:
            model_kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": getattr(
                    torch, cfg["model"]["compute_dtype"]
                ),
                "bnb_4bit_quant_type": cfg["model"]["quant_type"],
            }
        else:
            model_kwargs = {"load_in_8bit": True}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base, use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base, **model_kwargs
        )
    except Exception as e:
        log.error(f"❌ Failed to load model/tokenizer: {e}")
        sys.exit(1)

    # ③ LoRA injection
    if use_lora:
        if quant or eight_bit:
            model = prepare_model_for_kbit_training(model)
        targets = cfg["model"].get(
            "lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        lora_cfg = LoraConfig(
            r=cfg["model"]["lora_r"],
            lora_alpha=cfg["model"]["lora_alpha"],
            target_modules=targets,
            lora_dropout=cfg["model"]["lora_dropout"],
            bias=cfg["model"].get("lora_bias", "none"),
            task_type=cfg["model"].get("lora_task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, lora_cfg)
        log.info(f"LoRA adapters injected into: {targets} ✨")

    # ④ VRAM check
    if cfg.get("resources", {}).get("auto_check", False):
        seq_len = cfg["resources"].get("dummy_seq_len", 512)
        margin = cfg["resources"].get("vram_margin_mb", 0)
        ok, want, free = check_device_fits(
            model, cfg, margin, seq_len
        )
        if not ok:
            log.error(
                f"Need ≈{want//1024} GB, only {free//1024} GB free ‼️"
                " Adjust batch/seq_len/quant/LoRA"
            )
            return
        log.info(
            f"Resource check ✅ (need {want//1024} GB, have {free//1024} GB)"
        )

    # ⑤ TrainingArguments
    reporters = []
    if cfg.get("wandb", {}).get("use_wandb", False):
        reporters.append("wandb")
    if cfg.get("logging", {}).get("use_tensorboard", False):
        reporters.append("tensorboard")

    targs = TrainingArguments(
        output_dir=cfg["general"]["output_dir"],
        per_device_train_batch_size=
            cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=
            cfg["training"]["gradient_accumulation_steps"],
        max_steps=cfg["training"]["max_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=cfg["training"]["warmup_steps"],
        bf16=cfg["training"].get("bf16", False),
        fp16=cfg["training"].get("fp16", False),
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        report_to=reporters,
        logging_dir=cfg.get("logging", {}).get("tb_log_dir"),
        run_name=cfg["general"]["project_name"],
        push_to_hub=cfg["general"].get("push_to_hub", False),
        hub_model_id=cfg["general"].get("hub_model_id"),
    )

    # ⑥ Data collator & Trainer
    def collate_fn(batch):
        toks = tokenizer(
            [b[cfg["dataset"]["text_column"]] for b in batch],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        toks["labels"] = toks["input_ids"].clone()
        return toks

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collate_fn,
    )

    # ⑦ Train
    log.info("🚀 Starting training loop …")
    trainer.train()
    log.info("✅ Training complete")

    # ⑧ Push & deploy
    if cfg["general"].get("push_to_hub", False):
        log.info("📤 Pushing to Hugging Face Hub …")
        trainer.push_to_hub()
    if cfg["general"].get("deploy_after_training", False):
        log.info("⚡ Running inference…")
        run_inference(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
