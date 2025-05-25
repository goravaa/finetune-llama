#!/usr/bin/env python3
"""
🦙 Llama fine-tuning launcher
-------------------------------------
• QLoRA, LoRA-only, 4/8-bit or full FP16/32
• YAML-driven config
• Early VRAM check to avoid mid-run OOM
• Hugging Face authentication & model checks
"""
import argparse
import os
import sys
import yaml
import random
import logging

import torch
import numpy as np
from utils.logging_utils import setup_logger
from utils.resource_estimator import check_device_fits
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset import prepare_dataset
from deploy import run_inference
from huggingface_hub import HfApi, HfFolder
import transformers as _tfm
import datasets as _ds
import peft as _peft

log = setup_logger("finetune")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_hf_login():
    token = HfFolder.get_token()
    if not token:
        log.error("🔒 No Hugging Face token found. Run `huggingface-cli login`.")
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
        log.error(f"❌ Cannot load model '{repo_id}'. Check permissions.")
        sys.exit(1)


def main(cfg_path: str):
    # Load config
    cfg = load_yaml(cfg_path)
    # Version info
    log.info(f"Transformers {_tfm.__version__}, Datasets {_ds.__version__}, PEFT {_peft.__version__}")

    # Seed
    seed = cfg.get('general', {}).get('seed', 42)
    set_seed(seed)

    # HF login if pushing or base is HF repo
    base_model = cfg['model']['base_model']
    if cfg.get('general', {}).get('push_to_hub', False) or 'huggingface' in base_model:
        check_hf_login()
        verify_model_access(base_model)

    # Prepare dataset
    ds = prepare_dataset(cfg['dataset'])
    n_train = len(ds['train'])
    n_val = len(ds.get('validation', ds.get('test', [])))
    log.info(f"🏷️  Dataset: {n_train} train / {n_val} validation samples")

    # Load tokenizer & model
    quant = cfg['model'].get('load_in_4bit', False)
    eight_bit = cfg['model'].get('load_in_8bit', False)
    use_lora = cfg['model'].get('use_lora', False)

    # Validate compute dtype
    compute_dtype = None
    if quant:
        dtype_str = cfg['model'].get('bnb_4bit_compute_dtype', 'float16')
        if dtype_str not in ('float16', 'bfloat16'):
            log.error(f"Unsupported compute_dtype: {dtype_str}")
            sys.exit(1)
        compute_dtype = getattr(torch, dtype_str)

    # Assemble model kwargs
    load_kwargs = {}
    if quant:
        load_kwargs = {
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': compute_dtype,
            'bnb_4bit_quant_type': cfg['model'].get('bnb_4bit_quant_type', 'nf4'),
        }
    elif eight_bit:
        load_kwargs = {'load_in_8bit': True}

    # Load
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except Exception as e:
        log.error(f"❌ Failed to load model/tokenizer: {e}")
        sys.exit(1)

    # Quant preparation
    if quant or eight_bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA injection
    if use_lora:
        targets = cfg['model'].get('lora_target_modules',
                                   ['q_proj', 'v_proj', 'k_proj', 'o_proj'])
        lora_cfg = LoraConfig(
            r=cfg['model']['lora_r'],
            lora_alpha=cfg['model']['lora_alpha'],
            target_modules=targets,
            lora_dropout=cfg['model']['lora_dropout'],
            bias=cfg['model'].get('lora_bias', 'none'),
            task_type=cfg['model'].get('lora_task_type', 'CAUSAL_LM'),
        )
        model = get_peft_model(model, lora_cfg)
        log.info(f"🎯 LoRA adapters injected into: {targets}")

    # VRAM check
    if cfg.get('resources', {}).get('auto_check', False):
        seq_len = cfg['resources'].get('dummy_seq_len', 512)
        margin = cfg['resources'].get('vram_margin_mb', 0)
        ok, need, free = check_device_fits(model, cfg, margin, seq_len)
        if not ok:
            log.error(f"Need ~{need//1024} GB but only {free//1024} GB free. Adjust settings.")
            sys.exit(1)
        log.info(f"🖥️ VRAM check OK (need {need//1024} GB, have {free//1024} GB)")

    # Training arguments
    reporters = []
    if cfg.get('wandb', {}).get('use_wandb', False):
        reporters.append('wandb')
    if cfg.get('logging', {}).get('use_tensorboard', False):
        reporters.append('tensorboard')

    targs = TrainingArguments(
        output_dir=cfg['general']['output_dir'],
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        max_steps=cfg['training']['max_steps'],
        learning_rate=cfg['training']['learning_rate'],
        lr_scheduler_type=cfg['training']['lr_scheduler_type'],
        warmup_steps=cfg['training']['warmup_steps'],
        fp16=cfg['training'].get('fp16', False),
        bf16=cfg['training'].get('bf16', False),
        logging_steps=cfg['training']['logging_steps'],
        save_steps=cfg['training']['save_steps'],
        evaluation_strategy='steps',
        eval_steps=cfg['training']['eval_steps'],
        report_to=reporters,
        logging_dir=cfg.get('logging', {}).get('tb_log_dir'),
        run_name=cfg['general'].get('project_name'),
        push_to_hub=cfg['general'].get('push_to_hub', False),
        hub_model_id=cfg['general'].get('hub_model_id'),
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds['train'],
        eval_dataset=ds.get('validation', None),
        data_collator=data_collator,
    )

    # Train
    log.info("🚀 Starting training…")
    trainer.train()
    log.info("✅ Training complete")

    # Push & deploy
    if cfg['general'].get('push_to_hub', False):
        log.info("📤 Pushing to Hugging Face Hub…")
        trainer.push_to_hub()
    if cfg['general'].get('deploy_after_training', False):
        log.info("⚡ Launching inference…")
        run_inference(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='config.yaml')
    args = parser.parse_args()
    main(args.config)
