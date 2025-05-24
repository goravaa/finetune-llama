import torch, math

def _llama_param_count(base_model: str) -> int:
    """Rough param lookup for popular llama checkpoints (in billions)."""
    table = {
        "7b": 7, "13b": 13, "34b": 34, "70b": 70,
    }
    for key, val in table.items():
        if key in base_model.lower():
            return val * 1_000_000_000
    return 7_000_000_000  # default guess

def estimate_vram_mb(cfg) -> int:
    p = _llama_param_count(cfg["model"]["base_model"])
    bits = 4 if cfg["model"]["load_in_4bit"] else 16
    raw = p * bits / 8 / 1024 / 1024      # params → MB
    lora_mult = 1.05 if cfg["model"]["use_lora"] else 1.00
    batch_mult = cfg["training"]["per_device_train_batch_size"] / 4
    return int(raw * lora_mult * batch_mult + 1024)   # +1 GB safety

def check_device_fits(cfg, margin=0):
    want = estimate_vram_mb(cfg) + margin
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found 🥲")
    free = torch.cuda.mem_get_info()[0] // (1024 ** 2)
    return free >= want, want, free
