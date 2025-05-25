import torch


def measure_model_vram_mb(model: torch.nn.Module) -> int:
    """
    Move the given model to GPU (if not already) and measure its VRAM footprint.
    Returns the delta in MB.
    """
    device = torch.device("cuda")
    # Clear cache to get an accurate baseline
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    model.to(device)
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_allocated(device)
    return int((after - before) / (1024 ** 2))


def measure_activation_peak_mb(
    model: torch.nn.Module,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    """
    Perform a dummy forward pass to measure peak activation memory in MB.
    Assumes causal LM: input shape [batch_size, seq_len].
    """
    device = torch.device("cuda")
    # Prepare dummy inputs
    dummy_input = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )

    # Reset and track peak
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    peak = torch.cuda.max_memory_allocated(device)
    return int(peak / (1024 ** 2))


def estimate_training_vram_mb(
    model: torch.nn.Module,
    cfg: dict,
    seq_len: int = 512,
) -> int:
    """
    Estimate total VRAM required for training:
      - Model weight footprint
      - Activation peak (via dummy forward pass)
      - LoRA overhead (~5%) if enabled
      - Scaled by batch size
    seq_len: length of input sequence for dummy pass
    """
    # 1) Weight footprint
    weight_mb = measure_model_vram_mb(model)

    # 2) Activation peak
    batch_size = cfg.get("training", {}).get("per_device_train_batch_size", 1)
    act_peak_mb = measure_activation_peak_mb(model, seq_len, batch_size)

    # 3) Combine
    total_mb = weight_mb + act_peak_mb

    # 4) LoRA overhead
    if cfg.get("model", {}).get("use_lora", False):
        total_mb = int(total_mb * 1.05)

    return total_mb


def check_device_fits(
    model: torch.nn.Module,
    cfg: dict,
    margin_mb: int = 0,
    seq_len: int = 512,
):
    """
    Returns (fits: bool, required_mb: int, free_mb: int).
    Runs a dummy forward to gauge activation needs.
    """
    required = estimate_training_vram_mb(model, cfg, seq_len) + margin_mb
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found 🥲")
    free = torch.cuda.mem_get_info()[0] // (1024 ** 2)
    return free >= required, required, free
