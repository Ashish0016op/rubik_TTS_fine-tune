"""LoRA configuration and setup for Rumik-OSS-1 voice adaptation."""

import torch
import torch.nn as nn
from typing import List, Optional
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao as peft_torchao
        peft_torchao.is_torchao_available = lambda: False
        peft_torchao.dispatch_torchao = lambda *args, **kwargs: None
    except Exception:
        pass
except ImportError:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

def get_rumik_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> "LoraConfig":
    """Builds LoRA configuration targeting attention and MLP projection layers."""
    if LoraConfig is None:
        raise ImportError("PEFT library is required for LoRA. Install via 'pip install peft'.")
        
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False
    )


def apply_lora_to_rumik(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """Applies LoRA to the transformer backbone while keeping Mimi codec completely frozen."""
    # Ensure all base model parameters are frozen
    for p in model.parameters():
        p.requires_grad = False

    if get_peft_model is None:
        print("[!] Warning: PEFT not installed. Using un-adapted PyTorch model.")
        return model

    # If model is a wrapper, apply LoRA to the internal transformer
    transformer = getattr(model, "transformer", model)

    lora_config = get_rumik_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )

    peft_transformer = get_peft_model(transformer, lora_config)

    # If model was wrapped, replace the transformer reference
    if hasattr(model, "transformer"):
        model.transformer = peft_transformer
        # Keep stop head trainable if present
        if hasattr(model, "stop_head") and model.stop_head is not None:
            for p in model.stop_head.parameters():
                p.requires_grad = True
        return model

    return peft_transformer


def print_trainable_parameters(model: nn.Module):
    """Prints number of trainable vs total parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    pct = 100 * trainable_params / all_param if all_param > 0 else 0
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {pct:.4f}%"
    )
