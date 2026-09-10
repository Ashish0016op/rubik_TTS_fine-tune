"""Training loop and fine-tuning engine for Rumik TTS."""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from src.models.model_utils import RumikModelWrapper
from src.models.lora_setup import apply_lora_to_rumik, print_trainable_parameters
from src.models.speaker_encoder import SpeakerConditioningModule
from src.training.loss import RumikTTSLoss
from src.data.dataset import RumikTTSDataset, collate_fn_tts
from config.default_config import TrainingConfig, LoRAConfig

class RumikFineTuner:
    """Orchestrates LoRA fine-tuning for Rumik-OSS-1 voice adaptation."""

    def __init__(
        self,
        model_wrapper: RumikModelWrapper,
        training_config: Optional[TrainingConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        speaker_module: Optional[SpeakerConditioningModule] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.wrapper = model_wrapper
        self.train_cfg = training_config or TrainingConfig()
        self.lora_cfg = lora_config or LoRAConfig()
        self.speaker_module = speaker_module
        self.device = device

        # 1. Apply LoRA to transformer backbone
        print("[*] Applying LoRA adapters to attention and MLP projection layers...")
        self.model = apply_lora_to_rumik(
            self.wrapper,
            r=self.lora_cfg.r,
            lora_alpha=self.lora_cfg.lora_alpha,
            lora_dropout=self.lora_cfg.lora_dropout,
            target_modules=self.lora_cfg.target_modules
        )
        print_trainable_parameters(self.model)

        # 2. Setup Loss
        self.criterion = RumikTTSLoss(
            text_vocab_size=self.wrapper.token_layout.text_vocab_size,
            num_codebooks=self.wrapper.token_layout.num_codebooks,
            codebook_size=self.wrapper.token_layout.codebook_size
        )

        # 3. Setup Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.speaker_module is not None:
            trainable_params.extend([p for p in self.speaker_module.parameters() if p.requires_grad])

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            betas=(0.9, 0.95)
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        replay_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Runs one training epoch with gradient accumulation and optional replay mixing."""
        self.model.train()
        total_loss = 0.0
        steps = 0
        self.optimizer.zero_grad()

        replay_iter = iter(replay_dataloader) if replay_dataloader else None

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Replay mixing: occasionally swap or mix general batch to prevent catastrophic forgetting
            if replay_iter and torch.rand(1).item() < self.train_cfg.replay_ratio:
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_dataloader)
                    replay_batch = next(replay_iter)
                input_ids = replay_batch["input_ids"].to(self.device)
                labels = replay_batch["labels"].to(self.device)
                attention_mask = replay_batch["attention_mask"].to(self.device)

            # Forward pass
            outputs = self.wrapper.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits = outputs.logits

            # Stop head predictions
            stop_logits = None
            if hasattr(self.wrapper, "stop_head") and self.wrapper.stop_head is not None:
                last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
                if last_hidden is not None:
                    stop_logits = self.wrapper.stop_head(last_hidden)

            loss, metrics = self.criterion(
                lm_logits=logits,
                labels=labels,
                stop_logits=stop_logits,
                stop_labels=batch.get("stop_labels")
            )

            # Normalize for gradient accumulation
            loss = loss / self.train_cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.train_cfg.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += metrics["total_loss"]
            steps += 1
            progress_bar.set_postfix({"loss": f"{metrics['total_loss']:.4f}"})

        avg_loss = total_loss / max(1, steps)
        return {"avg_train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model on held-out validation set."""
        self.model.eval()
        total_loss = 0.0
        steps = 0

        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.wrapper.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits = outputs.logits

            stop_logits = None
            if hasattr(self.wrapper, "stop_head") and self.wrapper.stop_head is not None:
                last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
                if last_hidden is not None:
                    stop_logits = self.wrapper.stop_head(last_hidden)

            loss, metrics = self.criterion(
                lm_logits=logits,
                labels=labels,
                stop_logits=stop_logits,
                stop_labels=batch.get("stop_labels")
            )
            total_loss += metrics["total_loss"]
            steps += 1

        avg_loss = total_loss / max(1, steps)
        return {"val_loss": avg_loss}

    def save_checkpoint(self, output_dir: str):
        """Saves fine-tuned LoRA weights and optional speaker encoder."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Saving LoRA checkpoint to {output_dir}...")
        
        # Save LoRA adapter weights
        if hasattr(self.wrapper.transformer, "save_pretrained"):
            self.wrapper.transformer.save_pretrained(output_dir)

        # Save stop head if present
        if hasattr(self.wrapper, "stop_head") and self.wrapper.stop_head is not None:
            stop_head_path = os.path.join(output_dir, "stop_head.pt")
            torch.save(self.wrapper.stop_head.state_dict(), stop_head_path)

        # Save speaker conditioning module if present
        if self.speaker_module is not None:
            spk_path = os.path.join(output_dir, "speaker_module.pt")
            torch.save(self.speaker_module.state_dict(), spk_path)

        print(f"[+] Successfully saved checkpoint artifacts to {output_dir}")
