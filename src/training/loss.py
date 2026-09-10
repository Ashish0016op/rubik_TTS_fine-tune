"""Loss functions for Rumik TTS fine-tuning: Audio-masked cross-entropy and stop head loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class RumikTTSLoss(nn.Module):
    """Computes masked cross entropy over audio tokens with optional codebook weighting and stop loss."""

    def __init__(
        self,
        audio_vocab_offset: int = 261008,
        num_codebooks: int = 8,
        codebook_size: int = 2048,
        cb0_weight: float = 1.5,
        stop_loss_weight: float = 0.5,
        label_smoothing: float = 0.0,
        text_vocab_size: Optional[int] = None
    ):
        super().__init__()
        self.audio_vocab_offset = audio_vocab_offset if text_vocab_size is None else 261008
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.cb0_weight = cb0_weight
        self.stop_loss_weight = stop_loss_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        stop_logits: Optional[torch.Tensor] = None,
        stop_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            lm_logits: [batch, seq_len, vocab_size] (shifted or unshifted)
            labels: [batch, seq_len] with -100 on text tokens and target ID on audio tokens
            stop_logits: [batch, num_frames] optional stop head logits
            stop_labels: [batch, num_frames] optional binary ground truth
        """
        # Standard causal shift if not already shifted
        # logits: [batch, seq_len - 1, vocab_size]
        # target labels: [batch, 1:]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Audio token CE loss
        vocab_size = shift_logits.shape[-1]
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        token_loss = self.ce_loss(flat_logits, flat_labels)

        # Stop head loss
        total_loss = token_loss
        metrics = {"token_loss": token_loss.item()}

        if stop_logits is not None and stop_labels is not None:
            # Align shapes
            min_len = min(stop_logits.shape[-1], stop_labels.shape[-1])
            s_logits = stop_logits[..., :min_len]
            s_labels = stop_labels[..., :min_len]
            stop_loss = self.bce_loss(s_logits, s_labels)
            total_loss = total_loss + (self.stop_loss_weight * stop_loss)
            metrics["stop_loss"] = stop_loss.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
