"""Model architecture definitions, loading utilities, and Mimi codec interfaces for Rumik TTS."""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
try:
    from transformers import MimiModel
except ImportError:
    MimiModel = None

class TokenLayoutManager:
    """Manages conversion between flattened token IDs and (text, codebook_levels).
    
    Layout:
    - Text Tokens: [0, text_vocab_size - 1]
    - Codebook 0 Tokens: [text_vocab_size, text_vocab_size + 2047]
    - Codebook 1 Tokens: [text_vocab_size + 2048, text_vocab_size + 4095]
    ...
    - Codebook 7 Tokens: [text_vocab_size + 7*2048, text_vocab_size + 8*2048 - 1]
    """

    def __init__(self, text_vocab_size: int = 256000, num_codebooks: int = 8, codebook_size: int = 2048):
        self.text_vocab_size = text_vocab_size
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_vocab_offset = text_vocab_size
        self.total_vocab_size = text_vocab_size + (num_codebooks * codebook_size)

    def is_audio_token(self, token_id: int) -> bool:
        return self.audio_vocab_offset <= token_id < self.total_vocab_size

    def audio_code_to_token_id(self, codebook_idx: int, code_value: int) -> int:
        assert 0 <= codebook_idx < self.num_codebooks, f"Invalid codebook index: {codebook_idx}"
        assert 0 <= code_value < self.codebook_size, f"Invalid code value: {code_value}"
        return self.audio_vocab_offset + (codebook_idx * self.codebook_size) + code_value

    def token_id_to_audio_code(self, token_id: int) -> Tuple[int, int]:
        assert self.is_audio_token(token_id), f"Token {token_id} is not an audio token"
        offset_val = token_id - self.audio_vocab_offset
        codebook_idx = offset_val // self.codebook_size
        code_value = offset_val % self.codebook_size
        return codebook_idx, code_value

    def flatten_audio_frames(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Converts [batch, num_codebooks, num_frames] or [num_codebooks, num_frames] to interleaved token IDs.
        
        Order per frame: cb0, cb1, cb2, ..., cb7.
        Returns: 1D or 2D tensor of shape [..., num_frames * num_codebooks].
        """
        is_batched = audio_codes.dim() == 3
        if not is_batched:
            audio_codes = audio_codes.unsqueeze(0)
            
        b, k, t = audio_codes.shape
        assert k == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {k}"

        # Transpose to [b, t, k] for interleaved flattening
        transposed = audio_codes.permute(0, 2, 1).contiguous()
        
        # Create codebook offsets [0, 2048, 4096, ...]
        offsets = torch.arange(k, device=audio_codes.device) * self.codebook_size + self.audio_vocab_offset
        offsets = offsets.view(1, 1, k)
        
        flat_tokens = (transposed + offsets).view(b, t * k)
        return flat_tokens if is_batched else flat_tokens.squeeze(0)

    def unflatten_audio_frames(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        """Converts interleaved flat audio token IDs back to [..., num_codebooks, num_frames]."""
        is_batched = flat_tokens.dim() == 2
        if not is_batched:
            flat_tokens = flat_tokens.unsqueeze(0)
            
        b, seq_len = flat_tokens.shape
        assert seq_len % self.num_codebooks == 0, f"Token length {seq_len} not divisible by {self.num_codebooks}"
        num_frames = seq_len // self.num_codebooks
        
        tokens_3d = flat_tokens.view(b, num_frames, self.num_codebooks)
        offsets = torch.arange(self.num_codebooks, device=flat_tokens.device) * self.codebook_size + self.audio_vocab_offset
        offsets = offsets.view(1, 1, self.num_codebooks)
        
        raw_codes = tokens_3d - offsets
        # Permute to [b, num_codebooks, num_frames]
        audio_codes = raw_codes.permute(0, 2, 1).contiguous()
        return audio_codes if is_batched else audio_codes.squeeze(0)


class FrameStopHead(nn.Module):
    """Lightweight frame termination head predicting binary stop signal per frame."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = nn.GELU()
        self.classifier = nn.Linear(hidden_dim // 2, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns logits for stop probability [..., 1]."""
        x = self.dense(hidden_states)
        x = self.act(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)


class RumikModelWrapper(nn.Module):
    """Wrapper around Rumik-OSS-1 decoder transformer + frame stop head + Mimi codec interface."""

    def __init__(
        self,
        transformer: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mimi_model: Optional[Any] = None,
        stop_head: Optional[FrameStopHead] = None,
        token_layout: Optional[TokenLayoutManager] = None
    ):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.mimi = mimi_model
        self.token_layout = token_layout or TokenLayoutManager()
        
        # Attach or initialize stop head
        hidden_dim = getattr(transformer.config, "hidden_size", 2048)
        self.stop_head = stop_head or getattr(transformer, "stop_head", FrameStopHead(hidden_dim))

    @classmethod
    def load(
        cls,
        model_name_or_path: str = "rumik-ai/rumik-oss-1",
        mimi_model_id: str = "kyutai/mimi",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        load_mimi: bool = True
    ) -> "RumikModelWrapper":
        """Loads Rumik-OSS-1 checkpoint with tokenizer and frozen Mimi codec."""
        print(f"[*] Loading tokenizer for {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        print(f"[*] Loading base transformer {model_name_or_path}...")
        transformer = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device == "cpu":
            transformer = transformer.to(device)

        mimi_model = None
        if load_mimi:
            print(f"[*] Loading frozen Mimi codec ({mimi_model_id})...")
            try:
                if MimiModel is not None:
                    mimi_model = MimiModel.from_pretrained(mimi_model_id).to(device)
                else:
                    from transformers import AutoModel
                    mimi_model = AutoModel.from_pretrained(mimi_model_id, trust_remote_code=True).to(device)
                mimi_model.eval()
                for p in mimi_model.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[!] Warning: Could not load Mimi directly via HF ({e}). Initializing fallback Mimi interface.")
                mimi_model = None

        text_vocab_size = len(tokenizer)
        num_codebooks = 8
        codebook_size = 2048
        layout = TokenLayoutManager(
            text_vocab_size=text_vocab_size,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size
        )

        hidden_dim = getattr(transformer.config, "hidden_size", 2048)
        stop_head = getattr(transformer, "stop_head", None)
        if stop_head is None:
            stop_head = FrameStopHead(hidden_dim).to(device=device, dtype=torch_dtype)

        return cls(
            transformer=transformer,
            tokenizer=tokenizer,
            mimi_model=mimi_model,
            stop_head=stop_head,
            token_layout=layout
        )

    def get_config_summary(self) -> Dict[str, Any]:
        """Returns key architecture details for inspection."""
        cfg = self.transformer.config
        return {
            "model_type": getattr(cfg, "model_type", "cohere2"),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", "N/A")),
            "hidden_size": getattr(cfg, "hidden_size", getattr(cfg, "hidden_dim", "N/A")),
            "num_attention_heads": getattr(cfg, "num_attention_heads", "N/A"),
            "vocab_size": getattr(cfg, "vocab_size", "N/A"),
            "text_vocab_size": self.token_layout.text_vocab_size,
            "num_codebooks": self.token_layout.num_codebooks,
            "codebook_size": self.token_layout.codebook_size,
            "calculated_total_vocab": self.token_layout.total_vocab_size,
            "vocab_match": getattr(cfg, "vocab_size", None) == self.token_layout.total_vocab_size,
            "has_stop_head": self.stop_head is not None,
            "mimi_loaded": self.mimi is not None,
            "mimi_sample_rate": 24000,
            "mimi_frame_rate": 12.5,
        }
