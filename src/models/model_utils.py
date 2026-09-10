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
    
    Official Rumik-OSS-1 Layout:
    - Total Vocab Size: 277,404
    - Text Tokens & Special Tokens: [0, 261,007]
    - Audio Tokens (8 codebooks x 2048 entries): [261,008, 277,391]
        - Codebook 0: [261,008, 263,055]
        - Codebook 1: [263,056, 265,103]
        - ...
        - Codebook 7: [275,344, 277,391]
    - Special Tags:
        - text_start_token_id: 277,392
        - audio_start_token_id: 277,393
        - audio_end_token_id (Stop Token): 277,394
    """

    def __init__(
        self,
        audio_vocab_offset: int = 261008,
        num_codebooks: int = 8,
        codebook_size: int = 2048,
        total_vocab_size: int = 277404,
        text_start_token_id: int = 277392,
        audio_start_token_id: int = 277393,
        audio_end_token_id: int = 277394
    ):
        self.audio_vocab_offset = audio_vocab_offset
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.total_vocab_size = total_vocab_size
        self.last_audio_token_id = audio_vocab_offset + (num_codebooks * codebook_size) - 1 # 277391
        self.text_start_token_id = text_start_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

    def is_audio_token(self, token_id: int) -> bool:
        return self.audio_vocab_offset <= token_id <= self.last_audio_token_id

    def audio_code_to_token_id(self, codebook_idx: int, code_value: int) -> int:
        """Formula: tid = first_unit_id + code * num_codebooks + codebook_idx"""
        assert 0 <= codebook_idx < self.num_codebooks, f"Invalid codebook index: {codebook_idx}"
        code_value = max(0, min(self.codebook_size - 1, code_value))
        return self.audio_vocab_offset + (code_value * self.num_codebooks) + codebook_idx

    def token_id_to_audio_code(self, token_id: int) -> Tuple[int, int]:
        """Formula: code, q = divmod(tid - first_unit_id, num_codebooks)"""
        if not self.is_audio_token(token_id):
            return 0, 0
        code_val, cb_idx = divmod(token_id - self.audio_vocab_offset, self.num_codebooks)
        return int(cb_idx), int(code_val)

    def flatten_audio_frames(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Converts [batch, num_codebooks, num_frames] or [num_codebooks, num_frames] to interleaved token IDs.
        
        Official Rumik-OSS-1 Interleaving:
        Each frame has 8 tokens (q=0..7). Token ID = first_unit_id + code * 8 + q.
        """
        is_batched = audio_codes.dim() == 3
        if not is_batched:
            audio_codes = audio_codes.unsqueeze(0)
            
        b, k, t = audio_codes.shape
        assert k == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {k}"

        # Clamp input codes safely to [0, codebook_size - 1]
        audio_codes = torch.clamp(audio_codes, 0, self.codebook_size - 1).long()

        # Transpose to [b, t, k] for frame-major ordering
        transposed = audio_codes.permute(0, 2, 1).contiguous()
        
        # Quantizer indices [0, 1, 2, 3, 4, 5, 6, 7]
        q_indices = torch.arange(k, device=audio_codes.device, dtype=torch.long).view(1, 1, k)
        
        # Token ID = 261008 + code * 8 + q
        flat_tokens = (transposed * self.num_codebooks + q_indices + self.audio_vocab_offset).view(b, t * k)
        return flat_tokens if is_batched else flat_tokens.squeeze(0)

    def unflatten_audio_frames(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        """Converts interleaved flat audio token IDs back to [..., num_codebooks, num_frames].
        
        Matches official RumikOSSForCausalLM.audio_tokens_to_codes implementation.
        """
        is_batched = flat_tokens.dim() == 2
        if not is_batched:
            flat_tokens = flat_tokens.unsqueeze(0)
            
        b, seq_len = flat_tokens.shape
        valid_len = (seq_len // self.num_codebooks) * self.num_codebooks
        if valid_len < seq_len:
            flat_tokens = flat_tokens[:, :valid_len]
            
        num_frames = valid_len // self.num_codebooks
        if num_frames == 0:
            return torch.zeros((b, self.num_codebooks, 0), dtype=torch.long, device=flat_tokens.device)

        tokens_3d = flat_tokens.view(b, num_frames, self.num_codebooks)
        q_indices = torch.arange(self.num_codebooks, device=flat_tokens.device, dtype=torch.long).view(1, 1, self.num_codebooks)
        
        # code = (tid - 261008 - q) // 8
        raw_codes = (tokens_3d - self.audio_vocab_offset - q_indices) // self.num_codebooks
        raw_codes = torch.clamp(raw_codes, 0, self.codebook_size - 1).long()
        
        # Permute to [b, num_codebooks, num_frames]
        audio_codes = raw_codes.permute(0, 2, 1).contiguous()
        return audio_codes if is_batched else audio_codes.squeeze(0)


class FrameStopHead(nn.Module):
    """Lightweight frame termination head predicting binary stop signal per frame."""

    def __init__(self, hidden_dim: int = 2048):
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
        
        hidden_dim = getattr(transformer.config, "hidden_size", 2048)
        self.stop_head = stop_head or getattr(transformer, "stop_head", FrameStopHead(hidden_dim))

    @classmethod
    def load(
        cls,
        model_name_or_path: str = "rumik-ai/rumik-oss-1",
        mimi_model_id: str = "kyutai/mimi",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        load_mimi: bool = True
    ) -> "RumikModelWrapper":
        """Loads Rumik-OSS-1 checkpoint with tokenizer and frozen Mimi codec."""
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"[*] Loading tokenizer for {model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        print(f"[*] Loading base transformer {model_name_or_path} (device={device})...")
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
            print(f"[*] Loading frozen Mimi codec from {model_name_or_path} (subfolder='codec')...")
            try:
                if MimiModel is not None:
                    try:
                        mimi_model = MimiModel.from_pretrained(model_name_or_path, subfolder="codec").to(device)
                    except Exception:
                        mimi_model = MimiModel.from_pretrained(mimi_model_id).to(device)
                else:
                    from transformers import AutoModel
                    try:
                        mimi_model = AutoModel.from_pretrained(model_name_or_path, subfolder="codec", trust_remote_code=True).to(device)
                    except Exception:
                        mimi_model = AutoModel.from_pretrained(mimi_model_id, trust_remote_code=True).to(device)
                mimi_model.eval()
                for p in mimi_model.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[!] Warning: Could not load Mimi directly via HF ({e}). Initializing fallback Mimi interface.")
                mimi_model = None

        # Extract exact offsets from config if available
        cfg = transformer.config
        audio_vocab_offset = getattr(cfg, "first_unit_id", 261008)
        num_codebooks = getattr(cfg, "num_quantizers", 8)
        codebook_size = getattr(cfg, "codebook_size", 2048)
        total_vocab_size = getattr(cfg, "vocab_size", 277404)
        text_start_token_id = getattr(cfg, "text_start_token_id", 277392)
        audio_start_token_id = getattr(cfg, "audio_start_token_id", 277393)
        audio_end_token_id = getattr(cfg, "audio_end_token_id", 277394)

        layout = TokenLayoutManager(
            audio_vocab_offset=audio_vocab_offset,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            total_vocab_size=total_vocab_size,
            text_start_token_id=text_start_token_id,
            audio_start_token_id=audio_start_token_id,
            audio_end_token_id=audio_end_token_id
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

    def format_input_prompt(
        self,
        text: str,
        speaker: str = "Ira",
        description: str = "natural, clear Hindi pronunciation, conversational tone"
    ) -> torch.Tensor:
        """Formats input text with official Rumik-OSS-1 prompt structure:
        <text>{speaker}: <description="{description}"> {text}<audio>
        """
        prompt_str = f'<text>{speaker}: <description="{description}"> {text}<audio>'
        text_tokens = self.tokenizer.encode(prompt_str, add_special_tokens=True)
        return torch.tensor([text_tokens], dtype=torch.long)

    def get_config_summary(self) -> Dict[str, Any]:
        """Returns key architecture details for inspection."""
        cfg = self.transformer.config
        return {
            "model_type": getattr(cfg, "model_type", "rumik_oss"),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", 36),
            "hidden_size": getattr(cfg, "hidden_size", 2048),
            "num_attention_heads": getattr(cfg, "num_attention_heads", 16),
            "vocab_size": getattr(cfg, "vocab_size", 277404),
            "audio_vocab_offset": self.token_layout.audio_vocab_offset,
            "num_codebooks": self.token_layout.num_codebooks,
            "codebook_size": self.token_layout.codebook_size,
            "total_audio_vocab_range": f"[{self.token_layout.audio_vocab_offset} .. {self.token_layout.last_audio_token_id}]",
            "vocab_match": getattr(cfg, "vocab_size", 277404) == self.token_layout.total_vocab_size,
            "has_stop_head": self.stop_head is not None,
            "mimi_loaded": self.mimi is not None,
            "mimi_sample_rate": 24000,
            "mimi_frame_rate": 12.5,
            "speakers": getattr(cfg, "speakers", ["Ira", "Aisha", "Siya", "Zoya"])
        }
