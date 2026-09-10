"""Default configuration parameters for Rumik TTS training, streaming, and serving."""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AudioConfig:
    sample_rate: int = 24000
    frame_rate: float = 12.5  # 12.5 Hz (frames per second)
    num_codebooks: int = 8    # 8 RVQ codebooks per frame
    codebook_size: int = 2048 # 2048 entries per codebook
    total_tokens_per_sec: int = 100 # 12.5 * 8 = 100 tokens/sec
    pcm_format: str = "int16" # int16 or float32 for streaming chunks
    silence_db_threshold: float = 30.0 # for VAD trimming

@dataclass
class ModelConfig:
    model_name_or_path: str = "rumik-ai/rumik-oss-1"
    mimi_model_id: str = "kyutai/mimi"
    text_vocab_size: int = 256000 # default Cohere2 vocab
    audio_vocab_offset: int = 256000
    total_vocab_size: int = 256000 + (8 * 2048) # text_vocab + 8 * 2048 = 272384
    has_stop_head: bool = True
    stop_head_threshold: float = 0.5
    device: str = "cuda" # or "cpu"

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"

@dataclass
class SpeakerEncoderConfig:
    enabled: bool = False
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    inject_mode: str = "prefix" # "prefix" or "cross_attention"

@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints/rumik_lora_custom_voice"
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 15
    max_steps: int = -1
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    fp16: bool = True
    replay_ratio: float = 0.15 # General data mix to prevent catastrophic forgetting

@dataclass
class StreamingConfig:
    chunk_frames: int = 1 # 1 Mimi frame = 8 tokens = 80ms of audio
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_frames: int = 300 # ~24 seconds of audio (300 * 8 = 2400 tokens)
    stop_threshold: float = 0.5

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
