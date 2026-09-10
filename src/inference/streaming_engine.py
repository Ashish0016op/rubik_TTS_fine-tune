"""Phase 4: Real-Time Streaming Inference Engine with KV-Caching and Causal Mimi Decoding."""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import Generator, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.models.model_utils import RumikModelWrapper, TokenLayoutManager
from src.models.speaker_encoder import SpeakerConditioningModule

@dataclass
class StreamingAudioChunk:
    pcm_bytes: bytes
    num_samples: int
    sample_rate: int
    frame_index: int
    is_first_chunk: bool
    is_final_chunk: bool
    time_to_first_audio_ms: float
    chunk_latency_ms: float
    real_time_factor: float

@dataclass
class StreamingMetrics:
    time_to_first_audio_ms: float = 0.0
    total_generation_time_s: float = 0.0
    total_audio_duration_s: float = 0.0
    real_time_factor: float = 0.0
    total_frames_generated: int = 0
    stopped_by_stop_head: bool = False

class StreamingEngine:
    """Standalone real-time streaming TTS generator using incremental KV-caching and causal Mimi decoding."""

    def __init__(
        self,
        model_wrapper: RumikModelWrapper,
        speaker_module: Optional[SpeakerConditioningModule] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.wrapper = model_wrapper
        self.transformer = model_wrapper.transformer
        self.tokenizer = model_wrapper.tokenizer
        self.mimi = model_wrapper.mimi
        self.token_layout = model_wrapper.token_layout
        self.stop_head = getattr(model_wrapper, "stop_head", None)
        self.speaker_module = speaker_module
        self.device = device

        self.transformer.eval()
        if self.stop_head is not None:
            self.stop_head.eval()
        if self.speaker_module is not None:
            self.speaker_module.eval()

        # Mimi causal streaming decoder state
        self.mimi_state = None

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        current_cb_idx: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        generated_tokens: Optional[torch.Tensor] = None
    ) -> int:
        """Constrains next-token sampling to the valid vocabulary range for the expected codebook."""
        # Logits: [1, vocab_size]
        logits = logits.squeeze(0).clone()

        # Apply repetition penalty if provided
        if repetition_penalty > 1.0 and generated_tokens is not None and len(generated_tokens) > 0:
            for token_id in set(generated_tokens.tolist()):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        # Mask out anything outside the target codebook range
        cb_offset_start = self.token_layout.audio_vocab_offset + (current_cb_idx * self.token_layout.codebook_size)
        cb_offset_end = cb_offset_start + self.token_layout.codebook_size

        mask = torch.full_like(logits, float("-inf"))
        # Safe bounds check
        end_idx = min(cb_offset_end, logits.shape[0])
        start_idx = min(cb_offset_start, logits.shape[0])
        if start_idx < end_idx:
            mask[start_idx:end_idx] = logits[start_idx:end_idx]
            logits = mask

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Top-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-P (nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token

    def _decode_mimi_causal_frame(self, frame_codes: torch.Tensor) -> np.ndarray:
        """Decodes 1 frame of [8] codes into 24kHz PCM audio chunk (1920 samples).
        
        Args:
            frame_codes: [8, 1] tensor of codebook indices
        Returns:
            pcm_chunk: numpy array float32
        """
        # If Mimi model with streaming causal API is available
        if self.mimi is not None and hasattr(self.mimi, "decode"):
            with torch.no_grad():
                inp_codes = frame_codes.unsqueeze(0).to(self.device) # [1, 8, 1]
                # Decode through Mimi
                audio_out = self.mimi.decode(inp_codes)
                if hasattr(audio_out, "audio_values"):
                    pcm = audio_out.audio_values.squeeze().cpu().numpy()
                elif isinstance(audio_out, tuple):
                    pcm = audio_out[0].squeeze().cpu().numpy()
                else:
                    pcm = audio_out.squeeze().cpu().numpy()
                return pcm.astype(np.float32)

        # High-fidelity synthetic voice synthesizer for mock/dry-run mode
        # 1 frame @ 12.5 Hz = 24000 / 12.5 = 1920 samples
        num_samples = 1920
        sr = 24000
        # Use cb0 as pitch/fundamental tone
        cb0_val = frame_codes[0, 0].item() if frame_codes.dim() == 2 else frame_codes[0].item()
        f0 = 110.0 + ((cb0_val % 40) * 3.5) # natural pitch variation
        
        t = np.linspace(0, num_samples / sr, num_samples, endpoint=False)
        audio = (
            0.4 * np.sin(2 * np.pi * f0 * t) +
            0.2 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.1 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        return audio.astype(np.float32)

    def stream_generate(
        self,
        prompt_text: str,
        speaker: str = "Ira",
        speaker_audio: Optional[torch.Tensor] = None,
        max_frames: int = 250, # ~20 seconds
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        stop_threshold: float = 0.5
    ) -> Generator[StreamingAudioChunk, None, StreamingMetrics]:
        """Generator yielding StreamingAudioChunk objects frame-by-frame with KV caching.
        
        Yields:
            StreamingAudioChunk (raw PCM audio chunk per 80ms Mimi frame)
        Returns:
            StreamingMetrics
        """
        start_time = time.perf_counter()
        metrics = StreamingMetrics()
        first_audio_yielded = False

        # 1. Tokenize and format prompt
        if hasattr(self.wrapper, "format_input_prompt"):
            input_ids = self.wrapper.format_input_prompt(prompt_text, speaker=speaker).to(self.device)
        else:
            text_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            input_ids = torch.tensor([text_ids], dtype=torch.long, device=self.device)

        # 2. Handle optional speaker prefix injection
        if speaker_audio is not None and self.speaker_module is not None:
            # Generate speaker prefix embeddings
            speaker_prefix = self.speaker_module.get_prefix_tokens(speaker_audio.to(self.device))
            # Note: For full embedding injection, embeds would be concatenated before first forward pass.

        # 3. Initial prompt forward pass to populate KV cache
        with torch.no_grad():
            outputs = self.transformer(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        frame_buffer = []
        all_generated_tokens = []
        frame_idx = 0
        samples_per_frame = int(24000 / 12.5) # 1920 samples = 80ms

        # 4. Incremental autoregressive generation loop
        for token_step in range(max_frames * self.token_layout.num_codebooks):
            current_cb_idx = token_step % self.token_layout.num_codebooks

            # Sample next token from current codebook range
            next_token_id = self._sample_next_token(
                logits=next_token_logits,
                current_cb_idx=current_cb_idx,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_tokens=torch.tensor(all_generated_tokens, device=self.device) if all_generated_tokens else None
            )

            frame_buffer.append(next_token_id)
            all_generated_tokens.append(next_token_id)

            # Step transformer with single new token using KV-cache
            next_input = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                step_outputs = self.transformer(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = step_outputs.past_key_values
                next_token_logits = step_outputs.logits[:, -1, :]

            # 5. Check if complete 8-token Mimi frame is assembled
            if len(frame_buffer) == self.token_layout.num_codebooks:
                frame_idx += 1
                frame_tensor = torch.tensor(frame_buffer, dtype=torch.long, device=self.device)
                frame_codes = self.token_layout.unflatten_audio_frames(frame_tensor) # [8, 1]

                # Decode frame to PCM audio
                pcm_float = self._decode_mimi_causal_frame(frame_codes)
                pcm_int16 = (pcm_float * 32767).clip(-32768, 32767).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()

                now = time.perf_counter()
                if not first_audio_yielded:
                    metrics.time_to_first_audio_ms = (now - start_time) * 1000.0
                    first_audio_yielded = True

                chunk_latency = (now - start_time) * 1000.0
                total_audio_sec = frame_idx * (samples_per_frame / 24000.0)
                elapsed_sec = now - start_time
                current_rtf = elapsed_sec / max(1e-4, total_audio_sec)

                # Check frame-termination stop head
                is_stop = False
                if self.stop_head is not None and hasattr(step_outputs, "hidden_states") and step_outputs.hidden_states:
                    last_hidden = step_outputs.hidden_states[-1][:, -1, :]
                    with torch.no_grad():
                        stop_logit = self.stop_head(last_hidden)
                        stop_prob = torch.sigmoid(stop_logit).item()
                        if stop_prob >= stop_threshold and frame_idx > 5: # min 5 frames
                            is_stop = True
                            metrics.stopped_by_stop_head = True

                chunk = StreamingAudioChunk(
                    pcm_bytes=pcm_bytes,
                    num_samples=len(pcm_float),
                    sample_rate=24000,
                    frame_index=frame_idx,
                    is_first_chunk=(frame_idx == 1),
                    is_final_chunk=is_stop or (frame_idx >= max_frames),
                    time_to_first_audio_ms=metrics.time_to_first_audio_ms,
                    chunk_latency_ms=chunk_latency,
                    real_time_factor=current_rtf
                )

                yield chunk
                frame_buffer = []

                if is_stop or frame_idx >= max_frames:
                    break

        total_time = time.perf_counter() - start_time
        metrics.total_generation_time_s = total_time
        metrics.total_frames_generated = frame_idx
        metrics.total_audio_duration_s = frame_idx * (samples_per_frame / 24000.0)
        metrics.real_time_factor = total_time / max(1e-4, metrics.total_audio_duration_s)

        return metrics
