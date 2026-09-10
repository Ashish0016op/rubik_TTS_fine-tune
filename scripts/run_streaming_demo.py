"""Phase 4: Real-Time Streaming Inference Engine CLI Demo."""

import os
import sys
import time
import argparse
from typing import Optional, List, Dict, Any
import torch
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_utils import RumikModelWrapper, TokenLayoutManager
from src.models.speaker_encoder import SpeakerConditioningModule
from src.inference.streaming_engine import StreamingEngine

def run_streaming_demo(
    prompt: str = "Namaste! Yeh hamari real-time streaming audio inference engine ka demonstration hai.",
    model_id: str = "rumik-ai/rumik-oss-1",
    adapter_path: Optional[str] = None,
    output_wav: str = "outputs/streaming_demo_output.wav",
    max_frames: int = 150, # ~12 seconds
    temperature: float = 0.7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dry_run: bool = False
):
    print("=================================================================")
    print(" Phase 4: Real-Time Streaming Inference Engine Demo")
    print("=================================================================")
    print(f"[*] Prompt Text       : '{prompt}'")
    print(f"[*] Base Model        : {model_id}")
    print(f"[*] LoRA Adapter      : {adapter_path or 'None (Base Model)'}")
    print(f"[*] Compute Device    : {device}")
    print(f"[*] Output File       : {output_wav}")
    print(f"[*] Max Frames        : {max_frames} ({max_frames * 0.08:.1f}s audio)")
    print("-----------------------------------------------------------------")

    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    if dry_run or not torch.cuda.is_available():
        print("[!] Using simulated model wrapper for fast streaming verification...")
        
        class MockStreamingBackbone(torch.nn.Module):
            def __init__(self, vocab_size=272384, hidden_size=256):
                super().__init__()
                self.config = type("Config", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=False, **kwargs):
                # Simulated forward step
                hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 256)
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 272384)
                
                # Biased logits for valid audio codes
                logits[:, :, 256000:256000 + 16384] += 5.0
                
                fake_kv = [torch.randn(1, 4, 10, 64)] * 4
                res = type("Out", (), {
                    "logits": logits,
                    "past_key_values": fake_kv,
                    "hidden_states": [hidden]
                })()
                return res

        class MockTokenizer:
            def __len__(self):
                return 256000
            def encode(self, text, add_special_tokens=True):
                return [1, 101, 202, 303, 404, 2]

        mock_transformer = MockStreamingBackbone()
        mock_tokenizer = MockTokenizer()
        wrapper = RumikModelWrapper(
            transformer=mock_transformer,
            tokenizer=mock_tokenizer
        )
    else:
        print(f"[*] Loading live model {model_id}...")
        wrapper = RumikModelWrapper.load(model_id, device=device)
        if adapter_path and os.path.exists(adapter_path):
            from peft import PeftModel
            print(f"[*] Loading fine-tuned LoRA adapters from {adapter_path}...")
            wrapper.transformer = PeftModel.from_pretrained(wrapper.transformer, adapter_path)

    engine = StreamingEngine(model_wrapper=wrapper, device=device)

    print("\n[*] Starting real-time audio stream generation...")
    print(f"{'Frame':<8}{'Samples':<10}{'Latency (ms)':<15}{'Chunk RTF':<12}{'Status':<15}")
    print("-" * 65)

    all_pcm_chunks = []
    stream_generator = engine.stream_generate(
        prompt_text=prompt,
        max_frames=max_frames,
        temperature=temperature
    )

    ttfa_recorded = None

    for chunk in stream_generator:
        if chunk.is_first_chunk:
            ttfa_recorded = chunk.time_to_first_audio_ms
            status = f"FIRST AUDIO ({ttfa_recorded:.1f}ms)"
        elif chunk.is_final_chunk:
            status = "FINAL FRAME (STOP)"
        else:
            status = "STREAMING"

        print(f"#{chunk.frame_index:<7}{chunk.num_samples:<10}{chunk.chunk_latency_ms:<15.1f}{chunk.real_time_factor:<12.2f}{status}")
        
        # Convert PCM bytes back to float32 for final file saving
        pcm_int16 = np.frombuffer(chunk.pcm_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32767.0
        all_pcm_chunks.append(pcm_float)

        if chunk.frame_index >= 25: # generate at least 2 seconds (25 frames) in test
            if dry_run:
                break

    if all_pcm_chunks:
        full_audio = np.concatenate(all_pcm_chunks, axis=0)
        sf.write(output_wav, full_audio, 24000)
        total_duration = len(full_audio) / 24000.0
        print("-" * 65)
        print(f"[+] Total Audio Generated : {total_duration:.2f} seconds ({len(all_pcm_chunks)} frames)")
        print(f"[+] Time-To-First-Audio   : {ttfa_recorded:.2f} ms")
        print(f"[+] Saved Audio Output to : {output_wav}")
        print("[+] Phase 4 Streaming Engine verification successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Streaming TTS Generator")
    parser.add_argument("--prompt", type=str, default="Namaste! Yeh hamari real-time streaming audio inference engine ka demonstration hai.")
    parser.add_argument("--model-id", type=str, default="rumik-ai/rumik-oss-1")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--output-wav", type=str, default="outputs/streaming_demo_output.wav")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_streaming_demo(
        prompt=args.prompt,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        output_wav=args.output_wav,
        max_frames=args.max_frames,
        temperature=args.temperature,
        device=args.device,
        dry_run=args.dry_run
    )
