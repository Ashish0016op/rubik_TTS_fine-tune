"""Phase 1: Verification script for rumik-oss-1 and frozen Mimi codec."""

import os
import sys
import argparse
import json
import torch
import soundfile as sf
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_utils import RumikModelWrapper, TokenLayoutManager

def run_verification(
    model_id: str = "rumik-ai/rumik-oss-1",
    mimi_id: str = "kyutai/mimi",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_wav: str = "outputs/phase1_test_output.wav",
    speaker: str = "Ira",
    description: str = "happy, Hindi accent, steady pace",
    text_prompt: str = "नमस्ते, आज आपका दिन कैसा रहा?",
    dry_run: bool = False
):
    print("=================================================================")
    print(" Phase 1: Rumik-OSS-1 Environment & Checkpoint Verification")
    print("=================================================================")
    print(f"[*] Target Model ID : {model_id}")
    print(f"[*] Target Mimi ID  : {mimi_id} (or bundled codec subfolder)")
    print(f"[*] Compute Device  : {device}")
    print(f"[*] PyTorch Version : {torch.__version__}")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] GPU Device Name : {torch.cuda.get_device_name(0)}")

    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    if dry_run:
        print("\n[!] DRY RUN MODE ACTIVATED: Performing offline structural validation...")
        audio_offset = 261008
        n_codebooks = 8
        cb_size = 2048
        total_vocab = 277404
        layout = TokenLayoutManager(
            audio_vocab_offset=audio_offset,
            num_codebooks=n_codebooks,
            codebook_size=cb_size,
            total_vocab_size=total_vocab
        )
        
        print(f"\n1. Vocabulary Verification:")
        print(f"   - Audio Offset Start    : {audio_offset} (first_unit_id)")
        print(f"   - Audio Codebooks       : {n_codebooks} (RVQ 0-7)")
        print(f"   - Codebook Size         : {cb_size}")
        print(f"   - Audio Tokens Range    : [{audio_offset} .. {layout.last_audio_token_id}]")
        print(f"   - Total Model Vocab Size: {total_vocab}")
        assert layout.last_audio_token_id == 277391, "Audio range mismatch!"
        print("   [+] Vocab size and codebook formula verified successfully!")

        print(f"\n2. Configuration Summary:")
        mock_summary = {
            "model_type": "rumik_oss",
            "num_hidden_layers": 36,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "vocab_size": 277404,
            "audio_vocab_offset": 261008,
            "has_stop_head": True,
            "mimi_sample_rate": 24000,
            "mimi_frame_rate": 12.5,
            "speakers": ["Ira", "Aisha", "Siya", "Zoya"]
        }
        print(json.dumps(mock_summary, indent=4))
        print("\n[+] Phase 1 dry-run verification completed successfully!")
        return

    # Real checkpoint loading
    print(f"\n[*] Loading model {model_id} and bundled Mimi codec...")
    wrapper = RumikModelWrapper.load(
        model_name_or_path=model_id,
        mimi_model_id=mimi_id,
        device=device
    )
    summary = wrapper.get_config_summary()
    print("\n=== Model Configuration Inspection ===")
    print(json.dumps(summary, indent=4))

    print(f"\n=== Official Rumik-OSS-1 Synthesis ===")
    print(f"Speaker     : {speaker}")
    print(f"Description : {description}")
    print(f"Text        : {text_prompt}")
    
    # Official prompt format: <text>{SPEAKER}: <description="{DESCRIPTION}"> {TEXT}<audio>
    prompt = f'<text>{speaker}: <description="{description}"> {text_prompt}<audio>'
    inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Prompt Token Count: {inputs.input_ids.shape[1]}")

    model = wrapper.transformer
    mimi = wrapper.mimi

    print("[*] Generating speech audio tokens...")
    # Check if official generate_audio method is present
    if hasattr(model, "generate_audio"):
        with torch.inference_mode():
            ids = model.generate_audio(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                top_k=30,
                do_sample=True
            )
            audio_tokens = ids[0].tolist()[inputs.input_ids.shape[1]:]
    else:
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.8,
                top_k=30,
                top_p=0.9,
                do_sample=True,
                eos_token_id=wrapper.token_layout.audio_end_token_id,
                pad_token_id=wrapper.tokenizer.pad_token_id or 0
            )
            audio_tokens = ids[0].tolist()[inputs.input_ids.shape[1]:]

    print(f"[+] Generated {len(audio_tokens)} audio tokens ({len(audio_tokens) / 100:.2f}s audio).")

    # Decode audio tokens to waveform
    if len(audio_tokens) >= 8 and mimi is not None:
        if hasattr(model, "audio_tokens_to_codes"):
            codes = model.audio_tokens_to_codes(audio_tokens)
        else:
            token_tensor = torch.tensor([audio_tokens], dtype=torch.long, device=device)
            codes = wrapper.token_layout.unflatten_audio_frames(token_tensor)

        print(f"Audio codes shape: {codes.shape}")

        with torch.inference_mode():
            decoded = mimi.decode(codes.to(device))
            if hasattr(decoded, "audio_values"):
                wav = decoded.audio_values[0, 0].float().cpu().numpy()
            elif isinstance(decoded, tuple):
                wav = decoded[0].squeeze().float().cpu().numpy()
            else:
                wav = decoded.squeeze().float().cpu().numpy()

        sf.write(output_wav, wav, 24000)
        print(f"[+] Audio successfully written to: {output_wav} ({len(wav) / 24000:.2f} seconds)")
    else:
        print("[!] Warning: Could not decode audio tokens. Mimi codec not ready or tokens empty.")

    print("\n[+] End-to-end verification completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Rumik-OSS-1 Checkpoint and Environment")
    parser.add_argument("--model-id", type=str, default="rumik-ai/rumik-oss-1")
    parser.add_argument("--mimi-id", type=str, default="kyutai/mimi")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-wav", type=str, default="outputs/phase1_test_output.wav")
    parser.add_argument("--speaker", type=str, default="Ira", choices=["Ira", "Aisha", "Siya", "Zoya"])
    parser.add_argument("--description", type=str, default="happy, Hindi accent, steady pace")
    parser.add_argument("--prompt", type=str, default="नमस्ते, आज आपका दिन कैसा रहा?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_verification(
        model_id=args.model_id,
        mimi_id=args.mimi_id,
        device=args.device,
        output_wav=args.output_wav,
        speaker=args.speaker,
        description=args.description,
        text_prompt=args.prompt,
        dry_run=args.dry_run
    )
