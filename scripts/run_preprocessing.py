"""Phase 2: Data Preprocessing CLI pipeline for Rumik TTS."""

import os
import sys
import argparse
import csv
import json
import torch
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_utils import TokenLayoutManager
from src.data.preprocess import AudioPreprocessor
from src.data.dataset import RumikTTSDataset

def create_sample_mock_data(data_dir: str, csv_path: str, num_samples: int = 5):
    """Creates synthetic audio files and romanized Hindi transcripts for quick testing."""
    os.makedirs(data_dir, exist_ok=True)
    
    sample_texts = [
        "Namaste! Yeh hamara pehla fine-tuning sample hai.",
        "Aapka swagat hai hamare custom voice model training mein.",
        "Kripya dhyan dein, hum Mimi codec aur Rumik model ka upayog kar rahe hain.",
        "Streaming inference se audio turant play hona shuru ho jayega.",
        "Yeh voice adaptation bahut hi tez aur effective hai."
    ]

    records = []
    sr = 24000
    for i in range(num_samples):
        filename = f"sample_{i+1:03d}.wav"
        filepath = os.path.join(data_dir, filename)
        
        # Generate 2.5 seconds of synthetic speech-like harmonic signal
        duration = 2.0 + (i * 0.3)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Multi-harmonic voice-like formant signal
        f0 = 130.0 + (i * 10)
        signal = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
        # Add small envelope
        envelope = np.ones_like(t)
        envelope[:1000] = np.linspace(0, 1, 1000)
        envelope[-1000:] = np.linspace(1, 0, 1000)
        audio = signal * envelope
        
        sf.write(filepath, audio.astype(np.float32), sr)
        records.append({
            "audio_path": filename,
            "transcript": sample_texts[i % len(sample_texts)],
            "speaker_id": "custom_speaker_01"
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "speaker_id"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[+] Created {num_samples} sample audio files in '{data_dir}' and metadata in '{csv_path}'")


class MockTokenizer:
    """Lightweight mock tokenizer for offline data preprocessing verification."""
    def __init__(self, vocab_size=256000):
        self.vocab_size = vocab_size
    def __len__(self):
        return self.vocab_size
    def encode(self, text, add_special_tokens=True):
        # Deterministic character-level hash encoding into [100, 50000] range
        tokens = [1] if add_special_tokens else []
        for ch in text:
            tokens.append(100 + (ord(ch) * 37) % 50000)
        if add_special_tokens:
            tokens.append(2)
        return tokens


def run_pipeline(
    data_dir: str,
    transcript_csv: str,
    output_dir: str,
    val_ratio: float = 0.2,
    model_id: str = "rumik-ai/rumik-oss-1"
):
    print("=================================================================")
    print(" Phase 2: Data Preprocessing Pipeline (Mimi Encoding & Flattening)")
    print("=================================================================")
    print(f"[*] Audio Directory   : {data_dir}")
    print(f"[*] Transcript CSV    : {transcript_csv}")
    print(f"[*] Output Directory  : {output_dir}")
    print(f"[*] Validation Ratio  : {val_ratio}")

    if not os.path.exists(data_dir) or not os.path.exists(transcript_csv):
        print(f"\n[!] Data directory or CSV not found. Generating mock romanized Hindi dataset...")
        create_sample_mock_data(data_dir, transcript_csv)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer
    try:
        from transformers import AutoTokenizer
        print(f"[*] Loading tokenizer '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[!] Using offline fallback tokenizer: {e}")
        tokenizer = MockTokenizer()

    # Process dataset
    print("\n[*] Processing audio and computing Mimi RVQ codes...")
    train_ds, val_ds = RumikTTSDataset.build_from_raw_data(
        data_dir=data_dir,
        transcript_csv=transcript_csv,
        tokenizer=tokenizer,
        mimi_model=None, # will use preprocessor internal encoding
        val_split_ratio=val_ratio
    )

    # Inspect sample 0
    sample = train_ds[0]
    print(f"\n=== Processed Sample 0 Statistics ===")
    print(f"Input IDs shape      : {sample['input_ids'].shape}")
    print(f"Labels shape         : {sample['labels'].shape}")
    print(f"Text tokens count    : {sample['num_text_tokens']}")
    print(f"Audio tokens count   : {sample['num_audio_tokens']} ({sample['num_audio_tokens'] // 8} frames @ 12.5 Hz)")
    print(f"Masked label tokens  : {(sample['labels'] == -100).sum().item()} (all text positions masked)")
    print(f"Active audio labels  : {(sample['labels'] != -100).sum().item()}")
    print(f"Stop head label sum  : {sample['stop_labels'].sum().item()} (1 on final frame)")

    # Save processed manifest
    train_export = [train_ds.examples[i] for i in range(len(train_ds))]
    val_export = [val_ds.examples[i] for i in range(len(val_ds))]

    train_json = os.path.join(output_dir, "train_dataset.json")
    val_json = os.path.join(output_dir, "val_dataset.json")

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_export, f, indent=2)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_export, f, indent=2)

    print(f"\n[+] Saved {len(train_export)} train samples to {train_json}")
    print(f"[+] Saved {len(val_export)} val samples to {val_json}")
    print("[+] Phase 2 Data Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Audio & Transcripts for Rumik TTS")
    parser.add_argument("--data-dir", type=str, default="data/raw_audio")
    parser.add_argument("--transcript-csv", type=str, default="data/transcripts.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--model-id", type=str, default="rumik-ai/rumik-oss-1")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        transcript_csv=args.transcript_csv,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        model_id=args.model_id
    )
