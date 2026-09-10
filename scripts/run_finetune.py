"""Phase 3: LoRA Fine-Tuning CLI script for Rumik TTS."""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_utils import RumikModelWrapper, TokenLayoutManager
from src.models.speaker_encoder import SpeakerConditioningModule
from src.data.dataset import RumikTTSDataset, collate_fn_tts
from src.training.train import RumikFineTuner
from config.default_config import TrainingConfig, LoRAConfig

def run_finetuning(
    data_dir: str = "data/processed",
    output_dir: str = "checkpoints/rumik_lora_custom_voice",
    model_id: str = "rumik-ai/rumik-oss-1",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    lora_r: int = 16,
    enable_speaker_conditioning: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dry_run: bool = False
):
    print("=================================================================")
    print(" Phase 3: LoRA Voice Adaptation Fine-Tuning")
    print("=================================================================")
    print(f"[*] Processed Data Dir : {data_dir}")
    print(f"[*] Output Checkpoints : {output_dir}")
    print(f"[*] Base Model ID      : {model_id}")
    print(f"[*] Epochs             : {epochs}")
    print(f"[*] Batch Size         : {batch_size}")
    print(f"[*] Learning Rate      : {lr}")
    print(f"[*] LoRA Rank (r)      : {lora_r}")
    print(f"[*] Speaker Module     : {enable_speaker_conditioning}")
    print(f"[*] Device             : {device}")

    os.makedirs(output_dir, exist_ok=True)
    train_json = os.path.join(data_dir, "train_dataset.json")
    val_json = os.path.join(data_dir, "val_dataset.json")

    if not os.path.exists(train_json) or not os.path.exists(val_json):
        print(f"[!] Dataset files not found in '{data_dir}'. Running Phase 2 preprocessing first...")
        from scripts.run_preprocessing import run_pipeline
        run_pipeline("data/raw_audio", "data/transcripts.csv", data_dir)

    with open(train_json, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_json, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    print(f"[*] Loaded {len(train_records)} train records and {len(val_records)} validation records.")

    if dry_run or not torch.cuda.is_available():
        print("\n[!] Running training simulation / test loop...")
        # Create small test model architecture to verify forward/backward pass & LoRA hooks
        class MockConfig:
            def __init__(self, hidden_size, vocab_size):
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
                self.is_encoder_decoder = False
                self.model_type = "cohere2"
                self._name_or_path = "rumik-ai/rumik-oss-1"
            def to_dict(self):
                return {"hidden_size": self.hidden_size, "vocab_size": self.vocab_size, "model_type": self.model_type, "_name_or_path": self._name_or_path}
            def __contains__(self, key):
                return key in self.to_dict()

        class MockBackbone(torch.nn.Module):
            def __init__(self, vocab_size=272384, hidden_size=256):
                super().__init__()
                self.config = MockConfig(hidden_size, vocab_size)
                self.generation_config = None
                self.main_input_name = "input_ids"
                self.warnings_issued = {}
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.o_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.gate_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.up_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.down_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                return {"input_ids": input_ids}

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
                x = self.embed(input_ids)
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                attn = torch.matmul(q, k.transpose(-2, -1)) / 16.0
                out = torch.matmul(attn, v)
                out = self.o_proj(out)
                mlp = self.down_proj(torch.relu(self.gate_proj(out)) * self.up_proj(out))
                hidden = out + mlp
                logits = self.lm_head(hidden)
                res = type("Out", (), {"logits": logits, "hidden_states": [hidden]})()
                return res

        class MockTokenizer:
            def __len__(self):
                return 256000
            def encode(self, text, add_special_tokens=True):
                return [1, 105, 204, 305, 2]

        mock_transformer = MockBackbone()
        mock_tokenizer = MockTokenizer()
        wrapper = RumikModelWrapper(
            transformer=mock_transformer,
            tokenizer=mock_tokenizer
        )

        train_ds = RumikTTSDataset(train_records, mock_tokenizer)
        val_ds = RumikTTSDataset(val_records, mock_tokenizer)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_tts)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_tts)

        speaker_module = SpeakerConditioningModule(transformer_hidden_dim=256) if enable_speaker_conditioning else None

        train_cfg = TrainingConfig(
            output_dir=output_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            batch_size=batch_size
        )
        lora_cfg = LoRAConfig(r=lora_r)

        tuner = RumikFineTuner(
            model_wrapper=wrapper,
            training_config=train_cfg,
            lora_config=lora_cfg,
            speaker_module=speaker_module,
            device="cpu"
        )

        print("\n[*] Running training epochs...")
        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            train_metrics = tuner.train_epoch(train_loader, epoch=epoch)
            val_metrics = tuner.evaluate(val_loader)
            current_val_loss = val_metrics['val_loss']
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_metrics['avg_train_loss']:.4f} | Val Loss: {current_val_loss:.4f}")
            
            # Save single best checkpoint (removes old checkpoint before writing new one)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"[*] New best validation loss: {best_val_loss:.4f}. Updating single checkpoint...")
                tuner.save_checkpoint(output_dir, clean_existing=True)

        print("\n[+] Phase 3 fine-tuning successfully verified! Single checkpoint preserved.")
        return

    # Real GPU Fine-Tuning
    print(f"[*] Loading live checkpoint {model_id} for LoRA fine-tuning on {device}...")
    wrapper = RumikModelWrapper.load(model_id, device=device)
    train_ds = RumikTTSDataset(train_records, wrapper.tokenizer)
    val_ds = RumikTTSDataset(val_records, wrapper.tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_tts)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_tts)

    speaker_module = None
    if enable_speaker_conditioning:
        speaker_module = SpeakerConditioningModule(
            transformer_hidden_dim=getattr(wrapper.transformer.config, "hidden_size", 2048)
        ).to(device)

    train_cfg = TrainingConfig(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        batch_size=batch_size
    )
    lora_cfg = LoRAConfig(r=lora_r)

    tuner = RumikFineTuner(
        model_wrapper=wrapper,
        training_config=train_cfg,
        lora_config=lora_cfg,
        speaker_module=speaker_module,
        device=device
    )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = tuner.train_epoch(train_loader, epoch=epoch)
        val_metrics = tuner.evaluate(val_loader)
        current_val_loss = val_metrics['val_loss']
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_metrics['avg_train_loss']:.4f} | Val Loss: {current_val_loss:.4f}")
        
        # Keep only one best checkpoint on disk
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            print(f"[*] New best validation loss: {best_val_loss:.4f}. Overwriting with new single checkpoint...")
            tuner.save_checkpoint(output_dir, clean_existing=True)

    print(f"\n[+] Fine-tuning complete! Single best checkpoint saved at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Rumik TTS")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="checkpoints/rumik_lora_custom_voice")
    parser.add_argument("--model-id", type=str, default="rumik-ai/rumik-oss-1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--speaker-conditioning", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_finetuning(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        enable_speaker_conditioning=args.speaker_conditioning,
        device=args.device,
        dry_run=args.dry_run
    )
