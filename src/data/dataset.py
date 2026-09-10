"""Dataset definitions and preprocessing pipeline for Rumik TTS fine-tuning."""

import os
import csv
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from src.models.model_utils import TokenLayoutManager
from src.data.preprocess import AudioPreprocessor

try:
    from datasets import Dataset as HFDataset, DatasetDict
except ImportError:
    HFDataset = None
    DatasetDict = None

class RumikTTSDataset(Dataset):
    """PyTorch Dataset for supervised fine-tuning of Rumik-OSS-1 with audio-masked labels."""

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        token_layout: Optional[TokenLayoutManager] = None,
        max_seq_len: int = 4096
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.token_layout = token_layout or TokenLayoutManager()
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        
        # 1. Tokenize text prompt with official template <text>{speaker}: <description="..."> {text}<audio>
        text = item["transcript"]
        speaker = item.get("speaker_id", "Ira")
        formatted_prompt = f'<text>{speaker}: <description="natural clear Hindi pronunciation"> {text}<audio>'
        text_tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
        text_tensor = torch.tensor(text_tokens, dtype=torch.long)
        
        # 2. Get audio codes [8, num_frames] or flattened tokens
        if "audio_codes" in item:
            audio_codes = item["audio_codes"]
            if not isinstance(audio_codes, torch.Tensor):
                audio_codes = torch.tensor(audio_codes, dtype=torch.long)
            flat_audio_tokens = self.token_layout.flatten_audio_frames(audio_codes)
        elif "audio_tokens" in item:
            flat_audio_tokens = torch.tensor(item["audio_tokens"], dtype=torch.long)
        else:
            raise ValueError(f"Example {idx} has neither 'audio_codes' nor 'audio_tokens'")

        # 3. Concatenate text + audio tokens
        input_ids = torch.cat([text_tensor, flat_audio_tokens], dim=0)
        
        # 4. Create labels: MASK OUT text positions (-100), preserve audio tokens
        labels = torch.cat([
            torch.full_like(text_tensor, -100),
            flat_audio_tokens.clone()
        ], dim=0)

        # 5. Attention mask (all 1s)
        attention_mask = torch.ones_like(input_ids)

        # 6. Stop head labels: 0 for all audio frames except 1 on the last frame
        num_frames = flat_audio_tokens.shape[0] // self.token_layout.num_codebooks
        stop_labels = torch.zeros(num_frames, dtype=torch.float32)
        if num_frames > 0:
            stop_labels[-1] = 1.0

        # Truncate if exceeds max_seq_len
        if input_ids.shape[0] > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "stop_labels": stop_labels,
            "num_text_tokens": len(text_tensor),
            "num_audio_tokens": len(flat_audio_tokens)
        }

    @classmethod
    def build_from_raw_data(
        cls,
        data_dir: str,
        transcript_csv: str,
        tokenizer: PreTrainedTokenizer,
        mimi_model: Optional[Any] = None,
        val_split_ratio: float = 0.1,
        device: str = "cpu"
    ) -> Tuple["RumikTTSDataset", "RumikTTSDataset"]:
        """Loads audio files and transcript CSV, encodes through Mimi, and creates train/val splits."""
        preprocessor = AudioPreprocessor(target_sample_rate=24000)
        token_layout = TokenLayoutManager()
        
        processed_items = []
        with open(transcript_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"[*] Found {len(rows)} audio transcript entries. Starting preprocessing...")
        from tqdm import tqdm
        progress = tqdm(rows, desc="Processing Audio (24kHz + Mimi RVQ)", unit="file")
        
        for row in progress:
            audio_name = row.get("audio_path") or row.get("filename") or row.get("file")
            transcript = row.get("transcript") or row.get("text") or row.get("sentence")
            speaker_id = row.get("speaker_id", "default_speaker")
            
            full_audio_path = os.path.join(data_dir, audio_name) if not os.path.isabs(audio_name) else audio_name
            if not os.path.exists(full_audio_path):
                # check if direct relative path
                if os.path.exists(audio_name):
                    full_audio_path = audio_name
                else:
                    progress.write(f"[!] Skipping missing audio file: {full_audio_path}")
                    continue

            # Preprocess audio (24kHz mono, VAD trim)
            waveform = preprocessor.load_and_preprocess(full_audio_path)
            
            # Encode through frozen Mimi
            audio_codes = preprocessor.encode_with_mimi(waveform, mimi_model=mimi_model, device=device)
            
            duration = waveform.shape[-1] / 24000.0
            processed_items.append({
                "audio_path": full_audio_path,
                "transcript": transcript,
                "speaker_id": speaker_id,
                "audio_codes": audio_codes.tolist(),
                "duration_sec": duration
            })
            progress.set_postfix({"processed": len(processed_items), "dur_s": f"{duration:.1f}"})

        # Train / Validation Split
        n_total = len(processed_items)
        n_val = max(1, int(n_total * val_split_ratio)) if n_total > 1 else 0
        n_train = n_total - n_val

        train_items = processed_items[:n_train]
        val_items = processed_items[n_train:] if n_val > 0 else processed_items

        print(f"[*] Processed {n_total} total examples ({n_train} train, {n_val} val).")
        train_dataset = cls(train_items, tokenizer, token_layout)
        val_dataset = cls(val_items, tokenizer, token_layout)

        return train_dataset, val_dataset

    def to_huggingface_dataset(self):
        """Converts internal examples to HuggingFace Dataset format."""
        if HFDataset is None:
            raise ImportError("HuggingFace 'datasets' library is not installed.")
            
        data_records = []
        from tqdm import tqdm
        for i in tqdm(range(len(self)), desc="Formatting HuggingFace Dataset", unit="sample"):
            sample = self[i]
            data_records.append({
                "input_ids": sample["input_ids"].tolist(),
                "labels": sample["labels"].tolist(),
                "attention_mask": sample["attention_mask"].tolist(),
                "stop_labels": sample["stop_labels"].tolist()
            })
        return HFDataset.from_list(data_records)


def collate_fn_tts(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collates variable-length token sequences into padded batches."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    
    # Pad sequences
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_masks
    }
