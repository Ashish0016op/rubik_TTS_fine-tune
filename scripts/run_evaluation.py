"""Phase 6: Evaluation & Benchmarking CLI script for Rumik TTS."""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Any
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_utils import RumikModelWrapper
from src.inference.streaming_engine import StreamingEngine
from src.evaluation.evaluate import RumikEvaluator

DEFAULT_TEST_PROMPTS = [
    "Namaste! Yeh hamari pehli evaluation test sentence hai.",
    "Custom voice model fine-tuning ke baad natural sound kar raha hai ya nahi, yeh verify karna hai.",
    "Real time streaming latency kitni hai, kya audio turant sunai de raha hai?",
    "Aapka swagat hai hamare high performance Indic text to speech system mein.",
    "Mimi codec aur Rumik transformer milkar behtareen quality dete hain."
]

def run_evaluation_cli(
    model_id: str = "rumik-ai/rumik-oss-1",
    adapter_path: Optional[str] = "checkpoints/rumik_lora_custom_voice",
    output_dir: str = "outputs/evaluation",
    prompts_file: Optional[str] = None,
    device: str = "cpu",
    dry_run: bool = False
):
    print("=================================================================")
    print(" Phase 6: Evaluation & Side-by-Side Comparison Benchmark")
    print("=================================================================")
    print(f"[*] Base Model ID     : {model_id}")
    print(f"[*] Adapter Path      : {adapter_path}")
    print(f"[*] Output Directory  : {output_dir}")
    print(f"[*] Compute Device    : {device}")
    print(f"[*] Dry-Run Mode      : {dry_run}")
    print("=================================================================")

    # Load test prompts
    test_prompts = DEFAULT_TEST_PROMPTS
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                test_prompts = lines

    # Setup Engines
    if dry_run or not torch.cuda.is_available():
        class MockStreamingBackbone(torch.nn.Module):
            def __init__(self, vocab_size=272384, hidden_size=256):
                super().__init__()
                self.config = type("Config", (), {"hidden_size": hidden_size, "vocab_size": vocab_size, "model_type": "cohere2"})()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=False, **kwargs):
                hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], 256)
                logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 272384)
                logits[:, :, 256000:256000 + 16384] += 5.0
                fake_kv = [torch.randn(1, 4, 10, 64)] * 4
                return type("Out", (), {"logits": logits, "past_key_values": fake_kv, "hidden_states": [hidden]})()

        class MockTokenizer:
            def __len__(self):
                return 256000
            def encode(self, text, add_special_tokens=True):
                return [1, 101, 202, 303, 404, 2]

        base_wrapper = RumikModelWrapper(MockStreamingBackbone(), MockTokenizer())
        ft_wrapper = RumikModelWrapper(MockStreamingBackbone(), MockTokenizer())
        base_engine = StreamingEngine(base_wrapper, device="cpu")
        ft_engine = StreamingEngine(ft_wrapper, device="cpu")
    else:
        base_wrapper = RumikModelWrapper.load(model_id, device=device)
        base_engine = StreamingEngine(base_wrapper, device=device)

        ft_engine = None
        if adapter_path and os.path.exists(adapter_path):
            from peft import PeftModel
            ft_wrapper = RumikModelWrapper.load(model_id, device=device)
            ft_wrapper.transformer = PeftModel.from_pretrained(ft_wrapper.transformer, adapter_path)
            ft_engine = StreamingEngine(ft_wrapper, device=device)

    evaluator = RumikEvaluator(base_engine=base_engine, finetuned_engine=ft_engine)
    summary = evaluator.run_benchmark_suite(test_prompts=test_prompts, output_dir=output_dir)

    print("\n=================================================================")
    print(" Benchmark Summary Results")
    print("=================================================================")
    print(f"{'Metric':<25}{'Base Model':<20}{'Fine-Tuned Model':<20}")
    print("-" * 65)
    
    b_sum = summary["base_summary"]
    f_sum = summary.get("finetuned_summary", {})

    print(f"{'Mean TTFA (ms)':<25}{b_sum['mean_ttfa_ms']:<20.2f}{f_sum.get('mean_ttfa_ms', 0.0):<20.2f}")
    print(f"{'P95 TTFA (ms)':<25}{b_sum['p95_ttfa_ms']:<20.2f}{f_sum.get('p95_ttfa_ms', 0.0):<20.2f}")
    print(f"{'Mean RTF':<25}{b_sum['mean_rtf']:<20.2f}{f_sum.get('mean_rtf', 0.0):<20.2f}")
    print(f"{'Mean WER (%)':<25}{b_sum['mean_wer']*100:<20.2f}{f_sum.get('mean_wer', 0.0)*100:<20.2f}")
    print(f"{'Mean CER (%)':<25}{b_sum['mean_cer']*100:<20.2f}{f_sum.get('mean_cer', 0.0)*100:<20.2f}")
    print(f"{'Total Audio (s)':<25}{b_sum['total_audio_duration_s']:<20.2f}{f_sum.get('total_audio_duration_s', 0.0):<20.2f}")
    print("-" * 65)
    print(f"[+] Side-by-side WAV files written to: {os.path.join(output_dir, 'side_by_side')}")
    print("[+] Phase 6 Evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Rumik TTS Fine-Tuned Model")
    parser.add_argument("--model-id", type=str, default="rumik-ai/rumik-oss-1")
    parser.add_argument("--adapter-path", type=str, default="checkpoints/rumik_lora_custom_voice")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_evaluation_cli(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        prompts_file=args.prompts_file,
        device=args.device,
        dry_run=args.dry_run
    )
