"""Evaluation metrics: Word Error Rate (WER), Character Error Rate (CER), and Latency benchmarks."""

import re
import numpy as np
from typing import List, Dict, Any, Tuple

def normalize_text(text: str) -> str:
    """Normalizes romanized text (lowercasing, punctuation removal, whitespace collapsing)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """Computes Levenshtein distance between two token/character lists."""
    r_len, h_len = len(ref), len(hyp)
    dp = np.zeros((r_len + 1, h_len + 1), dtype=int)

    for i in range(r_len + 1):
        dp[i, 0] = i
    for j in range(h_len + 1):
        dp[0, j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # deletion
                    dp[i, j - 1] + 1,      # insertion
                    dp[i - 1, j - 1] + 1   # substitution
                )
    return int(dp[r_len, h_len])

def compute_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate between reference and hypothesis strings."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist / len(ref_words))

def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate between reference and hypothesis strings."""
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / len(ref_chars))

def aggregate_latency_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Computes statistical summary of latency benchmarks."""
    ttfa_vals = [m["ttfa_ms"] for m in metrics_list if "ttfa_ms" in m]
    rtf_vals = [m["rtf"] for m in metrics_list if "rtf" in m]
    duration_vals = [m["duration_s"] for m in metrics_list if "duration_s" in m]

    return {
        "mean_ttfa_ms": float(np.mean(ttfa_vals)) if ttfa_vals else 0.0,
        "p50_ttfa_ms": float(np.percentile(ttfa_vals, 50)) if ttfa_vals else 0.0,
        "p95_ttfa_ms": float(np.percentile(ttfa_vals, 95)) if ttfa_vals else 0.0,
        "mean_rtf": float(np.mean(rtf_vals)) if rtf_vals else 0.0,
        "total_audio_duration_s": float(np.sum(duration_vals)) if duration_vals else 0.0,
        "num_evaluated_prompts": len(metrics_list)
    }
