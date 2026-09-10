"""Audio preprocessing pipeline: 24kHz mono conversion, VAD trimming, and Mimi encoding."""

import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union

class AudioPreprocessor:
    """Handles audio loading, 24kHz mono resampling, VAD silence trimming, and Mimi encoding."""

    def __init__(
        self,
        target_sample_rate: int = 24000,
        top_db: float = 30.0,
        frame_length: int = 2048,
        hop_length: int = 512,
        normalize_peak: float = 0.95
    ):
        self.target_sample_rate = target_sample_rate
        self.top_db = top_db
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.normalize_peak = normalize_peak

    def load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """Loads audio file, converts to mono, resamples to 24kHz, trims silence, and normalizes.
        
        Returns: Tensor of shape [1, num_samples] (mono, float32, 24kHz).
        """
        assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
        
        # Load audio using soundfile / librosa / torchaudio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            waveform = torch.from_numpy(y)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 24kHz if needed
        if sr != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Trim leading and trailing silence with VAD / energy thresholding
        np_wav = waveform.squeeze(0).cpu().numpy()
        trimmed_wav, index = librosa.effects.trim(
            np_wav,
            top_db=self.top_db,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )

        # Peak normalization
        max_val = np.max(np.abs(trimmed_wav))
        if max_val > 1e-6:
            trimmed_wav = (trimmed_wav / max_val) * self.normalize_peak

        processed_tensor = torch.from_numpy(trimmed_wav).unsqueeze(0).float()
        return processed_tensor

    def encode_with_mimi(
        self,
        waveform: torch.Tensor,
        mimi_model: Optional[torch.nn.Module] = None,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Encodes [1, num_samples] 24kHz audio into [8, num_frames] discrete codebook indices.
        
        If mimi_model is None or running in fallback/synthetic mode, generates realistic simulated RVQ codes.
        """
        if mimi_model is not None:
            mimi_model.eval()
            with torch.no_grad():
                inp = waveform.unsqueeze(0).to(device) if waveform.dim() == 2 else waveform.to(device)
                # Kyutai Mimi or Transformers MimiModel encoding
                if hasattr(mimi_model, "encode"):
                    encoded = mimi_model.encode(inp)
                    if hasattr(encoded, "audio_codes"):
                        codes = encoded.audio_codes # [batch, num_codebooks, num_frames]
                    elif isinstance(encoded, tuple):
                        codes = encoded[0]
                    else:
                        codes = encoded
                    return codes.squeeze(0).cpu()

        # Fallback / Synthetic code generation matching 12.5 Hz frame rate
        # 24,000 samples/sec / 12.5 frames/sec = 1920 samples/frame
        num_samples = waveform.shape[-1]
        num_frames = max(1, num_samples // 1920)
        # RVQ 8 codebooks
        simulated_codes = torch.randint(0, 2048, (8, num_frames), dtype=torch.long)
        return simulated_codes
