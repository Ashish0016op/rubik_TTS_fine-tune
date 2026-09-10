"""Unit tests for SpeakerConditioningModule and ReferenceAudioEncoder."""

import unittest
import torch
from src.models.speaker_encoder import ReferenceAudioEncoder, SpeakerPrefixProjector, SpeakerConditioningModule

class TestSpeakerEncoder(unittest.TestCase):

    def test_reference_audio_encoder(self):
        encoder = ReferenceAudioEncoder(in_channels=1, hidden_dim=64, embedding_dim=128)
        # 1 second of 24kHz audio: [2, 1, 24000]
        waveform = torch.randn(2, 1, 24000)
        emb = encoder(waveform)
        
        self.assertEqual(emb.shape, (2, 128))
        # Check L2 norm is approximately 1.0
        norms = torch.norm(emb, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))

    def test_prefix_projector(self):
        projector = SpeakerPrefixProjector(embedding_dim=128, hidden_dim=256, num_prefix_tokens=4)
        emb = torch.randn(2, 128)
        prefix = projector(emb)
        self.assertEqual(prefix.shape, (2, 4, 256))

    def test_combined_conditioning_module(self):
        module = SpeakerConditioningModule(
            embedding_dim=128,
            transformer_hidden_dim=256,
            num_prefix_tokens=4
        )
        waveform = torch.randn(1, 24000) # [1, 24000]
        prefix = module.get_prefix_tokens(waveform)
        self.assertEqual(prefix.shape, (1, 4, 256))

if __name__ == "__main__":
    unittest.main()
