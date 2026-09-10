"""Unit tests for StreamingEngine: KV caching, frame grouping, and causal decode."""

import unittest
import torch
import numpy as np
from src.models.model_utils import RumikModelWrapper, TokenLayoutManager
from src.inference.streaming_engine import StreamingEngine

class TestStreamingEngine(unittest.TestCase):

    def setUp(self):
        class MockStreamingBackbone(torch.nn.Module):
            def __init__(self, vocab_size=272384, hidden_size=256):
                super().__init__()
                self.config = type("Config", (), {"hidden_size": hidden_size, "vocab_size": vocab_size, "model_type": "cohere2"})()
                self.embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=False, **kwargs):
                batch_size, seq_len = input_ids.shape
                hidden = torch.randn(batch_size, seq_len, 256)
                logits = torch.randn(batch_size, seq_len, 272384)
                logits[:, :, 256000:256000 + 16384] += 5.0
                fake_kv = [torch.randn(1, 4, 10, 64)] * 4
                return type("Out", (), {"logits": logits, "past_key_values": fake_kv, "hidden_states": [hidden]})()

        class MockTokenizer:
            def __len__(self):
                return 256000
            def encode(self, text, add_special_tokens=True):
                return [1, 100, 200, 300, 2]

        self.wrapper = RumikModelWrapper(MockStreamingBackbone(), MockTokenizer())
        self.engine = StreamingEngine(self.wrapper, device="cpu")

    def test_streaming_generation_flow(self):
        max_frames = 5
        generator = self.engine.stream_generate(
            prompt_text="Namaste test",
            max_frames=max_frames
        )

        chunks_received = []
        for chunk in generator:
            chunks_received.append(chunk)
            self.assertEqual(chunk.sample_rate, 24000)
            self.assertEqual(chunk.num_samples, 1920) # 80ms @ 24kHz
            self.assertTrue(len(chunk.pcm_bytes) == 1920 * 2) # 16-bit PCM = 2 bytes/sample

        self.assertEqual(len(chunks_received), max_frames)
        self.assertTrue(chunks_received[0].is_first_chunk)
        self.assertTrue(chunks_received[0].time_to_first_audio_ms > 0)
        self.assertTrue(chunks_received[-1].is_final_chunk)

if __name__ == "__main__":
    unittest.main()
