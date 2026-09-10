"""Unit tests for TokenLayoutManager: offsets, interleaving, and roundtrips."""

import unittest
import torch
from src.models.model_utils import TokenLayoutManager

class TestTokenLayout(unittest.TestCase):

    def setUp(self):
        self.text_vocab = 256000
        self.num_codebooks = 8
        self.codebook_size = 2048
        self.layout = TokenLayoutManager(
            text_vocab_size=self.text_vocab,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size
        )

    def test_vocab_arithmetic(self):
        expected_total = self.text_vocab + (self.num_codebooks * self.codebook_size)
        self.assertEqual(self.layout.total_vocab_size, expected_total)
        self.assertEqual(self.layout.total_vocab_size, 272384)

    def test_audio_token_identification(self):
        # Text token
        self.assertFalse(self.layout.is_audio_token(0))
        self.assertFalse(self.layout.is_audio_token(255999))
        
        # Audio tokens
        self.assertTrue(self.layout.is_audio_token(256000)) # start of cb0
        self.assertTrue(self.layout.is_audio_token(256000 + 2047)) # end of cb0
        self.assertTrue(self.layout.is_audio_token(256000 + 7 * 2048)) # start of cb7
        self.assertTrue(self.layout.is_audio_token(272383)) # last valid audio token
        
        # Out of bounds
        self.assertFalse(self.layout.is_audio_token(272384))

    def test_code_offset_mapping(self):
        # Codebook 0, Code 0 -> 256000
        t0 = self.layout.audio_code_to_token_id(0, 0)
        self.assertEqual(t0, 256000)
        cb_idx, code_val = self.layout.token_id_to_audio_code(t0)
        self.assertEqual(cb_idx, 0)
        self.assertEqual(code_val, 0)

        # Codebook 3, Code 512 -> 256000 + 3*2048 + 512 = 262656
        t3 = self.layout.audio_code_to_token_id(3, 512)
        self.assertEqual(t3, 256000 + 3 * 2048 + 512)
        cb_idx, code_val = self.layout.token_id_to_audio_code(t3)
        self.assertEqual(cb_idx, 3)
        self.assertEqual(code_val, 512)

    def test_flatten_unflatten_roundtrip(self):
        # 1D single example [8, 50] (50 frames = 400 tokens)
        original_codes = torch.randint(0, 2048, (8, 50))
        flat = self.layout.flatten_audio_frames(original_codes)
        self.assertEqual(flat.shape, (400,))
        
        # Reconstruct
        reconstructed = self.layout.unflatten_audio_frames(flat)
        self.assertTrue(torch.equal(original_codes, reconstructed))

        # Batched example [2, 8, 30] (30 frames = 240 tokens)
        batched_codes = torch.randint(0, 2048, (2, 8, 30))
        flat_batched = self.layout.flatten_audio_frames(batched_codes)
        self.assertEqual(flat_batched.shape, (2, 240))
        
        reconstructed_batched = self.layout.unflatten_audio_frames(flat_batched)
        self.assertTrue(torch.equal(batched_codes, reconstructed_batched))

if __name__ == "__main__":
    unittest.main()
