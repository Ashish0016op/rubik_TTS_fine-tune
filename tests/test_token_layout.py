"""Unit tests for TokenLayoutManager: official Rumik-OSS-1 offsets, interleaving, and roundtrips."""

import unittest
import torch
from src.models.model_utils import TokenLayoutManager

class TestTokenLayout(unittest.TestCase):

    def setUp(self):
        self.audio_offset = 261008
        self.num_codebooks = 8
        self.codebook_size = 2048
        self.total_vocab = 277404
        self.layout = TokenLayoutManager(
            audio_vocab_offset=self.audio_offset,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            total_vocab_size=self.total_vocab
        )

    def test_vocab_arithmetic(self):
        expected_last_audio = self.audio_offset + (self.num_codebooks * self.codebook_size) - 1
        self.assertEqual(self.layout.last_audio_token_id, expected_last_audio)
        self.assertEqual(self.layout.last_audio_token_id, 277391)
        self.assertEqual(self.layout.total_vocab_size, 277404)

    def test_audio_token_identification(self):
        self.assertFalse(self.layout.is_audio_token(0))
        self.assertFalse(self.layout.is_audio_token(261007))
        self.assertTrue(self.layout.is_audio_token(261008))
        self.assertTrue(self.layout.is_audio_token(277391))
        self.assertFalse(self.layout.is_audio_token(277392))

    def test_official_code_offset_mapping(self):
        # Formula: tid = first_unit_id + code * 8 + q
        # Code 0, Quantizer 0 -> 261008 + 0*8 + 0 = 261008
        t0 = self.layout.audio_code_to_token_id(0, 0)
        self.assertEqual(t0, 261008)
        cb_idx, code_val = self.layout.token_id_to_audio_code(t0)
        self.assertEqual(cb_idx, 0)
        self.assertEqual(code_val, 0)

        # Code 500, Quantizer 3 -> 261008 + 500*8 + 3 = 265011
        t3 = self.layout.audio_code_to_token_id(3, 500)
        self.assertEqual(t3, 261008 + 500 * 8 + 3)
        cb_idx, code_val = self.layout.token_id_to_audio_code(t3)
        self.assertEqual(cb_idx, 3)
        self.assertEqual(code_val, 500)

        # Last code 2047, Quantizer 7 -> 261008 + 2047*8 + 7 = 277391
        t_last = self.layout.audio_code_to_token_id(7, 2047)
        self.assertEqual(t_last, 277391)
        cb_idx, code_val = self.layout.token_id_to_audio_code(t_last)
        self.assertEqual(cb_idx, 7)
        self.assertEqual(code_val, 2047)

    def test_flatten_unflatten_roundtrip(self):
        # 1D single example [8, 50] (50 frames = 400 tokens)
        original_codes = torch.randint(0, 2048, (8, 50))
        flat = self.layout.flatten_audio_frames(original_codes)
        self.assertEqual(flat.shape, (400,))
        
        # Verify first frame tokens
        for q in range(8):
            expected_tid = 261008 + original_codes[q, 0].item() * 8 + q
            self.assertEqual(flat[q].item(), expected_tid)

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
