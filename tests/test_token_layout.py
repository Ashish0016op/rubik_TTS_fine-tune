"""Unit tests for TokenLayoutManager: offsets, interleaving, and roundtrips."""

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
        # Text token
        self.assertFalse(self.layout.is_audio_token(0))
        self.assertFalse(self.layout.is_audio_token(261007))
        
        # Audio tokens
        self.assertTrue(self.layout.is_audio_token(261008)) # start of cb0
        self.assertTrue(self.layout.is_audio_token(261008 + 2047)) # end of cb0
        self.assertTrue(self.layout.is_audio_token(261008 + 7 * 2048)) # start of cb7
        self.assertTrue(self.layout.is_audio_token(277391)) # last valid audio token
        
        # Special tokens
        self.assertFalse(self.layout.is_audio_token(277392)) # text_start
        self.assertFalse(self.layout.is_audio_token(277393)) # audio_start
        self.assertFalse(self.layout.is_audio_token(277394)) # audio_end

    def test_code_offset_mapping(self):
        # Codebook 0, Code 0 -> 261008
        t0 = self.layout.audio_code_to_token_id(0, 0)
        self.assertEqual(t0, 261008)
        cb_idx, code_val = self.layout.token_id_to_audio_code(t0)
        self.assertEqual(cb_idx, 0)
        self.assertEqual(code_val, 0)

        # Codebook 3, Code 512 -> 261008 + 3*2048 + 512 = 267664
        t3 = self.layout.audio_code_to_token_id(3, 512)
        self.assertEqual(t3, 261008 + 3 * 2048 + 512)
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
