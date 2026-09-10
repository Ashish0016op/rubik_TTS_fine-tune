"""Unit tests for FastAPI WebSocket and REST server."""

import unittest
import asyncio
from fastapi.testclient import TestClient
from src.server.app import app, initialize_engine

class TestServerEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize engine in dry-run mode
        initialize_engine(dry_run=True, device="cpu")
        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["engine_ready"])

    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Rumik TTS", response.text)

    def test_websocket_stream_flow(self):
        with self.client.websocket_connect("/ws/stream") as ws:
            # Send prompt request
            ws.send_json({
                "prompt": "Namaste unit test",
                "max_frames": 3,
                "temperature": 0.7
            })

            # Expect metadata start
            meta_start = ws.receive_json()
            self.assertEqual(meta_start["type"], "start")
            self.assertEqual(meta_start["sample_rate"], 24000)

            # Receive binary chunks or meta
            pcm_chunks = 0
            while True:
                msg = ws.receive()
                if "bytes" in msg:
                    pcm_chunks += 1
                    self.assertEqual(len(msg["bytes"]), 1920 * 2) # 1920 int16 samples = 3840 bytes
                elif "text" in msg:
                    import json
                    parsed = json.loads(msg["text"])
                    if parsed.get("type") == "end":
                        self.assertEqual(parsed["status"], "completed")
                        break
            
            self.assertGreaterEqual(pcm_chunks, 1)

if __name__ == "__main__":
    unittest.main()
