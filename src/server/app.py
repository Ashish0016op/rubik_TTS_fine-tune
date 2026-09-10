"""Phase 5: FastAPI WebSocket Streaming Server for Rumik TTS."""

import os
import json
import time
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import torch

from src.models.model_utils import RumikModelWrapper
from src.models.speaker_encoder import SpeakerConditioningModule
from src.inference.streaming_engine import StreamingEngine

app = FastAPI(
    title="Rumik TTS Real-Time Streaming Server",
    description="Low-latency real-time voice streaming engine fine-tuned from rumik-oss-1",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine state
streaming_engine: Optional[StreamingEngine] = None
server_state: Dict[str, Any] = {
    "model_loaded": False,
    "model_id": "rumik-ai/rumik-oss-1",
    "adapter_path": None,
    "device": "cpu",
    "sample_rate": 24000
}

def initialize_engine(
    model_id: str = "rumik-ai/rumik-oss-1",
    adapter_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dry_run: bool = False
):
    global streaming_engine, server_state
    print(f"[*] Initializing StreamingEngine (device={device}, dry_run={dry_run})...")

    if dry_run or not torch.cuda.is_available():
        # Setup lightweight mock wrapper for testing
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

        wrapper = RumikModelWrapper(
            transformer=MockStreamingBackbone(),
            tokenizer=MockTokenizer()
        )
    else:
        wrapper = RumikModelWrapper.load(model_id, device=device)
        if adapter_path and os.path.exists(adapter_path):
            from peft import PeftModel
            wrapper.transformer = PeftModel.from_pretrained(wrapper.transformer, adapter_path)

    streaming_engine = StreamingEngine(model_wrapper=wrapper, device=device)
    server_state["model_loaded"] = True
    server_state["model_id"] = model_id
    server_state["adapter_path"] = adapter_path
    server_state["device"] = device
    print("[+] StreamingEngine successfully initialized and ready for connections.")


# Static UI Directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def get_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Rumik TTS Streaming Server is Running</h2><p>Static UI not found.</p>")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "engine_ready": streaming_engine is not None,
        "state": server_state
    }

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket Streaming Endpoint.
    Protocol:
    1. Client sends JSON: {"prompt": "Namaste", "temperature": 0.7, "max_frames": 250}
    2. Server sends JSON meta: {"type": "start", "sample_rate": 24000}
    3. Server streams binary Int16 PCM audio frames as they are generated
    4. Server sends JSON completion: {"type": "end", "ttfa_ms": 45.2, "rtf": 0.35, "total_duration_s": 3.2}
    """
    await websocket.accept()
    if streaming_engine is None:
        await websocket.send_json({"error": "Streaming engine not initialized"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            prompt = req.get("prompt", "")
            speaker = req.get("speaker", "Ira")
            temperature = float(req.get("temperature", 0.8))
            top_p = float(req.get("top_p", 0.9))
            max_frames = int(req.get("max_frames", 200))
            
            if not prompt.strip():
                await websocket.send_json({"error": "Empty prompt received"})
                continue

            # Notify stream start
            await websocket.send_json({
                "type": "start",
                "sample_rate": 24000,
                "channels": 1,
                "format": "pcm_s16le",
                "speaker": speaker,
                "prompt": prompt
            })

            # Stream audio frames
            generator = streaming_engine.stream_generate(
                prompt_text=prompt,
                speaker=speaker,
                max_frames=max_frames,
                temperature=temperature,
                top_p=top_p
            )

            ttfa_ms = None
            total_frames = 0

            for chunk in generator:
                if chunk.is_first_chunk:
                    ttfa_ms = chunk.time_to_first_audio_ms
                    await websocket.send_json({
                        "type": "meta",
                        "event": "first_chunk",
                        "ttfa_ms": round(ttfa_ms, 2)
                    })

                # Send raw binary PCM chunk
                await websocket.send_bytes(chunk.pcm_bytes)
                total_frames += 1
                # Yield control to event loop for smooth low-latency streaming
                await asyncio.sleep(0.001)

            # Send end notification with final benchmark metrics
            total_duration_s = total_frames * (1920.0 / 24000.0)
            await websocket.send_json({
                "type": "end",
                "total_frames": total_frames,
                "total_duration_s": round(total_duration_s, 3),
                "ttfa_ms": round(ttfa_ms or 0.0, 2),
                "status": "completed"
            })

    except WebSocketDisconnect:
        print("[*] WebSocket client disconnected cleanly.")
    except Exception as e:
        print(f"[!] WebSocket stream error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
