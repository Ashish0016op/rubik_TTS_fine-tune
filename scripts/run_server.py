"""Phase 5: Server Launcher for Rumik TTS FastAPI WebSocket service."""

import os
import sys
import argparse
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server.app import app, initialize_engine

def start_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_id: str = "rumik-ai/rumik-oss-1",
    adapter_path: str = None,
    device: str = "cpu",
    dry_run: bool = False
):
    print("=================================================================")
    print(" Phase 5: Rumik TTS Real-Time Streaming Server")
    print("=================================================================")
    print(f"[*] Binding to        : http://{host}:{port}")
    print(f"[*] WebSocket URL     : ws://{host}:{port}/ws/stream")
    print(f"[*] Base Model        : {model_id}")
    print(f"[*] Adapter Path      : {adapter_path or 'None'}")
    print(f"[*] Compute Device    : {device}")
    print(f"[*] Dry-Run Mode      : {dry_run}")
    print("=================================================================")

    # Pre-initialize model engine
    initialize_engine(
        model_id=model_id,
        adapter_path=adapter_path,
        device=device,
        dry_run=dry_run
    )

    print(f"\n[+] Server starting! Open http://{host}:{port} in your browser.")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rumik TTS Streaming Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", type=str, default="rumik-ai/rumik-oss-1")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start_server(
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        device=args.device,
        dry_run=args.dry_run
    )
