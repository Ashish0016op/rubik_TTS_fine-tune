# Custom Voice Model (Fine-Tuned from Rumik-OSS-1) with Real-Time Streaming

A modular end-to-end framework for adapting the **Rumik-OSS-1** 3B foundation text-to-speech model (Cohere2 decoder-only transformer with flattened Mimi audio-codec tokens) to custom speaker voices (specifically optimized for Romanized Hindi and Indic speech), complete with **incremental KV-cached real-time streaming inference**, **WebSocket serving**, and **Web Audio API UI**.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Phase 1: Environment & Checkpoint Verification](#phase-1-environment--checkpoint-verification)
- [Phase 2: Dataset Preparation & Preprocessing](#phase-2-dataset-preparation--preprocessing)
- [Phase 3: Fine-Tuning (LoRA & Speaker Conditioning)](#phase-3-fine-tuning-lora--speaker-conditioning)
- [Phase 4: Real-Time Streaming Inference Engine](#phase-4-real-time-streaming-inference-engine)
- [Phase 5: WebSocket Server & Web Audio UI](#phase-5-websocket-server--web-audio-ui)
- [Phase 6: Evaluation & Side-by-Side Comparison](#phase-6-evaluation--side-by-side-comparison)
- [Running Unit Tests](#running-unit-tests)
- [Configuration Reference](#configuration-reference)
- [License & Usage Terms](#license--usage-terms)

---

## Architecture Overview

```
                      +-----------------------------------+
                      |   Input Text (Romanized Hindi)    |
                      +-----------------------------------+
                                        |
                             Tokenizer (Cohere2 Vocab)
                                        |
                                        v
+-----------------------+     +-----------------------------------+
|  Ref Audio Clip       | --> | Autoregressive Decoder Backbone   | <--- LoRA Adapters
|  (Few-Shot Speaker)   |     | (Step-by-Step KV-Cached LM)       |
+-----------------------+     +-----------------------------------+
                                        |
                         Iterative Token Stream (8 tokens/frame)
                                        |
                                        v
                        +-------------------------------+
                        |  Stop Head?  ---> [Terminate] |
                        +-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |  Frozen Mimi Causal Decoder   |
                        +-------------------------------+
                                        |
                           24kHz Raw PCM Audio Chunks
                                        |
                                        v
                        +-------------------------------+
                        | FastAPI WebSocket / Web Audio |
                        +-------------------------------+
```

### Key Architectural Concepts
1. **Frozen Mimi Codec**: 24kHz audio, 12.5 frames/sec, 8 RVQ codebooks per frame = 100 tokens/second total.
2. **Vocabulary Layout**:
   - `[0 .. 255,999]`: Text vocabulary (Cohere2 tokenizer).
   - `[256,000 .. 272,383]`: Interleaved audio tokens (8 codebooks $\times$ 2,048 entries each).
   - Total Vocabulary Size = `272,384`.
3. **Loss Masking**: Text token positions are masked (`-100`) so gradient updates only optimize audio token predictions and stop-signal classifications.
4. **Incremental KV-Caching**: Token-by-token generation with `past_key_values`, buffering 8 tokens (1 frame = 80ms) and decoding causally for sub-200ms Time-To-First-Audio (TTFA).

---

## Project Structure

```
rumik_tts/
├── config/
│   └── default_config.py        # Centralized configurations & hyperparameters
├── data/
│   ├── raw_audio/               # Raw audio input files (.wav, .mp3, .flac)
│   ├── transcripts.csv          # Audio-to-transcript mapping
│   └── processed/               # Preprocessed Mimi RVQ token manifests
├── src/
│   ├── models/
│   │   ├── model_utils.py       # RumikModelWrapper, TokenLayoutManager, StopHead
│   │   ├── lora_setup.py        # PEFT/LoRA adapter configuration
│   │   └── speaker_encoder.py   # Few-shot ReferenceAudioEncoder & PrefixProjector
│   ├── data/
│   │   ├── preprocess.py        # 24kHz resampling, VAD silence trimming
│   │   └── dataset.py           # RumikTTSDataset, token interleaving, label masking
│   ├── training/
│   │   ├── loss.py              # Audio-masked Cross-Entropy & BCE stop loss
│   │   └── train.py             # RumikFineTuner training loop with replay mixing
│   ├── inference/
│   │   └── streaming_engine.py  # StreamingEngine with KV cache & causal decode
│   ├── server/
│   │   ├── app.py               # FastAPI WebSocket streaming server
│   │   └── static/
│   │       └── index.html       # Web Audio API real-time player & visualizer
│   └── evaluation/
│       ├── metrics.py           # WER, CER, and Latency statistics
│       └── evaluate.py          # Side-by-side evaluator and benchmark suite
├── scripts/
│   ├── verify_checkpoint.py     # Phase 1: Verification CLI
│   ├── run_preprocessing.py     # Phase 2: Preprocessing CLI
│   ├── run_finetune.py          # Phase 3: Fine-tuning CLI
│   ├── run_streaming_demo.py    # Phase 4: Streaming inference CLI
│   ├── run_server.py            # Phase 5: Server launcher
│   └── run_evaluation.py        # Phase 6: Evaluation CLI
├── tests/                       # Unit test suite (11 test cases)
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone & Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Hardware Recommendations
- **Inference**: NVIDIA GPU with 8GB+ VRAM (or CPU mode for testing).
- **Fine-Tuning**: NVIDIA GPU with 16GB–24GB+ VRAM (e.g., RTX 3090/4090, A100).

---

## Phase 1: Environment & Checkpoint Verification

Before running fine-tuning, verify that the environment, tokenizer, and Mimi codec interfaces are functioning properly:

```bash
# Verify base model checkpoint and Mimi codec on GPU/CPU
python scripts/verify_checkpoint.py --model-id "rumik-ai/rumik-oss-1" --device cuda

# For quick offline testing (dry-run mode without downloading full weights):
python scripts/verify_checkpoint.py --dry-run
```

**Verification checks performed:**
- Model architecture (`n_layers`, `hidden_size`, `num_heads`).
- Vocabulary calculation (`text_vocab (256000) + 8 * 2048 = 272384`).
- Token flattening & unflattening roundtrips.
- Mimi codec sample rate (24kHz) and frame rate (12.5 Hz).

---

## Phase 2: Dataset Preparation & Preprocessing

Prepare your custom speaker audio and Romanized Hindi transcript CSV.

### 1. Prepare CSV Format
Create `data/transcripts.csv` with the following columns:
```csv
audio_path,transcript,speaker_id
sample_001.wav,"Namaste! Yeh hamara pehla fine-tuning sample hai.",custom_speaker_01
sample_002.wav,"Aapka swagat hai hamare voice model training mein.",custom_speaker_01
```

### 2. Run Preprocessing Pipeline
```bash
python scripts/run_preprocessing.py \
    --data-dir data/raw_audio \
    --transcript-csv data/transcripts.csv \
    --output-dir data/processed \
    --val-ratio 0.15
```

**What this does automatically:**
- Resamples all audio to **24kHz mono**.
- Trims leading/trailing silence via **VAD energy thresholding**.
- Encodes audio through the **frozen Mimi encoder** into 8 RVQ codebook indices.
- Interleaves codebook tokens with exact vocabulary offsets.
- Masks text tokens (`-100`) so loss is calculated exclusively on audio tokens.
- Exports `train_dataset.json` and `val_dataset.json`.

---

## Phase 3: Fine-Tuning (LoRA & Speaker Conditioning)

### 1. Standard LoRA Voice Fine-Tuning
Fine-tune attention and MLP projection layers while keeping the Mimi codec strictly frozen:

```bash
python scripts/run_finetune.py \
    --data-dir data/processed \
    --output-dir checkpoints/rumik_lora_custom_voice \
    --model-id "rumik-ai/rumik-oss-1" \
    --epochs 15 \
    --batch-size 2 \
    --lr 2e-4 \
    --lora-r 16 \
    --device cuda
```

### 2. Ultra-Low Data Few-Shot Adaptation (<30 mins of audio)
If you have very limited voice samples, enable the **Speaker Conditioning Module** (extracts a speaker embedding from reference audio and injects prefix embeddings):

```bash
python scripts/run_finetune.py \
    --data-dir data/processed \
    --output-dir checkpoints/rumik_speaker_conditioned \
    --speaker-conditioning \
    --epochs 10 \
    --lr 1e-4 \
    --device cuda
```

### Dry-Run Test
To verify the training loop, backward pass, and checkpoint saving on any machine without GPU:
```bash
python scripts/run_finetune.py --dry-run
```

---

## Phase 4: Real-Time Streaming Inference Engine

Run the standalone streaming generator in your terminal. Tokens are generated autoregressively with incremental KV-caching, assembled into 8-token Mimi frames, and causally decoded into 24kHz audio chunks in real time:

```bash
# Stream audio using the fine-tuned LoRA checkpoint
python scripts/run_streaming_demo.py \
    --prompt "Namaste! Yeh hamari real-time streaming audio engine ka test hai." \
    --adapter-path checkpoints/rumik_lora_custom_voice \
    --output-wav outputs/streaming_output.wav \
    --temperature 0.7 \
    --device cuda
```

### Real-Time Metrics Output
```
Frame   Samples   Latency (ms)   Chunk RTF   Status         
-----------------------------------------------------------------
#1      1920      140.5          1.76        FIRST AUDIO (140.5ms)
#2      1920      263.3          1.65        STREAMING
#3      1920      390.9          1.63        STREAMING
#4      1920      522.8          1.63        STREAMING
#5      1920      641.8          1.60        FINAL FRAME (STOP)
-----------------------------------------------------------------
[+] Total Audio Generated : 0.40 seconds (5 frames)
[+] Time-To-First-Audio   : 140.49 ms
```

---

## Phase 5: WebSocket Server & Web Audio UI

Launch the FastAPI WebSocket server and interactive web client:

```bash
# Launch server on port 8000
python scripts/run_server.py --port 8000 --adapter-path checkpoints/rumik_lora_custom_voice
```

Open your browser and navigate to:
```
http://localhost:8000
```

### Features in the Web Client
- **Gapless Web Audio API Playback**: Schedules incoming Int16 PCM chunks seamlessly without audio glitches or clicks.
- **Live Frequency Spectrum & Oscilloscope**: Real-time canvas visualizer.
- **HUD Latency Counters**: Displays live Time-To-First-Audio (TTFA ms), Real-Time Factor (RTF), and frame counts.
- **Romanized Hindi Presets**: Instant prompt buttons for quick testing.

### WebSocket Protocol Specification (`/ws/stream`)
1. **Client Sends JSON**:
   ```json
   {
     "prompt": "Namaste, aap kaise hain?",
     "temperature": 0.7,
     "top_p": 0.95,
     "max_frames": 200
   }
   ```
2. **Server Streams Metadata**:
   ```json
   {"type": "start", "sample_rate": 24000, "channels": 1, "format": "pcm_s16le"}
   ```
3. **Server Streams Binary Chunks**: Raw 16-bit signed PCM audio bytes (1920 samples $\times$ 2 bytes = 3840 bytes per frame).
4. **Server Sends Completion**:
   ```json
   {"type": "end", "total_frames": 45, "total_duration_s": 3.6, "ttfa_ms": 135.2, "status": "completed"}
   ```

---

## Phase 6: Evaluation & Side-by-Side Comparison

Benchmark Word Error Rate (WER), Character Error Rate (CER), Time-To-First-Audio (TTFA), and Real-Time Factor (RTF) while generating side-by-side audio comparison files:

```bash
python scripts/run_evaluation.py \
    --adapter-path checkpoints/rumik_lora_custom_voice \
    --output-dir outputs/evaluation
```

### Output Summary Table & Files
Side-by-side WAV files will be generated in `outputs/evaluation/side_by_side/`:
- `prompt_01_base.wav` vs `prompt_01_finetuned.wav`
- `prompt_02_base.wav` vs `prompt_02_finetuned.wav`

And a detailed benchmark report is saved to `outputs/evaluation/benchmark_report.json`.

---

## Running Unit Tests

Run the full automated unit test suite:
```bash
python -m unittest discover tests
```

**Test coverage:**
- `test_token_layout.py`: Codebook offset calculations, bounds checking, flatten/unflatten roundtrip.
- `test_speaker_encoder.py`: Reference audio feature extraction, attentive pooling, prefix projection.
- `test_streaming_engine.py`: Step-by-step KV cache updates, 8-token frame buffering, causal decoding.
- `test_server.py`: REST health check, static client serving, and WebSocket streaming transmission.

---

## Configuration Reference

Edit `config/default_config.py` to customize default parameters:

```python
@dataclass
class AudioConfig:
    sample_rate: int = 24000
    frame_rate: float = 12.5
    num_codebooks: int = 8
    codebook_size: int = 2048

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 15
    replay_ratio: float = 0.15 # general data replay to prevent catastrophic forgetting
```

---

## License & Usage Terms
- The base model `rumik-ai/rumik-oss-1` is released under the **CC-BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0) license.
- Ensure compliance with non-commercial licensing terms when deploying or distributing fine-tuned checkpoints.
