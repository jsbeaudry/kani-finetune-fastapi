# Kani TTS FastAPI

A production-ready **Text-to-Speech API** built on the [Kani TTS](https://huggingface.co/nineninesix/kani-tts-400m-0.3-pt) model (~400M parameters) with **LoRA fine-tuning** support and **NVIDIA NeMo Nano Codec** for audio decoding.

---

## Features

- **Multi-speaker TTS** -- Condition speech generation on a `speaker_id` for voice selection
- **LoRA fine-tuning** -- Launch background training jobs with custom HuggingFace datasets and track their status via API
- **Model hot-swap** -- Load a different checkpoint at runtime without restarting the server
- **HuggingFace Hub upload** -- Push models to the Hub automatically after training or via dedicated endpoint
- **Frame-level position encoding** -- Optimized RoPE positions where all 4 codec tokens per audio frame share a single position ID, improving long-form coherence and KV-cache efficiency
- **Flash Attention 2** -- Reduced memory usage and faster generation on Ampere+ GPUs
- **Optional `torch.compile`** -- Kernel fusion for additional throughput

---

## Architecture

```
Text prompt
    |
    v
[Tokenizer] --> [Causal LM (generate)] --> interleaved text + audio tokens
                                                    |
                                                    v
                                   [NeMo Nano Codec (decode)] --> PCM waveform
                                                                      |
                                                                      v
                                                                  WAV response (22050 Hz, 16-bit)
```

---

## Project Structure

```
kani-finetune-fastapi/
    requirements.txt              # Python dependencies
    app/
        __init__.py
        main.py                   # FastAPI app, lifespan, all routes
        config.py                 # Pydantic settings (env var driven)
        schemas.py                # Request/response Pydantic models
        models/
            __init__.py
            audio_player.py       # NeMo Nano Codec wrapper
            kani_model.py         # Kani inference engine
        training/
            __init__.py
            collator.py           # FramePosCollator for SFT training
            dataset.py            # HF dataset loading & preprocessing
            trainer.py            # LoRA fine-tuning orchestration
```

---

## Requirements

- **Python** 3.10+
- **CUDA GPU** (Ampere+ recommended for Flash Attention 2)
- **NVIDIA NeMo Toolkit** (for the audio codec)
- ~4 GB VRAM for inference, ~16 GB+ for fine-tuning

---

## Installation

```bash
git clone https://github.com/jsbeaudry/kani-finetune-fastapi.git
cd kani-finetune-fastapi

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: install Flash Attention 2 for faster inference
pip install flash-attn --no-build-isolation
```

---

## Quick Start

### 1. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On startup the server will:
1. Download and load the NeMo Nano Codec (`nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`)
2. Download and load the Kani TTS model (`nineninesix/kani-tts-400m-0.3-pt`)
3. Begin accepting requests once both are ready

### 2. Generate speech

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?", "speaker_id": "0047599005d8"}' \
  --output output.wav
```

### 3. Play the audio

```bash
# macOS
afplay output.wav

# Linux
aplay output.wav

# Or open with any media player
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check / readiness probe |
| `POST` | `/tts` | Generate speech from text (returns WAV) |
| `POST` | `/train` | Start a LoRA fine-tuning job (background) |
| `GET` | `/train/{job_id}` | Check training job status |
| `POST` | `/model/load` | Hot-swap the loaded model checkpoint |
| `POST` | `/model/upload` | Upload a checkpoint to HuggingFace Hub |

### Interactive docs

Once the server is running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Reference

### `GET /health`

Returns server status and model readiness.

**Response:**
```json
{
  "status": "ok",
  "model": "nineninesix/kani-tts-400m-0.3-pt",
  "model_loaded": true
}
```

---

### `POST /tts`

Generate speech audio from a text prompt.

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | `string` | Yes | -- | The text to synthesize |
| `speaker_id` | `string` | No | `null` | Speaker identity for voice conditioning |
| `temperature` | `float` | No | `0.8` | Sampling temperature (0.0 - 2.0) |
| `top_p` | `float` | No | `0.95` | Nucleus sampling threshold (0.0 - 1.0) |
| `max_new_tokens` | `int` | No | `1200` | Max generation length (1 - 4096) |
| `repetition_penalty` | `float` | No | `1.1` | Repetition penalty (1.0 - 3.0) |

**Example:**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Gen peyi ki gen bon sistèm siveyans maladi.",
    "speaker_id": "0047599005d8",
    "temperature": 0.6,
    "max_new_tokens": 800
  }' \
  --output speech.wav
```

**Response:** `audio/wav` file (16-bit PCM, 22050 Hz, mono)

**Error responses:**
- `422` -- Model failed to produce valid audio tokens
- `503` -- Model still loading

---

### `POST /train`

Launch a background LoRA fine-tuning job.

**Request body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `base_model_id` | `string` | No | `nineninesix/kani-tts-400m-0.3-pt` | Base model to fine-tune |
| `hf_datasets` | `array` | Yes | -- | List of HF dataset configurations (see below) |
| `max_duration_sec` | `int` | No | `30` | Max audio duration filter (seconds) |
| `n_shards_per_dataset` | `int` | No | `4` | Parallel processing shards per dataset |
| `lora_r` | `int` | No | `16` | LoRA rank |
| `lora_alpha` | `int` | No | `16` | LoRA scaling factor |
| `lora_dropout` | `float` | No | `0.1` | LoRA dropout |
| `target_modules` | `array` | No | `["q_proj","k_proj","v_proj","out_proj"]` | Modules to apply LoRA to |
| `num_train_epochs` | `int` | No | `4` | Training epochs |
| `per_device_train_batch_size` | `int` | No | `2` | Batch size per GPU |
| `gradient_accumulation_steps` | `int` | No | `4` | Gradient accumulation steps |
| `learning_rate` | `float` | No | `5e-5` | Peak learning rate |
| `lr_scheduler_type` | `string` | No | `cosine` | LR scheduler type |
| `warmup_ratio` | `float` | No | `0.1` | Warmup fraction |
| `weight_decay` | `float` | No | `0.02` | L2 regularization |
| `output_dir` | `string` | No | `./checkpoints` | Checkpoint save directory |
| `hf_token` | `string` | No | `null` | HuggingFace API token (enables auto-upload after training) |
| `dataset_name` | `string` | No | `null` | HF Hub repo ID to upload merged model to (required with `hf_token`) |

**Dataset configuration** (`hf_datasets` items):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `reponame` | `string` | Yes | -- | HuggingFace dataset repo ID |
| `name` | `string` | No | `null` | Dataset subset/config name |
| `split` | `string` | No | `train` | Dataset split |
| `text_col_name` | `string` | No | `text` | Text column name |
| `nano_layer_1..4` | `string` | No | `nano_layer_1..4` | Codec token column names |
| `encoded_len` | `string` | No | `encoded_len` | Audio length column name |
| `speaker_id` | `string` | No | `null` | Speaker label for voice conditioning |
| `max_len` | `int` | No | `null` | Random subsample limit |
| `categorical_filter` | `object` | No | `null` | Row filter (`column_name` + `value`) |

**Example -- single-speaker fine-tuning (save locally only):**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "nineninesix/kani-tts-400m-0.3-pt",
    "hf_datasets": [
      {
        "reponame": "jsbeaudry/kani-pretrain-data",
        "split": "train"
      }
    ],
    "num_train_epochs": 4,
    "lora_r": 16,
    "output_dir": "./checkpoints"
  }'
```

**Example -- train and auto-upload to HuggingFace Hub:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "hf_datasets": [
      {
        "reponame": "jsbeaudry/kani-pretrain-data",
        "split": "train"
      }
    ],
    "num_train_epochs": 4,
    "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
    "dataset_name": "jsbeaudry/haitian-kani-ht-v3"
  }'
```

**Example -- multi-speaker fine-tuning with filtering and Hub upload:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "hf_datasets": [
      {
        "reponame": "my-org/speaker-dataset",
        "speaker_id": "alice",
        "categorical_filter": {"column_name": "speaker", "value": "spk_001"},
        "max_len": 2000
      },
      {
        "reponame": "my-org/another-dataset",
        "speaker_id": "bob",
        "max_len": 2000
      }
    ],
    "max_duration_sec": 12,
    "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
    "dataset_name": "my-org/kani-multispeaker-v1"
  }'
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "started"
}
```

---

### `GET /train/{job_id}`

Check training job progress.

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "running",
  "error": null
}
```

Possible `status` values: `starting`, `running`, `completed`, `failed`

If `failed`, the `error` field contains the Python traceback.

---

### `POST /model/load`

Hot-swap the loaded model with a different checkpoint.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | `string` | Yes | Local path or HF repo ID |

**Example -- load a fine-tuned model:**
```bash
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./checkpoints/lora_kani_model_ft_exp"}'
```

**Example -- load from HuggingFace Hub:**
```bash
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "jsbeaudry/haitian-kani-ht-v3"}'
```

**Response:**
```json
{
  "status": "ok",
  "model": "./checkpoints/lora_kani_model_ft_exp"
}
```

---

### `POST /model/upload`

Upload a local model checkpoint to the HuggingFace Hub.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | `string` | Yes | Local path to the checkpoint directory |
| `hf_token` | `string` | Yes | HuggingFace API token with write access |
| `dataset_name` | `string` | Yes | Target Hub repo ID (e.g. `user/model-name`) |

**Example -- upload a training checkpoint:**
```bash
curl -X POST http://localhost:8000/model/upload \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./checkpoints/lora_kani_model_ft_exp",
    "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
    "dataset_name": "jsbeaudry/haitian-kani-ht-v3"
  }'
```

**Response:**
```json
{
  "status": "ok",
  "repo": "jsbeaudry/haitian-kani-ht-v3",
  "model_path": "./checkpoints/lora_kani_model_ft_exp"
}
```

**Error responses:**
- `404` -- Model directory not found at the given path
- `500` -- Upload failed (authentication error, network issue, etc.)

---

## Configuration

All settings are configured via environment variables prefixed with `KANI_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `KANI_MODEL_NAME` | `nineninesix/kani-tts-400m-0.3-pt` | Model to load at startup |
| `KANI_DEVICE_MAP` | `auto` | PyTorch device map (`auto`, `cpu`, `cuda:0`) |
| `KANI_CHECKPOINT_DIR` | `./checkpoints` | Default checkpoint directory |
| `KANI_MAX_NEW_TOKENS` | `1200` | Default max generation length |
| `KANI_TEMPERATURE` | `0.8` | Default sampling temperature |
| `KANI_TOP_P` | `0.95` | Default nucleus sampling threshold |
| `KANI_REPETITION_PENALTY` | `1.1` | Default repetition penalty |
| `KANI_USE_FLASH_ATTENTION` | `true` | Enable Flash Attention 2 |
| `KANI_USE_TORCH_COMPILE` | `false` | Enable torch.compile |
| `KANI_TOKENISER_LENGTH` | `64400` | Text tokenizer vocabulary size |

**Example:**
```bash
KANI_MODEL_NAME=jsbeaudry/haitian-kani-ht-v3 \
KANI_TEMPERATURE=0.6 \
KANI_USE_TORCH_COMPILE=true \
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Training Workflow

### End-to-end: train, upload, and use

```bash
# 1. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Launch training (with auto-upload to Hub)
JOB=$(curl -s -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "hf_datasets": [{"reponame": "jsbeaudry/kani-pretrain-data"}],
    "num_train_epochs": 4,
    "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
    "dataset_name": "jsbeaudry/haitian-kani-ht-v3"
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "Training job: $JOB"

# 3. Poll status until completed
curl http://localhost:8000/train/$JOB

# 4. Hot-swap to the fine-tuned model
curl -X POST http://localhost:8000/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./checkpoints/lora_kani_model_ft_exp"}'

# 5. Generate speech with the fine-tuned model
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from the fine-tuned model!"}' \
  --output finetuned_output.wav
```

### Upload a checkpoint manually (separate from training)

```bash
# Upload any local checkpoint to HuggingFace Hub
curl -X POST http://localhost:8000/model/upload \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./checkpoints/lora_kani_model_ft_exp",
    "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
    "dataset_name": "jsbeaudry/my-custom-kani-model"
  }'
```

### Training pipeline details

1. **Dataset loading** -- Downloads HF datasets and applies optional categorical filters
2. **Parallel preprocessing** -- Splits into shards, processes codec tokens in parallel workers
3. **Frame-level position encoding** -- Groups of 4 codec tokens share one RoPE position
4. **LoRA injection** -- Applies low-rank adapters to attention modules (configurable)
5. **SFT training** -- Supervised fine-tuning with cosine LR schedule and custom collator
6. **Merge & save** -- LoRA weights merged into base model for standalone deployment
7. **Hub upload** (optional) -- Auto-upload to HuggingFace Hub if `hf_token` + `dataset_name` are provided

---

## Audio Format

All generated audio is returned as:
- **Format**: WAV (RIFF)
- **Sample rate**: 22,050 Hz
- **Bit depth**: 16-bit signed integer (PCM)
- **Channels**: Mono

---

## Python Client Example

```python
import requests

# Generate speech
response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "Bonjour, comment allez-vous aujourd'hui?",
        "speaker_id": "0047599005d8",
        "temperature": 0.7,
    },
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(response.content)

# Or use with scipy/soundfile for further processing
import io
import scipy.io.wavfile

sample_rate, audio = scipy.io.wavfile.read(io.BytesIO(response.content))
print(f"Sample rate: {sample_rate}, Duration: {len(audio)/sample_rate:.2f}s")
```

---

## Deployment

### Docker (GPU)

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install flash-attn --no-build-isolation

COPY app/ app/
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t kani-tts-api .
docker run --gpus all -p 8000:8000 \
  -e KANI_MODEL_NAME=nineninesix/kani-tts-400m-0.3-pt \
  kani-tts-api
```

### RunPod / Cloud GPU

```bash
# Set environment variables in your pod template
export KANI_MODEL_NAME=nineninesix/kani-tts-400m-0.3-pt
export KANI_HF_TOKEN=hf_xxxxxxxxxxxxx
export KANI_USE_FLASH_ATTENTION=true

# Install and run
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Credits

- **Kani TTS Model**: [nineninesix/kani-tts-400m-0.3-pt](https://huggingface.co/nineninesix/kani-tts-400m-0.3-pt)
- **NeMo Nano Codec**: [nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)
- **Original Notebooks**: [jsbeaudry/kani-tts-finetune](https://gist.github.com/jsbeaudry/386f22b97bb625d55f7395ad858b4a86)

---

## License

This project wraps the Kani TTS model and NeMo Codec. Please refer to their respective licenses for usage terms.
