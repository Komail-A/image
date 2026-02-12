# FLUX API with EVA V1 LoRA Support

FastAPI server for FLUX.1-schnell image generation with LoRA support.

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX A6000)
- MinIO or S3-compatible storage

## Installation

```bash
# Install dependencies
pip install fastapi uvicorn diffusers torch boto3 transformers accelerate safetensors

# Download FLUX.1-schnell model
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir flux-model

# Create loras directory
mkdir loras
# Add your LoRA files (.safetensors) to loras/
```

## Configuration

Edit `flux_api.py` and update:

```python
API_KEY = "your-api-key"
S3_ENDPOINT = "http://your-minio-server:9010"
S3_BUCKET = "image-gen"
S3_ACCESS_KEY = "your-access-key"
S3_SECRET_KEY = "your-secret-key"
```

## Running

```bash
python3 flux_api.py
```

Server runs on `http://0.0.0.0:8999`

## API Usage

### Generate Image

**Endpoint:** `POST /generate`

**Headers:**
```
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**
```json
{
  "prompt": "woman sitting on a sofa, living room, casual outfit",
  "steps": 4,
  "width": 1024,
  "height": 1024,
  "lora": "EVA V1.safetensors",
  "lora_scale": 1.0,
  "seed": 777
}
```

**Example:**
```bash
curl -X POST "http://localhost:8999/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "woman sitting on a sofa, living room, casual outfit",
    "steps": 4,
    "width": 1024,
    "height": 1024,
    "lora": "EVA V1.safetensors",
    "lora_scale": 1.0,
    "seed": 777
  }'
```

**Response:**
```json
{
  "success": true,
  "image_id": "abc123",
  "filename": "flux_abc123_20260212_120000.png",
  "s3_url": "http://your-minio:9010/image-gen/flux_abc123_20260212_120000.png",
  "presigned_url": "http://...",
  "presigned_expires_in": 604800,
  "bucket": "image-gen",
  "prompt": "woman sitting on a sofa...",
  "lora_used": "EVA V1.safetensors",
  "seed": 777
}
```

## Model Files (Not Included)

Model files are too large for GitHub. Download separately:

- **FLUX.1-schnell**: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- **LoRA files**: Add your trained LoRA files to `loras/` directory

