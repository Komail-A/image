from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import FluxPipeline
import io
import uuid
import random
import os
from datetime import datetime
import boto3

# Config
API_KEY = "pzXluF464sC1LFYKZ_-vXKU4IPwMR4dFittsXjiP2cY"
S3_ENDPOINT = "http://hckd.synology.me:9010"
S3_BUCKET = "image-gen"
S3_ACCESS_KEY = "87r1ZuVsS4aYe4XpcnJt"
S3_SECRET_KEY = "99gz9RIraPVXemRHySqIwjGA9BTffttXLyHD1zOd"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(MODEL_DIR, "loras")

app = FastAPI(title="Flux API with EVA V1 LoRA")

# Global state
pipe = None
s3_client = None
current_lora = None
lora_fused = False

class ImageRequest(BaseModel):
    prompt: str
    steps: int = 4  # Flux schnell default
    width: int = 1024
    height: int = 1024
    filename: Optional[str] = None
    lora: Optional[str] = None
    lora_scale: float = 1.0
    seed: Optional[int] = None
    guidance_scale: float = 0.0  # Schnell uses 0.0

class ImageResponse(BaseModel):
    success: bool
    image_id: str
    filename: str
    s3_url: str
    bucket: str
    prompt: str
    lora_used: Optional[str]
    seed: int

@app.on_event("startup")
async def startup():
    global pipe, s3_client
    
    print("=" * 60)
    print("Flux API with EVA V1 LoRA Starting")
    print("=" * 60)
    
    # MinIO
    print("\n[1/2] MinIO...")
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    )
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"✓ Connected: {S3_BUCKET}")
    except:
        print(f"✗ Failed to connect to MinIO bucket: {S3_BUCKET}")
    
    # Load Flux
    print("\n[2/2] Loading Flux.1-schnell...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    pipe = FluxPipeline.from_pretrained(
        os.path.join(MODEL_DIR, "flux-model"),
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    print("✓ Flux loaded")
    print("\n" + "=" * 60)
    print(f"API Ready on port 8999")
    print(f"EVA V1 LoRA available")
    print("=" * 60 + "\n")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "Flux.1-schnell",
        "lora_active": current_lora,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=ImageResponse)
async def generate(req: ImageRequest, authorization: str = Header(None)):
    global current_lora, lora_fused
    
    # Auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Generate
    image_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = req.filename or f"flux_{image_id}.png"
    if not filename.endswith(".png"):
        filename = f"{filename}_{timestamp}.png"
    else:
        filename = f"{filename.replace(".png", "")}_{timestamp}.png"
    
    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    
    print(f"\n[{image_id}] Generating...")
    print(f"  Prompt: {req.prompt[:60]}...")
    print(f"  Size: {req.width}x{req.height}")
    print(f"  Steps: {req.steps}")
    print(f"  Seed: {seed}")
    
    try:
        # Handle LoRA
        lora_used = None
        if req.lora:
            lora_path = os.path.join(LORA_DIR, req.lora)
            
            if not os.path.exists(lora_path):
                raise HTTPException(status_code=404, detail=f"LoRA not found: {req.lora}")
            
            # Unfuse if previously fused
            if lora_fused:
                print(f"  Unfusing previous LoRA")
                pipe.unfuse_lora()
                lora_fused = False
            
            # Unload previous LoRA if different
            if current_lora and current_lora != req.lora:
                print(f"  Unloading previous LoRA: {current_lora}")
                pipe.unload_lora_weights()
                current_lora = None
            
            # Load new LoRA if needed
            if current_lora != req.lora:
                print(f"  Loading LoRA: {req.lora}")
                pipe.load_lora_weights(lora_path)
                current_lora = req.lora
            
            # Fuse LoRA with scale
            print(f"  Fusing LoRA with scale: {req.lora_scale}")
            pipe.fuse_lora(lora_scale=req.lora_scale)
            lora_fused = True
            print(f"  ✓ LoRA active: {req.lora}")
            lora_used = req.lora
        else:
            # Unfuse if no lora requested
            if lora_fused:
                pipe.unfuse_lora()
                lora_fused = False
            if current_lora:
                pipe.unload_lora_weights()
                current_lora = None
                print(f"  LoRA disabled")
        
        # Generate
        result = pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed)
        )
        
        image = result.images[0]
        
        # Upload to MinIO
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        
        s3_client.upload_fileobj(
            buf, S3_BUCKET, filename,
            ExtraArgs={"ContentType": "image/png"}
        )
        
        s3_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{filename}"
        print(f"  ✓ {s3_url}\n")
        
        return ImageResponse(
            success=True,
            image_id=image_id,
            filename=filename,
            s3_url=s3_url,
            bucket=S3_BUCKET,
            prompt=req.prompt,
            lora_used=lora_used,
            seed=seed
        )
    
    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8999)
