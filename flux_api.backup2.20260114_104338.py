from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, Dict
import torch
from diffusers import FluxPipeline
import io
import uuid
import random
import os
from datetime import datetime, timedelta
import boto3
import hashlib
import secrets

# Config
API_KEY = "pzXluF464sC1LFYKZ_-vXKU4IPwMR4dFittsXjiP2cY"
S3_ENDPOINT = "http://hckd.synology.me:9010"
S3_BUCKET = "image-gen"
S3_ACCESS_KEY = "87r1ZuVsS4aYe4XpcnJt"
S3_SECRET_KEY = "99gz9RIraPVXemRHySqIwjGA9BTffttXLyHD1zOd"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(MODEL_DIR, "loras")

app = FastAPI(title="Flux API with EVA V1 LoRA and Presigned Tokens")

# Global state
pipe = None
s3_client = None
current_lora = None
lora_fused = False

# Presigned token storage (in-memory - for production, use Redis)
presigned_tokens: Dict[str, dict] = {}

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

class PresignedTokenRequest(BaseModel):
    name: Optional[str] = None  # Optional name for tracking
    expires_in_hours: int = 24  # Default 24 hours
    max_uses: Optional[int] = None  # Optional usage limit

class PresignedTokenResponse(BaseModel):
    token: str
    expires_at: str
    max_uses: Optional[int]
    name: Optional[str]

class ImageResponse(BaseModel):
    success: bool
    image_id: str
    filename: str
    s3_url: str
    bucket: str
    prompt: str
    lora_used: Optional[str]
    seed: int

def verify_api_key(authorization: str = Header(None)) -> bool:
    """Verify the main API key"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

def verify_presigned_token(token: str) -> bool:
    """Verify a presigned token"""
    if token not in presigned_tokens:
        raise HTTPException(status_code=403, detail="Invalid or expired presigned token")
    
    token_data = presigned_tokens[token]
    
    # Check expiration
    if datetime.now() > token_data["expires_at"]:
        del presigned_tokens[token]
        raise HTTPException(status_code=403, detail="Presigned token has expired")
    
    # Check usage limit
    if token_data["max_uses"] is not None:
        if token_data["uses"] >= token_data["max_uses"]:
            raise HTTPException(status_code=403, detail="Presigned token usage limit exceeded")
        token_data["uses"] += 1
    
    return True

def cleanup_expired_tokens():
    """Remove expired tokens from storage"""
    now = datetime.now()
    expired = [token for token, data in presigned_tokens.items() if now > data["expires_at"]]
    for token in expired:
        del presigned_tokens[token]

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
    print(f"Presigned token system enabled")
    print("=" * 60 + "\n")

@app.get("/health")
async def health():
    cleanup_expired_tokens()
    return {
        "status": "healthy",
        "model": "Flux.1-schnell",
        "lora_active": current_lora,
        "gpu_available": torch.cuda.is_available(),
        "active_presigned_tokens": len(presigned_tokens)
    }

@app.post("/create-presigned-token", response_model=PresignedTokenResponse)
async def create_presigned_token(req: PresignedTokenRequest, authorization: str = Header(None)):
    """
    Create a presigned token for temporary access (requires main API key)
    """
    verify_api_key(authorization)
    cleanup_expired_tokens()
    
    # Generate secure token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=req.expires_in_hours)
    
    # Store token
    presigned_tokens[token] = {
        "created_at": datetime.now(),
        "expires_at": expires_at,
        "max_uses": req.max_uses,
        "uses": 0,
        "name": req.name
    }
    
    print(f"\n[Presigned Token Created]")
    print(f"  Name: {req.name or 'N/A'}")
    print(f"  Expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Max Uses: {req.max_uses or 'Unlimited'}\n")
    
    return PresignedTokenResponse(
        token=token,
        expires_at=expires_at.isoformat(),
        max_uses=req.max_uses,
        name=req.name
    )

@app.post("/generate", response_model=ImageResponse)
async def generate(req: ImageRequest, authorization: str = Header(None)):
    """
    Generate image with main API key authentication
    """
    global current_lora, lora_fused
    
    verify_api_key(authorization)
    
    return await _generate_image(req)

@app.post("/generate-presigned", response_model=ImageResponse)
async def generate_presigned(req: ImageRequest, x_presigned_token: str = Header(None)):
    """
    Generate image with presigned token (no main API key needed)
    """
    global current_lora, lora_fused
    
    if not x_presigned_token:
        raise HTTPException(status_code=401, detail="Missing X-Presigned-Token header")
    
    verify_presigned_token(x_presigned_token)
    
    return await _generate_image(req)

async def _generate_image(req: ImageRequest):
    """
    Internal function to generate image (shared by both endpoints)
    """
    global current_lora, lora_fused
    
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
