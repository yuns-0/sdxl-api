from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionXLPipeline
import base64
from io import BytesIO
import os
import gc
import time
import math
import uvicorn

app = FastAPI(title="Stable Diffusion XL API", description="API for generating images with Stable Diffusion XL optimized for portrait images")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System info and memory management
def get_gpu_memory():
    """Return detailed GPU memory info"""
    if torch.cuda.is_available():
        # Force garbage collection and cache clearing
        torch.cuda.empty_cache()
        gc.collect()
        
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)    # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        
        return {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(total, 2),
            "allocated_memory_gb": round(allocated, 2),
            "reserved_memory_gb": round(reserved, 2),
            "free_memory_gb": round(total - reserved, 2)
        }
    return {"error": "CUDA not available"}

print("Loading SDXL model optimized for portrait images (1200x1800) on RTX 3060...")

# Memory optimizations
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SDXL model with memory optimizations
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Progressive loading to avoid VRAM spikes
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,  # Half precision is essential
    use_safetensors=True,
    variant="fp16"
)

# Apply memory optimizations
pipe.enable_attention_slicing(slice_size="auto")  # Help with large images
pipe = pipe.to(device)
pipe = pipe.to(torch.float16)

# VAE tiling is essential for large portrait images
if hasattr(pipe, 'enable_vae_tiling'):
    pipe.enable_vae_tiling()
    print("VAE tiling enabled for generating large images")
else:
    print("WARNING: VAE tiling not available, large images may fail")

# Enable xformers for efficient memory attention
if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
    pipe.enable_xformers_memory_efficient_attention()
    print("Using xformers for efficient attention")
else:
    print("xformers not available, using standard attention")

print("Model loaded successfully with portrait image optimizations!")
print(f"GPU info: {get_gpu_memory()}")

# Defining request models
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1200
    height: int = 1800
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    seed: int = None

@app.get("/")
def root():
    return {
        "message": "Stable Diffusion XL API is running!",
        "endpoints": {
            "/health": "Check system health and GPU status",
            "/generate": "Generate images with custom parameters",
            "/generate_portrait": "Generate optimized portrait images"
        },
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    gpu_info = get_gpu_memory()
    free_gb = gpu_info.get("free_memory_gb", 0)
    can_generate_large = free_gb > 10  # Need ~10GB for 1200x1800
    
    return {
        "status": "healthy",
        "gpu_info": gpu_info,
        "can_generate_1200x1800": can_generate_large
    }

@app.post("/generate")
async def generate(request: GenerationRequest):
    prompt = request.prompt
    negative_prompt = request.negative_prompt
    
    # Default to portrait dimensions
    width = request.width
    height = request.height
    
    # Safety check for dimensions - RTX 3060 has limitations
    max_pixels = 1200 * 1800  # Target resolution
    if width * height > max_pixels:
        # Scale down proportionally if exceeds target
        scale_factor = math.sqrt(max_pixels / (width * height))
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        print(f"Requested dimensions too large, scaled to: {width}x{height}")
    
    # Optimize steps for large images
    num_inference_steps = request.num_inference_steps
    guidance_scale = request.guidance_scale
    seed = request.seed
    
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    
    # Memory cleanup before generation
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Generating {width}x{height} image with prompt: '{prompt}'")
    print(f"Current GPU memory state: {get_gpu_memory()}")
    
    start_time = time.time()
    
    try:
        # Generate image with portrait optimizations
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        generation_time = time.time() - start_time
        print(f"Image generated successfully in {generation_time:.2f} seconds!")
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Memory cleanup after generation
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'image': img_str,
            'dimensions': f"{width}x{height}",
            'steps': num_inference_steps,
            'generation_time_seconds': round(generation_time, 2)
        }
    
    except RuntimeError as e:
        # Handle out of memory errors gracefully
        torch.cuda.empty_cache()
        gc.collect()
        
        if "CUDA out of memory" in str(e):
            # Suggest optimizations specific to portrait images
            raise HTTPException(
                status_code=507,
                detail={
                    'error': 'GPU out of memory',
                    'message': 'Try reducing image height or width by ~20%, or reduce steps to 20',
                    'gpu_info': get_gpu_memory(),
                    'suggestion': 'Portrait images need more VRAM. Try 1024x1536 instead.'
                }
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_portrait")
async def generate_portrait(request: GenerationRequest):
    """Specialized endpoint for portrait-oriented images with optimized settings"""
    # Force portrait dimensions but allow customization
    width = request.width
    height = request.height
    
    # Ensure it's actually portrait orientation
    if width > height:
        width, height = height, width
        print("Swapped dimensions to ensure portrait orientation")
    
    # Add portrait-specific prompt enhancements
    prompt = request.prompt
    if not any(term in prompt.lower() for term in ['portrait', 'vertical']):
        prompt += ", portrait orientation, vertical composition"
    
    # Create a new request with the updated values
    new_request = GenerationRequest(
        prompt=prompt,
        negative_prompt=request.negative_prompt,
        width=width,
        height=height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed
    )
    
    # Pass to regular generate function
    return await generate(new_request)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)