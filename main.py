from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import io
import os

from stable_diffusion_cpp import StableDiffusion

app = FastAPI(
    title="Web API for stable-diffusion.cpp",
    description="Generate images using a local stable-diffusion.cpp backend.",
    version="1.0.0"
)

MODEL_PATH = os.path.join("/Volumes/TOSHIBA EXT/GGUF", "chroma-unlocked-v20-Q4_K_S.gguf")

sd_pipeline: Optional[StableDiffusion] = None

@app.on_event("startup")
async def startup_event():
    global sd_pipeline
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        print(f"Loading Stablecpp model from {MODEL_PATH}...")
        sd_pipeline = StableDiffusion(model_path=MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load Stablecpp model: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global sd_pipeline
    sd_pipeline = None
    print("Stablecpp model unloaded.")


class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1 

class ImageToImageRequest(BaseModel):
    pass


@app.post("/generate/text-to-image")
async def generate_text_to_image(request: TextToImageRequest):
    if sd_pipeline is None:
        raise HTTPException(status_code=503, detail="Stablecpp model not loaded.")

    print(f"Generating image for prompt: '{request.prompt}'")
    try:
        images = sd_pipeline.txt_to_img(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            sample_steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            progress_callback=lambda step, steps, time: print(f"Step {step}/{steps}") 
        )

        if not images:
            raise HTTPException(status_code=500, detail="Image generation failed, no image returned.")

        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format="PNG") 
        img_byte_arr.seek(0) 

        print("Image generated successfully.")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"Error during image generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {e}")


@app.get("/")
async def read_root():
    return {"message": "Stablecpp API is running. Use /docs for API reference."}

# To run this:
# 1. Ensure you have activated your virtual environment: `source sd_api_env/bin/activate`
# 2. Run from your terminal in the directory containing main.py:
#    `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
#    Remove `--reload` for production.