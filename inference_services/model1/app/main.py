from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import base64
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Make sure your project root is on sys.path so `model1_inference` can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model1_inference import generate

# === ADDED: Set image_size explicitly ===
image_size = 512  # or whatever your model expects

@app.get("/")
def read_root():
    return {"message": "Inference Service Model 1"}

def cleanup_files(files: list):
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)

def resolve_ckpt_prefix(prefix: str) -> str:
    base = os.path.dirname(prefix)
    name = os.path.basename(prefix)
    full = os.path.join(base, name)

    shards = [
        full + ".data-00000-of-00001",
        full + ".index",
        full + ".meta"
    ]
    missing = [p for p in shards if not os.path.exists(p)]
    if missing:
        logger.error(f"Missing checkpoint shards: {missing}")
        raise HTTPException(status_code=400,
                            detail=f"Missing checkpoint shards: {missing}")
    return full

@app.post("/infer")
async def infer(
    content_image: UploadFile = File(...),
    model_path: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    logger.info(f"Raw model_path received: {model_path}")
    ckpt_prefix = resolve_ckpt_prefix(model_path)
    logger.info(f"Resolved checkpoint prefix: {ckpt_prefix}")

    temp_content = f"/tmp/{content_image.filename}"
    rand_hex = os.urandom(4).hex()
    prefix = f"stylized_{os.path.splitext(content_image.filename)[0]}_{rand_hex}-"
    temp_output_dir = "/tmp"

    try:
        with open(temp_content, "wb") as f:
            shutil.copyfileobj(content_image.file, f)

        generate(
            contents_path=temp_content,
            model_path=ckpt_prefix,
            is_same_size=True,
            resize_height=image_size,
            resize_width=image_size,
            save_path=temp_output_dir,
            prefix=prefix,
            suffix=""
        )

        basename = os.path.splitext(os.path.basename(temp_content))[0]
        expected = f"{prefix}{basename}.jpg"
        temp_output = os.path.join(temp_output_dir, expected)
        if not os.path.exists(temp_output):
            candidates = [f for f in os.listdir(temp_output_dir)
                          if f.startswith(prefix) and f.endswith(".jpg")]
            if not candidates:
                raise Exception("Stylized image was not generated.")
            temp_output = os.path.join(temp_output_dir, candidates[0])

        with open(temp_output, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if background_tasks:
            background_tasks.add_task(cleanup_files, [temp_content, temp_output])

        return JSONResponse({
            "stylized_image_base64": img_base64,
            "output_filename": os.path.basename(temp_output)
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        cleanup_files([temp_content])
        return JSONResponse(status_code=500, content={"error": str(e)})
