from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import base64
import logging
# Fix import for run_inference to work both as a module and as a script
try:
    from .model2_inference import run_inference
except ImportError:
    from model2_inference import run_inference

# --- Configuration ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Helper Functions ---

@app.get("/")
def read_root():
    return {"message": "Inference Service Model 2"}

def cleanup_files(files: list):
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/infer")
async def infer(
    background_tasks: BackgroundTasks,
    content_image: UploadFile = File(...),
    model_path: str = Form(...),
    image_size: int = Form(512)
):
    temp_content = f"/tmp/{content_image.filename}"
    output_filename = f"stylized_{os.path.splitext(content_image.filename)[0]}_{os.urandom(4).hex()}.png"
    temp_output = f"/tmp/{output_filename}"

    # --- Model path resolution and error handling ---
    # Convert relative model_path to absolute path if needed
    if not os.path.isabs(model_path):
        # Assume model_path is relative to the current working directory
        # or to the app root; resolve relative to this file's parent directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_model_path = os.path.abspath(os.path.join(base_dir, model_path))
        model_path = abs_model_path

    if os.path.isdir(model_path):
        latest_txt = os.path.join(model_path, "latest.txt")
        if os.path.exists(latest_txt):
            with open(latest_txt, "r") as f:
                latest_model_file = f.read().strip()
            model_path = os.path.join(model_path, latest_model_file)
        else:
            # Fallback to v1
            model_name = os.path.basename(model_path)
            fallback = os.path.join(model_path, f"{model_name}_v1.pth")
            if os.path.exists(fallback):
                model_path = fallback
            else:
                logger.error(f"No valid model file found in directory {model_path}")
                cleanup_files([temp_content, temp_output])
                return JSONResponse(status_code=404, content={"error": f"No valid model file found in directory {model_path}"})

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        cleanup_files([temp_content, temp_output])
        return JSONResponse(status_code=404, content={"error": f"Model file not found at {model_path}"})

    try:
        with open(temp_content, "wb") as f:
            shutil.copyfileobj(content_image.file, f)

        run_inference(
            model_path=model_path,
            content_image_path=temp_content,
            output_path=temp_output,
            image_size=image_size,
        )

        with open(temp_output, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        background_tasks.add_task(cleanup_files, [temp_content, temp_output])

        return JSONResponse(content={
            "stylized_image_base64": img_base64,
            "output_filename": output_filename
        })
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        cleanup_files([temp_content, temp_output])
        return JSONResponse(status_code=500, content={"error": str(e)})
