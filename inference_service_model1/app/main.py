from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import base64
import logging
from .model1_inference import run_inference

# --- Configuration ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Helper Functions ---

@app.get("/")
def read_root():
    return {"message": "Inference Service Model 1"}

def cleanup_files(files: list):
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/infer")
async def infer(
    background_tasks: BackgroundTasks,  # Keep BackgroundTasks for cleanup
    content_image: UploadFile = File(...),
    model_path: str = Form(...),
    image_size: int = Form(512)
):
    temp_content = f"/tmp/{content_image.filename}"
    output_filename = f"stylized_{os.path.splitext(content_image.filename)[0]}_{os.urandom(4).hex()}.png"
    temp_output = f"/tmp/{output_filename}"

    try:
        with open(temp_content, "wb") as f:
            shutil.copyfileobj(content_image.file, f)

        # run_inference no longer returns loss
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
