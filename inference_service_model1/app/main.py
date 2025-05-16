from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import os
import base64
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model1_inference import generate

@app.get("/")
def read_root():
    return {"message": "Inference Service Model 1"}

def cleanup_files(files: list):
    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)

def resolve_model_path(model_path: str):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../persistent_storage/models/model1"))
    if not model_path or not os.path.isabs(model_path):
        model_dir = base_dir
    else:
        model_dir = model_path
    if os.path.isdir(model_dir):
        latest_txt = os.path.join(model_dir, "latest.txt")
        if os.path.exists(latest_txt):
            with open(latest_txt, "r") as f:
                latest_model_file = f.read().strip()
            ckpt_prefix = os.path.join(model_dir, latest_model_file)
            ckpt_index = ckpt_prefix + ".index"
            ckpt_meta = ckpt_prefix + ".meta"
            ckpt_data = ckpt_prefix + ".data-00000-of-00001"
            if all(os.path.exists(p) for p in [ckpt_index, ckpt_meta, ckpt_data]):
                return ckpt_prefix
            pth_file = os.path.join(model_dir, latest_model_file)
            if os.path.exists(pth_file):
                return pth_file
            fallback_ckpt = os.path.join(model_dir, "model1_v1.ckpt")
            if all(os.path.exists(fallback_ckpt + ext) for ext in [".index", ".meta", ".data-00000-of-00001"]):
                return fallback_ckpt
            fallback_pth = os.path.join(model_dir, "model1_v1.pth")
            if os.path.exists(fallback_pth):
                return fallback_pth
            raise FileNotFoundError(f"No valid model checkpoint (.ckpt) or .pth file found in {model_dir}")
        else:
            raise FileNotFoundError(f"latest.txt not found in {model_dir}")
    elif os.path.isfile(model_dir):
        return model_dir
    else:
        raise FileNotFoundError(f"Model path {model_dir} does not exist")

@app.post("/infer")
async def infer(
    background_tasks: BackgroundTasks,
    content_image: UploadFile = File(...),
    model_path: str = Form(""),
    image_size: int = Form(512)
):
    temp_content = f"/tmp/{content_image.filename}"
    prefix = f"stylized_{os.path.splitext(content_image.filename)[0]}_{os.urandom(4).hex()}-"
    temp_output_dir = "/tmp"

    try:
        with open(temp_content, "wb") as f:
            shutil.copyfileobj(content_image.file, f)

        resolved_model_path = resolve_model_path("")
        logger.info(f"Resolved model1 checkpoint path: {resolved_model_path}")

        try:
            generate(
                contents_path=temp_content,
                model_path=resolved_model_path,
                is_same_size=True,
                resize_height=image_size,
                resize_width=image_size,
                save_path=temp_output_dir,
                prefix=prefix,
                suffix=""
            )
        except Exception as tf_e:
            logger.error(f"TensorFlow error during generate(): {tf_e}")
            cleanup_files([temp_content])
            return JSONResponse(status_code=500, content={"error": f"TensorFlow error: {str(tf_e)}"})

        basename = os.path.splitext(os.path.basename(temp_content))[0]
        output_filename = f"{prefix}{basename}.jpg"
        temp_output = os.path.join(temp_output_dir, output_filename)
        if not os.path.exists(temp_output):
            candidates = [f for f in os.listdir(temp_output_dir) if f.startswith(prefix) and f.endswith(".jpg")]
            if not candidates:
                raise Exception("Stylized image not generated.")
            temp_output = os.path.join(temp_output_dir, candidates[0])
            output_filename = candidates[0]

        with open(temp_output, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        background_tasks.add_task(cleanup_files, [temp_content, temp_output])

        return JSONResponse(content={
            "stylized_image_base64": img_base64,
            "output_filename": output_filename
        })
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        cleanup_files([temp_content])
        return JSONResponse(status_code=500, content={"error": str(e)})
