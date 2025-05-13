from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import logging
import os
from PIL import Image
import torch

try:
    from .model import TransformerNet
    from .training_utils import train_style_transfer
except ImportError:
    # Fallback for local dev if not running as a package
    from model import TransformerNet  # type: ignore
    from training_utils import train_style_transfer  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSISTENT_STORAGE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../persistent_storage")
)
MODELS_DIR = os.path.join(PERSISTENT_STORAGE, "models")
CONTENT_DIR = os.path.join(PERSISTENT_STORAGE, "input_images")
STYLE_IMAGE_PATH = os.path.join(PERSISTENT_STORAGE, "input_images", "style.jpg")

app = FastAPI()

class FineTuneRequest(BaseModel):
    model_name: str
    epochs: int = 2
    batch_size: int = 2
    image_size: int = 256
    lr: float = 1e-3
    style_weight: float = 1e5
    content_weight: float = 1e0

def get_next_model_version(model_dir: str, model_name: str) -> int:
    """Finds the next version number for the model."""
    if not os.path.exists(model_dir):
        return 1
    versions = []
    for fname in os.listdir(model_dir):
        if fname.startswith(f"{model_name}_v") and fname.endswith(".pth"):
            try:
                v = int(fname.split("_v")[-1].split(".pth")[0])
                versions.append(v)
            except Exception:
                continue
    return max(versions, default=0) + 1

def update_latest_version_file(model_dir: str, latest_model_filename: str):
    """Writes the latest model filename to latest.txt."""
    latest_path = os.path.join(model_dir, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(latest_model_filename)

def run_actual_finetuning(
    model_name: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    lr: float,
    style_weight: float,
    content_weight: float
):
    """
    Fine-tune the selected model using images from persistent storage.
    The new model version will overwrite the old one and be used for inference.
    """
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    # Find latest version to load
    latest_path = os.path.join(model_dir, "latest.txt")
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            latest_model_file = f.read().strip()
        model_path = os.path.join(model_dir, latest_model_file)
    else:
        # Fallback to v1 if no latest.txt
        model_path = os.path.join(model_dir, f"{model_name}_v1.pth")
    logger.info(f"Fine-tuning model: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return

    # Load content images from persistent storage
    if not os.path.exists(CONTENT_DIR):
        logger.error(f"Content directory not found: {CONTENT_DIR}")
        return
    content_image_files = [
        os.path.join(CONTENT_DIR, fname)
        for fname in os.listdir(CONTENT_DIR)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not content_image_files:
        logger.error("No content images found for fine-tuning.")
        return
    content_images = []
    for p in content_image_files:
        try:
            content_images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            logger.error(f"Could not load content image {p}: {e}")
    logger.info(f"Loaded {len(content_images)} content images for fine-tuning.")

    # Load style image
    if not os.path.exists(STYLE_IMAGE_PATH):
        logger.error(f"Style image not found at {STYLE_IMAGE_PATH}")
        return
    try:
        style_image = Image.open(STYLE_IMAGE_PATH).convert("RGB")
    except Exception as e:
        logger.error(f"Could not load style image: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = TransformerNet().to(device)
    try:
        model_instance.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return
    logger.info("Loaded model for fine-tuning.")

    # Fine-tune the model
    try:
        trained_model = train_style_transfer(
            content_images=content_images,
            style_image=style_image,
            model_instance=model_instance,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            lr=lr,
            style_weight=style_weight,
            content_weight=content_weight,
            device=device,
            progress_callback=lambda ep, eps, b_idx, total_b: logger.info(
                f"Epoch {ep}/{eps}, Batch {b_idx}/{total_b}"
            ),
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

    # Save as new version
    next_version = get_next_model_version(model_dir, model_name)
    new_model_filename = f"{model_name}_v{next_version}.pth"
    new_model_path = os.path.join(model_dir, new_model_filename)
    try:
        torch.save(trained_model.state_dict(), new_model_path)
        logger.info(f"Saved new model version to {new_model_path}")
        update_latest_version_file(model_dir, new_model_filename)
        logger.info(f"Updated latest.txt to {new_model_filename}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

@app.post("/finetune")
async def finetune_model(request: FineTuneRequest, background_tasks: BackgroundTasks):
    """
    Start fine-tuning a model in the background.
    """
    logger.info(f"Received fine-tuning request for model: {request.model_name}")
    model_dir = os.path.join(MODELS_DIR, request.model_name)
    latest_path = os.path.join(model_dir, "latest.txt")
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            latest_model_file = f.read().strip()
        model_path = os.path.join(model_dir, latest_model_file)
    else:
        model_path = os.path.join(model_dir, f"{request.model_name}_v1.pth")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {model_path}.")

    # Check content and style images exist
    if not os.path.exists(CONTENT_DIR) or not os.listdir(CONTENT_DIR):
        raise HTTPException(status_code=400, detail="No content images found for fine-tuning.")
    if not os.path.exists(STYLE_IMAGE_PATH):
        raise HTTPException(status_code=400, detail="Style image not found for fine-tuning.")

    background_tasks.add_task(
        run_actual_finetuning,
        request.model_name,
        request.epochs,
        request.batch_size,
        request.image_size,
        request.lr,
        request.style_weight,
        request.content_weight,
    )
    return {"message": f"Fine-tuning started for model '{request.model_name}'. A new version will be created and used for inference after completion."}

@app.get("/")
def read_root():
    return {"message": "Fine-tuning Service. POST to /finetune with model_name."}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

