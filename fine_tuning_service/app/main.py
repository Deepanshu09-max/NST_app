from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import logging
import os
from imageio import imread # Import imread here
import numpy as np # Import numpy for shape check

try:
    from .training_utils import train
except ImportError:
    from training_utils import train  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



PERSISTENT_STORAGE = "/persistent_storage"

MODELS_DIR = os.path.join(PERSISTENT_STORAGE, "models")
CONTENT_DIR = os.path.join(PERSISTENT_STORAGE, "input_images")
VGG_WEIGHTS_PATH = os.path.join(PERSISTENT_STORAGE, "vgg19.npz") # Corrected VGG weights path

app = FastAPI()

class FineTuneRequest(BaseModel):
    model_name: str
    epochs: int = 2
    batch_size: int = 2
    image_size: int = 256
    lr: float = 1e-3
    style_weight: float = 1e5
    content_weight: float = 1e0

try:
    import tensorflow as tf
    try:
        tf1 = tf.compat.v1
        tf1.disable_v2_behavior()
    except AttributeError:
        tf1 = tf  # For TF1.x fallback
except ImportError:
    tf1 = None

def get_next_model_version(model_dir: str, model_name: str) -> int:
    logger.info(f"Determining next model version for '{model_name}' in directory '{model_dir}'")
    if not os.path.exists(model_dir):
        logger.info(f"Model directory '{model_dir}' does not exist. Starting with version 1.")
        return 1
    
    versions = []
    try:
        filenames = os.listdir(model_dir)
        logger.info(f"Files in '{model_dir}': {filenames}")
    except Exception as e:
        logger.error(f"Could not list files in '{model_dir}': {e}")
        logger.info("Assuming version 1 due to directory listing error.")
        return 1

    expected_prefix = f"{model_name}_v"
    # Checkpoint files are saved with extensions like .index, .meta, .data-00000-of-00001
    # We'll use .index as a representative file for a checkpoint prefix.
    # The filename stored in latest.txt is model_name_vN.ckpt, which is a prefix convention.
    expected_suffix_for_version_detection = ".index" 
    
    for fname in filenames:
        logger.debug(f"Checking file for versioning: {fname}")
        # We are looking for files like 'model_name_vN.index' or 'model_name_vN-STEP.index'
        if fname.startswith(expected_prefix) and fname.endswith(expected_suffix_for_version_detection):
            try:
                # Extract version part: model_name_v<VERSION>.index or model_name_v<VERSION>-<STEP>.index
                version_part_with_step = fname[len(expected_prefix):-len(expected_suffix_for_version_detection)]
                
                # Handle cases like model_name_v<VERSION>-<STEP>
                version_str = version_part_with_step.split('-')[0]

                v = int(version_str)
                versions.append(v)
                logger.debug(f"Found matching file: {fname}, extracted base version: {v}")
            except ValueError:
                logger.warning(f"Could not parse version from filename: {fname}. Extracted version string: '{version_str}'. Skipping.")
            except Exception as e:
                logger.warning(f"Error processing filename {fname} for versioning: {e}. Skipping.")
        else:
            logger.debug(f"File {fname} does not match versioning pattern '{expected_prefix}*{expected_suffix_for_version_detection}'.")
            
    if not versions:
        logger.info(f"No existing version files found for '{model_name}' with pattern '{expected_prefix}*{expected_suffix_for_version_detection}'. Starting with version 1.")
        return 1
    
    current_max_version = max(versions)
    next_version = current_max_version + 1
    logger.info(f"Found versions from .index files: {sorted(list(set(versions)))}. Max existing version: {current_max_version}. Next version: {next_version}.")
    return next_version

def update_latest_version_file(model_dir: str, latest_model_filename: str):
    latest_path = os.path.join(model_dir, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(latest_model_filename)

def resolve_model_path(model_name: str):
    model_dir = os.path.join(MODELS_DIR, model_name)
    latest_path = os.path.join(model_dir, "latest.txt")
    if os.path.isdir(model_dir):
        if os.path.exists(latest_path):
            with open(latest_path, "r") as f:
                latest_model_file = f.read().strip()
            # latest_model_file is now just the base name, e.g., model3_v2
            ckpt_prefix = os.path.join(model_dir, latest_model_file) 
            ckpt_index = ckpt_prefix + ".index"
            ckpt_meta = ckpt_prefix + ".meta"
            ckpt_data = ckpt_prefix + ".data-00000-of-00001"
            if all(os.path.exists(p) for p in [ckpt_index, ckpt_meta, ckpt_data]):
                return ckpt_prefix
            # Fallback if latest.txt points to a name that needs .ckpt appended (older logic)
            ckpt_prefix_with_ext = os.path.join(model_dir, latest_model_file + ".ckpt")
            ckpt_index_with_ext = ckpt_prefix_with_ext + ".index"
            ckpt_meta_with_ext = ckpt_prefix_with_ext + ".meta"
            ckpt_data_with_ext = ckpt_prefix_with_ext + ".data-00000-of-00001"
            if all(os.path.exists(p) for p in [ckpt_index_with_ext, ckpt_meta_with_ext, ckpt_data_with_ext]):
                 return ckpt_prefix_with_ext

        # Fallback to v1 if latest.txt is missing or invalid
        logger.info(f"latest.txt not found or invalid for {model_name}. Trying fallback to _v1.")
        fallback_ckpt_prefix = os.path.join(model_dir, f"{model_name}_v1")
        fallback_ckpt_index = fallback_ckpt_prefix + ".index"
        fallback_ckpt_meta = fallback_ckpt_prefix + ".meta"
        fallback_ckpt_data = fallback_ckpt_prefix + ".data-00000-of-00001"
        if all(os.path.exists(p) for p in [fallback_ckpt_index, fallback_ckpt_meta, fallback_ckpt_data]):
            return fallback_ckpt_prefix
        # Further fallback for older .ckpt extension in v1 name
        fallback_ckpt_prefix_ext = os.path.join(model_dir, f"{model_name}_v1.ckpt")
        fallback_ckpt_index_ext = fallback_ckpt_prefix_ext + ".index"
        fallback_ckpt_meta_ext = fallback_ckpt_prefix_ext + ".meta"
        fallback_ckpt_data_ext = fallback_ckpt_prefix_ext + ".data-00000-of-00001"
        if all(os.path.exists(p) for p in [fallback_ckpt_index_ext, fallback_ckpt_meta_ext, fallback_ckpt_data_ext]):
            return fallback_ckpt_prefix_ext
            
        raise FileNotFoundError(f"latest.txt not found or invalid in {model_dir}, and v1 fallback failed.")
    raise FileNotFoundError(f"Model directory {model_dir} does not exist or is not a directory.")

def file_exists_and_log(path, description):
    if not os.path.exists(path):
        logger.error(f"{description} not found at {path}")
        return False
    logger.info(f"{description} found at {path}")
    return True

def validate_image(image_path: str, image_type: str) -> bool:
    """Helper function to validate a single image."""
    if not file_exists_and_log(image_path, image_type):
        return False
    try:
        img = imread(image_path)
        if img is None:
            logger.error(f"{image_type} at {image_path} could not be loaded (None).")
            return False
        if not isinstance(img, np.ndarray) or len(img.shape) != 3 or img.shape[2] != 3:
            logger.error(f"{image_type} at {image_path} is not a valid RGB image (shape: {getattr(img, 'shape', 'N/A')}).")
            return False
    except Exception as e:
        logger.error(f"{image_type} at {image_path} could not be loaded or is invalid: {e}")
        return False
    return True

@app.post("/finetune")
async def finetune_model(request: FineTuneRequest, background_tasks: BackgroundTasks):
    model_dir = os.path.join(MODELS_DIR, request.model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Define model-specific style image path
    model_specific_style_image_path = os.path.join(model_dir, "style.jpg")

    try:
        # This resolve_model_path is for the model being fine-tuned, not VGG
        resolve_model_path(request.model_name) 
        logger.info(f"Existing model checkpoint for '{request.model_name}' resolved (will be used as a base if applicable, or a new one created).")
    except FileNotFoundError:
        logger.info(f"No existing model checkpoint found for '{request.model_name}'. A new model will be trained from scratch or base VGG.")
    except Exception as e:
        logger.error(f"Model resolve error for '{request.model_name}': {e}")
        # Not raising HTTPException here as we might be training a new model version from scratch
        # and don't strictly need a pre-existing one for the fine-tuning process itself.
        # The `train` function handles loading or initializing weights.

    if not file_exists_and_log(CONTENT_DIR, "Content directory"):
        raise HTTPException(status_code=400, detail="Content directory not found.")
    if not os.listdir(CONTENT_DIR):
        logger.error("No files found in the content directory for fine-tuning.")
        raise HTTPException(status_code=400, detail="No content images found for fine-tuning.")
    
    if not validate_image(model_specific_style_image_path, f"Style image for {request.model_name}"):
        raise HTTPException(status_code=400, detail=f"Style image 'style.jpg' is invalid or not found in model directory {model_dir}.")
    
    if not file_exists_and_log(VGG_WEIGHTS_PATH, "VGG19 weights file"):
        raise HTTPException(status_code=400, detail=f"VGG19 weights file not found at {VGG_WEIGHTS_PATH}")

    def run_finetune_task(): # Renamed to avoid conflict if run_finetune is imported elsewhere
        all_content_image_paths = [
            os.path.join(CONTENT_DIR, fname)
            for fname in os.listdir(CONTENT_DIR)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        logger.info(f"Initial content image candidates: {all_content_image_paths}")

        # Filter out the style image from content images and validate
        valid_content_images = []
        for img_path in all_content_image_paths:
            if os.path.abspath(img_path) == os.path.abspath(model_specific_style_image_path):
                logger.info(f"Skipping style image '{img_path}' (model-specific) from content images list.")
                continue
            if validate_image(img_path, "Content image"):
                valid_content_images.append(img_path)
        
        if not valid_content_images:
            logger.error("No valid content images found after filtering and validation.")
            # Optionally, raise an error or send a notification
            return

        logger.info(f"Final valid content images for training: {valid_content_images}")
        logger.info(f"Style image to be used (model-specific): {model_specific_style_image_path}")

        next_version = get_next_model_version(model_dir, request.model_name)
        new_model_filename_base = f"{request.model_name}_v{next_version}"
        # The .ckpt extension is typically added by the TensorFlow saver, 
        # so we pass the prefix to the train function.
        new_model_save_prefix = os.path.join(model_dir, new_model_filename_base)

        try:
            import training_utils # Ensure tf1 is available in training_utils
            training_utils.tf1 = tf1 

            train(
                content_targets_path=valid_content_images,
                style_target_path=model_specific_style_image_path, # Use model-specific style image
                content_weight=request.content_weight,
                style_weight=request.style_weight,
                tv_weight=1e-2, # This is a common default, can be made configurable
                vgg_path=VGG_WEIGHTS_PATH,
                save_path=new_model_save_prefix, # Pass prefix, saver adds step and .ckpt
                debug=False, # Set to True for more verbose TensorFlow logging
                logging_period=100
            )
            # After successful training, update latest.txt to point to the new model prefix (without .ckpt)
            update_latest_version_file(model_dir, new_model_filename_base) 
            logger.info(f"Fine-tuned model saved with prefix {new_model_save_prefix} and latest.txt updated to '{new_model_filename_base}'.")
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}", exc_info=True)

    background_tasks.add_task(run_finetune_task)
    return {"message": f"Fine-tuning started for model '{request.model_name}'. A new version will be created."}

@app.get("/")
def read_root():
    return {"message": "Fine-tuning Service. POST to /finetune with model_name."}

@app.get("/health")
def health():
    return {"status": "ok"}

# To run the fine-tuning service:
# 1. Open a terminal and navigate to the fine_tuning_service/app directory.
# 2. Start the FastAPI server with uvicorn:
#    uvicorn main:app --host 0.0.0.0 --port 8001

# To check if the service is running:
# - Open http://localhost:8001/health in your browser or run:
#   curl http://localhost:8001/health

# To trigger fine-tuning (example using curl):
# curl -X POST "http://localhost:8001/finetune" \
#      -H "Content-Type: application/json" \
#      -d '{"model_name": "model3", "epochs": 2, "batch_size": 2, "image_size": 256, "lr": 0.001, "style_weight": 1e5, "content_weight": 1.0}'

# - You can also use the Swagger UI at http://localhost:8001/docs to POST to /finetune interactively.

# The fine-tuning will run in the background. Check logs for progress and completion.
# The new model checkpoint will be saved in persistent_storage/models/<model_name>/ and latest.txt will be updated.

