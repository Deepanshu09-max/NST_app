from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, Response  # Keep both imports for now
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import requests
import json
import base64
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Adjust path assuming script is run from routing_service/app directory
PERSISTENT_STORAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../persistent_storage"))
INPUT_IMAGES = os.path.join(PERSISTENT_STORAGE, "input_images")
OUTPUT_IMAGES = os.path.join(PERSISTENT_STORAGE, "output_images")
MODELS_DIR = os.path.join(PERSISTENT_STORAGE, "models")  # Ensure this points to /persistent_storage/models
FEEDBACK_FILE = os.path.join(PERSISTENT_STORAGE, "feedback.jsonl")

os.makedirs(INPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuration for inference services
# These URLs now use the service names as hostnames and the internal container port (8000)
# INFERENCE_SERVICE_URLS = {
#     "model1": "http://inference_service_model1:8000/infer",
#     "model2": "http://inference_service_model2:8000/infer",
#     "model3": "http://inference_service_model3:8000/infer",
#     "model4": "http://inference_service_model4:8000/infer",
#     # Add other models and their inference service URLs here
# }

def get_latest_model_path(model_name: str):
    model_dir = os.path.join(MODELS_DIR, model_name)
    latest_path = os.path.join(model_dir, "latest.txt")
    logging.info(f"Checking for latest model in directory: {model_dir}")
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            latest_model_file = f.read().strip()
        logging.info(f"Found latest model file: {latest_model_file}")
        ckpt_prefix = os.path.join(model_dir, latest_model_file)
        ckpt_index = ckpt_prefix + ".index"
        ckpt_data = ckpt_prefix + ".data-00000-of-00001"
        ckpt_meta = ckpt_prefix + ".meta"
        
        if all(os.path.exists(p) for p in [ckpt_index, ckpt_data, ckpt_meta]):
            return ckpt_prefix
        if os.path.exists(ckpt_prefix) and ckpt_prefix.endswith(".pth"):
            return ckpt_prefix
        ckpt_file = ckpt_prefix + ".ckpt"
        if os.path.exists(ckpt_file):
            return ckpt_file
    logging.error(f"No valid model found for {model_name} in {model_dir}")
    return None

@app.get("/")
def read_root():
    return {"message": "Routing Service"}

@app.post("/stylize")
async def stylize(
    model: str = Form(...),
    content_image: UploadFile = File(...)
):
    logging.info(f"Stylize endpoint hit with model: {model}, content_image: {content_image.filename}")
    try:
        # Save input images
        content_path = os.path.join(INPUT_IMAGES, content_image.filename)
        with open(content_path, "wb") as f:
            shutil.copyfileobj(content_image.file, f)
        logging.info(f"Saved input image to {content_path}")

        # Get the latest model file path
        model_path_local_check = get_latest_model_path(model)
        logging.info(f"Resolved model path: {model_path_local_check}")

        # Adjust the model path to match the inference service's expectations
        model_path_in_inference = model_path_local_check.replace(
            "/persistent_storage/models", "/persistent_storage/models"
        )
        logging.info(f"Model path sent to inference service: {model_path_in_inference}")

        # Route to inference service - Use service names for Docker/K8s compatibility
        if model == "model1":
            url = "http://inference_service_model1:8000/infer"
        elif model == "model2":
            url = "http://inference_service_model2:8000/infer"
        elif model == "model3":
            url = "http://inference_service_model3:8000/infer"
        elif model == "model4":
            url = "http://inference_service_model4:8000/infer"
        else:
            logging.error(f"Invalid model specified: {model}")
            raise HTTPException(status_code=400, detail="Invalid model specified")

        logging.info(f"Routing request for model '{model}' to inference service at {url}")

        content_file = None
        try:
            content_file = open(content_path, "rb")
            files = {
                "content_image": (content_image.filename, content_file, content_image.content_type)
            }
            data = {"model_path": model_path_in_inference}
            logging.info(f"Sending request to {url} with data: {data}")
            resp = requests.post(url, files=files, data=data, timeout=120)
            resp.raise_for_status()
            logging.info(f"Received response from {url} with status code {resp.status_code}")

            # Handle JSON Response
            inference_result = resp.json()
            logging.info(f"Inference result: {inference_result}")
            img_base64 = inference_result.get("stylized_image_base64")
            output_filename_from_inference = inference_result.get("output_filename", f"stylized_{content_image.filename}")

            if not img_base64:
                logging.error("Inference service response did not contain 'stylized_image_base64'")
                raise HTTPException(status_code=500, detail="Inference service did not return image data.")

            # Decode base64 image
            img_data = base64.b64decode(img_base64)

            # Save the decoded image
            base, _ = os.path.splitext(output_filename_from_inference)
            output_filename = base + ".png"
            output_path = os.path.join(OUTPUT_IMAGES, output_filename)
            with open(output_path, "wb") as f:
                f.write(img_data)
            logging.info(f"Saved stylized image to {output_path}")

            # Return JSON Response (Compatible with Frontend)
            return JSONResponse(content={
                "stylized_image_base64": img_base64,
                "output_filename": output_filename,
                "message": "Stylization successful"
            })

        except requests.exceptions.Timeout:
            logging.error(f"Request to inference service at {url} timed out.")
            raise HTTPException(status_code=504, detail="Inference service request timed out")
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Could not connect to inference service at {url}. Error: {e}")
            raise HTTPException(status_code=503, detail=f"Could not connect to inference service at {url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request to inference service at {url} failed. Error: {e}")
            raise HTTPException(status_code=502, detail="Failed to get response from inference service")
        except Exception as e:
            logging.exception("An internal server error occurred during stylization.")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        finally:
            if content_file and not content_file.closed:
                content_file.close()
                logging.info(f"Closed content file {content_path}")
    except Exception as e:
        logging.exception("An internal server error occurred during stylization.")
        return JSONResponse(content={"error": "Internal server error", "details": str(e)}, status_code=500)

@app.post("/feedback")  # Ensure this matches the rewritten path
async def feedback(request: Request):
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON received"}, status_code=400)

    # Save feedback to persistent file
    try:
        with open(FEEDBACK_FILE, "a") as f:
            f.write(json.dumps(data) + "\n")
    except IOError as e:
        print(f"Error writing feedback file: {e}")
        return JSONResponse({"error": "Failed to save feedback due to server file issue"}, status_code=500)
    except Exception as e:
        print(f"Unexpected error saving feedback: {e}")
        return JSONResponse({"error": "An unexpected error occurred while saving feedback"}, status_code=500)

    print("Feedback received:", data)
    return {"status": "ok"}
