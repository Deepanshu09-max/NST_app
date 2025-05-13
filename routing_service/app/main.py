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

# Adjust path assuming script is run from routing_service/ directory
PERSISTENT_STORAGE = "../persistent_storage"
INPUT_IMAGES = os.path.join(PERSISTENT_STORAGE, "input_images")
OUTPUT_IMAGES = os.path.join(PERSISTENT_STORAGE, "output_images")
MODELS_DIR = os.path.join(PERSISTENT_STORAGE, "models")
FEEDBACK_FILE = os.path.join(PERSISTENT_STORAGE, "feedback.jsonl")

os.makedirs(INPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def get_latest_model_path(model_name):
    model_dir = os.path.join(MODELS_DIR, model_name)
    latest_path = os.path.join(model_dir, "latest.txt")
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            latest_model_file = f.read().strip()
        return os.path.join(model_dir, latest_model_file)
    else:
        # Fallback to v1 if no latest.txt
        fallback = os.path.join(model_dir, f"{model_name}_v1.pth")
        if os.path.exists(fallback):
            return fallback
        return None

@app.get("/")
def read_root():
    return {"message": "Routing Service"}

@app.post("/api/stylize")
async def stylize(
    model: str = Form(...),
    content_image: UploadFile = File(...)
):
    # Save input images
    content_path = os.path.join(INPUT_IMAGES, content_image.filename)
    with open(content_path, "wb") as f:
        shutil.copyfileobj(content_image.file, f)
    logging.info(f"Saved input image to {content_path}")

    # Get the latest model file path
    model_path_local_check = get_latest_model_path(model)

    # Basic check if model file exists before calling inference
    if not model_path_local_check or not os.path.exists(model_path_local_check):
        logging.error(f"Model file not found at {model_path_local_check}")
        raise HTTPException(status_code=404, detail=f"Model file not found at {model_path_local_check}")

    # Send the actual model file path to the inference service
    model_path_in_inference = model_path_local_check

    # Route to inference service - Use localhost for local testing
    if model == "model1":
        # Assuming model1 runs on port 8001 locally
        url = "http://localhost:8001/infer"
    elif model == "model2":
        # Assuming model2 runs on port 8002 locally
        url = "http://localhost:8002/infer"
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
        logging.info(f"Sending request to {url} with model path {model_path_in_inference}")
        resp = requests.post(url, files=files, data=data, timeout=120)
        resp.raise_for_status()
        logging.info(f"Received response from {url} with status code {resp.status_code}")

        # Handle JSON Response
        inference_result = resp.json()
        img_base64 = inference_result.get("stylized_image_base64")
        output_filename_from_inference = inference_result.get("output_filename", f"stylized_{content_image.filename}")

        if not img_base64:
            logging.error("Inference service response did not contain 'stylized_image_base64'")
            raise HTTPException(status_code=500, detail="Inference service did not return image data.")
        else:
            logging.info(f"Received base64 image data (length: {len(img_base64)}, start: {img_base64[:30]}...)")

        # Decode base64 image
        img_data = base64.b64decode(img_base64)

        # Ensure the output filename has a standard image extension
        base, _ = os.path.splitext(output_filename_from_inference)
        output_filename = base + ".png"
        output_path = os.path.join(OUTPUT_IMAGES, output_filename)

        # Save the decoded image
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
        error_details = f"Failed to get response from inference service: {str(e)}"
        status_code = 502
        try:
            inference_error = e.response.json()
            error_details = {
                "error": "Inference service failed",
                "inference_service_status": e.response.status_code,
                "inference_service_error": inference_error
            }
            status_code = e.response.status_code
        except:
            pass
        logging.error(f"Request to inference service at {url} failed. Status: {e.response.status_code if e.response else 'N/A'}. Details: {error_details}")
        raise HTTPException(status_code=status_code, detail=error_details)
    except Exception as e:
        logging.exception("An internal server error occurred during stylization.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if content_file and not content_file.closed:
            content_file.close()
            logging.info(f"Closed content file {content_path}")

@app.post("/api/feedback")
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
