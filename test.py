import requests
import os
import base64
import time
import sys
import shutil # For file operations if needed by services, not directly by tests here

# --- Configuration ---
BASE_URL_FRONTEND = "http://localhost:80"
BASE_URL_ROUTING = "http://localhost:8000"
BASE_URL_INFERENCE_MODEL1 = "http://localhost:8001"
BASE_URL_INFERENCE_MODEL2 = "http://localhost:8002"
BASE_URL_INFERENCE_MODEL3 = "http://localhost:8003"
BASE_URL_INFERENCE_MODEL4 = "http://localhost:8004"

# Path to a sample content image for testing uploads
# This script will create a dummy one if it doesn't exist.
SAMPLE_CONTENT_IMAGE_FILENAME = "sample_content_test.jpg"

# For latest model test (now repurposed for dynamic prefix)
PERSISTENT_STORAGE_MODELS_HOST_PATH = "/persistent_storage/models"
LATEST_MODEL_TXT_FILENAME = "latest.txt" # Expects a checkpoint file name like "udnie.ckpt-done"

# --- Helper Functions ---
def _get_checkpoint_name_from_latest_txt(model_dir_name, default_checkpoint_name):
    """Reads a checkpoint file name from latest.txt in the specified model directory."""
    latest_txt_host_path = os.path.join(PERSISTENT_STORAGE_MODELS_HOST_PATH, model_dir_name, LATEST_MODEL_TXT_FILENAME)
    checkpoint_name = default_checkpoint_name
    
    if os.path.exists(latest_txt_host_path):
        try:
            with open(latest_txt_host_path, 'r') as f:
                content = f.read().strip()
            if content:
                checkpoint_name = content
                print(f"INFO: Using checkpoint '{checkpoint_name}' from {latest_txt_host_path} for {model_dir_name}.")
            else:
                print(f"INFO: {latest_txt_host_path} is empty. Using default checkpoint '{default_checkpoint_name}' for {model_dir_name}.")
        except Exception as e:
            print(f"WARNING: Error reading {latest_txt_host_path}: {e}. Using default checkpoint '{default_checkpoint_name}' for {model_dir_name}.")
    else:
        print(f"INFO: {latest_txt_host_path} not found. Using default checkpoint '{default_checkpoint_name}' for {model_dir_name}.")
    return checkpoint_name

_MODEL_ROOT_IN_CONTAINER = "/persistent_storage/models"

# Define default checkpoint names
DEFAULT_MODEL1_CHECKPOINT = "udnie.ckpt-done"
DEFAULT_MODEL2_CHECKPOINT = "la_muse.ckpt-done"
DEFAULT_MODEL3_CHECKPOINT = "rain_princess.ckpt-done"
DEFAULT_MODEL4_CHECKPOINT = "the_scream.ckpt-done"

# Get checkpoint names, potentially overridden by latest.txt in each model folder
MODEL1_CHECKPOINT_NAME = _get_checkpoint_name_from_latest_txt("model1", DEFAULT_MODEL1_CHECKPOINT)
MODEL2_CHECKPOINT_NAME = _get_checkpoint_name_from_latest_txt("model2", DEFAULT_MODEL2_CHECKPOINT)
MODEL3_CHECKPOINT_NAME = _get_checkpoint_name_from_latest_txt("model3", DEFAULT_MODEL3_CHECKPOINT)
MODEL4_CHECKPOINT_NAME = _get_checkpoint_name_from_latest_txt("model4", DEFAULT_MODEL4_CHECKPOINT)

# Model path as expected by inference services (inside their containers)
MODEL1_PATH_IN_INFERENCE = f"{_MODEL_ROOT_IN_CONTAINER}/model1/{MODEL1_CHECKPOINT_NAME}"
MODEL2_PATH_IN_INFERENCE = f"{_MODEL_ROOT_IN_CONTAINER}/model2/{MODEL2_CHECKPOINT_NAME}"
MODEL3_PATH_IN_INFERENCE = f"{_MODEL_ROOT_IN_CONTAINER}/model3/{MODEL3_CHECKPOINT_NAME}"
MODEL4_PATH_IN_INFERENCE = f"{_MODEL_ROOT_IN_CONTAINER}/model4/{MODEL4_CHECKPOINT_NAME}"


def create_dummy_image_if_not_exists(filepath, size=(100, 100), color="blue"):
    if os.path.exists(filepath):
        return
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", size, color=color)
        # Add some text to make it slightly more unique if needed
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), os.path.basename(filepath), fill=(0,0,0))
        img.save(filepath)
        print(f"INFO: Created dummy image: {filepath}")
    except ImportError:
        print(f"WARNING: PIL/Pillow not installed. Cannot create dummy image {filepath}. Creating a text placeholder.")
        with open(filepath, "w") as f:
            f.write(f"dummy image data for {os.path.basename(filepath)}")

def check_service_health(service_name, base_url, endpoint="/health"):
    """Generic health check for a service."""
    # If the default "/health" is used, we assume the service root provides a JSON message
    # Otherwise, we use the provided endpoint.
    is_default_health_check = (endpoint == "/health")
    check_url = base_url if is_default_health_check else f"{base_url}{endpoint}"

    print(f"TEST: Health check for {service_name} at {check_url}")
    try:
        response = requests.get(check_url, timeout=10)
        if response.status_code == 200:
            if is_default_health_check: # Expecting JSON message like {"message": "Service Name"}
                try:
                    json_response = response.json()
                    if "message" in json_response:
                        print(f"PASS: {service_name} is healthy. Message: '{json_response['message']}'")
                        return True
                    else:
                        print(f"FAIL: {service_name} root responded 200 OK, but 'message' key missing in JSON. Response: {response.text[:200]}")
                        return False
                except ValueError: # Includes JSONDecodeError
                    print(f"FAIL: {service_name} root responded 200 OK, but response is not valid JSON. Response: {response.text[:200]}")
                    return False
            else: # For specific endpoints like frontend's "/"
                 print(f"PASS: {service_name} is healthy ({endpoint} path). Status: {response.status_code}")
                 return True
        else:
            print(f"FAIL: {service_name} health check failed. Status: {response.status_code}, Response: {response.text[:200]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAIL: {service_name} health check failed. Error: {e}")
        return False

# --- Test Functions ---

def test_frontend_loads():
    return check_service_health("Frontend", BASE_URL_FRONTEND, "/") # Frontend root should be 200

def test_routing_service_health():
    return check_service_health("Routing Service", BASE_URL_ROUTING)

def test_inference_service_health(model_num_str, base_url):
    return check_service_health(f"Inference Service {model_num_str}", base_url)

def _test_direct_inference_endpoint(model_name_display, inference_base_url, model_path_in_inference):
    print(f"TEST: Direct Inference endpoint ({model_name_display}) using {SAMPLE_CONTENT_IMAGE_FILENAME}")
    if not model_path_in_inference: # Handles case where latest model path couldn't be determined
        print(f"FAIL: Model path for {model_name_display} is not available. Skipping test.")
        return False
    if not os.path.exists(SAMPLE_CONTENT_IMAGE_FILENAME):
        print(f"FAIL: Sample content image '{SAMPLE_CONTENT_IMAGE_FILENAME}' not found for direct inference test ({model_name_display}).")
        return False

    url = f"{inference_base_url}/infer"
    try:
        with open(SAMPLE_CONTENT_IMAGE_FILENAME, 'rb') as content_file:
            files = {'content_image': (SAMPLE_CONTENT_IMAGE_FILENAME, content_file, 'image/jpeg')}
            data = {'model_path': model_path_in_inference}

            response = requests.post(url, files=files, data=data, timeout=180) # Increased timeout

        if response.status_code == 200:
            json_response = response.json()
            if "stylized_image_base64" in json_response and "output_filename" in json_response:
                try:
                    base64.b64decode(json_response["stylized_image_base64"])
                    print(f"PASS: Direct inference ({model_name_display}) returned 200 OK with expected JSON structure and valid base64 image.")
                    return True
                except Exception as e:
                    print(f"FAIL: Direct inference ({model_name_display}) returned 200 OK, but base64 decoding failed: {e}. Response: {json_response}")
                    return False
            else:
                print(f"FAIL: Direct inference ({model_name_display}) returned 200 OK, but JSON response structure is incorrect: {json_response}")
                return False
        elif response.status_code == 400 and "Model path not found or invalid" in response.text:
            print(f"FAIL: Direct inference ({model_name_display}) returned 400. Model path '{model_path_in_inference}' likely invalid or files missing. Response: {response.text[:200]}")
            return False
        else:
            print(f"FAIL: Direct inference ({model_name_display}) failed. Status: {response.status_code}, Response: {response.text[:500]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAIL: Direct inference ({model_name_display}) failed. Error: {e}")
        return False
    except FileNotFoundError:
        print(f"FAIL: Could not open sample image '{SAMPLE_CONTENT_IMAGE_FILENAME}' for direct inference test ({model_name_display}).")
        return False

def test_stylize_endpoint_via_routing_service():
    print(f"TEST: Stylize endpoint (model1) via Routing Service using {SAMPLE_CONTENT_IMAGE_FILENAME}")
    if not os.path.exists(SAMPLE_CONTENT_IMAGE_FILENAME):
        print(f"FAIL: Sample content image '{SAMPLE_CONTENT_IMAGE_FILENAME}' not found for stylize test.")
        return False

    url = f"{BASE_URL_ROUTING}/stylize"
    try:
        with open(SAMPLE_CONTENT_IMAGE_FILENAME, 'rb') as content_file:
            files = {'content_image': (SAMPLE_CONTENT_IMAGE_FILENAME, content_file, 'image/jpeg')}
            data = {'model': 'model1'} # Test with model1

            response = requests.post(url, files=files, data=data, timeout=180) # Increased timeout

        if response.status_code == 200:
            json_response = response.json()
            if "stylized_image_base64" in json_response and "output_filename" in json_response:
                try:
                    base64.b64decode(json_response["stylized_image_base64"])
                    print("PASS: Stylize endpoint returned 200 OK with expected JSON structure and valid base64 image.")
                    return True
                except Exception as e:
                    print(f"FAIL: Stylize endpoint returned 200 OK, but base64 decoding failed: {e}. Response: {json_response}")
                    return False
            else:
                print(f"FAIL: Stylize endpoint returned 200 OK, but JSON response structure is incorrect: {json_response}")
                return False
        else:
            print(f"FAIL: Stylize endpoint failed. Status: {response.status_code}, Response: {response.text[:500]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"FAIL: Stylize endpoint failed. Error: {e}")
        return False
    except FileNotFoundError:
        print(f"FAIL: Could not open sample image '{SAMPLE_CONTENT_IMAGE_FILENAME}' for stylize test.")
        return False


def test_direct_inference_endpoint_model1():
    return _test_direct_inference_endpoint("Model 1", BASE_URL_INFERENCE_MODEL1, MODEL1_PATH_IN_INFERENCE)

def test_direct_inference_endpoint_model2():
    return _test_direct_inference_endpoint("Model 2", BASE_URL_INFERENCE_MODEL2, MODEL2_PATH_IN_INFERENCE)

def test_direct_inference_endpoint_model3():
    return _test_direct_inference_endpoint("Model 3", BASE_URL_INFERENCE_MODEL3, MODEL3_PATH_IN_INFERENCE)

def test_direct_inference_endpoint_model4():
    return _test_direct_inference_endpoint("Model 4", BASE_URL_INFERENCE_MODEL4, MODEL4_PATH_IN_INFERENCE)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Automated Application Tests ---")

    # Create a dummy sample content image if it doesn't exist in the current directory
    create_dummy_image_if_not_exists(SAMPLE_CONTENT_IMAGE_FILENAME)

    # Allow some time for services to fully initialize after docker-compose up
    # Adjust this based on your system and service startup times
    initial_wait_time = 10
    print(f"\nWaiting for {initial_wait_time} seconds for services to initialize...")
    time.sleep(initial_wait_time)

    test_results = {} # Using a dictionary to store results with test names

    # --- Health Checks ---
    print("\n--- Running Health Checks ---")
    test_results["Frontend Load"] = test_frontend_loads()
    test_results["Routing Service Health"] = test_routing_service_health()
    test_results["Inference Service Model 1 Health"] = test_inference_service_health("Model 1", BASE_URL_INFERENCE_MODEL1)
    test_results["Inference Service Model 2 Health"] = test_inference_service_health("Model 2", BASE_URL_INFERENCE_MODEL2)
    test_results["Inference Service Model 3 Health"] = test_inference_service_health("Model 3", BASE_URL_INFERENCE_MODEL3)
    test_results["Inference Service Model 4 Health"] = test_inference_service_health("Model 4", BASE_URL_INFERENCE_MODEL4)

    # --- Functional Tests ---
    print("\n--- Running Functional Tests ---")
    # Only run functional tests if core dependent services are healthy
    if test_results.get("Routing Service Health") and test_results.get("Inference Service Model 1 Health"):
        test_results["Stylize Endpoint (Routing)"] = test_stylize_endpoint_via_routing_service()
        test_results["Direct Inference (Model 1)"] = test_direct_inference_endpoint_model1()
    else:
        print("\nSKIPPING Stylize/Direct Inference (Model 1) tests due to health check failures of dependent services.")
        test_results["Stylize Endpoint (Routing)"] = "SKIPPED" # Mark as skipped instead of False
        test_results["Direct Inference (Model 1)"] = "SKIPPED"

    if test_results.get("Inference Service Model 2 Health"):
        test_results["Direct Inference (Model 2)"] = test_direct_inference_endpoint_model2()
    else:
        print("\nSKIPPING Direct Inference (Model 2) test due to its service health check failure.")
        test_results["Direct Inference (Model 2)"] = "SKIPPED"

    if test_results.get("Inference Service Model 3 Health"):
        test_results["Direct Inference (Model 3)"] = test_direct_inference_endpoint_model3()
    else:
        print("\nSKIPPING Direct Inference (Model 3) test due to its service health check failure.")
        test_results["Direct Inference (Model 3)"] = "SKIPPED"

    if test_results.get("Inference Service Model 4 Health"):
        test_results["Direct Inference (Model 4)"] = test_direct_inference_endpoint_model4()
    else:
        print("\nSKIPPING Direct Inference (Model 4) test due to its service health check failure.")
        test_results["Direct Inference (Model 4)"] = "SKIPPED"
    

    # --- Test Summary ---
    print("\n--- Test Summary ---")
    all_tests_passed = True
    has_failures = False
    for test_name, result in test_results.items():
        if result == "SKIPPED":
            status = "SKIPPED"
        elif result: # True
            status = "PASS"
        else: # False
            status = "FAIL"
            has_failures = True
        
        print(f"{test_name}: {status}")
        # all_tests_passed is now effectively "no failures"
        if status == "FAIL":
            all_tests_passed = False


    if all_tests_passed and not has_failures: # Check both if you want to distinguish between all pass and some skips
        print("\nSUCCESS: All runable tests passed!")
        sys.exit(0)
    elif has_failures:
        print("\nFAILURE: Some tests failed.")
        sys.exit(1)
    else: # No failures, but some tests were skipped
        print("\nCOMPLETED: Some tests were skipped, but no failures.")
        sys.exit(0) # Or a different exit code for "passed with skips" if desired