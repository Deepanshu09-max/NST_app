#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Define base directory for the application
APP_DIR="."
PERSISTENT_STORAGE_DIR="$APP_DIR/persistent_storage"
INPUT_IMAGES_DIR="$PERSISTENT_STORAGE_DIR/input_images"
MODELS_DIR="$PERSISTENT_STORAGE_DIR/models"
MODEL1_DIR="$MODELS_DIR/model1" # Example for fine-tuning test

ROUTING_SERVICE_URL="http://localhost:8000"
INFERENCE_SERVICE_MODEL1_URL="http://localhost:8001"
INFERENCE_SERVICE_MODEL2_URL="http://localhost:8002"
INFERENCE_SERVICE_MODEL3_URL="http://localhost:8003"
INFERENCE_SERVICE_MODEL4_URL="http://localhost:8004"
FINE_TUNING_SERVICE_URL="http://localhost:8005" # Assumed port

TEST_CONTENT_IMAGE_NAME="test_content_image.jpg"
TEST_CONTENT_IMAGE_PATH="$INPUT_IMAGES_DIR/$TEST_CONTENT_IMAGE_NAME"
DUMMY_STYLE_IMAGE_PATH="$MODEL1_DIR/style.jpg"
DUMMY_VGG_WEIGHTS_PATH="$PERSISTENT_STORAGE_DIR/vgg19.npz"

# --- Helper Functions ---
cleanup() {
  echo "Cleaning up..."
  docker-compose -f "$APP_DIR/docker-compose.yml" down -v --remove-orphans
  rm -f "$TEST_CONTENT_IMAGE_PATH"
  rm -f "$DUMMY_STYLE_IMAGE_PATH"
  rm -f "$DUMMY_VGG_WEIGHTS_PATH"
  # Potentially remove other test-generated files from persistent_storage if needed
  echo "Cleanup finished."
}

# Trap errors and ensure cleanup runs
trap cleanup EXIT ERR INT TERM

# --- Test Setup ---
echo "Setting up test environment..."
mkdir -p "$INPUT_IMAGES_DIR"
mkdir -p "$MODEL1_DIR"

# Create a dummy content image for testing
echo "dummy image data" > "$TEST_CONTENT_IMAGE_PATH"
# Create dummy style image and VGG weights for fine-tuning endpoint test
echo "dummy style data" > "$DUMMY_STYLE_IMAGE_PATH"
touch "$DUMMY_VGG_WEIGHTS_PATH"

echo "Building and starting services via Docker Compose..."
docker-compose -f "$APP_DIR/docker-compose.yml" build
docker-compose -f "$APP_DIR/docker-compose.yml" up -d

echo "Waiting for services to become healthy (sleeping for 60 seconds)..."
# In a production Jenkins pipeline, use a more robust health check loop
sleep 60

# --- Run Tests ---
echo "Starting API tests..."

# 1. Health Checks
echo "Performing health checks..."
curl -f "$ROUTING_SERVICE_URL/" || { echo "Routing service health check failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL1_URL/" || { echo "Inference service model1 health check failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL2_URL/" || { echo "Inference service model2 health check failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL3_URL/" || { echo "Inference service model3 health check failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL4_URL/" || { echo "Inference service model4 health check failed"; exit 1; }
curl -f "$FINE_TUNING_SERVICE_URL/health" || { echo "Fine-tuning service health check failed"; exit 1; }
echo "All services are healthy."

# 2. Stylize Endpoint Test
echo "Testing /api/stylize endpoint..."
response_stylize=$(curl -s -X POST \
  -F "model=model1" \
  -F "content_image=@$TEST_CONTENT_IMAGE_PATH" \
  "$ROUTING_SERVICE_URL/api/stylize")

if echo "$response_stylize" | grep -q "stylized_image_base64"; then
  echo "/api/stylize test PASSED"
else
  echo "/api/stylize test FAILED. Response: $response_stylize"
  exit 1
fi

# 3. Feedback Endpoint Test
echo "Testing /api/feedback endpoint..."
response_feedback=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"feedback": "good", "model": "model1", "content_image_filename": "'"$TEST_CONTENT_IMAGE_NAME"'", "output_image_filename": "stylized_output.jpg", "timestamp": "2023-01-01T12:00:00Z"}' \
  "$ROUTING_SERVICE_URL/api/feedback")

if echo "$response_feedback" | grep -q "ok"; then
  echo "/api/feedback test PASSED"
else
  echo "/api/feedback test FAILED. Response: $response_feedback"
  exit 1
fi

# # 4. Fine-tuning Endpoint Test (Initiation)
# echo "Testing /finetune endpoint..."
# # Ensure at least one content image exists for the fine-tuning service to find
# cp "$TEST_CONTENT_IMAGE_PATH" "$INPUT_IMAGES_DIR/content_for_finetune.jpg"

# response_finetune=$(curl -s -X POST \
#   -H "Content-Type: application/json" \
#   -d '{"model_name": "model1", "epochs": 1, "batch_size": 1, "image_size": 256, "lr": 0.001, "style_weight": 10.0, "content_weight": 1.0}' \
#   "$FINE_TUNING_SERVICE_URL/finetune")

# if echo "$response_finetune" | grep -q "Fine-tuning started"; then
#   echo "/finetune test PASSED"
# else
#   echo "/finetune test FAILED. Response: $response_finetune"
#   exit 1
# fi

echo "All tests passed successfully!"

# --- Cleanup is handled by the trap ---
exit 0