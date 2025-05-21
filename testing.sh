#!/bin/bash
set -e

APP_DIR="."
PERSISTENT_STORAGE_DIR="$APP_DIR/persistent_storage"
INPUT_IMAGES_DIR="$PERSISTENT_STORAGE_DIR/input_images"
MODELS_DIR="$PERSISTENT_STORAGE_DIR/models"
MODEL1_DIR="$MODELS_DIR/model1"

ROUTING_SERVICE_URL="http://localhost:8000"
INFERENCE_SERVICE_MODEL1_URL="http://localhost:8001"
INFERENCE_SERVICE_MODEL2_URL="http://localhost:8002"
INFERENCE_SERVICE_MODEL3_URL="http://localhost:8003"
INFERENCE_SERVICE_MODEL4_URL="http://localhost:8004"
FINE_TUNING_SERVICE_URL="http://localhost:8005"

TEST_CONTENT_IMAGE_NAME="test_content_image.jpg"
TEST_CONTENT_IMAGE_PATH="$INPUT_IMAGES_DIR/$TEST_CONTENT_IMAGE_NAME"
DUMMY_STYLE_IMAGE_PATH="$MODEL1_DIR/style.jpg"
DUMMY_VGG_WEIGHTS_PATH="$PERSISTENT_STORAGE_DIR/vgg19.npz"

cleanup() {
  echo "Cleaning up test files…"
  rm -f "$TEST_CONTENT_IMAGE_PATH" "$DUMMY_STYLE_IMAGE_PATH" "$DUMMY_VGG_WEIGHTS_PATH"
}
trap cleanup EXIT ERR INT TERM

echo "Preparing test artifacts…"
mkdir -p "$INPUT_IMAGES_DIR" "$MODEL1_DIR"
echo "dummy image data" > "$TEST_CONTENT_IMAGE_PATH"
echo "dummy style data" > "$DUMMY_STYLE_IMAGE_PATH"
touch "$DUMMY_VGG_WEIGHTS_PATH"

echo "Running API tests…"

# HEALTH CHECKS
curl -f "$ROUTING_SERVICE_URL/"                        || { echo "Routing failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL1_URL/"                || { echo "Inference1 failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL2_URL/"                || { echo "Inference2 failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL3_URL/"                || { echo "Inference3 failed"; exit 1; }
curl -f "$INFERENCE_SERVICE_MODEL4_URL/"                || { echo "Inference4 failed"; exit 1; }
curl -f "$FINE_TUNING_SERVICE_URL/health"               || { echo "Fine-tune health failed"; exit 1; }

# /api/stylize
response_stylize=$(curl -s -X POST \
  -F "model=model1" \
  -F "content_image=@$TEST_CONTENT_IMAGE_PATH" \
  "$ROUTING_SERVICE_URL/api/stylize")
echo "$response_stylize" | grep -q "stylized_image_base64" \
  && echo "stylize PASSED" \
  || { echo "stylize FAILED: $response_stylize"; exit 1; }

# /api/feedback
response_feedback=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"feedback":"good","model":"model1","content_image_filename":"'"$TEST_CONTENT_IMAGE_NAME"'","output_image_filename":"stylized_output.jpg","timestamp":"2023-01-01T12:00:00Z"}' \
  "$ROUTING_SERVICE_URL/api/feedback")
echo "$response_feedback" | grep -q "ok" \
  && echo "feedback PASSED" \
  || { echo "feedback FAILED: $response_feedback"; exit 1; }

echo "All tests passed!"
