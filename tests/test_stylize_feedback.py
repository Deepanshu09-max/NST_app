import base64
import requests
import os
import pytest

# Base URL for routing service
BASE = "http://localhost:8000"

# Path to this test file’s directory, so we can locate fixtures
HERE = os.path.abspath(os.path.dirname(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")
TEST_IMG = os.path.join(FIXTURES_DIR, "dummy.jpg")

@pytest.fixture(autouse=True)
def verify_fixture_exists():
    if not os.path.isfile(TEST_IMG):
        pytest.skip(f"Fixture not found: {TEST_IMG}")

def test_stylize_endpoint():
    """
    Send a dummy JPEG to POST /api/stylize with model=model1.
    Expect a JSON response containing 'stylized_image_base64',
    and that the returned value decodes to valid JPEG bytes.
    """
    with open(TEST_IMG, "rb") as f:
        files = {
            "content_image": ("dummy.jpg", f, "image/jpeg"),
            "model": (None, "model1")
        }
        r = requests.post(f"{BASE}/api/stylize", files=files)
    assert r.status_code == 200, f"Stylize returned {r.status_code}: {r.text}"

    data = r.json()
    assert "stylized_image_base64" in data, f"Key missing in response: {data}"
    img_b64 = data["stylized_image_base64"]

    # Verify base64 decode and JPEG magic bytes (0xFFD8FFE0 or 0xFFD8FF)
    decoded = base64.b64decode(img_b64)
    assert decoded[:2] == b"\xff\xd8", "Returned data is not a JPEG"

def test_feedback_endpoint():
    """
    Send a dummy JSON to POST /api/feedback.
    Expect a 200 response with JSON {'status': 'ok'}.
    """
    payload = {
        "feedback": "good",
        "model": "model1",
        "content_image_filename": "dummy.jpg",
        "output_image_filename": "stylized_output.jpg",
        "timestamp": "2025-01-01T12:00:00Z"
    }
    r = requests.post(f"{BASE}/api/feedback", json=payload)
    assert r.status_code == 200, f"Feedback returned {r.status_code}: {r.text}"

    data = r.json()
    assert data.get("status") == "ok", f"Unexpected feedback response: {data}"
