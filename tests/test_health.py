import requests

BASE = "http://localhost"
PORTS = {
    "routing": 8000,
    "model1": 8001,
    "model2": 8002,
    "model3": 8003,
    "model4": 8004,
    "finetune": 8005,
}

def test_routing_health():
    # Routing endpoint responds at GET /
    r = requests.get(f"{BASE}:{PORTS['routing']}/")
    assert r.status_code == 200

import pytest

@pytest.mark.parametrize("svc,endpoint", [
    ("model1", "/"),
    ("model2", "/"),
    ("model3", "/"),
    ("model4", "/"),
])
def test_inference_health(svc, endpoint):
    # Inference modelX should respond at GET /
    url = f"{BASE}:{PORTS[svc]}{endpoint}"
    r = requests.get(url)
    assert r.status_code == 200

def test_finetune_health():
    # Fine-tune service exposes GET /health
    r = requests.get(f"{BASE}:{PORTS['finetune']}/health")
    assert r.status_code == 200
