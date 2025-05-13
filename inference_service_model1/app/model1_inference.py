import torch
from PIL import Image
import os

from .model import TransformerNet, deprocess, test_transform

def load_image(image_path, image_size=None):
    image = Image.open(image_path).convert("RGB")
    transform = test_transform(image_size=image_size)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def save_image(img_tensor, output_path):
    image = deprocess(img_tensor.clone().detach())
    image = Image.fromarray(image)
    image.save(output_path)

def run_inference(model_path, content_image_path, output_path, image_size=512):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    content_image = load_image(content_image_path, image_size=image_size).to(device)

    with torch.no_grad():
        output = model(content_image)

    save_image(output.cpu(), output_path)
