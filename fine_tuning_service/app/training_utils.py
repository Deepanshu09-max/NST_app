import torch
import torch.nn as nn
from .model import TransformerNet, VGG16, train_transform, style_transform, gram_matrix
import logging

logger = logging.getLogger(__name__)

def train_style_transfer(
    content_images,
    style_image,
    model_instance=None, # Existing model instance for fine-tuning
    epochs=2,
    batch_size=2,
    image_size=256,
    lr=1e-3,
    style_weight=1e5,
    content_weight=1e0,
    device=None,
    progress_callback=None # Optional progress callback
):
    """
    Trains or fine-tunes a style transfer model.
    If model_instance is provided, it fine-tunes that model. Otherwise, it trains a new one.
    Returns the trained/fine-tuned model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare content images
    # content_images are expected to be a list of PIL Images
    content_tensors = [train_transform(image_size)(img) for img in content_images]
    if not content_tensors:
        logger.error("No content images provided for training.")
        raise ValueError("Content images list cannot be empty.")
    content_dataset = torch.stack(content_tensors)

    # Prepare style image
    # style_image is expected to be a PIL Image
    style_tensor = style_transform(image_size)(style_image).unsqueeze(0).to(device)

    if model_instance:
        transformer = model_instance.to(device)
        logger.info("Fine-tuning existing model.")
    else:
        transformer = TransformerNet().to(device)
        logger.info("Training new model.")

    optimizer = torch.optim.Adam(transformer.parameters(), lr)
    mse_loss = nn.MSELoss()
    vgg = VGG16(requires_grad=False).to(device)

    # Compute style features once
    with torch.no_grad():
        style_features_vgg = vgg(style_tensor)
        gram_style = [gram_matrix(y) for y in style_features_vgg]

    transformer.train() # Set model to training mode

    total_batches = (len(content_dataset) + batch_size - 1) // batch_size
    logger.info(f"Starting training for {epochs} epochs, {total_batches} batches per epoch.")

    for epoch in range(epochs):
        epoch_content_loss = 0.0
        epoch_style_loss = 0.0
        epoch_total_loss = 0.0
        num_batches_processed = 0

        for i in range(0, len(content_dataset), batch_size):
            batch_idx = i // batch_size + 1
            batch_content = content_dataset[i:i+batch_size].to(device)

            optimizer.zero_grad()

            output_stylized = transformer(batch_content)
            # Clamp values to be between 0 and 1, as images are normalized this way
            output_stylized = torch.clamp(output_stylized, 0, 1)


            features_output = vgg(output_stylized)
            with torch.no_grad(): # Content features should not require gradients for vgg
                features_content = vgg(batch_content)

            # Content Loss
            content_loss = content_weight * mse_loss(features_output.relu2_2, features_content.relu2_2)

            # Style Loss
            style_loss = 0.0
            for ft_y, gm_s in zip(features_output, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s.expand_as(gm_y))
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            epoch_content_loss += content_loss.item()
            epoch_style_loss += style_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches_processed +=1

            if progress_callback:
                progress_callback(epoch + 1, epochs, batch_idx, total_batches)
            
            logger.debug(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{total_batches} - Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

        avg_content_loss = epoch_content_loss / num_batches_processed
        avg_style_loss = epoch_style_loss / num_batches_processed
        avg_total_loss = epoch_total_loss / num_batches_processed
        logger.info(f"Epoch {epoch+1} Summary - Avg Content Loss: {avg_content_loss:.4f}, Avg Style Loss: {avg_style_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}")

    logger.info("Training/Fine-tuning finished.")
    return transformer
