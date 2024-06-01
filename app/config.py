import os

import torch
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel


def setup_environment(
    image_directory="images",
    collection_name="images",
    model_name="openai/clip-vit-base-patch32",
):
    """
    Set up the environment for working with images.

    Args:
        image_directory (str): The directory containing the image dataset.
        collection_name (str): The name of the Qdrant collection for indexing images.
        model_name (str): The name of the CLIP model to use.

    Returns:
        Tuple[List[str], torch.device, QdrantClient, CLIPProcessor, CLIPModel]: A tuple containing image paths,
        device, Qdrant client, CLIP processor, and CLIP model.
    """
    # Define the path to your image dataset
    image_paths = [
        os.path.join(image_directory, filename)
        for filename in os.listdir(image_directory)
        if filename.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Qdrant configuration
    qdrant_client = QdrantClient(":memory:")
    collection_name = "images"
    # CLIP model and processor
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    return image_paths, device, qdrant_client, collection_name, processor, model


# Usage
image_paths, device, qdrant_client, collection_name, processor, model = (
    setup_environment()
)
