import logging

import torch
from PIL import Image
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams, Distance
from torch.utils.data import DataLoader, Dataset

from app.config import device, collection_name, processor, model, qdrant_client


class ImageDataset(Dataset):
    """
    Dataset class for loading and preprocessing images.
    """

    def __init__(self, image_paths):
        """
        Initialize the ImageDataset class.

        Parameters:
        - image_paths (List[str]): List of paths to image files.
        """
        self.image_paths = image_paths

    def __len__(self):
        """
        Get the number of images in the dataset.

        Returns:
        - int: Number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image and its associated data at the given index.

        Parameters:
        - idx (int): Index of the image to retrieve.

        Returns:
        - Tuple[torch.Tensor, str]: Pixel values of the image and its path.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", do_rescale=False)
        return inputs["pixel_values"], self.image_paths[idx]


def create_collection_if_not_exists(collection_name):
    """
    Create a Qdrant collection if it does not exist.

    Parameters:
    - collection_name (str): Name of the collection.
    """
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        logging.info(f"Collection {collection_name} already exists.")
    except ValueError as e:
        if "not found" in str(e):
            logging.info(
                f"Collection {collection_name} not found. Creating a new collection."
            )
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            logging.info(f"Collection {collection_name} created.")
        else:
            raise e


def index_images(image_paths):
    """
    Index images in the Qdrant collection.

    Parameters:
    - image_paths (List[str]): List of paths to image files.
    """
    create_collection_if_not_exists(collection_name)

    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=False
    )

    points = []
    for inputs, paths in dataloader:
        inputs = inputs.squeeze(1).to(device, non_blocking=True)
    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=inputs).cpu().numpy()
        for i, path in enumerate(paths):
            point = PointStruct(id=i, vector=outputs[i].tolist())
            points.append(point)

    logging.info(f"Upserting {len(points)} points into Qdrant.")
    qdrant_client.upsert(collection_name=collection_name, points=points)
    torch.cuda.empty_cache()  # Clear GPU cache after upserting
    logging.info("Upsert completed.")


def search_images(query):
    """
    Search for images based on a text query.

    Parameters:
    - query (str): Text query for image search.

    Returns:
    - List[str]: List of image paths matching the query.
    """
    inputs = processor(text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).cpu().numpy()

    results = qdrant_client.search(
        collection_name=collection_name, query_vector=text_features[0], limit=5
    )
    return results
