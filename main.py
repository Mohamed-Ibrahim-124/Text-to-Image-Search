# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
import os
from PIL import Image
from qdrant_client import QdrantClient
from collections import Counter
from qdrant_client.models import VectorParams, Distance
import numpy as np
from qdrant_client.models import PointStruct

def explore_dataset(dataset_path):
  """
  Analyzes an image-only dataset and generates a report.

  Args:
      dataset_path (str): Path to the directory containing the image dataset.

  Returns:
      dict: A dictionary containing information about the dataset,
          including number of images, format distribution, size distribution,
          dominant color analysis (optional), and presence of duplicates.
  """
  dataset_summary = {
      "num_images": 0,
      "format_distribution": Counter(),
      "size_distribution": Counter(),
      "dominant_colors": None,  # Optional: Replace with analysis result
      "num_duplicates": 0,
  }

  for main_folder_name in os.listdir(dataset_path):
    main_folder_path = os.path.join(dataset_path, main_folder_name)
    if os.path.isdir(main_folder_path):
      for sub_folder_name in os.listdir(main_folder_path):
        sub_folder_path = os.path.join(main_folder_path, sub_folder_name)
        if os.path.isdir(sub_folder_path):
          for image_name in os.listdir(sub_folder_path):
            image_path = os.path.join(sub_folder_path, image_name)
            dataset_summary["num_images"] += 1

            # Analyze image format
            _, ext = os.path.splitext(image_path)
            dataset_summary["format_distribution"][ext.lower()] += 1

            try:
              with Image.open(image_path) as image:
                # Analyze image size
                width, height = image.size
                dataset_summary["size_distribution"][(width, height)] += 1

                # Optional: Analyze dominant colors (replace with your logic)
                # ... (code to analyze and store dominant colors)
                # dataset_summary["dominant_colors"] = ...

                # Check for duplicates (compare fingerprints or hash values)
                # ... (code to compare image data and identify duplicates)
                # dataset_summary["num_duplicates"] += ...

            except (IOError, OSError) as e:
              print(f"Error opening image: {image_path}. Skipping...")

  return dataset_summary


def load_images(dataset_path, batch_size=100):
    images = []
    for main_folder_name in os.listdir(dataset_path):
        main_folder_path = os.path.join(dataset_path, main_folder_name)
        if os.path.isdir(main_folder_path):
            for sub_folder_name in os.listdir(main_folder_path):
                sub_folder_path = os.path.join(main_folder_path, sub_folder_name)
                if os.path.isdir(sub_folder_path):
                    for image_name in os.listdir(sub_folder_path):
                        image_path = os.path.join(sub_folder_path, image_name)
                        try:
                            with Image.open(image_path) as image:
                                # Preprocess the image (e.g., resize, convert to RGB)
                                processed_image = image.resize((224, 224))  # Example resize
                                processed_image = processed_image.convert('RGB')  # Example conversion
                                images.append((processed_image, main_folder_name))
                                if len(images) == batch_size:
                                    yield images
                                    images = []
                        except (IOError, OSError) as e:
                            print(f"Error opening image: {image_path}. Skipping...")
    if images:
        yield images


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_path = "Dataset"
    images = load_images(dataset_path)
    # Example usage
    dataset_summary = explore_dataset(dataset_path)
    print("Dataset Summary:", dataset_summary)

    images_dataset = load_images(dataset_path, batch_size=100000)
    for batch_images in images_dataset:
        # Process batch_images
        print(f"Processed {len(batch_images)} images")
    qdrant_client = QdrantClient(host="localhost", port=6333)
    qdrant_client.recreate_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=100, distance=Distance.COSINE),
    )


    vectors = np.random.rand(100, 100)
    qdrant_client.upsert(
        collection_name="my_collection",
        points=[
            PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={"color": "red", "rand_number": idx % 10}
            )
            for idx, vector in enumerate(vectors)
        ]
    )