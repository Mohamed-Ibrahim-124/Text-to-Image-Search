import logging
import time

from fastapi import FastAPI

from app.config import image_paths
from app.utils import index_images, search_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """
    Startup event to index images.
    """
    start_time = time.time()
    logging.info("Indexing images...")
    index_images(image_paths)
    logging.info(f"Indexing completed in {time.time() - start_time:.2f} seconds")


@app.get(
    "/search/",
    summary="Search for images based on a text query",
    response_description="List of image paths matching the query",
)
async def search(query: str):
    """
    Search for images based on a text query.

    Args:
        query (str): The text query for image search.

    Returns:
        List[dict]: List of dictionaries representing the search results.
    """
    logging.info(f"Received search query: {query}")
    start_time = time.time()
    results = search_images(query)
    logging.info(f"Search results for '{query}': {results}")
    logging.info(f"Query completed in {time.time() - start_time:.2f} seconds")
    return results
