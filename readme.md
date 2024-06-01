Text2Image Search System
Objective

This project aims to develop a system capable of searching for similar images based on textual queries using machine
learning.
Dataset

The Advertisement Image Dataset was used for this project. It was downloaded from the following links:

    Subfolder 0: https://storage.googleapis.com/ads-dataset/subfolder-0.zip
    Subfolder 1: https://storage.googleapis.com/ads-dataset/subfolder-1.zip

Data Exploration

A summarized report of the dataset contents was provided, along with visualizations of sample images.
Implementation

The system was implemented using FastAPI with in-memory Qdrant for vector search. The service retrieves relevant images
based on textual input.
Example Queries

    "red car"
    "black cat"

Query Evaluation

Examples of successful and unsuccessful queries were provided, along with potential methods for quantitative evaluation.
Running the Project

    Clone the Repository:

    bash

    git clone https://github.com/Mohamed-Ibrahim-124/Text-to-Image-Search.git
    cd text2image-search

Install Dependencies:

    bash
    pip install -r requirements.txt

Run the FastAPI Service:

bash

    uvicorn app.main:app --reload

    Test the Service:

    Visit http://127.0.0.1:8000/docs and use the /search/ endpoint with text queries.

Challenges and Improvements

    Challenges: Handling large datasets efficiently, ensuring accurate text-to-image matching.
    Improvements: Enhancing the model with more robust image and text embeddings, using a larger and more diverse dataset for better generalization.

Contact

For any questions, please contact:

    Name: Mohamed Ibrahim
    Email: mohamedibrahim_124@outlook.com

