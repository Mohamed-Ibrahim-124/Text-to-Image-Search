import requests

# URL of the FastAPI service


def test_search(base_url, query):
    response = requests.get(f"{base_url}/search/", params={"query": query})
    if response.status_code == 200:
        print(f"Query: {query}\nResults: {response.json()}\n")
    else:
        print(f"Query: {query}\nFailed with status code: {response.status_code}\n")


if __name__ == "__main__":
    url = "http://localhost:8000"

    # Sample queries to test
    queries = [
        "black cat",
        "red car",
        "sunset",
        "mountain",
        "flower",
        "advertisement",
        "girl",
    ]

    for query in queries:
        test_search(url, query)
