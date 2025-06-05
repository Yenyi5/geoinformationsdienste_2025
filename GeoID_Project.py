import requests

BASE_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

def list_collections():
    response = requests.get(f"{BASE_URL}/collections")
    response.raise_for_status()
    collections = response.json()
    for collection in collections.get("collections", []):
        print(f"- {collection['id']}: {collection['title']}")
        
list_collections()