# stac_pipeline.py
import os
import json
import requests
import folium
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# --- Endpoints ---
BASE_URL_STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
BASE_URL_OSM = "https://nominatim.openstreetmap.org"

# --- LLM factory ---
def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """
    Uses env vars:
      OPENAI_API_KEY
      OPENAI_API_BASE
    """
    # Streamlit's sidebar will set these env vars before calling this.
    try:
        return ChatOpenAI(model=model, temperature=temperature, timeout = 20, max_retries=1)
    except TypeError: 
        return ChatOpenAI(model=model, temperature=temperature)

# --- STAC collections ---
def get_stac_collections() -> List[dict]:
    r = requests.get(f"{BASE_URL_STAC}/collections", timeout=30)
    r.raise_for_status()
    cols = r.json()["collections"]
    # Slim them down so prompt stays short
    return [
        {
            "id": c.get("id"),
            "description": c.get("description"),
            "keywords": c.get("keywords"),
            "extent": c.get("extent"),
        }
        for c in cols
    ]

# --- Structured output schema ---
class StacSearchParams(BaseModel):
    location: list = Field(description="Geocodable strings for Nominatim")
    datetime_range: Optional[str] = Field(
        description="YYYY-MM-DD/YYYY-MM-DD or omitted"
    )
    collectionid: str = Field(description="Chosen collection id")

# --- LLM: extract search params from NL ---
def extract_search_params(query: str, collections: List[dict], llm: ChatOpenAI) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=StacSearchParams)
    prompt = PromptTemplate(
        template=(
            "You translate user questions into STAC API parameters.\n"
            "Question: {query}\n\n"
            "Choose the best fitting collection id from:\n{collections}\n\n"
            "If no time span is given in the question, leave it blank.\n"
            "Extract exactly: {format_instructions}"
        ),
        input_variables=["query"],
        partial_variables={
            "collections": json.dumps(collections, indent=2),
            "format_instructions": parser.get_format_instructions(),
        },
    )
    result = (prompt | llm | parser).invoke({"query": query})
    return {
        "location": result.location,
        "datetime_range": result.datetime_range or None,
        "collectionid": [result.collectionid],
    }

# --- Geocode via OSM Nominatim ---
def geocode_first_bbox(location: List[str]) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Returns bbox as [min_lon, min_lat, max_lon, max_lat] for the FIRST location string.
    """
    if not location:
        return None, None
    params = {"q": location[0], "format": "json", "limit": 1}
    headers = {"User-Agent": "stac-chat-app/1.0 (lucie.kluwe@mailbox.tu-dresden.de)"}  # set your email
    r = requests.get(f"{BASE_URL_OSM}/search", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None, None
    info = data[0]
    # Nominatim boundingbox: [south, north, west, east]
    bbox = [
        float(info["boundingbox"][2]),  # min_lon (west)
        float(info["boundingbox"][0]),  # min_lat (south)
        float(info["boundingbox"][3]),  # max_lon (east)
        float(info["boundingbox"][1]),  # max_lat (north)
    ]
    return bbox, info.get("display_name")

# --- Folium map for bbox ---
def folium_bbox_map(bbox: List[float], tooltip: Optional[str]):
    min_lon, min_lat, max_lon, max_lat = bbox
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    m = folium.Map(location=center, zoom_control=True)
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        tooltip=tooltip or "Selected area",
        fill=True,
        fill_opacity=0.2,
    ).add_to(m)
    folium.Marker(center, tooltip="Center").add_to(m)
    return m

# --- STAC /search ---
def search_stac(collection_ids: List[str], bbox=None, datetime_range: Optional[str] = None, limit: int = 5):
    url = f"{BASE_URL_STAC}/search"
    payload: Dict[str, Any] = {"collections": collection_ids, "limit": limit}
    if bbox:
        payload["bbox"] = bbox
    if datetime_range:
        payload["datetime"] = datetime_range
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    summaries = [
        {
            "id": f.get("id"),
            "datetime": f.get("properties", {}).get("datetime"),
            "collection": f.get("collection"),
            "bbox": f.get("bbox"),
        }
        for f in feats
    ]
    return summaries, feats

# --- LLM summary ---
def summarize_results(query: str, items: Any, llm: ChatOpenAI) -> str:
    msg = (
        f'These are results from a STAC API request. For the request: "{query}", '
        f"evaluate the items and recommend one. Be concise (<= 6 sentences).\n"
        f"Items: {items}"
    )
    return llm.invoke(msg).content
