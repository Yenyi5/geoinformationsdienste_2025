# stac_pipeline.py
import os
import json
import requests
import folium
from typing import List, Dict, Any, Optional, Tuple
import random

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from shapely.geometry import box, shape
from datetime import datetime


# --- Endpoints ---
BASE_URL_STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
#BASE_URL_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BASE_URL_OSM = "https://nominatim.openstreetmap.org"

# --- LLM factory ---
def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """
    Uses env vars:
      OPENAI_API_KEY
      OPENAI_API_BASE
    """
    # Streamlit's sidebar will set these env vars before calling this.
    """ try:
        return ChatOpenAI(model=model, temperature=temperature, timeout = , max_retries=1)
    except TypeError: """ 
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

# --- LLM: extract search params  ---
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
def geocode_first_bbox(location: List[str]) -> Tuple[Optional[List[float]], Optional[str], Optional[dict]]:
    """
    Returns bbox as [min_lon, min_lat, max_lon, max_lat] for the FIRST location string.
    """
    if not location:
        return None, None, None
    params = {"q": location[0], "format": "json", "polygon_geojson": 1, "limit": 1}
    headers = {"User-Agent": "stac-chat-app/1.0 (lucie.kluwe@mailbox.tu-dresden.de)"}  # set your email
    r = requests.get(f"{BASE_URL_OSM}/search", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None, None, None
    info = data[0]
    # Nominatim boundingbox: [south, north, west, east]
    bbox = [
        float(info["boundingbox"][2]),  # min_lon (west)
        float(info["boundingbox"][0]),  # min_lat (south)
        float(info["boundingbox"][3]),  # max_lon (east)
        float(info["boundingbox"][1]),  # max_lat (north)
    ]
    polygon = info.get("geojson") 
    return bbox, info.get("display_name"), polygon

# --- Folium map for bbox ---
def folium_bbox_map(bbox: List[float], tooltip: Optional[str], location_polygon: Optional[dict] = None):
    min_lon, min_lat, max_lon, max_lat = bbox
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    m = folium.Map(location=center, zoom_control=True)
    
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        tooltip=tooltip or "Selected area",
        fill=True,
        fill_opacity=0.2,
    ).add_to(m)

    # render polygon on map
    if location_polygon:
        folium.GeoJson(
            location_polygon,
            name="location_polygon",
            tooltip=tooltip or "Area polygon",
            style_function=lambda f: {
                "color": "purple",
                "weight": 2, 
                "fillColor": "red", 
                "fillOpacity": 0.15,
            },
        ).add_to(m)
 
    folium.Marker(center, tooltip="Center").add_to(m)
    return m

# --- STAC /search ---
def search_stac(collection_ids: List[str], bbox=None, datetime_range: Optional[str] = None, limit: int = 200):
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

def calculate_overlap(bbox: List[float], geometry: dict) -> float:
    bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
    geom_shape = shape(geometry)
    inter = bbox_geom.intersection(geom_shape)
    if bbox_geom.area == 0:
        return 0.0
    return (inter.area / bbox_geom.area) * 100.0


def build_items_eval(items: List[dict], bbox: List[float]) -> List[dict]:
    items_eval = []
    for item in items:
        item_id = item.get("id")
        dt_iso = item.get("properties", {}).get("datetime")
        date_fmt = None
        if dt_iso:
            try:
                date_fmt = datetime.fromisoformat(dt_iso.replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_fmt = dt_iso

        geometry = item.get("geometry")
        overlap = calculate_overlap(bbox, geometry) if (bbox and geometry) else 0.0

        url = None
        for link in item.get("links", []):
            if link.get("rel") == "self":
                url = link.get("href")
                break

        assets_new = {}
        for k, v in (item.get("assets") or {}).items():
            assets_new[k] = v.get("title")

        items_eval.append({
            "id": item_id,
            "date": date_fmt,
            "geometry": geometry,
            "url": url,
            "overlap_percentage": {
                "value": overlap,
                "description": "Overlap of item polygon with requested bounding box",
            },
            "assets": assets_new,
            "properties": item.get("properties"),
        })

    items_eval.sort(key=lambda x: x["overlap_percentage"]["value"], reverse=True)
    return items_eval


def add_top_items_to_map(m: folium.Map, items_eval: List[dict], top_k: int = 10) -> folium.Map:
    def random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    for it in (items_eval or [])[:top_k]:
        geom = it.get("geometry")
        if not geom:
            continue
        color = random_color()
        popup = folium.Popup(
            f"<b>ID:</b> {it.get('id')}<br>"
            f"<b>Date:</b> {it.get('date')}<br>"
            f"<b>Overlap %:</b> {round(it.get('overlap_percentage', {}).get('value', 0.0), 2)}<br>"
            f"<b>URL:</b> <a href='{it.get('url')}' target='_blank'>open</a>",
            max_width=450,
        )
        folium.GeoJson(
            geom,
            name=it.get("id"),
            style_function=lambda f, c=color: {"color": c, "fillColor": c, "weight": 2, "fillOpacity": 0.2},
            popup=popup,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# --- LLM summary ---
def summarize_results(query: str, items_eval: Any, llm: ChatOpenAI, top_n: int = 100) -> str:
    top_items = items_eval[:top_n] if isinstance(items_eval, list) else items_eval
    msg = (
        f'These are properties of the top {top_n} results with the most of item polygon with requested bounding box '
        f'from a STAC API request.\n'
        f'For the request: "{query}", evaluate the items and recommend one. '
        f'Be concise (<= 6 sentences).\n'
        f"Items: {top_items}"
    )
    return llm.invoke(msg).content
