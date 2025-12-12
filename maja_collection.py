import requests
import json
import folium
from langchain_openai.chat_models.base import BaseChatOpenAI
import os
from datetime import datetime
import random
from shapely.geometry import box, shape

from typing import TypedDict, List, Dict, Any, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

#!pip install -qU langsmith

#Define API key, endpoint, llm
# academic cloud
API_Key = os.getenv("ACADEMIC_API_KEY")
API_Endpoint = os.getenv("ACADEMIC_API_ENDPOINT")
#Model =  "deepseek-r1" 
Model =  "llama-3.3-70b-instruct"

# openAI:
API_Key = os.getenv("OPENAI_API_KEY")
API_Endpoint = os.getenv("OPENAI_API_ENDPOINT")
Model = "gpt-4o-mini"

os.environ["OPENAI_API_KEY"] = API_Key
os.environ["OPENAI_API_BASE"] = API_Endpoint

# langsmith
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# API URLs
#BASE_URL_STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
BASE_URL_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BASE_URL_OSM = "https://nominatim.openstreetmap.org"



class LocationState(TypedDict):

    # location
    location: Optional[str]

    # location_polygon
    location_polygon: Optional[str]

    # bbox
    bbox: Optional[List[int]]

    # bbox display name
    bbox_name : Optional[str]

    # datetime range
    datetime_range : Optional[str]

    # map object of bbox
    bboxmap: Optional[Any]

    # processing metadata
    messages: List[Dict[str, Any]]  # Track conversation with LLM for analysis

    # resulting Scene IDs
    scene_ids: Optional[Dict[str, str]]

    # resulting Items
    items : Optional[Any]

    # item info to evaluate
    items_eval: Optional[Any]

    # output message, after the validation of the geojson
    query: str

    # Collections in catalog
    catalogcollections: List[dict]

    # Collection ID
    collectionid : Optional[List[str]]

    # map object of bbox
    resultsmap: Optional[Any]


# model
llm = BaseChatOpenAI(model=Model, temperature=0)

# get stac collections
def get_stac_collections():
    response = requests.get(f"{BASE_URL_STAC}/collections")
    response.raise_for_status()
    collections = response.json()["collections"]
    filtered = []
    for col in collections:
        # filter to relevant info because else llm query exceeds allowed length
        filtered.append({
            "id": col.get("id"),
            "description": col.get("description"),
            "keywords": col.get("keywords"),
            "extent": col.get("extent")
        })
    return filtered


### NODES

def generate_searchparams(state: LocationState):
    ''' Generate search parameters from user input based on structured output'''
    start = datetime.now()
    print("Extracting search parameters...")
    class StacSearchParams(BaseModel):
        location: list = Field(description="The geographic location indicated in the search as a geocodable location string to be used for geocoding via Nominatim to find its coordinates")
        datetime_range: str = Field(description="The time span in the format YYYY-MM-DD/YYYY-MM-DD")
        collectionid: str = Field(description = "The collection id from the STAC catalog that best matches the user's query.")
    parser = PydanticOutputParser(pydantic_object=StacSearchParams)
    prompt = PromptTemplate(
        template=(
            "You are a system that translates user questions in natural language into STAC API parameters.\n"
            "Given the question: {query}, choose the best fitting collection id from the following options:\n{collections}\n"
            "If no time span is given in the question, do not set a temporal extent. \n"
            "Extract the following: {format_instructions}"
        ),
        input_variables=["query"],
        partial_variables={
            "collections" : json.dumps(state["catalogcollections"], indent=2),
            "format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    response = chain.invoke({"query": state["query"]})
    print(f"LLM extracted \n - location: {response.location}\n - date range: {response.datetime_range}\n - collection ID: {response.collectionid}")
    #print(response)

    # Update messages for tracking
    message = [HumanMessage(prompt.format(query=state["query"]))]
    new_message = state.get("messages", []) + [
        {"role": "user", "content": message},
        {"role": "agent", "content": response}
    ]
    
    # update the state and the message
    print(" * Parameter extraction time: ", datetime.now()- start)
    return {"location": response.location,  "datetime_range" : response.datetime_range, "collectionid" : [response.collectionid], "messages": new_message }

def getgeometry(state:LocationState):
    start = datetime.now()
    url = f"{BASE_URL_OSM}/search" # OSM Nominatim search endpoint 
    params = {
        "q": state["location"],
        "format": "json",
        "polygon_geojson": 1,
        "limit": 1
    }
    headers = {"User-Agent": "geoid-stac-client/1.0 (lucie.kluwe@mailbox.tu-dresden.de)"}
    print("Sending request to Nominatim API...")
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    results = response.json()
    print(" * Response time:", datetime.now() - start)
    if results:
        location_info = results[0]
        #print(location_info)
        bbox = [float(location_info["boundingbox"][2]), float(location_info["boundingbox"][0]),
                float(location_info["boundingbox"][3]), float(location_info["boundingbox"][1])]
        print("Extracted location: ", location_info["display_name"])
        return {"bbox": bbox, "bbox_name": location_info["display_name"], "location_polygon": location_info["geojson"]}
    else:
        return {"bbox": None, "bbox_name": None}

def show_on_map(state:LocationState):
    "Showing the bounding box on a map"

    print("Creating the map...")

    if state["bbox"]:
        bbox = state["bbox"]
    else:
        print("No bbox provided.")
        return {"bboxmap":None}

    # bbox: [min_lon, min_lat, max_lon, max_lat]
    min_lon, min_lat, max_lon, max_lat = bbox
    # Center of bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    # Create map centered on bbox
    m = folium.Map(location=[center_lat, center_lon])
    # Draw rectangle for bbox
    folium.Rectangle(
        bounds=[
            [min_lat, min_lon],
            [max_lat, max_lon]
        ],
        color='cornflowerblue',
        tooltip=state["bbox_name"],
        fill=True,
        fill_opacity=0.2
    ).add_to(m)

    # add area polygon
    folium.GeoJson(
        state["location_polygon"],
        color='red',
        tooltip=state["bbox_name"],
        fill=True,
        fill_opacity=0.2
    )

    # zoom map to rectangle
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    m.save("bbox_map.html")
    print(" * Map saved to bbox_map.html. To view in live tab in VS Code, install the extension Live Preview, open map_bbox.html, press ctrl+shift+p and run Live Preview: Show Preview (Internal Browser)")
    #display(m)
    return {"bboxmap":m}

def print_dict(items):
    for item in items[0:1]:
        for key, value in item.items():
            print(f"{key}: {value}\n")

#Access our STAC data collection
def search_stac(state: LocationState):
    start = datetime.now()
    
    url = f"{BASE_URL_STAC}/search" #STAC search endpoint 
    payload = {
        "collections": state["collectionid"]#,
        #"limit": 20
    }
    #Parameters (can be adjusted)
    if state["bbox"]:
        payload["bbox"] = state["bbox"]
    if state["datetime_range"]:
        payload["datetime"] = state["datetime_range"]

    headers = {"Content-Type": "application/json"}
    # Makes request to the STAC API
    print("Sending request to STAC API...\n >", payload)
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    items = response.json().get("features", [])
    print_dict(items)
    scene_ids = {item['id']:{item['properties']['datetime']} for item in items}
    print(f"Found {len(items)} items:")
    for item in items:
        print(f"ID: {item['id']}, Date: {item['properties']['datetime']}")
    print(" * Response time:", datetime.now()- start)
    return {"scene_ids":scene_ids, "items":items}


def calculate_overlap(bbox, geometry):
    bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
    geometry_shape = shape(geometry)

    intersection = bbox_geom.intersection(geometry_shape)
    overlap_percentage = (intersection.area / bbox_geom.area) * 100
    return overlap_percentage

def get_evaluation_info(state: LocationState):
    # items is a list of dictionaries
    items = state["items"]
    bbox = state["bbox"]

    # Initialize the new list of dictionaries
    items_eval = []

    # Iterate over the items
    for item in items:
        # Extract id and geometry
        item_id = item["id"]
        date = datetime.fromisoformat(item['properties']['datetime']).strftime("%Y-%m-%d %H:%M")
        geometry = item["geometry"]
        overlap_percentage = calculate_overlap(bbox, geometry)
        links = item["links"]
        url = next((link['href'] for link in links if link['rel'] == 'self'), None)
        properties = item["properties"]
        assets = item["assets"]
        assets_new = {}
        for k,v in assets.items():
            title = v["title"]
            assets_new[k] = title

        # Add the new dictionary to items_eval
        items_eval.append({
            "id": item_id,
            "date": date,
            "geometry": geometry,
            "url": url,
            "overlap_percentage": {
                "value": overlap_percentage,
                "description": "Overlap of item polygon with requested bounding box"
            },
            "assets": assets_new,
            "properties": properties
        })
    
    # sort by overlap percentage
    items_eval = sorted(items_eval, key=lambda x: x["overlap_percentage"]["value"], reverse=True)
    #print_dict(items_eval)

    return {"items_eval": items_eval }
                                

def summarise_result(state:LocationState):
    start = datetime.now()
    items_eval = state["items_eval"][:100]
    message = f""" These are properties of the 100 results with the most overlap with the requested bbox from a request sent to the Stac API. Evaluate the contents of the items regarding the initial request {state["query"]}. Recommend one of the items to use. These are the items that were found: {items_eval}"""
    # message = [HumanMessage(content=message)] ?
    response = llm.invoke(message)
    print("Result Summary: ", response.content)
    new_message = state.get("messages", []) + [
        {"role": "user", "content": message},
        {"role": "agent", "content": response.content}
    ]
    print(" * Summary time:", datetime.now() - start)
    return {"messages": new_message }

def show_results_on_map(state:LocationState):
    "Showing the bounding box on a map"

    print("Showing 10 result items with largest overlap with requested bounding box on the map...")

    if state["items_eval"]:
        items = state["items_eval"][:10]
    else:
        print("No result items provided.")
        return {"resultsmap": None}
    
    if state["bboxmap"]:
        m = state["bboxmap"]
    else:
        print("No bbox map provided.")
        return {"resultsmap": None}


    scene_info = {item['id']: {'date': item['date'], 'geom': item['geometry'], 'url': item['url'], 'overlap': item['overlap_percentage']['value']} for item in items}


    # Generate random colors for each polygon
    def random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    for scene_id, infos  in scene_info.items():
        color = random_color()

        geom = infos["geom"]
        url = infos["url"]
        date = infos["date"]
        overlap = round(infos["overlap"], 2)

        popup_content = f"""
        <b>Scene ID:</b> {scene_id}<br>
        <b>Date:</b> {date}<br>
        <b>Overlap Percentage:</b> {overlap}%<br>
        <b>URL:</b> <a href="{url}" target="_blank">View item in browser</a><br>
    """

        folium.GeoJson(
            geom,
            name=scene_id,
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": color,
                "weight": 2,
                "fillOpacity": 0.2
            },
            popup=folium.Popup(popup_content, max_width=500)  # show scene_id on hover
        ).add_to(m)

    # Fit map to all geometries
    bounds = []
    for infos in scene_info.values():
        geom = infos["geom"]
        coords = geom["coordinates"][0]  # assuming Polygon
        # swap to [lat, lon] for Leaflet
        bounds.extend([[lat, lon] for lon, lat in coords])

    m.fit_bounds(bounds)
    folium.LayerControl().add_to(m)

    m.save("results_map.html")
    print(" * Map saved to results_map.html")
    #display(m)
    return {"resultsmap":m}

# Create the graph
graph = StateGraph(LocationState)

# Add nodes
graph.add_node("extract_search", generate_searchparams)
graph.add_node("get_geom", getgeometry)
graph.add_node("show_map", show_on_map)
graph.add_node("search_stac", search_stac)
graph.add_node("create_evaluation_dict", get_evaluation_info)
graph.add_node("summarise_results", summarise_result)
graph.add_node("show_results_map", show_results_on_map)

# Add Edges
graph.add_edge(START, "extract_search")
graph.add_edge("extract_search", "get_geom")
graph.add_edge("get_geom", "show_map")
graph.add_edge("show_map", "search_stac")
graph.add_edge("search_stac", "create_evaluation_dict")
graph.add_edge("create_evaluation_dict", "summarise_results")
graph.add_edge("summarise_results", "show_results_map")
graph.add_edge("show_results_map", END)

# Compile the graph
compiled_graph = graph.compile()

# draw architecture graph
try:
    with open("graph_output.png", "wb") as f:
        f.write(compiled_graph.get_graph().draw_mermaid_png())
    print("Graph displayed successfully.")
except Exception as e:
    print(f"Fehler beim Speichern des Graphen: {e}")

# Initiate
start_all = datetime.now()

print("\nProcessing the user input...")
result = compiled_graph.invoke({
    "location": None,
    "location_polygon" : None,
    "bbox": None,
    "datetime_range": None,
    "bboxmap": None,
    "messages": [],
    "scene_ids": None,
    "items": None,
    "items_eval": None,
    "query": "Finde Sentinel-2 Daten für Lüttich in 2024",
    "catalogcollections": get_stac_collections(),
    "collectionid": None,
    "resultsmap": None

})

print("Total execution time:", datetime.now() - start_all)

