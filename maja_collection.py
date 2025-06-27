import requests
import json
import folium
from langchain_openai.chat_models.base import BaseChatOpenAI
import os

from typing import TypedDict, List, Dict, Any, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

#!pip install -qU langsmith

#Define API key, endpoint, llm, URL for STAC data collection
API_Key = "7f966d739f12900214b52741e3f80ff2" 
API_Endpoint = "https://chat-ai.academiccloud.de/v1"
#Model =  "deepseek-r1" 
Model =  "llama-3.3-70b-instruct"
BASE_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

os.environ["OPENAI_API_KEY"] = API_Key
os.environ["OPENAI_API_BASE"] = API_Endpoint

# langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"


class LocationState(TypedDict):

    # location
    location: Optional[str]

    # bbox
    bbox: Optional[List[int]]

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

    # output message, after the validation of the geojson
    query: str

    # Collection ID
    collectionid : str


# model
llm = BaseChatOpenAI(model=Model, temperature=0)

### NODES

def generate_searchparams(state: LocationState):
    ''' Generate search parameters from user input based on structured output'''
    class StacSearchParams(BaseModel):
        location: list = Field(description="The geographic location indicated in the search as a geocodable location string to be used for geocoding via Nominatim to find its coordinates")
        datetime_range: str = Field(description="The time span in the format YYYY-MM-DD/YYYY-MM-DD")
    parser = PydanticOutputParser(pydantic_object=StacSearchParams)
    prompt = PromptTemplate(
        template=(
            "You are a system that translates user questions in natural language into STAC API parameters."
            "Given the question: {query}, extract the following: {format_instructions} "
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    message = [HumanMessage(content=prompt)]
    chain = message | llm | parser
    response = chain.invoke({"query": state["query"]})

    print(response)

    # Update messages for tracking
    new_message = state.get("messages", []) + [
        {"role": "user", "content": message},
        {"role": "agent", "content": response.content}
    ]

    # update the state and the message
    return {"location": response.location,  "datetime_range" : response.datetime_range, "messages": new_message }

#def getgeometry(state:LocationState):
#    return {"bbox": bbox}

def show_on_map(state:LocationState):
    "Show the bounding box on a map"

    print("Creating the map...")

    if state["bbox"]:
        bbox = state["bbox"]
    else:
        "No bbox provided."

    # bbox: [min_lon, min_lat, max_lon, max_lat]
    min_lon, min_lat, max_lon, max_lat = bbox
    # Center of bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    # Create map centered on bbox
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    # Draw rectangle for bbox
    folium.Rectangle(
        bounds=[
            [min_lat, min_lon],
            [max_lat, max_lon]
        ],
        color='cornflowerblue',
        fill=True,
        fill_opacity=0.2
    ).add_to(m)

    folium.Marker(
        location=[center_lat, center_lon],
        tooltip=state["location"],
        popup=state["location"],
        icon=folium.Icon(color="cornflowerblue"),
    ).add_to(m)

    m.save("bbox_map.html")
    print("Map saved to bbox_map.html. To view in live tab in VS Code, install the extension Live Preview, open map_bbox.html, press ctrl+shift+p and run Live Preview: Show Preview (Internal Browser)")
    #display(m)
    return {"map_bbox":m}


#Access our STAC data collection
def search_stac(state: LocationState):
    url = f"{BASE_URL}/search" #STAC search endpoint 
    payload = {
        "collections": state["collectionid"],
        "limit": 10
    }
    #Parameters (can be adjusted)
    if state["bbox"]:
        payload["bbox"] = state["bbox"]
    if state["datetime_range"]:
        payload["datetime"] = state["datetime_range"]

    headers = {"Content-Type": "application/json"}
    #Makes request to the STAC API
    print("Sending request to STAC API")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    items = response.json().get("features", [])

    print(f"Found {len(items)} items.")
    
    scene_ids = {item['id']:{item['properties']['datetime']} for item in items}
    
    return {"scene_ids":scene_ids, "items":items}

def summarise_result(state:LocationState):
    items = state["items"]
    message = f""" These are the results from a request sent to the Stac API based on this request: {state["query"]}. Summarise and evaluate them. These are the results: {items}"""
    # message = [HumanMessage(content=message)] ?
    response = llm.invoke(message)
    print("Result Summary: ", response)
    new_message = state.get("messages", []) + [
        {"role": "user", "content": message},
        {"role": "agent", "content": response.content}
    ]
    return {"messages": new_message }

# Create the graph
graph = StateGraph(LocationState)

# Add nodes
graph.add_node("extract_search", generate_searchparams)
graph.add_node("get_geom", getgeometry)
graph.add_node("show_map", show_on_map)
graph.add_node("search_stac", search_stac)
graph.add_node("summarise_results", summarise_result)

# Add Edges
graph.add_edge(START, "extract_search")
graph.add_edge("extract_search", "get_geom")
graph.add_edge("get_geom", "show_map")
graph.add_edge("show_map", "search_stac")
graph.add_edge("search_stac", "summarise_results")
graph.add_edge("summarise_results", END)

# Compile the graph
compiled_graph = graph.compile()

# draw mermaid graph

# Initiate
print("\nProcessing the user input...")
result = compiled_graph.invoke({
    "location": None,
    "bbox": None,
    "datetime_range": None,
    "bboxmap": None,
    "messages": [],
    "scene_ids": None,
    "items": None,
    "query": "Find Sentinel-2 MAJA data over Cologne.",
    "collectionid": "S2_L2A_MAJA" 
})