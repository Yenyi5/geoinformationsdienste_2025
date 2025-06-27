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

    # geojson errors
    geojson_errors: Optional[Dict[str, Any]]

    # analysis and decision about the geojson
    is_valid_geojson: Optional[bool]

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
    scene_ids: Optional[List[str]]

    # resulting Items
    items : Optional[Any]

    # output message, after the validation of the geojson
    output_message: Optional[str]


# model
llm = BaseChatOpenAI(model=Model, temperature=0)

### NODES

def generate_searchparams(state: LocationState, query):
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
    response = chain.invoke({"query": query})

    print(response)

    # Update messages for tracking
    new_message = state.get("messages", []) + [
        {"role": "user", "content": message},
        {"role": "agent", "content": response.content}
    ]

    # update the state and the message
    return {"location": response.location,  "datetime_range" : response.datetime_range, "messages": new_message }

#def getgeometry():
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
def search_stac(state: LocationState, collection_id):
    url = f"{BASE_URL}/search" #STAC search endpoint 
    payload = {
        "collections": [collection_id],
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


def main():
    #Step 1: Natural Language Query form user 
    user_question = "Find Sentinel-2 MAJA data over Cologne."

    #Call llm and print response 
    llm_output = call_llm(user_question)
    print("LLM Output:", llm_output)
    
    #Step 2: parse the string as JSON
    try:
        search_params = json.loads(llm_output)
    except json.JSONDecodeError:
        print("Failed to parse LLM output.")
        return

    #Use one specific collection as an example 
    collection_id = "S2_L2A_MAJA"

    #Step 3:Perform the STAC search within the given collection
    search_stac(
        collection_id=collection_id,
        bbox=search_params.get("bbox"),
        datetime_range=search_params.get("datetime_range")
    )

    # Step 4: Display bbox in leaflet view
    map_bbox(bbox=search_params.get("bbox"))

if __name__ == "__main__":
    main()