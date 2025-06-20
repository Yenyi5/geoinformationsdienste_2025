import requests
import json
import folium
from langchain_openai.chat_models.base import BaseChatOpenAI
import os

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

#Define API key, endpoint, llm, URL for STAC data collection
API_Key = "7f966d739f12900214b52741e3f80ff2" 
API_Endpoint = "https://chat-ai.academiccloud.de/v1"
#Model =  "deepseek-r1" 
Model =  "meta-llama-3.1-8b-instruct"
BASE_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

# Predefine structure of desired LLM output
class StacSearchParams(BaseModel):
    bbox: list = Field(description="The area's bounding box in the format [min_lon, min_lat, max_lon, max_lat] with the coordiantes as integers")
    datetime_range: str = Field(description="The time span in the format YYYY-MM-DD/YYYY-MM-DD")

# Calls the LLM with a prompt and return a raw text output 
def call_llm(query):
    os.environ["OPENAI_API_KEY"] = API_Key
    os.environ["OPENAI_API_BASE"] = API_Endpoint
    llm = BaseChatOpenAI(
        model=Model,
        temperature=0,
        max_tokens=1024
    )
    parser = PydanticOutputParser(pydantic_object=StacSearchParams)
    
    prompt = PromptTemplate(
        template=(
            "You are a system that translates user questions in natural language into STAC API parameters."
            "Given the question: {query}, extract the following: {format_instructions} "
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    response = chain.invoke({"query": query})
    return response.model_dump_json()

#Access our STAC data collection
def search_stac(collection_id, bbox=None, datetime_range=None):
    url = f"{BASE_URL}/search" #STAC search endpoint 
    payload = {
        "collections": [collection_id],
        "limit": 10
    }
    #Parameters (can be adjusted)
    if bbox:
        payload["bbox"] = bbox
    if datetime_range:
        payload["datetime"] = datetime_range

    headers = {"Content-Type": "application/json"}
    #Makes request to the STAC API
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    items = response.json().get("features", [])
    
    for item in items:
        print(f"ID: {item['id']}, Date: {item['properties']['datetime']}")
    
    return items

# Create leaflet view of bounding box
def map_bbox(bbox=None):
    ''' Draws the generated bbox into a map interface.'''
    if bbox is None:
        print("No bbox provided.")
        return None
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

    # Display map in notebook or save to HTML
    m.save("bbox_map.html")
    print("Map saved to bbox_map.html. To view in live tab in VS Code, install the extension Live Preview, open map_bbox.html, press ctrl+shift+p and run Live Preview: Show Preview (Internal Browser)")
    return m


def main():
    #Step 1: Natural Language Query form user 
    user_question = "Find Sentinel-2 MAJA data over Berlin."

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