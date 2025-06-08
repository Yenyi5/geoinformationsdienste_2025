import requests
import json
import folium

#Define API key, endpoint, llm, URL for STAC data collection
API_Key = "7f966d739f12900214b52741e3f80ff2" 
API_Endpoint = "https://chat-ai.academiccloud.de/v1"
Model = "meta-llama-3.1-8b-instruct"
BASE_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"

#Calls the LLM with a prompt and return a raw text output 
def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_Key}",
        "Content-Type": "application/json",  #sending in Json
    }
    payload = {
        "model": Model,
        "messages": [{"role": "user", "content": prompt}],
    }
    #Makes request to the LLM API
    response = requests.post(API_Endpoint + "/chat/completions", headers=headers, json=payload)
    response.raise_for_status() 
    return response.json()["choices"][0]["message"]["content"] 


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
    print("Map saved to bbox_map.html")
    return m


def main():
    #Step 1: Natural Language Query form user 
    user_question = "Find Sentinel-2 MAJA data over Berlin in February 2024"

    #Step 2: Prompt the LLM to extract bbox and date range only 
    prompt = f"""
You are a system that translates user questions in natural language into STAC API parameters.
Given the question: '{user_question}', extract the following as a JSON:
- "bbox": [min_lon, min_lat, max_lon, max_lat]
- "datetime_range": "YYYY-MM-DD/YYYY-MM-DD"
Return only raw JSON — no markdown, no code formatting.
"""
    #Call llm and print response 
    llm_output = call_llm(prompt)
    print("LLM Output:", llm_output)

    #Clean the output from possible formatting to only include content within the most outer curly braces
    start = llm_output.find('{')
    end = llm_output.rfind('}')
    if start != -1 and end != -1 and end > start:
        cleaned_output = llm_output[start:end+1]
    else:
        cleaned_output = llm_output.strip()
    
    #Step 3: parse the string as JSON
    try:
        search_params = json.loads(cleaned_output)
    except json.JSONDecodeError:
        print("Failed to parse LLM output.")
        return

    #Use one specific collection as an example 
    collection_id = "S2_L2A_MAJA"

    #Step 4:Perform the STAC search within the given collection
    search_stac(
        collection_id=collection_id,
        bbox=search_params.get("bbox"),
        datetime_range=search_params.get("datetime_range")
    )

    # Step 5: Display bbox in leaflet view
    map_bbox(bbox=search_params.get("bbox"))

if __name__ == "__main__":
    main()
