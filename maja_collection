import requests
import json

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

    #Clean the output from possible formatting 
    cleaned_output = llm_output.strip().strip("```").strip()
    
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

if __name__ == "__main__":
    main()
