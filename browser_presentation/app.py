import os
import streamlit as st
from streamlit_folium import st_folium
import time

from dotenv import load_dotenv
load_dotenv("../.env")


# import functions from stac_pipepine.py
from stac_pipeline import (
    make_llm,
    get_stac_collections,
    extract_search_params,
    geocode_first_bbox,
    folium_bbox_map,
    search_stac,    #not yet accessed
    summarize_results, #not yet accessed 

)

 
#streamlit app configuration
st.set_page_config(page_title="STAC Geosearch Chat", page_icon="🗺️", layout="wide")
st.title("STAC Geosearch Chat")

 
#sidebar UI for configuring LLM connection
with st.sidebar:
    st.subheader("LLM Settings")

    # Dropdown menu to select AI provider
    ai_provider = st.selectbox("Select AI Provider", ["Academic Cloud", "OpenAI"])

    # Dynamically populate fields based on the selected provider
    if ai_provider == "Academic Cloud":
        api_base = st.text_input("API Base", value=os.getenv("ACADEMIC_API_ENDPOINT", "https://chat-ai.academiccloud.de/v1"))
        api_key = st.text_input("API Key", type="password", value=os.getenv("ACADEMIC_API_KEY", ""))
        model_name = st.text_input("Model", value=os.getenv("ACADEMIC_MODEL", "llama-3.3-70b-instruct"))
    elif ai_provider == "OpenAI":
        api_base = st.text_input("API Base", value=os.getenv("OPENAI_API_ENDPOINT", "https://api.openai.com/v1"))
        api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Temperature slider
    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.0,
        0.1,
        help="For more logical, reproducible results, a low temperature is recommended."
    )

    st.caption("Tip: set these in your environment so you don't need to paste them every time.")

# Set environment variables based on the selected provider
os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_KEY"] = api_key

 
#initialise streamlit sesseion state 
if "current_state" not in st.session_state:
    st.session_state.current_state = "get_user_query"
    st.session_state.last_bbox = None
    st.session_state.last_bbox_name = None
    st.session_state.last_params = None
    st.session_state.last_query = None



# initialize LLM using settings from sidebar 
llm = make_llm(model=model_name, temperature=temperature)

# Chat input box 
query = st.chat_input("Ask for data (e.g., 'Find LST data for the summer 2018 for the two largest cities in Germany.')")

 
# if user submitts query, handle it 
if query:
    # display the user's message in chat UI
    with st.chat_message("user"):
        st.markdown(query)
    # extract stact search parameters using LLM
    with st.spinner("Extracting parameters with LLM…"):
        t1 = time.time()
        # get available Stac collections 
        collections = get_stac_collections()
        print("Parameter extraction started...")
        
        #LLM parses the natural-language query into structured parameters
        params = extract_search_params(query, collections, llm)
        t2 = time.time()

        print("Parameter extracted: ", params)
        print(f'Parameter extraction took {t2-t1} seconds')


        # update the session states with the new values
        with st.spinner(" Geocoding location…"):
            print("Location geocoding started: ", params)

            
            bbox, bbox_name = geocode_first_bbox(params["location"])
            t3 = time.time()

            print ("Bounding box extracted is: ", bbox)
            
            #store results in session state 
            st.session_state.last_bbox = bbox
            st.session_state.last_bbox_name = bbox_name
            st.session_state.last_params = params
            st.session_state.current_state = "results_available" # state that the results are available
        
        
# if results have been computed, display them
if (st.session_state.current_state == "results_available"):
    # retrieve the latest values from the session states variable
    extracted_params = st.session_state.last_params
    extracted_bbox = st.session_state.last_bbox
    extracted_bbox_name = st.session_state.last_bbox_name

    # display the parameters extracted
    with st.expander("Debug: extracted parameters", expanded=True):
            st.json(extracted_params)

    # alternative way of showing the parameters extracted
    st.write({"bbox": extracted_bbox, "bbox_name": extracted_bbox_name})
    st.subheader("Area of Interest")
    # map create the map with the bbox and the bbox names extracted
    fmap = folium_bbox_map(extracted_bbox, extracted_bbox_name)
    st_folium(fmap, width=900, height=500) 
 

        