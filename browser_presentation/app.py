import os

import streamlit as st

from streamlit_folium import st_folium

import time

 

# import functions from stac_pipepine.py

from stac_pipeline import (

    make_llm,

    get_stac_collections,

    extract_search_params,

    geocode_first_bbox,

    folium_bbox_map,

    search_stac,

    summarize_results,

)

 

st.set_page_config(page_title="STAC Geosearch Chat", page_icon="🗺️", layout="wide")

st.title("STAC Geosearch Chat")

 

with st.sidebar:

    st.subheader("LLM Settings")

   

    api_base = st.text_input("API Base", value=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))

    api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", "")) 
   # model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "llama-3.3-70b-instruct"))

    model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1) # controls deterministic vs. random

 

    st.caption("Tip: set these in your environment so you don't need to paste them every time.")

 

os.environ["OPENAI_API_BASE"] = api_base

os.environ["OPENAI_API_KEY"] = api_key

 

# Build LLM

llm = make_llm(model=model_name, temperature=temperature)

 

# Chat input

query = st.chat_input("Ask for data (e.g., 'Find LST data for the summer 2018 for the two largest cities in Germany.')")

 

if query:

    #st.session_state.chat.append({"role": "user", "content": query})

    with st.chat_message("user"):

        st.markdown(query)

 

    with st.spinner(" Extracting parameters with LLM…"):

        t1 = time.time()

        collections = get_stac_collections()

        print("Parameter extraction started...")

        params = extract_search_params(query, collections, llm)

        print("parameter extracted: ", params)

        t2 = time.time()

        with st.expander("Debug: extracted parameters", expanded=True):

            st.json(params)

 

        with st.spinner(" Geocoding location…"):

            bbox, bbox_name = geocode_first_bbox(params["location"])

            t3 = time.time()

            print ("bounding box extracted is: ", bbox)

            st.write({"bbox": bbox, "bbox_name": bbox_name})

     

        st.subheader("Area of Interest")

        fmap = folium_bbox_map(bbox, bbox_name)

        st_folium(fmap, width=900, height=500)        

 

        st.session_state.last_bbox = bbox

        st.session_state.last_bbox_name = bbox_name

        st.session_state.last_params = params