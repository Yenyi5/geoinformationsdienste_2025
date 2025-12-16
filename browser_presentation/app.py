import os
import streamlit as st
from streamlit_folium import st_folium
import time
import streamlit.components.v1 as components
 

# import functions from stac_pipepine.py
from stac_pipeline import (
    make_llm,
    get_stac_collections,
    extract_search_params,
    geocode_first_bbox,
    folium_bbox_map,
    search_stac,    #not yet accessed
    summarize_results, #not yet accessed 
    build_items_eval, 
    add_top_items_to_map,  

)

 
#streamlit app configuration
st.set_page_config(page_title="STAC Geosearch Chat", page_icon="🗺️", layout="wide")
st.title("STAC Geosearch Chat")

 
#sidebar UI for configuring LLM connection
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

 
#initialise streamlit sesseion state 
if "current_state" not in st.session_state:
    st.session_state.current_state = "get_user_query"
    st.session_state.last_bbox = None
    st.session_state.last_bbox_name = None
    st.session_state.last_polygon = None
    st.session_state.last_params = None
    st.session_state.last_query = None
    st.session_state.last_items_eval = None
    st.session_state.last_summary = None



# initialize LLM using settings from sidebar 
llm = make_llm(model=model_name, temperature=temperature)

# Chat input box 
query = st.chat_input("Ask for data (e.g., 'Find LST data for the summer 2018 for the two largest cities in Germany.')")

 
# if user submitts query, handle it 
if query:
    # display the user's message in chat UI
    with st.chat_message("user"):
        st.markdown(query)
    # llm processes the query to extract parameters
    with st.spinner("Extracting parameters with LLM…"):
        # get available Stac collections 
        collections = get_stac_collections()
        
        #LLM parses the natural-language query into structured parameters
        params = extract_search_params(query, collections, llm)

    # geocode the location to bbox and polygon using Nominatim
    with st.spinner("Geocoding location…"):
        bbox, bbox_name, polygon = geocode_first_bbox(params["location"])
  
    #store results in session state 
    st.session_state.last_bbox = bbox
    st.session_state.last_bbox_name = bbox_name
    st.session_state.last_polygon = polygon
    st.session_state.last_params = params
    st.session_state.last_query = query
        
    # search stac and rank results by overlap with the bbox
    if bbox:
        with st.spinner("Searching STAC…"):
            summaries, feats = search_stac(
                params["collectionid"],
                bbox=bbox,
                datetime_range=params.get("datetime_range"),
                limit=200,
            )

        items_eval = build_items_eval(feats, bbox)
        st.session_state.last_items_eval = items_eval
        # ask llm to summarize top results
        with st.spinner("Summarizing top results with LLM…"):
            st.session_state.last_summary = summarize_results(query, items_eval, llm, top_n=100)
    # if no bbox could be geocoded, show empty reuslts 
    else:
        st.session_state.last_items_eval = []
        st.session_state.last_summary = "Could not geocode the location."

    # set the current state to results available
    st.session_state.current_state = "results_available" # state that the results are available
        
        
# rendering results : if results are available 
if (st.session_state.current_state == "results_available"):
    # retrieve the latest values from the session states variable
    extracted_params = st.session_state.last_params
    extracted_bbox = st.session_state.last_bbox
    extracted_bbox_name = st.session_state.last_bbox_name
    polygon = st.session_state.last_polygon
    items_eval = st.session_state.last_items_eval or []
    summary = st.session_state.last_summary

    # display the parameters extracted
    with st.expander("Debug: extracted parameters", expanded=True):
            st.json(extracted_params)

    # alternative way of showing the parameters extracted
    st.write({"bbox": extracted_bbox, "bbox_name": extracted_bbox_name})
    st.subheader("Area of Interest")
    # map create the map with the bbox and the bbox names extracted
    
    # --- MAP: build once per query and reuse ---
    if "map_html" not in st.session_state:
        st.session_state.map_html = None
        st.session_state.map_sig = None


    # create a signature for this map : change this only when the content should change
    map_sig = (
        st.session_state.last_query,
        tuple(extracted_bbox) if extracted_bbox else None,
        extracted_bbox_name,
        len(items_eval),
    )
    # only rebuild the map if the content has changed => solution to flash problem
    if st.session_state.map_sig != map_sig:
        fmap = folium_bbox_map(extracted_bbox, extracted_bbox_name, polygon)
        add_top_items_to_map(fmap, items_eval, top_k=10)  
        st.session_state.map_html = fmap.get_root().render()
        st.session_state.map_sig = map_sig

    components.html(st.session_state.map_html, height=520)

    # display top results in a table 
    st.subheader("Top results (ranked by overlap)")
    st.dataframe([
        {
            "id": it["id"],
            "date": it["date"],
            "overlap_%": round(it["overlap_percentage"]["value"], 2),
            "url": it["url"],
        }
        for it in items_eval[:10]
    ])

    # display the summary/recommendation by the llm
    if summary:
        st.subheader("Recommendation")
        st.write(summary)


 

        