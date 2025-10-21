import os
import streamlit as st
from streamlit_folium import st_folium

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
    
    api_base = st.text_input("API Base", value=os.getenv("OPENAI_API_BASE", "https://chat-ai.academiccloud.de/v1"))
    api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", "7f966d739f12900214b52741e3f80ff2"))
    model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "llama-3.3-70b-instruct"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1) # controls deterministic vs. random

    st.caption("Tip: set these in your environment so you don't need to paste them every time.")


os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_KEY"] = api_key

# Build LLM
llm = make_llm(model=model_name, temperature=temperature)

# Session state for chat history (optional)
if "chat" not in st.session_state:
    st.session_state.chat = []

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# for the persistence of last search parameters and results 
if "last_bbox" not in st.session_state:
    st.session_state.last_bbox = None
if "last_bbox_name" not in st.session_state:
    st.session_state.last_bbox_name = None
if "last_params" not in st.session_state:
    st.session_state.last_params = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

# Chat input
query = st.chat_input("Ask for data (e.g., 'Find LST data for the summer 2018 for the two largest cities in Germany.')")

if query:
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            import time
            t0 = time.time()

            if not api_key:
                st.warning("Please provide your API Key in the sidebar.")
                st.stop()

            # Connectivity sanity checks (small timeouts)
            import requests
            with st.status("🔌 Checking connectivity...", expanded=False) as s:
                ok = True
                try:
                    r = requests.get("https://geoservice.dlr.de/eoc/ogc/stac/v1", timeout=5)
                    st.write("STAC reachable " if r.ok else f"STAC status {r.status_code}")
                except Exception as e:
                    ok = False
                    st.error(f"STAC unreachable: {e}")

                try:
                    r = requests.get("https://nominatim.openstreetmap.org/status.php", timeout=5,
                                     headers={"User-Agent": "stac-chat-app/1.0 (your.email@example.com)"})
                    st.write("Nominatim reachable " if r.ok else f"Nominatim status {r.status_code}")
                except Exception as e:
                    ok = False
                    st.error(f"Nominatim unreachable: {e}")

                if not ok:
                    st.stop()
                s.update(label="Connectivity OK", state="complete")

            with st.spinner(" Extracting parameters with LLM…"):
                t1 = time.time()
                collections = get_stac_collections()
                #llm check
                pong = llm.invoke("Reply with the single word: pong").content.strip()
                if "pong" not in pong.lower():
                    st.warning(f"LLM healthcheck unexpected: {pong}")
                    st.stop()

                params = extract_search_params(query, collections, llm)
                t2 = time.time()
                with st.expander("Debug: extracted parameters", expanded=False):
                    st.json(params)

            with st.spinner(" Geocoding location…"):
                bbox, bbox_name = geocode_first_bbox(params["location"])
                t3 = time.time()
                with st.expander("Debug: geocode result", expanded=False):
                    st.write({"bbox": bbox, "bbox_name": bbox_name})
                if not bbox:
                    st.info("No bounding box found. Try a clearer location like 'Berlin, Germany'.")
                    st.stop()

            st.subheader("Area of Interest")
            fmap = folium_bbox_map(bbox, bbox_name)
            st_folium(fmap, width=900, height=500)

            with st.spinner("🔎 Searching STAC…"):
                results, feats = search_stac(
                    params["collectionid"],
                    bbox=bbox,
                    datetime_range=params["datetime_range"],
                    limit=3,  # smaller = faster
                )
                t4 = time.time()

            st.write(
                f" Times — LLM: {t2-t1:.1f}s, Geocode: {t3-t2:.1f}s, STAC: {t4-t3:.1f}s, Total: {t4-t0:.1f}s"
            )

            st.subheader("🔎 STAC Results")
            if results:
                st.dataframe(results, use_container_width=True)
                with st.spinner("Summarizing results…"):
                    summary = summarize_results(query, feats, llm)
            else:
                summary = "No results found. Try broadening the date range, changing the collection, or using a larger area."

            st.markdown(
                f"**Collection:** {params['collectionid'][0]}  \n"
                f"**Datetime:** {params['datetime_range'] or 'not set'}  \n"
                f"**Location:** {bbox_name or '—'}"
            )
            st.markdown("Summary")
            st.write(summary)
            #  save last search info 
            st.session_state.last_bbox = bbox
            st.session_state.last_bbox_name = bbox_name
            st.session_state.last_params = params
            st.session_state.last_results = results
            st.session_state.last_summary = summary

            # Save to chat history (optional)
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Collection: {params['collectionid'][0]}\n"
                        f"Datetime: {params['datetime_range'] or 'not set'}\n"
                        f"Location: {bbox_name or '—'}\n\n"
                        f"Summary: {summary}"
                    ),
                }
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.chat.append({"role": "assistant", "content": f"Error: {e}"})

# --- Persistent display of last results ---

st.markdown("## Current Area & Results")

if st.session_state.last_bbox:
    fmap = folium_bbox_map(
        st.session_state.last_bbox, st.session_state.last_bbox_name
    )
    st_folium(fmap, width=900, height=500)

    p = st.session_state.last_params or {}
    st.markdown(
        f"**Collection:** {(p.get('collectionid') or ['—'])[0]}  \n"
        f"**Datetime:** {p.get('datetime_range') or 'not set'}  \n"
        f"**Location:** {st.session_state.last_bbox_name or '—'}"
    )

    if st.session_state.last_results:
        st.subheader("STAC Results")
        st.dataframe(st.session_state.last_results, use_container_width=True)

    if st.session_state.last_summary:
        st.markdown("Summary")
        st.write(st.session_state.last_summary)

else:
    st.info("No area selected yet. Ask a query to get started.")
