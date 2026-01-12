import os

import streamlit as st
import streamlit.components.v1 as components

from maja_collection import init_llm, run_query

st.set_page_config(page_title="STAC Geosearch Chat", page_icon="🗺️", layout="wide")
st.title("STAC Geosearch Chat")

with st.sidebar:
    st.subheader("LLM Settings")
    api_base = st.text_input("API Base", value=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    api_key = st.text_input("API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    st.caption("Tip: set these in your environment so you don't need to paste them every time.")

# Session state
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "map_html" not in st.session_state:
    st.session_state.map_html = None
if "map_sig" not in st.session_state:
    st.session_state.map_sig = None

# Chat input
query = st.chat_input("Ask for data (e.g., 'Find Sentinel-2 data for Berlin in 2024')")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Initialize backend LLM (must be called before run_query)
    init_llm(model_name, temperature, api_key=api_key, api_base=api_base)

    with st.spinner("Running backend pipeline…"):
        result = run_query(query)

    st.session_state.last_query = query
    st.session_state.last_result = result

# Render results
result = st.session_state.last_result
if result:
    bbox = result.get("bbox")
    bbox_name = result.get("bbox_name")
    items_eval = result.get("items_eval") or []

    # Debug (optional): show some extracted state
    with st.expander("Debug: state", expanded=False):
        st.json({
            "bbox": bbox,
            "bbox_name": bbox_name,
            "datetime_range": result.get("datetime_range"),
            "collectionid": result.get("collectionid"),
            "items_count": len(result.get("items") or []),
        })

    st.subheader("Area of Interest")

    # Use results map if available; otherwise bbox map
    m = result.get("resultsmap") or result.get("bboxmap")

    # avoid flicker: only rebuild HTML when content changed
    map_sig = (
        st.session_state.last_query,
        tuple(bbox) if bbox else None,
        bbox_name,
        len(items_eval),
    )
    if m and st.session_state.map_sig != map_sig:
        st.session_state.map_html = m.get_root().render()
        st.session_state.map_sig = map_sig

    if st.session_state.map_html:
        components.html(st.session_state.map_html, height=520)
    else:
        st.info("No map available (bbox not found or no results).")

    st.subheader("Top results (ranked by overlap)")
    st.dataframe([
        {
            "id": it.get("id"),
            "date": it.get("date"),
            "overlap_%": round((it.get("overlap_percentage") or {}).get("value", 0.0), 2),
            "url": it.get("url"),
        }
        for it in items_eval[:10]
    ])

    # Recommendation / summary
    summary = result.get("summary")
    if not summary:
        # fallback: try to extract last agent message from messages
        msgs = result.get("messages") or []
        # your code stores agent responses as {"role": "agent", "content": ...}
        agent_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "agent"]
        if agent_msgs:
            last = agent_msgs[-1].get("content")
            # sometimes it's a string, sometimes an object
            summary = last if isinstance(last, str) else getattr(last, "content", None)

    if summary:
        st.subheader("Recommendation")
        st.write(summary)
