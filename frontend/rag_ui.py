
import nltk
from pathlib import Path
import json

# Add project-local nltk_data folder
LOCAL_NLTK_PATH = Path(__file__).resolve().parent / "nltk_data_local/"
nltk.data.path.append(str(LOCAL_NLTK_PATH))

# Optional: log what paths NLTK is using
# print("NLTK search paths:", nltk.data.path)

import streamlit as st
from frontend.app_retrievers import VANILLA_RETRIEVER, BM25_RETRIEVER, COMPOSITE_RETRIEVER, bm25_preprocessor
import pandas as pd
# * The way the retrievers are currently set up does not allow for filtering



# * ==========================================================================================
# * Set up Retrievers -- Should just be setting vars
# * ==========================================================================================


# TODO GET IT TO FUCKING WORK
vanilla_retiever = VANILLA_RETRIEVER
bm25_retriever = BM25_RETRIEVER
composite_retriever = COMPOSITE_RETRIEVER


# * ==========================================================================================
# * State Variable Set-up
# * ==========================================================================================

if 'retriever' not in st.session_state:
    st.session_state['retriever'] = VANILLA_RETRIEVER
if 'retriever_name' not in st.session_state:
    st.session_state['retriever_name'] = "Dense"
if 'num_papers' not in st.session_state:
    st.session_state['num_papers'] = 10 # @ Hunter - Does this work the way we need it to? 
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0  # zero-indexed



# Title
st.title("Streamlining NASA Research Retrieval with RAG-Inspired Semantic Search: DEMO/Rapid Prototype Version")

# * ==========================================================================================
# * Sideabar Set-up 
# * ==========================================================================================

# Slider for number of papers to retrieve
num_papers = st.sidebar.slider("Number of Papers to Retrieve", min_value=1, max_value=30, key="num_papers")

# Retriever selection
retriever_name = st.sidebar.radio(
    "Select Retriever Type",
    options=["Dense", "Sparse", "Composite"],
    key="retriever_name"
)


if retriever_name != st.session_state.get('last_selected_retriever'):
    if retriever_name == 'Dense':
        st.session_state['retriever'] = VANILLA_RETRIEVER
    elif retriever_name == 'Sparse':
        st.session_state['retriever'] = BM25_RETRIEVER
    else:
        st.session_state['retriever'] = COMPOSITE_RETRIEVER
    st.session_state['last_selected_retriever'] = retriever_name

# * ==========================================================================================
# * Search Bar (Dynamic Expansion)
# * ==========================================================================================

# Inject CSS for the dynamic search box
st.markdown("""
<style>
.dynamic-input-container {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 6px 10px;
    max-height: 150px;  /* Max height before scroll */
    overflow-y: auto;
    background-color: white;
}
textarea {
    width: 100% !important;
    resize: none;  /* Prevent manual resizing */
    font-size: 16px;
    line-height: 1.4;
    min-height: 32px;
    border: none;
    outline: none;
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    st.markdown('<div class="dynamic-input-container">', unsafe_allow_html=True)
    user_query = st.text_area(
        label="",
        value=st.session_state.get("user_query", ""),
        key="main_query_input",
        height=70,
        label_visibility="collapsed",
        placeholder="Enter Query"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["user_query"] = user_query

with col2:
    submit_clicked = st.button("🔍")

if submit_clicked and user_query:
    st.markdown(f"<p style='font-size:18px; color:gray;'>Searching for: <b>{user_query}</b></p>", unsafe_allow_html=True)

    if retriever_name == 'Sparse':
        user_query = bm25_preprocessor.preprocess(user_query)

    docs, distances, meta = st.session_state['retriever'].top_k(user_query, st.session_state['num_papers'])

st.subheader("Search Results")

# Ensure the search only runs after the user submits a query
if submit_clicked and user_query:
    st.markdown(f"<p style='font-size:18px; color:gray;'>Searching for: <b>{user_query}</b></p>", unsafe_allow_html=True)

    if retriever_name == 'Sparse':
        user_query = bm25_preprocessor.preprocess(user_query)

    # Run the retrieval logic here
    docs, distances, meta = st.session_state['retriever'].top_k(user_query, st.session_state['num_papers'])

    # Save docs and meta to session state to persist across interactions
    st.session_state["search_results"] = {
        "docs": docs,
        "meta": meta
    }

    # Initialize or reset the current page to 0
    st.session_state["current_page"] = 0

# Only proceed with displaying results if they're available
if "search_results" in st.session_state:
    docs = st.session_state["search_results"]["docs"]
    meta = st.session_state["search_results"]["meta"]

    # Check if there are any results
    if docs:
        # Get current page index
        current_page = st.session_state.get("current_page", 0)
        max_page = len(docs) - 1

        # Display one result at a time
        abstract = docs[current_page]
        metadata = meta[current_page]

        # Extract metadata safely
        title = metadata.get("title", "No Title")
        keywords = metadata.get("keywords", "N/A")
        try:
            keywords = json.loads(keywords)
        except:
            pass
        if isinstance(keywords, list):
            keywords = ', '.join(keywords).title()
        date = metadata.get("date", "N/A")
        url = metadata.get("paper_url", "#")

        # Display result
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"<h5 style='margin-bottom: 0;'>{title}</h5>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 14px; color: #444; padding-top: 8px;'>{abstract}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Keywords**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{keywords}</div>", unsafe_allow_html=True)

            st.markdown("**Date**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{date}</div>", unsafe_allow_html=True)

            st.markdown("**Paper URL**")
            st.markdown(f"<a href='{url}' target='_blank'>{url}</a>", unsafe_allow_html=True)

        # Navigation controls
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous", disabled=(current_page == 0)):
                st.session_state["current_page"] = max(current_page - 1, 0)
                st.rerun()

        with col_page:
            st.markdown(f"<center>Page {current_page + 1} of {max_page + 1}</center>", unsafe_allow_html=True)

        with col_next:
            if st.button("Next ➡️", disabled=(current_page >= max_page)):
                st.session_state["current_page"] = min(current_page + 1, max_page)
                st.rerun()
