
import nltk
from pathlib import Path
import json

# nltk.download('wordnet')
# nltk.download('omw-1.4') 

# Add project-local nltk_data folder
LOCAL_NLTK_PATH = Path(__file__).resolve().parent / "nltk_data_local/"
nltk.data.path.append(str(LOCAL_NLTK_PATH))


import streamlit as st
from frontend.app_retrievers import VANILLA_RETRIEVER, BM25_RETRIEVER, COMPOSITE_RETRIEVER, bm25_preprocessor


# * The way the retrievers are currently set up does not allow for filtering

# * ==========================================================================================
# * Set up Retrievers -- Should just be setting vars
# * ==========================================================================================

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
    st.session_state['num_papers'] = 10
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0  # zero-indexed

# Title
st.title("ICES Search")
st.subheader("A prototype for Streamlining NASA Research Retrieval with RAG-Inspired Semantic Search ")

# * ==========================================================================================
# * Sideabar Set-up 
# * ==========================================================================================

# Image
with st.sidebar:
    st.image(r'frontend\MU_AIM Wordmark-MB-BG (3997x2035).png', use_container_width=True)
    css = """
            <style>
            button[title="View fullscreen"]{
                visibility: hidden;}
            </style>
            """
    st.markdown(css, unsafe_allow_html=True)

    @st.dialog("Quick Start Guide")
    def quick_start():
        st.markdown("""
    ### 🚀 Quick Start Guide

    This tool helps you search through NASA research using three advanced methods:

    - **Dense Retrieval (Vanilla RAG)**  
      Uses AI-generated embeddings to match the meaning of your query to document content. Great for conceptual or natural language questions.

    - **Sparse Retrieval (BM25)**  
      Matches keywords directly. Best when you know the specific terms or technical jargon used in papers.

    - **Composite Retrieval (RRF)**  
      Combines both methods using Reciprocal Rank Fusion — balancing the precision of keywords with the flexibility of semantic matching.

    ---
    ### 💡 Tips for Better Results

    - **Use longer, more specific queries** — our models perform best with full questions or descriptions like:  
      “How does NASA use generative design in spacecraft manufacturing?”  
      or  
      “Applications of thermal shielding on Mars rovers during dust storms.”

    - Adjust the **number of papers** using the slider to control result breadth.

    ---

    _Still stuck? Keep in mind that results are pulled only from the ICES [ICES paper repository](https://hdl.handle.net/2346/58495) paper repository._
    """)
        if st.button("Close"):
            st.rerun()
    if "quick_start" not in st.session_state:
        if st.button("Quick Start"):
            quick_start()


    # Slider for number of papers to retrieve
    num_papers = st.slider(label = "Number of Papers to Retrieve", min_value=1, max_value=30, key="num_papers")

    # Retriever selection
    retriever_name = st.radio(
        "Select Retriever Type",
        options=["Dense", "Sparse", "Composite"],
        key="retriever_name"
    )
    # Footer 
    footer = """
    <span style="font-size:12px;"><br><br>Developed by:<br> 
    <a href="https://linkedin.com/in/samuel-brooks-59730a1b1" style="color:#FC3D21;">Samuel Brooks</a><br>
    <a href="https://linkedin.com/in/iannicholas-ortega" style="color:#FC3D21;">Ian Ortega</a><br>
    <a href="https://linkedin.com/in/hunter-sandidge" style="color:#FC3D21;">Hunter Sandidge</a><br>
    © 2025
    </span>
    """

    st.markdown(footer, unsafe_allow_html=True)


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
st.markdown("""
<style>
/* Make all text white by default */
body, .markdown-text-container, .stText, .stMarkdown, div, p, span {
    color: white !important;
}

/* Fix table-style metadata blocks */
div[data-testid="column"] {
    color: white !important;
}

/* Optional: make links more visible */
a {
    color: #e8c309 !important;
    text-decoration: underline;
}

/* Align the button to the top of the column */
button[kind="primary"] {
    margin-top: 6px !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1], gap="small")

with col1:
    st.markdown('<div class="dynamic-input-container">', unsafe_allow_html=True)
    user_query = st.text_area(
        label="",
        value=st.session_state.get("user_query", ""),
        key="main_query_input",
        height=120,
        label_visibility="collapsed",
        placeholder="Enter Query"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["user_query"] = user_query

with col2:
    submit_clicked = st.button("🔍")


if submit_clicked and user_query:
    if retriever_name == 'Sparse':
        user_query = bm25_preprocessor.preprocess(user_query)

    docs, distances, meta = st.session_state['retriever'].top_k(user_query, st.session_state['num_papers'])

    st.subheader("Search Results")
    st.session_state["search_results"] = {
        "docs": docs,
        "meta": meta
    }
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
        
        authors = metadata.get("authors","N/A")
        try:
            authors = json.loads(authors)
        except:
            pass
        if isinstance(authors, list):
            authors = ', '.join(authors).title()
        
        date = metadata.get("date", "N/A")
        url = metadata.get("paper_url", "#")
    
        # Display result
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"<h5 style='margin-bottom: 0;'>{title}</h5>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 14px; color: #444; padding-top: 8px;'>{abstract}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Paper URL**")
            st.markdown(f"<a href='{url}' target='_blank'>{url}</a>", unsafe_allow_html=True)
            
            st.markdown("**Authors**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{authors}</div>", unsafe_allow_html=True)
            
            st.markdown("**Keywords**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{keywords}</div>", unsafe_allow_html=True)

            st.markdown("**Date**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{date}</div>", unsafe_allow_html=True)
            

        
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
