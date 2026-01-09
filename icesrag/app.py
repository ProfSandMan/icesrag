
import json
import os
# Disable Streamlit's file watcher to prevent torch.classes errors
# This must be set before importing streamlit
# The error occurs because Streamlit's watcher tries to inspect torch.classes.__path__
# which doesn't exist in PyTorch's custom __getattr__ implementation
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
from pathlib import Path

import nltk
# Add project-local nltk_data folder
LOCAL_NLTK_PATH = Path(__file__).resolve().parent / "nltk_data_local/"
nltk.data.path.append(str(LOCAL_NLTK_PATH))
import streamlit as st

from icesrag.app_retrievers import VANILLA_RETRIEVER, BM25_RETRIEVER, COMPOSITE_RETRIEVER, bm25_preprocessor
from icesrag.utils.llms import OpenAILLM
from icesrag.utils.hyde import HyDELLM

# Configure Streamlit
st.set_page_config(
    page_title="ICES Search",
    page_icon="🧑‍🚀",
    layout="wide"
)

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
    st.session_state['current_page'] = 0
if 'active_search' not in st.session_state:
    st.session_state['active_search'] = False
if 'prior_user_query' not in st.session_state:
    st.session_state['prior_user_query'] = ""
if 'user_query' not in st.session_state:
    st.session_state['user_query'] = ""
if 'hyde' not in st.session_state:
    st.session_state['hyde'] = False
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None
if 'hyde_query' not in st.session_state:
    st.session_state['hyde_query'] = None

# * ==========================================================================================
# * Config
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
        color: #f6c308 !important;
        text-decoration: underline;
    }

    /* Align the button to the top of the column */
    button[kind="primary"] {
        margin-top: 6px !important;
    }

    /* Disable text area resize */
    textarea {
        resize: none !important;
    }

    /* Reduce padding after horizontal line */
    hr {
        margin: 0.0rem 0 !important;
    }

    /* Reduce padding before input box */
    .dynamic-input-container {
        margin-top: 0.0rem !important;
    }

    </style>
    """, unsafe_allow_html=True)

# * ==========================================================================================
# * Functions
# * ==========================================================================================

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

    - **Use longer, more specific queries or your actual working abstract** — our models perform best with full questions or descriptions like:  
        "I am exploring whether or not human beings can survive on Mars. The focus of my research is on whether or not Mars has the chemical compounds necessary to support human life. I care both about the elemental composition and compound structures we have discovered so far on the planet."

    - Adjust the **number of papers** using the slider to control result breadth.

    ---

    _Still stuck? Keep in mind that results are pulled only from the ICES [ICES paper repository](https://hdl.handle.net/2346/58495) paper repository._
    """)
    if st.button("Close"):
        st.rerun()

# * ==========================================================================================
# * Sideabar Set-up 
# * ==========================================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
logo_path = BASE_DIR / "assets" / "adv logo.png"
# Image
with st.sidebar:
    st.image(logo_path, use_container_width=False)
    css = """
            <style>
            button[title="View fullscreen"]{
                visibility: hidden;}
            </style>
            """
    st.markdown(css, unsafe_allow_html=True)
    st.write("---")

    # Slider for number of papers to retrieve
    num_papers = st.slider(label = "Number of Papers to Retrieve", min_value=1, max_value=30, key="num_papers")

    # Retriever selection
    retriever_name = st.radio(
        "Select Retriever Type",
        options=["Dense", "Sparse", "Composite"],
        key="retriever_name"
    )

    # Toggle for Hypothetical Document Embedding search
    st.session_state['hyde'] = False
    st.session_state['api_key'] = None
    if st.toggle("HyDE Search", key="hyde_search"):
        st.session_state['hyde'] = True
        openaikey = st.text_input("OpenAI API Key", placeholder = "Enter your API Key", type = "password")
        if openaikey != '':
            st.session_state['api_key'] = openaikey

    # Footer 
    footer = """
    <span style="font-size:12px;"><br><br>Developed by:<br> 
    <a href="https://linkedin.com/in/samuel-brooks-59730a1b1" style="color:#f6c308;">Samuel Brooks</a><br>
    <a href="https://www.linkedin.com/in/lilly-hanslik-124149295" style="color:#f6c308;">Lilly Hanslik</a><br>
    <a href="https://linkedin.com/in/iannicholas-ortega" style="color:#f6c308;">Ian Ortega</a><br>
    <a href="https://linkedin.com/in/hunter-sandidge" style="color:#f6c308;">Hunter Sandidge</a><br>
    <a href="https://www.linkedin.com/in/joseph-wall-75432114" style="color:#f6c308;">Joseph Wall</a><br>
    © 2026 Marquette University
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
# * Home Page 
# * ==========================================================================================

# $ Search Bar ========================================================
# Title
st.title("ICES Search", anchor=False)
st.subheader("A prototype for Streamlining NASA Research Retrieval with RAG-Inspired Semantic Search", anchor=False)
if "quick_start" not in st.session_state:
    if st.button("Quick Start Guide"):
        quick_start()

st.markdown("---")

col1, col2 = st.columns([14, 1], gap="small")

with col1:
    st.markdown('<div class="dynamic-input-container">', unsafe_allow_html=True)

    if st.session_state['user_query'] != "" and st.session_state['active_search'] == True:
        user_query = st.text_area(
            label="Search Query",
            value=st.session_state.get("user_query", ""),
            key="main_query_input",
            height=120,
            label_visibility="collapsed",
            placeholder="Enter Query (Press Ctrl+Enter to search)"
        )
    else: # No not yet active search, use default input text
        user_query = st.text_area(
            label="Search Query",
            key="main_query_input",
            height=120,
            label_visibility="collapsed",
            placeholder="Enter Query (Press Ctrl+Enter to search)"
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["user_query"] = user_query.rstrip()

    # if st.session_state['active_search'] == False and st.session_state['current_page'] == 0 and st.session_state['user_query'] != "":
    #     st.session_state['active_search'] = True

    # If button hit, but no query, reset active search
    if st.session_state['active_search'] == True and st.session_state['user_query'] == "":
        st.session_state['active_search'] = False
        st.session_state['current_page'] = 0
        st.session_state.pop("search_results", None)
        st.session_state['hyde_query'] = None

    # If user query has changed, reset current page and query
    if st.session_state['user_query'] != st.session_state['prior_user_query']:
        st.session_state['current_page'] = 0
        st.session_state['prior_user_query'] = st.session_state['user_query']
        st.session_state['hyde_query'] = None

with col2:
    # Add empty space to push button to bottom
    st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
    user_submitted = st.button("🔍")

    # User submitted and active search is false and user query is not empty
    if user_submitted == True and st.session_state['active_search'] == False and st.session_state['user_query'] != "":
        st.session_state['active_search'] = True


# $ HyDE Search ========================================================
# Determine if HyDE search should be performed
if user_submitted == True and st.session_state['active_search'] == True and st.session_state['user_query'] != "" and st.session_state['current_page'] == 0:
    if st.session_state['hyde'] == True:
        if st.session_state['api_key'] is None:
            st.error('WARNING! You have not loaded an API Key, so no HYDE search was performed', icon="🚨")
        else:
            # Use OpenAI API to generate hypothetical document embeddings
            try:
                with st.spinner('Generating HyDE query...'):
                    llm = OpenAILLM(st.session_state['api_key'], model = 'gpt-4o-mini')
                    hyde_llm = HyDELLM(llm)
                    hyde_query = hyde_llm.query(user_query)
                    st.session_state['hyde_query'] = hyde_query
                    user_query = hyde_query
            except Exception as e:
                st.error(f"Error: {e}", icon="🚨")

st.markdown("---")

# $ Search Results ========================================================
if st.session_state['active_search'] == True and st.session_state['current_page'] == 0:

    if retriever_name == 'Sparse':
        user_query = bm25_preprocessor.preprocess(user_query)

    docs, distances, meta = st.session_state['retriever'].top_k(user_query, st.session_state['num_papers'])
    st.session_state["search_results"] = {
        "docs": docs,
        "meta": meta
    }
    st.session_state["current_page"] = 0
    prior_user_query = st.session_state['user_query']

# Only proceed with displaying results if they're available
if st.session_state['active_search'] == True:
    st.subheader("Search Results")
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
            authors = '; '.join(authors).title()
        
        date = metadata.get("date", "N/A")
        url = metadata.get("paper_url", "#")
    
        # Display result
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"<h5 style='margin-bottom: 0;'>{title}</h5>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 14px; color: #fff; padding-top: 8px;'>{abstract}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Paper URL**")
            st.markdown(f"<a href='{url}' target='_blank'>{url}</a>", unsafe_allow_html=True)
            
            st.markdown("**Authors**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{authors}</div>", unsafe_allow_html=True)
            
            st.markdown("**Keywords**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{keywords}</div>", unsafe_allow_html=True)

            st.markdown("**Date**")
            st.markdown(f"<div style='margin-bottom: 10px;'>{date}</div>", unsafe_allow_html=True)
        
        st.text('\n\n')
        st.markdown("---")

        # Navigation controls
        col_prev, col_page, col_next = st.columns([1, 2, 1], border=False)
        with col_prev:
            if st.button("⬅️ Previous", disabled=(current_page == 0)):
                st.session_state["current_page"] = max(current_page - 1, 0)
                st.rerun()

        with col_page:
            st.markdown(f"<center>Page {current_page + 1} of {max_page + 1}</center>", unsafe_allow_html=True)

        with col_next:
            left_col, right_col = st.columns([1.5, 1])
            with right_col:
                if st.button("Next ➡️", disabled=(current_page >= max_page)):
                    st.session_state["current_page"] = min(current_page + 1, max_page)
                    st.rerun()