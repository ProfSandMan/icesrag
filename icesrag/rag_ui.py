import streamlit as st

st.sidebar.title("🔍 Filter Parameters") #*You can use emojis!

# TODO Select Params to limit in side bar

# Toggleable filters - First bits of meta data I thought would be useful to toggle by
filter_author = st.sidebar.checkbox("Author")
filter_date = st.sidebar.checkbox("Date")
filter_publisher = st.sidebar.checkbox("Publisher")
filter_keyword = st.sidebar.checkbox("Keyword")  #Whole point of RAG is to avoid specific kword matching alone so idk

# Input for Selected Filters
if filter_author:
    st.sidebar.text_input("Enter Author Name")
if filter_date:
    st.sidebar.date_input("Select Date")
if filter_publisher:
    st.sidebar.text_input("Enter Publisher")
if filter_keyword:
    st.sidebar.text_input("Enter Keyword")


# Title
st.title("Streamlining NASA Research Retrieval with RAG-Inspired Semantic Search: DEMO/Rapid Prototype Version")

#Input Box
# Input box with enter querry as placeholder 

#default value using the examples 
default_query = st.session_state.get("user_query", "")

col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_input("", placeholder="Enter Query", label_visibility="collapsed", value=default_query, key="main_query_input")
    st.session_state["user_query"] = user_query
with col2:
    submit_clicked = st.button("🔍")

if submit_clicked and user_query:
    st.markdown(f"<p style='font-size:18px; color:gray;'>Searching for: <b>{user_query}</b></p>", unsafe_allow_html=True)

# TODO: Stitch together with actual functionality

# Exampel Queries - These are placeholders
st.markdown("""
<div style="margin-top: 20px;">
    <p style="font-size: 20px; font-weight: bold;">Example Queries:</p>
</div>
""", unsafe_allow_html=True)

#TODO have relevant example queries 
col1, col2, col3 = st.columns(3)
example_queries = [
    "What papers were published by Dr. Smith in 2023?",
    "Show me papers on renewable energy.",
    "Find papers from Nature journal."
]

with col1:
    if st.button("EX: " + example_queries[0]):
        st.session_state["user_query"] = example_queries[0]
        #st.write(f"Selected: {example_queries[0]}")
with col2:
    if st.button("EX: " + example_queries[1]):
        st.session_state["user_query"] = example_queries[1]
        #st.write(f"Selected: {example_queries[1]}")
with col3:
    if st.button("EX: " + example_queries[2]):
        st.session_state["user_query"] = example_queries[2]
        #st.write(f"Selected: {example_queries[2]}")



#TODO After the query is run display the output as well as the updated site page (See Front End Design "Post Query")