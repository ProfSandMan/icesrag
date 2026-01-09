from pydantic import BaseModel, Field
from openai import OpenAI
from numpy.random import rand
from tqdm import tqdm
import os
from dotenv import load_dotenv

from sqlite3 import connect
import pandas as pd

from icesrag.utils.embed.strategy_pattern import EmbeddingEngine
from icesrag.utils.embed.base_embedders import SentenceTransformEmbedder

from icesrag.utils.text_preprocess.strategy_pattern import TextPreprocessingEngine
from icesrag.utils.text_preprocess.bm25_preprocess import BM25PreProcess

from icesrag.retrieve.rerank.rrf import ReciprocalRerankFusion
from icesrag.retrieve.rerank.strategy_pattern import ReRankEngine

from icesrag.retrieve.retrievers.chroma import ChromaRetriever
from icesrag.retrieve.retrievers.sqlite import SQLiteRetriever
from icesrag.retrieve.retrievers.strategy_pattern import RetrieverEngine

from icesrag.retrieve.pipeline.retriever import CompositeRetriever




# FIXME: use this script to ONLY generate the queries, do not evaluate
# FIXME:    Will want to evaluate chroma, bm25, and composite separately
# FIXME:    Blow away stuff in the ices.db (view included) and do it smarter


# * ==================================================================================
# * Support Functions/Classes
# * ==================================================================================

# Define Pydantic model for structured output
class QueriesFromAbstracts(BaseModel):
    short_form_queries: list[str] = Field(description="Short-form queries", 
                                          example = ["real-time monitoring of life support systems on the International Space Station", 
                                                     "anomaly detection in ISS environmental control systems",
                                                     "integrated monitoring frameworks for astronaut life support"])
    long_form_queries: list[str] = Field(description="Long-form queries", 
                                        example = ["I am exploring existing approaches to real-time monitoring of life support systems aboard the International Space Station. Specifically, I’m interested in how telemetry data is collected, processed, and utilized to maintain system stability and crew safety in microgravity environments. What frameworks, sensors, or computational methods have been developed to enable continuous monitoring, and how have they been validated in operational settings?",
                                                   "As part of my research, I am investigating techniques for detecting anomalies within the environmental control systems of the ISS. I’m looking for studies that discuss methods—either rule-based or machine learning-driven—for identifying deviations in system performance that could compromise environmental stability or crew health. What types of anomalies have been observed in the past, and what approaches have proven most effective for real-time detection and response?",
                                                   "My work focuses on developing integrated monitoring frameworks for astronaut life support systems, and I’m seeking prior research that addresses how multiple subsystems—such as atmosphere regulation, water recycling, and thermal control—are coordinated into a unified monitoring architecture. How have past efforts combined telemetry from these subsystems to generate actionable insights, and what design principles or technologies have enabled reliable, end-to-end system integration in spaceflight environments?"])

# Get the position of the paper in the database
def get_paper_position(papers, target_paper_id):
    for index, paper in enumerate(papers, start=1):
        if paper.get("metadata", {}).get("paper_id") == target_paper_id:
            return index
    raise ValueError(f"Paper with ID {target_paper_id} not found in the database.")

# * ==================================================================================
# * Define variables
# * ==================================================================================

# $ Crtitical variables
total_abstracts = 500
total_queries = 3

# Set up OpenAI client
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
model = "gpt-4o-mini"

abstract_db_path = "./data/ices.db"

system_prompt = """You are an expert in academic research and information retrieval. Your task is to generate realistic, high-quality research queries that a domain expert might use to retrieve a given scientific abstract. You will generate two forms of queries:

1. Short-form queries: These queries should be phrased as search engine inputs and tailored to return the given abstract as the most relevant result.
2. Long-form queries: These queries should be phrased as natural language questions and tailored to return the given abstract as the most relevant result.

"""

user_prompt = f"""Given the following research abstract, generate {total_queries} realistic short-form queries and {total_queries} realistic long-form queries that a researcher might use to find this work. The queries should be:
1. Representative of domain-specific search behavior.
2. Focused on retrieving this abstract as the top result.
3. Varied in phrasing and scope (from specific to general).

Example Abstract:

This paper presents a modular sensor integration framework for real-time monitoring of life support systems aboard the International Space Station. The system leverages telemetry fusion, anomaly detection algorithms, and redundancy-aware failover mechanisms to ensure astronaut safety and operational continuity.

Example Queries:

Short-form queries:
1. real-time monitoring of life support systems on the International Space Station
2. anomaly detection in ISS environmental control systems
3. integrated monitoring frameworks for astronaut life support

Long-form queries:
1. I am exploring existing approaches to real-time monitoring of life support systems aboard the International Space Station. Specifically, I’m interested in how telemetry data is collected, processed, and utilized to maintain system stability and crew safety in microgravity environments. What frameworks, sensors, or computational methods have been developed to enable continuous monitoring, and how have they been validated in operational settings?
2. As part of my research, I am investigating techniques for detecting anomalies within the environmental control systems of the ISS. I’m looking for studies that discuss methods—either rule-based or machine learning-driven—for identifying deviations in system performance that could compromise environmental stability or crew health. What types of anomalies have been observed in the past, and what approaches have proven most effective for real-time detection and response?
3. My work focuses on developing integrated monitoring frameworks for astronaut life support systems, and I’m seeking prior research that addresses how multiple subsystems—such as atmosphere regulation, water recycling, and thermal control—are coordinated into a unified monitoring architecture. How have past efforts combined telemetry from these subsystems to generate actionable insights, and what design principles or technologies have enabled reliable, end-to-end system integration in spaceflight environments?

Abstract to generate queries for:

"""

# * ==================================================================================
# * Generate Queries
# * ==================================================================================

# Import abstracts
print("Importing abstracts...")

conn = connect(abstract_db_path)
data = pd.read_sql("SELECT id, abstract FROM abstracts", conn)
n = len(data)
data['random'] = rand(len(data))
data.sort_values(by='random', inplace=True)

data = data[data['id'] == 2628] # 1035 leto, 2744 cost prediction, 2605 caving on the moon, 2628 Mars elements for human life
# data = data[:total_abstracts]
abstracts = {row['id']: row['abstract'] for _, row in data.iterrows()}

# Import prior queries
already_evaluated = pd.read_sql("SELECT DISTINCT paper_id FROM evaluation_queries", conn)
already_evaluated = already_evaluated['paper_id'].to_list()

# Loop through abstracts and generate queries
print("Generating queries...")
for paper_id, abstract in tqdm(abstracts.items(), desc="Generating Queries", total=len(abstracts), unit="abstracts"):
    if paper_id not in already_evaluated:
        print(f"\nProcessing paper ID: {paper_id}")
        response = client.beta.chat.completions.parse(model = model,
                                                      messages = [{"role": "system", "content": system_prompt},
                                                                  {"role": "user", "content": user_prompt + abstract}],
                                                      response_format = QueriesFromAbstracts)
        result = response.choices[0].message.parsed
        for query in result.short_form_queries:
            # Store
            conn.execute("INSERT INTO evaluation_queries (paper_id, query_type, query) VALUES (?, ?, ?)", (paper_id, "short", query))
        for query in result.long_form_queries:
            conn.execute("INSERT INTO evaluation_queries (paper_id, query_type, query) VALUES (?, ?, ?)", (paper_id, "long", query))
        conn.commit()

conn.close()
print("Done!")