from pydantic import BaseModel
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




# Define Pydantic model for structured output
class QueriesFromAbstracts(BaseModel):
    queries = list[str]

# Get the position of the paper in the database
def get_paper_position(papers, target_paper_id):
    for index, paper in enumerate(papers, start=1):
        if paper.get("metadata", {}).get("paper_id") == target_paper_id:
            return index
    raise ValueError(f"Paper with ID {target_paper_id} not found in the database.")

# Define variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
model = "gpt-4o-mini"

total_abstracts = 1
total_queries = 3

abstract_db_path = "./ices.db"

system_prompt = """You are an expert in academic research and information retrieval. Your task is to generate realistic, high-quality research queries that a domain expert might use to retrieve a given scientific abstract. These queries should be phrased as search engine inputs and tailored to return the given abstract as the most relevant result.
"""

user_prompt = f"""Given the following research abstract, generate {total_queries} realistic research questions or search queries that a researcher might use to find this work. The queries should be:
1. Representative of domain-specific search behavior.
2. Focused on retrieving this abstract as the top result.
3. Varied in phrasing and scope (from specific to general).

Example Abstract:

This paper presents a modular sensor integration framework for real-time monitoring of life support systems aboard the International Space Station. The system leverages telemetry fusion, anomaly detection algorithms, and redundancy-aware failover mechanisms to ensure astronaut safety and operational continuity.

Example Queries:

1. telemetry fusion for ISS life support systems
2. real-time monitoring of life support systems on the International Space Station
3. redundancy-aware sensor networks for spacecraft life support
4. anomaly detection in ISS environmental control systems
5. integrated monitoring frameworks for astronaut life support

Abstract to generate queries for:

"""

# Set up Retriever
print("Setting up Retriever...")

collection = "ices"
chroma_path = "./chroma.db"
bm25_path = "./bm25.db"

vanilla_embedder = EmbeddingEngine(SentenceTransformEmbedder())
vanilla_retriever = RetrieverEngine(ChromaRetriever())
vanilla_retriever.connect(chroma_path, collection)

bm25_preprocessor = TextPreprocessingEngine(BM25PreProcess())
bm25_retriever = RetrieverEngine(SQLiteRetriever())
bm25_retriever.connect(bm25_path, collection)

reranker = ReRankEngine(ReciprocalRerankFusion())

strategies = [
                {'name':'vanilla',
                 'embed': vanilla_embedder,
                 'retriever': vanilla_retriever
                 },
                
                {'name':'bm25',
                 'preprocess': bm25_preprocessor,
                 'retriever': bm25_retriever
                } 
            ]

retriever = CompositeRetriever(strategies, reranker)

# Import abstracts
print("Importing abstracts...")

conn = connect(abstract_db_path)
data = pd.read_sql("SELECT abstract FROM abstracts", conn)
n = len(data)
data['random'] = rand(len(data))
data.sort_values(by='random', inplace=True)
abstracts = data[:total_abstracts].set_index('index')['abstract'].to_dict()

# Import prior queries
already_evaluated = pd.read_sql("SELECT paper_id FROM evaluation", conn)
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
        
        for query in result.queries:
            # Retrieve
            retrieved_papers = retriever.retrieve(query, k = n)
            # Find the position of paper_id
            position = get_paper_position(retrieved_papers, paper_id)
            print(f"Paper found at position: {position}")
            # Store
            conn.execute("INSERT INTO evaluations (paper_id, query, retrieved_position) VALUES (?, ?, ?)", (paper_id, query, position))
        conn.commit()

conn.close()
print("Done!")