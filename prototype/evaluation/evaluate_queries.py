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

from sqlite3 import connect
import pandas as pd
from tqdm import tqdm

# * ==================================================================================
# * Support Functions/Classes
# * ==================================================================================

def get_paper_position(papers, target_paper_id):
    for index, paper in enumerate(papers, start=1):
        if paper.get("paper_id") == target_paper_id:
            return index
    raise ValueError(f"Paper with ID {target_paper_id} not found in the database.")

# * ==================================================================================
# * Set up retrievers
# * ==================================================================================

collection = "ices"
chroma_path = "./data/chroma.db"
bm25_path = "./data/bm25.db"

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

# * ==================================================================================
# * Gather data
# * ==================================================================================

conn = connect("./data/ices.db")
data = pd.read_sql("SELECT id, paper_id, query_type, query FROM evaluation_queries", conn)

for index, row in tqdm(data.iterrows(), total=len(data), unit="queries", desc="Evaluating queries"):
    eval_id = row['id']
    paper_id = row['paper_id']
    query = row['query']

    # Vanilla retriever
    retrieved_papers, distances, metadatas = vanilla_retriever.top_k(query, top_k = len(data))
    position = get_paper_position(metadatas, paper_id)
    conn.execute("INSERT INTO evaluations (evaluation_query_id, paper_id, retrieval_method, retrieved_position) VALUES (?, ?, ?, ?)", (eval_id, paper_id, "chroma", position))


    # BM25 retriever
    bm25_query = bm25_preprocessor.preprocess(query)
    retrieved_papers, distances, metadatas = bm25_retriever.top_k(bm25_query, top_k = len(data))
    position = get_paper_position(metadatas, paper_id)
    conn.execute("INSERT INTO evaluations (evaluation_query_id, paper_id, retrieval_method, retrieved_position) VALUES (?, ?, ?, ?)", (eval_id, paper_id, "bm25", position))

    # Composite retriever
    docs, scores, metadatas = retriever.top_k(query, k = len(data))
    position = get_paper_position(metadatas, paper_id)
    conn.execute("INSERT INTO evaluations (evaluation_query_id, paper_id, retrieval_method, retrieved_position) VALUES (?, ?, ?, ?)", (eval_id, paper_id, "composite", position))

conn.commit()
conn.close()
print("Done!")