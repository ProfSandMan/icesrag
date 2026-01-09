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
from pathlib import Path

# * ==================================================================================
# * Support Functions/Classes
# * ==================================================================================

def get_paper_position(papers, target_paper_id, source: str):
    for index, paper in enumerate(papers, start=1):
        if paper.get("paper_id") == target_paper_id:
            return index
    print(f"Paper with ID {target_paper_id} not found in the {source} database.")
    return None

# * ==================================================================================
# * Set up retrievers
# * ==================================================================================

collection = "ices"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
chroma_path = str(BASE_DIR / "data" / "chroma.db")
bm25_path = str(BASE_DIR / "data" / "bm25.db")

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

db_path = str(BASE_DIR / "data" / "ices.db")
conn = connect(db_path)
data = pd.read_sql("SELECT id, paper_id, query_type, query FROM evaluation_queries", conn)

for index, row in tqdm(data.iterrows(), total=len(data), unit="queries", desc="Evaluating queries"):
    eval_id = row['id']
    paper_id = row['paper_id']
    query = row['query']
    valid = True

    # Vanilla retriever
    retrieved_papers, distances, metadatas = vanilla_retriever.top_k(query, top_k = len(data))
    chroma_position = get_paper_position(metadatas, paper_id, "chroma")
    
    # BM25 retriever
    bm25_query = bm25_preprocessor.preprocess(query)
    retrieved_papers, distances, metadatas = bm25_retriever.top_k(bm25_query, top_k = len(data))
    bm25_position = get_paper_position(metadatas, paper_id, "bm25")

    # Composite retriever
    docs, scores, metadatas = retriever.top_k(query, k = len(data))
    composite_position = get_paper_position(metadatas, paper_id, "composite")

    if chroma_position is None or bm25_position is None or composite_position is None:
        valid = False
    
    if valid:
        conn.execute("INSERT INTO evaluations (evaluation_query_id, paper_id, retrieval_method, retrieved_position) VALUES (?, ?, ?, ?)", (eval_id, paper_id, "chroma", chroma_position))
        conn.execute("INSERT INTO evaluations (evaluation_query_id, paper_id, retrieval_method, retrieved_position) VALUES (?, ?, ?, ?)", (eval_id, paper_id, "bm25", bm25_position))
        conn.execute("INSERT INTO evaluations (evaluation_query_id, paper_id, retrieval_method, retrieved_position) VALUES (?, ?, ?, ?)", (eval_id, paper_id, "composite", composite_position))

conn.commit()
conn.close()
print("Done!")