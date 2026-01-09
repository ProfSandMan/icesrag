from pathlib import Path

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

def get_paper_position(papers, target_paper_id, source: str):
    for index, paper in enumerate(papers, start=1):
        if paper.get("paper_id") == target_paper_id:
            return index
    print(f"Paper with ID {target_paper_id} not found in the database: {source}.")
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
                {
                    'name':'vanilla',
                    'embed': vanilla_embedder,
                    'retriever': vanilla_retriever
                 },
                
                {
                    'name':'bm25',
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
data = pd.read_sql("SELECT id, paper_id, query_type, query, hyde_query FROM evaluation_queries WHERE hyde_query IS NOT NULL", conn)

batch_size = 25
chroma_updates = []
bm25_updates = []
composite_updates = []

total_abstracts = int(pd.read_sql("SELECT COUNT(*) AS ct FROM abstracts", conn)['ct'].values[0])
for index, row in tqdm(data.iterrows(), total=len(data), unit="queries", desc="Evaluating queries"):
    eval_id = row['id']
    paper_id = row['paper_id']
    query = row['hyde_query']

    # Vanilla retriever
    retrieved_papers, distances, metadatas = vanilla_retriever.top_k(query, top_k = total_abstracts)
    position = get_paper_position(metadatas, paper_id, "chroma")
    chroma_updates.append((position, eval_id))

    # BM25 retriever
    bm25_query = bm25_preprocessor.preprocess(query)
    retrieved_papers, distances, metadatas = bm25_retriever.top_k(bm25_query, top_k = total_abstracts)
    position = get_paper_position(metadatas, paper_id, "bm25")
    bm25_updates.append((position, eval_id))

    # Composite retriever
    docs, scores, metadatas = retriever.top_k(query, k = total_abstracts)
    position = get_paper_position(metadatas, paper_id, "composite")
    composite_updates.append((position, eval_id))
    
    # Once we have batch_size items, perform batch updates and clear buffers
    if len(chroma_updates) >= batch_size:
        conn.executemany("UPDATE evaluations SET hyde_retrieved_position = ? WHERE evaluation_query_id = ? AND retrieval_method = 'chroma'", chroma_updates)
        chroma_updates.clear()
        conn.executemany("UPDATE evaluations SET hyde_retrieved_position = ? WHERE evaluation_query_id = ? AND retrieval_method = 'bm25'", bm25_updates)
        bm25_updates.clear()
        conn.executemany("UPDATE evaluations SET hyde_retrieved_position = ? WHERE evaluation_query_id = ? AND retrieval_method = 'composite'", composite_updates)
        composite_updates.clear()
        conn.commit()

# Insert any remaining updates after the loop
if chroma_updates:
    conn.executemany("UPDATE evaluations SET hyde_retrieved_position = ? WHERE evaluation_query_id = ? AND retrieval_method = 'chroma'", chroma_updates)
if bm25_updates:
    conn.executemany("UPDATE evaluations SET hyde_retrieved_position = ? WHERE evaluation_query_id = ? AND retrieval_method = 'bm25'", bm25_updates)
if composite_updates:
    conn.executemany("UPDATE evaluations SET hyde_retrieved_position = ? WHERE evaluation_query_id = ? AND retrieval_method = 'composite'", composite_updates)

conn.commit()
conn.close()

print("Done!")