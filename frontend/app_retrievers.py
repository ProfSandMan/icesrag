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

from pathlib import Path # * Absolute Path Construction 


# * ==================================================================================
# * Set up retrievers
# * ==================================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

 
collection = "ices"

# chroma_path = "./data/chroma.db"
# chroma_path = r"C:\Users\samue\OneDrive\Desktop\Local Folder\icesrag\data\chroma.db"
chroma_path = BASE_DIR / "data" / "chroma.db"

# bm25_path = "./data/bm25.db"
# bm25_path = r"C:\Users\samue\OneDrive\Desktop\Local Folder\icesrag\data\bm25.db"
bm25_path = BASE_DIR / "data" / "bm25.db"

vanilla_embedder = EmbeddingEngine(SentenceTransformEmbedder())
vanilla_retriever = RetrieverEngine(ChromaRetriever())
vanilla_retriever.connect(str(chroma_path), collection)
# vanilla_retriever.connect(chroma_path, collection)
VANILLA_RETRIEVER = vanilla_retriever

bm25_preprocessor = TextPreprocessingEngine(BM25PreProcess())
bm25_retriever = RetrieverEngine(SQLiteRetriever())
bm25_retriever.connect(str(bm25_path), collection)
# bm25_retriever.connect(bm25_path, collection)
BM25_RETRIEVER = bm25_retriever

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

COMPOSITE_RETRIEVER = CompositeRetriever(strategies, reranker)

# print(vanilla_retriever.top_k("GIVE ME MARS GOD DAMMIT", 5))