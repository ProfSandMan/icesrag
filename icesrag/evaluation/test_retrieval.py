import logging

# ANSI escape sequences for colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(
            f"{log_fmt}%(asctime)s - %(name)s - %(levelname)s - %(message)s{self.reset}"
        )
        return formatter.format(record)

# Configure logging with colors
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

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

def get_paper_position(papers, target_paper_id):
    for index, paper in enumerate(papers, start=1):
        if paper.get("paper_id") == target_paper_id:
            return index
    raise ValueError(f"Paper with ID {target_paper_id} not found in the database.")

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

query = "artificial intelligence Leto anomaly detection, prognostics, and telemetry monitoring"
# query = "commercial space industry accelerates humanity set sight deepspace destination human capital required support bold endeavor grows dramatically alleviate strain industry workforce reduce cost optimize operation collins aerospace rtx business developing spaceflight intelligence system leto™ integrates existing future system leto™ fullstack system us physicsbased model aipowered algorithm continuously monitor inform ground support personnel onboard crew performance vehicle environmental control life support system eclss leto™ trained verified decade spaceflight telemetry survey vehicle eclss provides performance metric prognostic anomaly detection function alert user system degradation advise upcoming maintenance event occur intelligent eclss monitoring leto™ allows greater insight system performance reducing labor required letting critical engineering staff focus valueadded activity paper introduces leto™ discusses capability actual use case concludes next step"


# # Test vanilla retriever
# retrieved_papers, metadatas = vanilla_retriever.top_k(query, top_k = 3000)
# position = get_paper_position(metadatas, 1035)
# print(position)

# # Test bm25 retriever
# retrieved_papers, metadatas = bm25_retriever.top_k(query, top_k = 3000)
# position = get_paper_position(metadatas, 1035)
# print(position)

docs, scores, metadatas = retriever.retrieve(query, k = 3000)
position = get_paper_position(metadatas, 1035)
print(position)