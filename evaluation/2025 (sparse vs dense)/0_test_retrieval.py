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

# * Leto
paper_id =  1035
sf = ['Leto spaceflight intelligence system', 
      'AI-based monitoring of Environmental Control and Life Support Systems', 
      'Collins Aerospace ECLSS performance monitoring']
lf = ["I'm researching advancements in spaceflight intelligence systems specifically focusing on Leto™ developed by Collins Aerospace. How does this system utilize AI and physics-based models to monitor Environmental Control and Life Support Systems (ECLSS), and what benefits are derived from its integration with existing systems?", 
      "As part of my study into optimizing operations within the commercial space industry, I'm looking for insights on workforce management through advanced monitoring technologies. How does Leto™ streamline ground support operations and reduce labor demands while improving monitoring of a vehicle's ECLSS performance?", 
      'I am investigating how artificial intelligence can enhance the monitoring of Environmental Control and Life Support Systems in space missions. Could you provide examples of the capabilities of Leto™ in terms of anomaly detection and performance metrics, and how it has been validated through use cases in actual spaceflight scenarios?']

# * Predict component life
paper_id =  2744
sf = ['commercialization of space low earth orbit destinations', 
      'hardware reliability analysis method for space systems', 
      'impact of component life predictions on space program design']
lf = ["Can you provide insights into how the commercialization of space is transforming design strategies for systems used in Commercial Low Earth Orbit (LEO) destinations? Specifically, I'd like to understand the roles of hardware reliability and predicted life in shaping non-recurring and sustaining engineering efforts.", 
    "I'm researching methods for predicting component life and its implications for hardware reliability in space systems. How does Collins Aerospace's analysis methodology influence initial design decisions and sustaining engineering plans for products in the Commercial Low Earth Orbit market?", 
    "What are the critical factors and methodologies for ensuring hardware reliability in the design of space systems, particularly in the context of the Commercial Low Earth Orbit destination market? I am particularly interested in the application of analysis methods to predict life values and their effects on program performance and design metrics, as illustrated in the case study on the Water Processor Assembly."]

# * Caving on the moon
paper_id =  2605
sf = ['caves and pits on the Moon and Mars', 
      'EVA systems for cave exploration on the Moon', 
      'habitats in planetary caves for astrobiology']
lf = ['What are the environmental benefits of exploring caves and pits on the Moon and Mars, particularly in relation to astrobiological research and safeguarding astronauts from surface conditions?', 
      'I am conducting research on the exploration of lunar and Martian caves and pits, focusing on the requirements for EVA systems and operations that ensure safety and efficiency during these explorations. What were the key findings from the recent field tests at Skylight Cave regarding spacesuit design and operational recommendations?', 
      'My work addresses the potential scientific value of lunar and planetary caves as sheltered environments for life. Can you provide insights into the field studies conducted to assess EVA system enhancements for cave exploration, particularly in terms of features like protective garments and visibility improvements?']


sf_res_vanilla = []
sf_res_bm25 = []
sf_res_retriever = []

lf_res_vanilla = []
lf_res_bm25 = []
lf_res_retriever = []

for query in sf:
    # Test vanilla retriever
    retrieved_papers, distances, metadatas = vanilla_retriever.top_k(query, top_k = 3000)
    position = get_paper_position(metadatas, paper_id)
    sf_res_vanilla.append(position)

    # Test bm25 retriever
    retrieved_papers, distances, metadatas = bm25_retriever.top_k(query, top_k = 3000)
    position = get_paper_position(metadatas, paper_id)
    sf_res_bm25.append(position)

    docs, scores, metadatas = retriever.top_k(query, k = 3000)
    position = get_paper_position(metadatas, paper_id)
    sf_res_retriever.append(position)

for query in lf:
    # Test vanilla retriever
    retrieved_papers, distances, metadatas = vanilla_retriever.top_k(query, top_k = 3000)
    position = get_paper_position(metadatas, paper_id)
    lf_res_vanilla.append(position)

    # Test bm25 retriever
    retrieved_papers, distances, metadatas = bm25_retriever.top_k(query, top_k = 3000)
    position = get_paper_position(metadatas, paper_id)
    lf_res_bm25.append(position)

    docs, scores, metadatas = retriever.top_k(query, k = 3000)
    position = get_paper_position(metadatas, paper_id)
    lf_res_retriever.append(position)


print("Short Form Vanilla: ", sf_res_vanilla)
print("Short Form BM25: ", sf_res_bm25)
print("Short Form Retriever: ", sf_res_retriever)

print("Long Form Vanilla: ", lf_res_vanilla)
print("Long Form BM25: ", lf_res_bm25)
print("Long Form Retriever: ", lf_res_retriever)