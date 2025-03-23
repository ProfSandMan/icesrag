from typing import Dict, List, Optional, Tuple
import logging

from icesrag.retrieve.rerank.strategy_pattern import ReRankEngine
from icesrag.utils.embed.strategy_pattern import EmbeddingEngine
from icesrag.utils.text_preprocess.strategy_pattern import \
    TextPreprocessingEngine

logger = logging.getLogger(__name__)

class CompositeRetriever():
    def __init__(self, strategies: List[Dict], reranker: ReRankEngine):
        """
        The CompositeRetriever acts as a pipeline for multi-strategy retrieval

        Args:
            strategies (List[Dict]): a dictionary with the following key:value pairs:
                a. 'name': str
                b. 'preprocess': TextPreprocessingEngine (Optional)
                c. 'embedder': EmbeddingEngine (Optional)
                d. 'retrieve': RetrieverEngine
            reranker (ReRankEngine): the methodology to rerank and fuse retrieved chunks

        """
        logger.info(f"Initializing CompositeRetriever with {len(strategies)} strategies")
        self.strategies_ = strategies
        self.reranker_ = reranker
        for i, strategy in enumerate(strategies):
            logger.debug(f"Strategy {i+1}: {strategy.get('name', 'unnamed')}")
    
    def retrieve(self, query: str, k: Optional[int] = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Retrieves the (top k) results for a user-input query

        Args:
            query (str): the user's query
            k (Optional[int]): the desired number of returned chunks
        
        returns:
            Tuple[List[str], List[Dict], List[Dict]]: List (in descending order) of dictionaries corresponding to the resultant query
            List[str]: List of documents
            List[float]: List of fusion scores
            List[Dict]: List of metadata
        """
        logger.info(f"Starting retrieval for query: {query[:50]}...")
        results = {}
        fuse_results = {}

        # Perform initial preprocess/embed/retrieval by strategy
        for i, strategy in enumerate(self.strategies_):
            name = strategy['name']
            logger.info(f"Processing strategy {i+1}/{len(self.strategies_)}: {name}")
            temp_query = query

            if 'preprocess' in strategy.keys():
                logger.debug(f"Applying preprocessing for strategy {name}")
                preprocesser = strategy['preprocess']
                if isinstance(preprocesser, TextPreprocessingEngine):
                    temp_query = preprocesser.preprocess(temp_query)
                    logger.debug("Preprocessing complete")

            if 'embed' in strategy.keys():
                logger.debug(f"Generating embeddings for strategy {name}")
                embedder = strategy['embed']
                if isinstance(embedder, EmbeddingEngine):
                    temp_query = embedder.process(temp_query)
                    logger.debug("Embedding generation complete")

            logger.debug(f"Performing retrieval for strategy {name}")
            res = strategy['retriever'].rank_all(temp_query)
            results[name] = res
            logger.debug(f"Retrieved {len(res['document_ids'])} documents for strategy {name}")

            # Build fusion-ready dictionary
            rank_d = {}
            for i, chunk_id in enumerate(res['document_ids']):
                rank_d[chunk_id] = res['rankings'][i]
            fuse_results[name] = rank_d
            logger.debug(f"Built fusion dictionary for strategy {name}")

        # Fuse
        logger.info("Starting fusion of results from all strategies")
        fusion_rank = self.reranker_.rerank(fuse_results)
        logger.debug(f"Fusion complete, generated {len(fusion_rank)} fused results")

        # Identify top k chunk ids from fusion
        logger.info(f"Selecting top {k} results from fusion")
        if k > len(fusion_rank):
            k = len(fusion_rank)
            logger.debug(f"Adjusted k to {k} as it exceeded available results")
        fused_ids = list(fusion_rank.keys())
        
        docs = []
        scores = []
        metadatas = []
        for i in range(0, k):
            logger.debug(f"Processing result {i+1}/{k}")
            temp = {}
            id = fused_ids[i]
            scores.append(fusion_rank[id])
            for result in results.keys():
                result = results[result]
                try:
                    chunk_pos = result['document_ids'].index(id)
                    if 'document' not in temp.keys():
                        docs.append(result['documents'][chunk_pos])
                    if 'metadata' not in temp.keys():
                        metadatas.append(result['metadatas'][chunk_pos])
                    else: # melt metadata (each strategy may contain different metadata tags)
                        # * note: if similar keys exists, only the first strategy keys are retained
                        metadatas.append(result['metadatas'][chunk_pos] | temp['metadata'])
                except ValueError:
                    logger.warning(f"Document ID {id} not found in results from fusion rank")
        
        logger.info(f"Retrieval complete, returning {len(docs)} results")
        return docs, scores, metadatas