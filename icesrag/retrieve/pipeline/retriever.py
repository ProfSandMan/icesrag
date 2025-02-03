from typing import Dict, List, Optional

from icesrag.retrieve.rerank.strategy_pattern import ReRankEngine
from icesrag.utils.embed.strategy_pattern import EmbeddingEngine
from icesrag.utils.text_preprocess.strategy_pattern import \
    TextPreprocessingEngine


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
        self.strategies_ = strategies
        self.reranker_ = reranker
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Retrieves the (top k) results for a user-input query

        Args:
            query (str): the user's query
            k (Optionla[int]): the desired number of returned chunks
        
        returns:
            List[Dict]: List (in descending order) of dictionaries corresponding to the resultant query
                Dictionary has the following form:
                    {'document':document,
                     'document_id':document_id,
                     'fusion_socre':score,
                     'metadatas':metadatas}  
        """
        results = {}
        fuse_results = {}

        # Perform initial preprocess/embed/retrieval by strategy
        for strategy in self.strategies_:
            name = strategy['name']
            temp_query = query

            if 'preprocess' in strategy.keys():
                preprocesser = strategy['preprocess']
                if isinstance(preprocesser, TextPreprocessingEngine):
                    temp_query = preprocesser.preprocess(temp_query)

            if 'embed' in strategy.keys():
                embedder = strategy['embed']
                if isinstance(embedder, EmbeddingEngine):
                    temp_query = embedder.process(temp_query)

            res = strategy['retriever'].rank_all(temp_query)
            results[name] = res

            # Build fusion-ready dictionary
            rank_d = {}
            for i, chunk_id in enumerate(res['document_ids']):
                rank_d[chunk_id] = res['rankings'][i]
            fuse_results[name] = rank_d

        # Fuse
        fusion_rank = self.reranker_.rerank(fuse_results)

        # Identify top k chunk ids from fusion
        final_results = []
        if k > len(fusion_rank):
            k = len(fusion_rank)
        fused_ids = list(fusion_rank.keys())
        for i in range(0, k):
            temp = {}
            id = fused_ids[i]
            temp['document_id'] = id
            temp['fusion_score'] = fusion_rank[id]
            for result in results.keys():
                result = results[result]
                chunk_pos = result['document_ids'].index(id)
                if 'document' not in temp.keys():
                    temp['document'] = result['documents'][chunk_pos]
                if 'metadata' not in temp.keys():
                    temp['metadata'] = result['metadatas'][chunk_pos]
                else: # melt metadata (each strategy may contain different metadata tags)
                    # * note: if similar keys exists, only the first strategy keys are retained
                    temp['metadata'] = result['metadatas'][chunk_pos] | temp['metadata']
            final_results.append(temp)
        
        return final_results