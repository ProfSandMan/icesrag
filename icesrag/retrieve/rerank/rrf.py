from typing import Dict

from icesrag.retrieve.rerank.strategy_pattern import ReRankStrategy


class ReciprocalRerankFusion(ReRankStrategy):
    def __init__(self):
        pass

    def rerank(self, rankings: Dict[Dict[str:int]], **kwargs) -> Dict[str:int]:
        """
        Reranks all chunks across all strategies.
        
        Args:
            rankings (Dict[Dict[str:int]]): A dictionary with keys for each strategy title and values of dictionaries with keys of chunk id and values of rankings
                ex: 
                    {
                    'strategy1':
                        {
                            chunk_id1:rank, 
                            chunk_id2:rank
                        },
                     'strategy2':
                        {
                            chunk_id1:rank, 
                            chunk_id2:rank
                        }
                    }
        Returns:
            Dict[str:int]: dictionary of the chunk ids and their reranked rankings
        """
        #
        #
        # @ Hunter return here
        #
        # note: don't forget to handle rank ids that only occur in one of the strategies




# def reciprocal_rank_fusion(rankings, k=60):
#     """
#     Implements the Reciprocal Rank Fusion algorithm.

#     Parameters:
#         rankings: A list of lists, where each inner list represents a ranking of items.
#         k: A constant used in the RRF formula (default is 60).

#     Returns:
#         A dictionary where keys are items and values are their RRF scores.
#     """

#     scores = {}
#     for ranking in rankings:
#         for i, item in enumerate(ranking):
#             if item not in scores:
#                 scores[item] = 0
#             scores[item] += 1 / (k + i + 1)

#     return scores