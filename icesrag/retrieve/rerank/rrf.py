from typing import Dict, List

from icesrag.retrieve.rerank.strategy_pattern import ReRankStrategy


def _rrf(ranking: List[int], k=60) -> int:
    """
    Implements the Reciprocal Rank Fusion algorithm.

    Parameters:
        rankings: A list of lists, where each inner list represents a ranking of items.
        k: A constant used in the RRF formula (default is 60).

    Returns:
        A dictionary where keys are items and values are their RRF scores.
    """
    final_rank = 0
    for rank in ranking:
        final_rank += 1 / (k + rank + 1)
    return final_rank

class ReciprocalRerankFusion(ReRankStrategy):
    def __init__(self):
        pass

    def rerank(self, rankings: Dict[str, Dict[str, int]], **kwargs) -> Dict[str, float]:
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
            Dict[str:float]: dictionary of the chunk ids and their reranked rankings

        """
        n_strategies = len(rankings.keys())
        final_d = {}
        max_rank = 0

        # Assemble rankings
        for strategy in rankings:
            strategy = rankings[strategy]
            for chunk in strategy.keys():
                rank = strategy[chunk]
                if rank > max_rank:
                    max_rank = rank 
                if chunk in final_d.keys():
                    curr = final_d[chunk]
                    curr.append(rank)
                    final_d[chunk] = curr
                else:
                    final_d[chunk] = [rank]

        # Perform ranking
        for chunk in final_d.keys():
            k = len(final_d[chunk])
            # Ensure all chunks have equal representation
            if k < n_strategies:
                curr = final_d[chunk]
                delta = n_strategies - k
                for i in range(0, delta):
                    curr.append(max_rank + 1) # use maximum rank found and add one for missing representation (conservative)
                final_d[chunk] = curr
            # RRF ranking
            final_d[chunk] = _rrf(final_d[chunk])

        # Sort (highest score to lowest)
        final_d = dict(sorted(final_d.items(), key=lambda item: item[1], reverse=True))

        return final_d