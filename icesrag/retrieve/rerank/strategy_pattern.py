from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class ReRankStrategy(ABC):
    """
    Strategy interface for 

    Methods:
        rerank
    """

    @abstractmethod
    def rerank(self, rankings: Dict[str, Dict[str, int]], **kwargs) -> Dict[str, int]:
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
        pass

class ReRankEngine:
    """
    The ReRankEngine uses a strategy to rerank all chunks.
    """

    def __init__(self, strategy: ReRankStrategy):
        """
        Initializes the engine with a given strategy. The strategy will define how the engine interacts with the vector database.
        
        Args:
            strategy (ReRankStrategy): The strategy to use for reranking chunks.
        """
        self.strategy = strategy

    def rerank(self, rankings: Dict[str, Dict[str, int]], **kwargs) -> Dict[str, int]:
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
        return self.strategy.rerank(rankings, **kwargs)