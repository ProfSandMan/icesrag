from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class RetrieverStrategy(ABC):
    """
    Strategy interface for interacting with a vector database. Implementations of this interface
    should provide methods for connecting to the database, retrieving the top K results, and ranking all results.

    Methods:
        connect(dbpath: str, collection_name: str, **kwargs) -> None:
            Establishes a connection to the vector database.
        
        top_k(query: str, top_k: int, **kwargs) -> Tuple[List[str], List[Dict]]:
            Retrieves the top k results for a given query.
        
        rank_all(query: str, **kwargs) -> Tuple[List[str], List[int], List[Dict]]:
            Ranks all results for a given query.
    """

    @abstractmethod
    def connect(self, dbpath: str, collection_name: str, **kwargs) -> None:
        """Establish a connection to the vector database."""
        pass
    
    @abstractmethod
    def top_k(self, query: str, top_k: int, **kwargs) -> Tuple[List[str], List[float],List[Dict]]:
        """Retrieve the top K results for a given query."""
        pass
    
    @abstractmethod
    def rank_all(self, query: str, **kwargs) -> Dict:
        """Rank all results for a given query."""
        pass

class RetrieverEngine:
    """
    The VectorDatabaseRetrieverEngine uses a strategy to interact with a vector database.
    It delegates operations like connect, top_k, and rank_all to the strategy class.
    """

    def __init__(self, strategy: RetrieverStrategy):
        """
        Initializes the engine with a given strategy. The strategy will define how the engine interacts with the vector database.
        
        Args:
            strategy (VectorDatabaseStrategy): The strategy to use for connecting and querying the vector database.
        """
        self.strategy = strategy

    def connect(self, dbpath: str, collection_name: str, **kwargs) -> None:
        """
        Connects to the vector database using the strategy's connect method.
        
        Args:
            dbpath (str): The path to the database storage.
            collection_name (str): The name of the collection to interact with.
            **kwargs: Additional arguments to pass to the connect method.
        """
        self.strategy.connect(dbpath, collection_name, **kwargs)

    def top_k(self, query: str, top_k: int, **kwargs) -> Tuple[List[str], List[float],List[Dict]]:
        """
        Retrieves the top k most relevant documents for a query using the strategy's top_k method.
        
        Args:
            query (str): The query string.
            top_k (int): The number of results to retrieve.
            **kwargs: Additional arguments for customizing the search.
        
        Returns:
            Tuple of:
                - List of document
                - List of distances
                - List of metadata dictionaries
        """
        return self.strategy.top_k(query, top_k, **kwargs)

    def rank_all(self, query: str, **kwargs) -> Dict:
        """
        Ranks all documents based on the query using the strategy's rank_all method.
        
        Args:
            query (str): The query string to rank documents by.
            **kwargs: Additional arguments for customizing the ranking.
        
        Returns:
            Form of:
                {'documents':documents,
                'document_ids':document_ids,
                'rankings':rankings,
                'metadatas':metadatas}                   
        """
        return self.strategy.rank_all(query, **kwargs)