from abc import ABC, abstractmethod
from typing import Dict, Any


# ----------------------------------------------
# Vector Database Strategy Interface
# ----------------------------------------------

class VectorDatabaseStrategy(ABC):
    """
    Abstract base class for interacting with different vector databases.
    
    Concrete strategies should implement the methods to connect, delete, and add embeddings to the database.
    """

    @abstractmethod
    def connect(self, dbpath: str, collection_name: str, **kwargs: Any) -> None:
        """
        Connect to a vector database collection.
        
        Args:
            dbpath (str): The location of the vector database
            collection_name (str): The name of the collection to connect to.
            **kwargs: Additional parameters (e.g., authentication, host, etc.) for the connection.

        Returns:
            None
        """
        pass

    @abstractmethod
    def delete(self, collection_name: str, **kwargs: Any) -> None:
        """
        Delete a vector database collection.
        
        Args:
            collection_name (str): The name of the collection to delete.
            **kwargs: Additional parameters for the deletion (e.g., force delete flag).
        
        Returns:
            None
        """
        pass

    @abstractmethod
    def add(self, embeddings: Dict[str, Any], **kwargs: Any) -> None:
        """
        Add embeddings to the vector database collection.
        
        Args:
            embeddings (dict): A dictionary where keys represent identifiers and values are the embeddings.
            **kwargs: Additional parameters for adding embeddings (e.g., metadata, timestamp).
        
        Returns:
            None
        """
        pass

class VectorDatabaseEngine:
    """
    Engine that uses a specific vector database strategy for managing collections.
    
    This class allows switching strategies at runtime and delegates the tasks of connecting, deleting, and adding embeddings
    to the strategy currently in use.
    """

    def __init__(self, strategy: VectorDatabaseStrategy):
        """
        Initialize the engine with a specific vector database strategy.

        Args:
            strategy (VectorDatabaseStrategy): The strategy to use for interacting with the vector database (e.g., FAISS).
        """
        self._strategy = strategy

    def set_strategy(self, strategy: VectorDatabaseStrategy):
        """
        Set or switch the strategy at runtime.

        Args:
            strategy (VectorDatabaseStrategy): The new vector database strategy to use.
        """
        self._strategy = strategy

    def connect(self, dbpath: str, collection_name: str, **kwargs: Any) -> None:
        """
        Connect to a vector database collection using the current strategy.
        
        Args:
            dbpath (str): The location of the vector database
            collection_name (str): The name of the collection to connect to.
            **kwargs: Additional parameters for the connection.

        Returns:
            None
        """
        self._strategy.connect(dbpath = dbpath, collection_name=collection_name, **kwargs)

    def delete(self, collection_name: str, **kwargs: Any) -> None:
        """
        Delete a vector database collection using the current strategy.
        
        Args:
            collection_name (str): The name of the collection to delete.
            **kwargs: Additional parameters for the deletion.

        Returns:
            None
        """
        self._strategy.delete(collection_name=collection_name, **kwargs)

    def add(self, embeddings: Dict[str, Any], **kwargs: Any) -> None:
        """
        Add embeddings to the vector database using the current strategy.
        
        Args:
            embeddings (dict): A dictionary of embeddings to be added to the collection.
            **kwargs: Additional parameters for adding embeddings.

        Returns:
            None
        """
        self._strategy.add(embeddings=embeddings, **kwargs)