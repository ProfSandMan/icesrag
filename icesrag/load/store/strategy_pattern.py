from abc import ABC, abstractmethod
from typing import Any, Dict

# ----------------------------------------------
# Database Strategy Interface
# ----------------------------------------------

class DatabaseStrategy(ABC):
    """
    Abstract base class for interacting with different databases.
    
    Concrete strategies should implement the methods to connect, delete, and add embeddings to the database.
    """

    @abstractmethod
    def connect(self, dbpath: str, collection_name: str, **kwargs: Any) -> None:
        """
        Connect to a database collection.
        
        Args:
            dbpath (str): The location of the database
            collection_name (str): The name of the collection to connect to.
            **kwargs: Additional parameters (e.g., authentication, host, etc.) for the connection.

        Returns:
            None
        """
        pass

    @abstractmethod
    def delete(self, collection_name: str, **kwargs: Any) -> None:
        """
        Delete a database collection.
        
        Args:
            collection_name (str): The name of the collection to delete.
            **kwargs: Additional parameters for the deletion (e.g., force delete flag).
        
        Returns:
            None
        """
        pass

    @abstractmethod
    def add(self, data: Dict[str, Any], **kwargs: Any) -> None:
        """
        Add data to the database collection.
        
        Args:
            data (dict): A dictionary where keys represent identifiers and values are the embeddings.
            **kwargs: Additional parameters for adding embeddings (e.g., metadata, timestamp).
        
        Returns:
            None
        """
        pass

class DatabaseEngine:
    """
    Engine that uses a specific database strategy for managing collections.
    
    This class allows switching strategies at runtime and delegates the tasks of connecting, deleting, and adding embeddings
    to the strategy currently in use.
    """

    def __init__(self, strategy: DatabaseStrategy):
        """
        Initialize the engine with a specific database strategy.

        Args:
            strategy (DatabaseStrategy): The strategy to use for interacting with the database (e.g., FAISS).
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DatabaseStrategy):
        """
        Set or switch the strategy at runtime.

        Args:
            strategy (DatabaseStrategy): The new database strategy to use.
        """
        self._strategy = strategy

    def connect(self, dbpath: str, collection_name: str, **kwargs: Any) -> None:
        """
        Connect to a database collection using the current strategy.
        
        Args:
            dbpath (str): The location of the database
            collection_name (str): The name of the collection to connect to.
            **kwargs: Additional parameters for the connection.

        Returns:
            None
        """
        self._strategy.connect(dbpath = dbpath, collection_name=collection_name, **kwargs)

    def delete(self, collection_name: str, **kwargs: Any) -> None:
        """
        Delete a database collection using the current strategy.
        
        Args:
            collection_name (str): The name of the collection to delete.
            **kwargs: Additional parameters for the deletion.

        Returns:
            None
        """
        self._strategy.delete(collection_name=collection_name, **kwargs)

    def add(self, data: Dict[str, Any], **kwargs: Any) -> None:
        """
        Add embeddings to the database using the current strategy.
        
        Args:
            data (dict): A dictionary of data to be added to the collection.

        Returns:
            None
        """
        self._strategy.add(data=data, **kwargs)