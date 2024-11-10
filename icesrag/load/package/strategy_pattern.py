from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Any

class PackageStrategy(ABC):
    """
    Abstract base class for packaging documents, embeddings, and metadata.
    
    Concrete strategies should implement the methods for creating unique IDs, 
    packaging single items, and packaging batches of items.
    """

    @abstractmethod
    def create_ids(self, tag: Optional[str], chunks: Union[str, List[str]]) -> List[str]:
        """
        Create unique IDs for one or more document chunks.
        
        Args:
            tag (Optional[str]): An optional tag to prepend to the IDs.
            chunks (Union[str, List[str]]): A single chunk or a list of chunks for which to generate IDs.
        
        Returns:
            List[str]: A list of generated unique IDs.
        """
        pass

    @abstractmethod
    def package(self, 
                chunk: str, 
                metadata: Dict, 
                id: str, 
                embedding: Optional[List[float]],
                tag: Optional[str] = None) -> Any:
        """
        Package a single chunk of text along with its embedding, metadata, and ID.
        
        Args:
            chunk (str): The chunk of text.
            metadata (Dict): Metadata associated with the chunk.
            id (str): The unique ID for the chunk.
            embedding (List[float]): The embedding associated with the chunk.
            tag (Optional[str]): An optional tag to prepend to the IDs.
        
        Returns:
            Dict: A dictionary containing the chunk, embedding, metadata, and ID.
        """
        pass

    @abstractmethod
    def batch_package(self, chunks: List[str], 
                      metadatas: List[Dict], 
                      ids: List[str], 
                      embeddings: Optional[List[List[float]]],
                      tag: Optional[List[str]] = None) -> Any:
        """
        Package a batch of chunks, embeddings, metadata, and IDs.
        
        Args:
            chunks (List[str]): A list of document chunks.
            metadatas (List[Dict]): A list of metadata dictionaries, each corresponding to a chunk.
            ids (List[str]): A list of IDs for the chunks.
            embeddings (Optional[List[List[float]]]): A list of embeddings, each corresponding to a chunk.
            tag (Optional[List[str]]): An optional tag to prepend to the IDs.

        Returns:
            List[Dict]: A list of dictionaries, each containing a chunk, embedding, metadata, and ID.
        """
        pass

class PackageEngine:
    """
    Engine that uses a specific packaging strategy to package documents, embeddings, and metadata.
    
    This class allows switching strategies at runtime and delegates the tasks of creating IDs,
    packaging single items, and packaging batches to the strategy currently in use.
    """

    def __init__(self, strategy: PackageStrategy):
        """
        Initialize the engine with a specific document packaging strategy.

        Args:
            strategy (DocumentPackagingStrategy): The strategy to use for packaging documents (e.g., SimpleDocumentPackagingStrategy).
        """
        self._strategy = strategy

    def set_strategy(self, strategy: PackageStrategy):
        """
        Set or switch the strategy at runtime.

        Args:
            strategy (DocumentPackagingStrategy): The new document packaging strategy to use.
        """
        self._strategy = strategy

    def create_ids(self, tag: Optional[str], chunks: Union[str, List[str]]) -> List[str]:
        """
        Create unique IDs for document chunks using the current strategy.
        
        Args:
            tag (Optional[str]): An optional tag to prepend to the IDs.
            chunks (Union[str, List[str]]): A single chunk or a list of chunks for which to generate IDs.
        
        Returns:
            List[str]: A list of generated unique IDs.
        """
        return self._strategy.create_ids(tag=tag, chunks=chunks)

    def package(self, 
                chunk: str, 
                metadata: Dict, 
                id: str, 
                embedding: Optional[List[float]],
                tag: Optional[str] = None) -> Any:
        """
        Package a single chunk of text along with its embedding, metadata, and ID.
        
        Args:
            chunk (str): The chunk of text.
            metadata (Dict): Metadata associated with the chunk.
            id (str): The unique ID for the chunk.
            embedding (List[float]): The embedding associated with the chunk.
            tag (Optional[str]): An optional tag to prepend to the IDs.
        
        Returns:
            Dict: A dictionary containing the chunk, embedding, metadata, and ID.
        """
        return self._strategy.package(chunk=chunk, 
                                      metadata=metadata,
                                      id = id,
                                      embedding=embedding,
                                      tag = tag)

    def batch_package(self, 
                      chunks: List[str], 
                      metadatas: List[Dict], 
                      ids: List[str], 
                      embeddings: Optional[List[List[float]]],
                      tag: Optional[List[str]] = None) -> Any:
        """
        Package a batch of chunks, embeddings, metadata, and IDs.
        
        Args:
            chunks (List[str]): A list of document chunks.
            metadatas (List[Dict]): A list of metadata dictionaries, each corresponding to a chunk.
            ids (List[str]): A list of IDs for the chunks.
            embeddings (Optional[List[List[float]]]): A list of embeddings, each corresponding to a chunk.
            tag (Optional[List[str]]): An optional tag to prepend to the IDs.

        Returns:
            List[Dict]: A list of dictionaries, each containing a chunk, embedding, metadata, and ID.
        """
        return self._strategy.batch_package(chunks=chunks, 
                                            metadatas=metadatas,
                                            ids = ids,
                                            embedding=embeddings,
                                            tag=tag)