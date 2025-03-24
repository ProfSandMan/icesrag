from typing import Dict, List, Tuple, Union
import logging

import chromadb
import numpy as np

from icesrag.retrieve.retrievers.strategy_pattern import RetrieverStrategy

logger = logging.getLogger(__name__)

class ChromaRetriever(RetrieverStrategy):
    """
    Concrete implementation of the VectorDatabaseStrategy interface using ChromaDB as the vector database.
    """

    def __init__(self):
        """
        Initializes the ChromaDBRetriever instance. The client and collection are initially set to None.
        """
        logger.info("Initializing ChromaRetriever")
        self.client = None
        self.collection = None

    def connect(self, dbpath: str, collection_name: str, **kwargs) -> None:
        """
        Connects to the ChromaDB instance, creating or opening the specified collection.

        Args:
            dbpath (str): Path to the directory where ChromaDB data is persisted.
            collection_name (str): The name of the collection to retrieve or create.
            **kwargs: Additional keyword arguments passed to the ChromaDB client (if needed).
        """
        logger.info(f"Connecting to ChromaDB at {dbpath}")
        self.client = chromadb.PersistentClient(dbpath)
        logger.debug("Successfully created ChromaDB client")
        
        logger.debug(f"Getting collection {collection_name}")
        self.collection = self.client.get_collection(name=collection_name)
        logger.info(f"Successfully connected to collection {collection_name}")

    def top_k(self, query: Union[str, List[float]], top_k: int, **kwargs) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Retrieves the top K most relevant documents for a given query.

        Args:
            query (Union[str, List[float]]): The query string or embeddings to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments that may be used for customizing the search.

        Returns:
            Tuple:
                - A list of document (strings) representing the top K results.
                - A list of distances for each of the top K results.
                - A list of metadata dictionaries for each of the top K results.
        """
        logger.info(f"Retrieving top {top_k} results for query")
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.") 
        
        query_type = "text" if isinstance(query, str) else "embedding"
        logger.debug(f"Query type: {query_type}")
        
        if isinstance(query, str):
            results = self.collection.query(query_texts=[query], n_results=top_k, **kwargs)
        else:
            results = self.collection.query(query_embeddings=[query], n_results=top_k, **kwargs)
            
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        logger.info(f"Successfully retrieved {len(documents)} results")
        return documents, distances, metadatas

    def rank_all(self, query: Union[str, List[float]], **kwargs) -> Dict:
        """
        Ranks all documents based on their relevance to a given query.

        Args:
            query (Union[str, List[float]]): The query string or embeddings to search for.
            **kwargs: Additional arguments for customizing the ranking.
        
        Returns:
            Form of:
                {'documents':documents,
                'document_ids':document_ids,
                'rankings':rankings,
                'metadatas':metadatas}            
        """
        logger.info("Ranking all documents")
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")  
              
        total_k = self.collection.count()
        logger.debug(f"Total documents in collection: {total_k}")
        
        query_type = "text" if isinstance(query, str) else "embedding"
        logger.debug(f"Query type: {query_type}")
        
        if isinstance(query, str):
            results = self.collection.query(query_texts=[query], n_results=total_k, **kwargs)
        else:
            results = self.collection.query(query_embeddings=[query], n_results=total_k, **kwargs)
            
        documents = results['documents'][0]
        document_ids = results['ids'][0]
        rankings = [i + 1 for i in range(0, len(document_ids))] 
        metadatas = results['metadatas'][0]
        
        d = {'documents':documents,
             'document_ids':document_ids,
             'rankings':rankings,
             'metadatas':metadatas}
        logger.info(f"Successfully ranked {len(documents)} documents")
        return d