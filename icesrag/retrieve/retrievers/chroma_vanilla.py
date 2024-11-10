import chromadb
from typing import List, Dict, Tuple
from icesrag.retrieve.retrievers.strategy_pattern import RetrieverStrategy

class ChromaDBRetriever(RetrieverStrategy):
    """
    Concrete implementation of the VectorDatabaseStrategy interface using ChromaDB as the vector database.
    """

    def __init__(self):
        """
        Initializes the ChromaDBRetriever instance. The client and collection are initially set to None.
        """
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
        self.client = chromadb.Client(persist_directory=dbpath)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def top_k(self, query: str, top_k: int, **kwargs) -> Tuple[List[str], List[Dict]]:
        """
        Retrieves the top K most relevant documents for a given query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top results to retrieve.
            **kwargs: Additional keyword arguments that may be used for customizing the search.

        Returns:
            Tuple:
                - A list of document (strings) representing the top K results.
                - A list of metadata dictionaries for each of the top K results.
        """
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")        
        results = self.collection.query(query_texts=[query], n_results=top_k, **kwargs)
        documents = results['documents']
        metadatas = results['metadatas']
        return documents, metadatas

    def rank_all(self, query: str, **kwargs) -> Dict[List[str], List[str], List[int], List[Dict]]:
        """
        Ranks all documents based on their relevance to a given query.

        Args:
            query (str): The query string to rank documents by.
            **kwargs: Additional arguments for customizing the ranking.
        
        Returns:
            Dictionary of:
                - List of documents.
                - List of document IDs.
                - List of rankings (or scores).
                - List of metadata dictionaries.
            Form of:
                {'documents':documents,
                'document_ids':document_ids,
                'rankings':rankings,
                'metadatas':metadatas}            
        """
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")        
        total_k = self.collection.count()
        results = self.collection.query(query_texts=[query], n_results=total_k, **kwargs)
        documents = results['documents']
        document_ids = results['ids']
        rankings = [i for i in range(len(document_ids))]
        metadatas = results['metadatas']
        d = {'documents':documents,
             'document_ids':document_ids,
             'rankings':rankings,
             'metadatas':metadatas}
        return d