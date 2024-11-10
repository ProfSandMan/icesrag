import chromadb
from typing import List, Dict, Tuple
from icesrag.retrieve.retrievers.strategy_pattern import RetrieverStrategy
from icesrag.utils.text_preprocess.strategy_pattern import TextPreprocessingEngine
from rank_bm25 import BM25Okapi

def reverse_rank_list(nums):
    # Create a sorted version of the original list (descending order) along with the original indices
    sorted_nums = sorted(enumerate(nums), key=lambda x: x[1], reverse=True)

    # Create a list to hold the ranks
    ranks = [0] * len(nums)

    # Assign reverse ranks
    current_rank = 1
    for i in range(len(sorted_nums)):
        # If it's not the first item and the current value is equal to the previous one,
        # we assign it the same rank
        if i > 0 and sorted_nums[i][1] == sorted_nums[i - 1][1]:
            ranks[sorted_nums[i][0]] = ranks[sorted_nums[i - 1][0]]
        else:
            ranks[sorted_nums[i][0]] = current_rank
        
        current_rank += 1

    return ranks

class BM25Retriever(RetrieverStrategy):
    """
    Concrete implementation of the VectorDatabaseStrategy interface using ChromaDB as the vector database.
    """

    def __init__(self, preprocessor:TextPreprocessingEngine):
        """
        Initializes the ChromaDBRetriever instance. The client and collection are initially set to None.
        """
        self.client = None
        self.collection = None
        self.preprocessor = preprocessor

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

        Notes:
            Requires 'bm25' to be a keyword in metadata

        Returns:
            Tuple:
                - A list of document (strings) representing the top K results.
                - A list of metadata dictionaries for each of the top K results.
        """
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")   

        # Gather all 
        total_k = self.collection.count()
            # can probably just do a straight query (faster) than having it calc distances and return)
        results = self.collection.query(query_texts=[query], n_results=total_k, **kwargs)        
        documents = results['documents']
        metadatas = results['metadatas']
        bm25_docs = [m['bm25'] for m in metadatas]

        # Process query
        query = self.preprocessor.preprocess(query)

        # Initialize BM25 with the tokenized documents
        bm25 = BM25Okapi(bm25_docs)

        # Get the BM25 scores for the query across all documents
        scores = bm25.get_scores(query)

        # Get indices of the top-k documents
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Get best documents and metadata
        bestdocs = []
        bestmeta = []
        for i in top_k_indices:
            bestdocs.append(documents[i])
            bestmeta.append(metadatas[i])

        return bestdocs, bestmeta

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

        # Gather all 
        total_k = self.collection.count()
            # can probably just do a straight query (faster) than having it calc distances and return)        
        results = self.collection.query(query_texts=[query], n_results=total_k, **kwargs)        
        documents = results['documents']
        metadatas = results['metadatas']
        document_ids = results['ids']
        bm25_docs = [m['bm25'] for m in metadatas]

        # Process query
        query = self.preprocessor.preprocess(query)

        # Initialize BM25 with the tokenized documents
        bm25 = BM25Okapi(bm25_docs)

        # Get the BM25 scores for the query across all documents
        scores = bm25.get_scores(query)

        # Get indices of the top-k documents
        rankings = reverse_rank_list(scores)
        # base 0 to stay consistent w/ vanilla
        rankings = [r-1 for r in rankings]

        d = {'documents':documents,
             'document_ids':document_ids,
             'rankings':rankings,
             'metadatas':metadatas}
        return d