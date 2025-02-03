import json
import sqlite3
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from icesrag.retrieve.retrievers.strategy_pattern import RetrieverStrategy


def _retrieve_and_rank(query: str, collection_name: str, client: Any) -> pd.DataFrame:
    """
    Retrieves and ranks all data within the the SQLite db

    Args:
        query (str): The query string to search for.
        collection_name (str): The name of the collection to interact with or create.
        client (SQL Engine): The SQL engine

    Returns:
        Dataframe
    """
    # Grab 
    sql = f"SELECT * FROM {collection_name}"
    data = pd.read_sql(sql, client)

    # Initialize BM25 with the tokenized documents
    bm25 = BM25Okapi(data['embeddings'])

    # Get the BM25 scores for the query across all documents
    scores = bm25.get_scores(query)
    data['score'] = scores

    # Get indices of the top-k documents
    data.sort_values(by='score', ascending=True, inplace=True)
    data['rank'] = np.arange(1, len(data)+1)    
    return data

class SQLiteRetriever(RetrieverStrategy):
    """
    Concrete implementation of the Retriever interface using SQLite.
    """

    def __init__(self):
        """
        Initializes the SQLiteRetriever instance. The client and collection are initially set to None.
        """
        self.client = None
        self.collection = None

    def connect(self, dbpath: str, collection_name: str) -> None:
        """
        Establish a connection to the SQLiteDB instance and select/create a collection.

        This method initializes a SQLite client and connects to the SQLiteDB instance.

        Args:
            dbpath (str): The directory path where SQLiteDB will persist its data.
            collection_name (str): The name of the collection to interact with or create.

        Raises:
            ValueError: If the provided dbpath is invalid or cannot be accessed.
        """
        # Create a SQLite client and connect to the database
        self.client = sqlite3.connect(dbpath)

        # Ensure 'corpus' is not taken
        assert collection_name.lower() != 'corpus', "'corpus' is a reserved table name."

        # Ensure the collection exists or create it if necessary
        try:
            check = pd.read_sql(f"SELECT * FROM {collection_name} LIMIT 1", self.client)
        except:
            raise Exception(f"The collection '{collection_name}' does not exist.")
        self.collection = collection_name

    def top_k(self, query: str, top_k: int, **kwargs) -> Tuple[List[str], List[Dict]]:
        """
        Retrieves the top K most relevant documents for a given query.
        * The query must already have gone through preprocessing (if any) before using this method.

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
            raise ValueError("SQLite DB has not been connected. Please use .connect() first.")

        # Retrieve and rank
        data = _retrieve_and_rank(query, self.collection, self.client)
        data = data.iloc[:top_k]

        # Package
        documents = list(data['documents'])
        metadatas = list(data['metadatas'])
        metadatas = [json.loads(meta) for meta in metadatas]
        return documents, metadatas

    def rank_all(self, query: str, **kwargs) -> Tuple[List[str], List[str], List[int], List[Dict]]:
        """
        Ranks all documents based on their relevance to a given query.
        The query must already have gone through preprocessing (if any) before using this method.

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
            raise ValueError("SQLite has not been connected. Please use .connect() first.")   

        # Retrieve and rank
        data = _retrieve_and_rank(query, self.collection, self.client)

        # Package
        documents = list(data['documents'])
        document_ids = list(data['ids'])
        rankings = list(data['rank'])
        metadatas = list(data['metadatas'])
        metadatas = [json.loads(meta) for meta in metadatas]

        d = {'documents':documents,
             'document_ids':document_ids,
             'rankings':rankings,
             'metadatas':metadatas}
        return d