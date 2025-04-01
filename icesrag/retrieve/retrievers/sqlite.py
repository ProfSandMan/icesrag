import json
import sqlite3
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from icesrag.retrieve.retrievers.strategy_pattern import RetrieverStrategy

logger = logging.getLogger(__name__)

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

    logger.debug(f"Retrieving data from collection {collection_name}")
    # Grab 
    sql = f"SELECT * FROM {collection_name}"
    data = pd.read_sql(sql, client)
    logger.debug(f"Retrieved {len(data)} documents from database")

    # Initialize BM25 with the tokenized documents
    logger.debug("Initializing BM25 with document embeddings")
    bm25 = BM25Okapi(data['embeddings'])

    # Get the BM25 scores for the query across all documents
    logger.debug("Calculating BM25 scores for query")
    scores = bm25.get_scores(query)
    data['score'] = scores

    # Get indices of the top-k documents
    logger.debug("Sorting documents by score")
    data.sort_values(by='score', ascending=False, inplace=True)
    data['rank'] = np.arange(1, len(data)+1)    
    logger.debug("Ranking complete")

    return data

class SQLiteRetriever(RetrieverStrategy):
    """
    Concrete implementation of the Retriever interface using SQLite.
    """

    def __init__(self):
        """
        Initializes the SQLiteRetriever instance. The client and collection are initially set to None.
        """
        logger.info("Initializing SQLiteRetriever")
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
        logger.info(f"Connecting to SQLite database at {dbpath}")
        # Create a SQLite client and connect to the database
        self.client = sqlite3.connect(dbpath, check_same_thread=False)

        # Ensure 'corpus' is not taken
        assert collection_name.lower() != 'corpus', "'corpus' is a reserved table name."

        # Ensure the collection exists or create it if necessary
        try:
            logger.debug(f"Checking if collection {collection_name} exists")
            check = pd.read_sql(f"SELECT * FROM {collection_name} LIMIT 1", self.client)
            logger.info(f"Successfully connected to existing collection {collection_name}")
        except:
            raise Exception(f"The collection '{collection_name}' does not exist.")
        self.collection = collection_name

    def top_k(self, query: str, top_k: int, **kwargs) -> Tuple[List[str], List[float],List[Dict]]:
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
                - A list of distances for each of the top K results.
                - A list of metadata dictionaries for each of the top K results.
        """
        logger.info(f"Retrieving top {top_k} results for query: {query[:50]}...")
        if self.client is None:
            raise ValueError("SQLite DB has not been connected. Please use .connect() first.")

        # Retrieve and rank
        data = _retrieve_and_rank(query, self.collection, self.client)
        data = data.iloc[:top_k]

        # Package
        logger.debug("Packaging results")
        documents = list(data['documents'])
        distances = list(data['score'])
        metadatas = list(data['metadatas'])
        metadatas = [json.loads(meta) for meta in metadatas]
        logger.info(f"Successfully retrieved {len(documents)} results")
        return documents, distances, metadatas

    def rank_all(self, query: str, **kwargs) -> Dict:
        """
        Ranks all documents based on their relevance to a given query.
        The query must already have gone through preprocessing (if any) before using this method.

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
        logger.info(f"Ranking all documents for query: {query[:50]}...")
        if self.client is None:
            raise ValueError("SQLite has not been connected. Please use .connect() first.")   

        # Retrieve and rank
        data = _retrieve_and_rank(query, self.collection, self.client)

        # Package
        logger.debug("Packaging results")
        documents = list(data['documents'])
        document_ids = list(data['ids'])
        rankings = list(data['rank'])
        metadatas = list(data['metadatas'])
        metadatas = [json.loads(meta) for meta in metadatas]

        d = {'documents':documents,
             'document_ids':document_ids,
             'rankings':rankings,
             'metadatas':metadatas}
        logger.info(f"Successfully ranked {len(documents)} documents")
        return d