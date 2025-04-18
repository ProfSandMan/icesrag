import json
import sqlite3
import logging

import pandas as pd

from icesrag.load.store.strategy_pattern import DatabaseStrategy

logger = logging.getLogger(__name__)

class SQLiteDBStore(DatabaseStrategy):
    """
    SQLiteDB is a concrete implementation of the DatabaseStrategy pattern, 
    which manages interactions with the SQLite database. This class provides
    functionality to connect to a SQLiteDB instance, add data to a collection, and delete collections.

    Attributes:
        client (SQLitedb.Client): The SQLiteDB client used to interact with the database.
        collection (SQLitedb.Collection): The SQLiteDB table that holds the data.
    """

    def __init__(self):
        """
        Initializes the SQLiteDB instance.

        Sets the client to None initially, as the connection has not been established.
        """
        logger.info("Initializing SQLiteDBStore")
        self.client = None
        self.collection = None

    def connect(self, dbpath: str, collection_name: str) -> None:
        """
        Establish a connection to the SQLiteDB instance and select/create a collection.

        This method initializes a SQLite client and connects to the SQLiteDB instance.
        If a collection with the specified name does not exist, it will be created.

        Args:
            dbpath (str): The directory path where SQLiteDB will persist its data.
            collection_name (str): The name of the collection to interact with or create.

        Raises:
            ValueError: If the provided dbpath is invalid or cannot be accessed.
        """
        logger.info(f"Connecting to SQLite database at {dbpath}")
        # Create a SQLite client and connect to the database
        self.client = sqlite3.connect(dbpath)

        # Ensure 'corpus' is not taken
        assert collection_name.lower() != 'corpus', "'corpus' is a reserved table name."

        # Ensure the collection exists or create it if necessary
        try:
            logger.debug(f"Checking if collection {collection_name} exists")
            check = pd.read_sql(f"SELECT * FROM {collection_name} LIMIT 1", self.client)
            logger.info(f"Collection {collection_name} already exists")
        except:
            # Create table
            logger.info(f"Creating new collection {collection_name}")
            create = f"""CREATE TABLE {collection_name}  (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        documents  TEXT    COLLATE NOCASE,
                        embeddings TEXT    COLLATE NOCASE
                                           UNIQUE ON CONFLICT REPLACE,
                        ids        TEXT    COLLATE NOCASE,
                        metadatas  TEXT    COLLATE NOCASE
                     );
                     """
            self.client.execute(create)
            self.client.commit()
            logger.debug(f"Successfully created collection {collection_name}")
        self.collection = collection_name
        logger.info(f"Successfully connected to collection {collection_name}")

    def delete(self, collection_name: str) -> None:
        """
        Delete a SQLiteDB table by name.

        This method deletes the specified table from the SQLiteDB instance.
        If the table does not exist, it will raise an error.

        Args:
            collection_name (str): The name of the table to delete.

        Raises:
            ValueError: If SQLiteDB has not been connected.
        """
        logger.info(f"Attempting to delete collection {collection_name}")
        if self.client is None:
            raise ValueError("SQLiteDB has not been connected. Please use .connect() first.")
        
        # Ensure 'corpus' is not taken
        assert collection_name.lower() != 'corpus', "'corpus' is a reserved table name."        

        # Delete the collection
        self.client.execute(f"DROP TABLE {collection_name}")
        self.client.commit()
        logger.info(f"Successfully deleted collection {collection_name}")

    def add(self, data: dict) -> None:
        """
        Add data (documents, embeddings, metadata, and ids) to the SQLiteDB collection.

        This method adds the data provided in the `data` dictionary to the SQLiteDB collection.
        The dictionary should contain keys: 'documents', 'embeddings', 'metadatas', and 'ids',
        which are required for adding data to the collection.

        Args:
            data (dict): A dictionary containing the following keys:
                - 'documents': List of documents to add.
                - 'embeddings': List of embeddings corresponding to the documents.
                - 'metadatas': List of metadata objects corresponding to the documents.
                - 'ids': List of unique IDs for the documents.

        Raises:
            ValueError: If SQLiteDB has not been connected.
        """
        logger.info(f"Adding data to collection {self.collection}")
        if self.client is None:
            raise ValueError("SQLiteDB has not been connected. Please use .connect() first.")
        
        # Ensure 'corpus' is not taken
        assert self.collection.lower() != 'corpus', "'corpus' is a reserved table name."

        # Expected keys
        expected = ['documents','metadatas','ids','embeddings']
        for e in expected:
            if e not in data.keys():
                raise Exception(f"data must contain: {e} in its keys")
        for k in data.keys():
            if k not in expected:
                raise Exception(f"data contained unexpected key: {k}")

        # Convert metadatas to JSON for storage reasons
        logger.debug("Converting metadata to JSON format")
        data = data.copy()
        data['metadatas'] = [json.dumps(meta) for meta in data['metadatas']]

        # Unpack the dictionary into the collection.add method
        logger.debug(f"Adding {len(data['documents'])} documents to database")
        data = pd.DataFrame(data)
        data.to_sql(name = self.collection, con = self.client, if_exists = 'append', index = False)
        logger.info(f"Successfully added {len(data['documents'])} documents to collection {self.collection}")