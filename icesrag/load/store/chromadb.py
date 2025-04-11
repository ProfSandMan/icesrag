import chromadb
import logging

from icesrag.load.store.strategy_pattern import DatabaseStrategy

logger = logging.getLogger(__name__)

class ChromaDBStore(DatabaseStrategy):
    """
    ChromaDB is a concrete implementation of the DatabaseStrategy pattern, 
    which manages interactions with the Chroma database. This class provides
    functionality to connect to a ChromaDB instance, add data to a collection, and delete collections.

    Attributes:
        client (chromadb.Client): The ChromaDB client used to interact with the database.
        collection (chromadb.Collection): The ChromaDB collection that holds the data.
    """

    def __init__(self):
        """
        Initializes the ChromaDB instance.

        Sets the client to None initially, as the connection has not been established.
        """
        logger.info("Initializing ChromaDBStore")
        self.client = None
        self.collection = None

    def connect(self, dbpath: str, collection_name: str) -> None:
        """
        Establish a connection to the ChromaDB instance and select/create a collection.

        This method initializes a Chroma client and connects to the ChromaDB instance.
        If a collection with the specified name does not exist, it will be created.

        Args:
            dbpath (str): The directory path where ChromaDB will persist its data.
            collection_name (str): The name of the collection to interact with or create.

        Raises:
            ValueError: If the provided dbpath is invalid or cannot be accessed.
        """
        logger.info(f"Connecting to ChromaDB at {dbpath}")
        # Create a Chroma client and connect to the database
        self.client = chromadb.PersistentClient(dbpath)
        logger.debug("Successfully created ChromaDB client")

        # Ensure the collection exists or create it if necessary
        logger.debug(f"Getting or creating collection {collection_name}")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Successfully connected to collection {collection_name}")

    def delete(self, collection_name: str) -> None:
        """
        Delete a ChromaDB collection by name.

        This method deletes the specified collection from the ChromaDB instance.
        If the collection does not exist, it will raise an error.

        Args:
            collection_name (str): The name of the collection to delete.

        Raises:
            ValueError: If ChromaDB has not been connected.
        """
        logger.info(f"Attempting to delete collection {collection_name}")
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")
        
        # Delete the collection
        self.client.delete_collection(name=collection_name)
        logger.info(f"Successfully deleted collection {collection_name}")

    def add(self, data: dict) -> None:
        """
        Add data (documents, embeddings, metadata, and ids) to the ChromaDB collection.

        This method adds the data provided in the `data` dictionary to the ChromaDB collection.
        The dictionary should contain keys: 'documents', 'embeddings', 'metadatas', and 'ids',
        which are required for adding data to the collection.

        Args:
            data (dict): A dictionary containing the following keys:
                - 'documents': List of documents to add.
                - 'embeddings': List of embeddings corresponding to the documents.
                - 'metadatas': List of metadata objects corresponding to the documents.
                - 'ids': List of unique IDs for the documents.

        Raises:
            ValueError: If ChromaDB has not been connected.
        """
        logger.info(f"Adding data to collection {self.collection}")
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")
        
        # Unpack the dictionary into the collection.add method
        logger.debug(f"Adding {len(data['documents'])} documents to collection")
        self.collection.add(**data)
        logger.info(f"Successfully added {len(data['documents'])} documents to collection {self.collection}")