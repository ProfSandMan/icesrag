import chromadb
from icesrag.load.store.strategy_pattern import VectorDatabaseStrategy

class ChromaDB(VectorDatabaseStrategy):
    """
    ChromaDB is a concrete implementation of the VectorDatabaseStrategy pattern, 
    which manages interactions with the Chroma vector database. This class provides
    functionality to connect to a ChromaDB instance, add data to a collection, and delete collections.

    Attributes:
        client (chromadb.Client): The ChromaDB client used to interact with the database.
        collection (chromadb.Collection): The ChromaDB collection that holds the vector data.
    """

    def __init__(self):
        """
        Initializes the ChromaDB instance.

        Sets the client to None initially, as the connection has not been established.
        """
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
        # Create a Chroma client and connect to the database
        self.client = chromadb.Client(persist_directory=dbpath)

        # Ensure the collection exists or create it if necessary
        self.collection = self.client.get_or_create_collection(name=collection_name)

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
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")
        
        # Delete the collection
        self.client.delete_collection(name=collection_name)

    def add(self, data: dict) -> None:
        """
        Add data (documents, embeddings, metadata, and ids) to the ChromaDB collection.

        This method adds the data provided in the `data` dictionary to the ChromaDB collection.
        The dictionary should contain keys: 'documents', 'embeddings', 'metadatas', and 'ids',
        which are required for adding data to the collection.

        Args:
            data (dict): A dictionary containing the following keys:
                - 'documents': List of documents to add.
                - 'embeddings': List of vector embeddings corresponding to the documents.
                - 'metadatas': List of metadata objects corresponding to the documents.
                - 'ids': List of unique IDs for the documents.

        Raises:
            ValueError: If ChromaDB has not been connected.
        """
        if self.client is None:
            raise ValueError("ChromaDB has not been connected. Please use .connect() first.")
        
        # Unpack the dictionary into the collection.add method
        self.collection.add(**data)