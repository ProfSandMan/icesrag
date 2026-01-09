"""
Abstract embedding and vector database storage module.

This module generates embeddings for paper abstracts and stores them in vector databases
for semantic search. It supports two retrieval strategies:
1. Dense embeddings (ChromaDB) - for semantic similarity search
2. Sparse BM25 index (SQLite) - for keyword-based retrieval

The module uses a strategy pattern to allow flexible embedding and storage backends.
It should be run after scrape_ices_repo.py and authors_and_key_words.py have populated
the abstracts table.

Workflow:
    1. Loads abstracts and metadata from SQLite database
    2. Generates dense embeddings using sentence transformers
    3. Preprocesses text for BM25 indexing
    4. Stores both dense and sparse representations in their respective databases
"""
import os
from sqlite3 import connect

import pandas as pd

from icesrag.utils.embed.strategy_pattern import EmbeddingEngine
from icesrag.utils.embed.base_embedders import SentenceTransformEmbedder

from icesrag.utils.text_preprocess.strategy_pattern import TextPreprocessingEngine
from icesrag.utils.text_preprocess.bm25_preprocess import BM25PreProcess

from icesrag.load.package.strategy_pattern import PackageEngine
from icesrag.load.package.chroma_packager import PackageChroma
from icesrag.load.package.sqlite_packager import PackageSQLite

from icesrag.load.store.strategy_pattern import DatabaseEngine
from icesrag.load.store.chromadb import ChromaDBStore
from icesrag.load.store.sqlitedb import SQLiteDBStore

from icesrag.load.pipeline.loader import CompositeLoader


def embed_abstracts(db_path: str) -> None:
    """
    Generate embeddings for abstracts and store them in vector databases.
    
    This function processes all abstracts from the database, generates embeddings
    using sentence transformers, and stores them in both ChromaDB (for semantic
    search) and a BM25 index in SQLite (for keyword-based retrieval).
    
    Parameters
    ----------
    year : int
        The year of papers to process (currently used for logging/identification).
        Note: All abstracts in the database are processed regardless of year.
    db_path : str
        Path to the directory containing the database files. Expected files:
        - {db_path}/ices.db - Source database with abstracts
        - {db_path}/chroma.db - ChromaDB storage (created if needed)
        - {db_path}/bm25.db - BM25 index storage (created if needed)
    
    Returns
    -------
    None
        Results are stored directly in the vector databases.
    
    Raises
    ------
    FileNotFoundError
        If the ices.db file is not found at the specified path.
    sqlite3.OperationalError
        If there are issues accessing the database.
    """

    print("Preparing abstracts for embedding...")

    # Variables:
    collection = "ices"

    abstract_path = db_path + "/ices.db"
    
    chroma_path = db_path + "/chroma.db"
    # Delete chroma.db if it exists
    if os.path.exists(chroma_path):
        os.remove(chroma_path)

    bm25_path = db_path + "/bm25.db"
    # Clear ices table in bm25.db
    conn = connect(bm25_path)
    conn.execute("DELETE FROM ices")
    conn.commit()
    conn.close()

    # Retrieve abstracts and metadata
    conn = connect(abstract_path)
    data = pd.read_sql("SELECT * FROM abstracts", conn)
    data.rename(columns={'id': 'paper_id'}, inplace=True)
    columns = list(data.columns)
    abstracts = data['abstract'].to_list()
    columns.remove('abstract')
    metadata = data[columns].to_dict(orient='records')

    # Build strategy and load
    vanilla_embedder = EmbeddingEngine(SentenceTransformEmbedder())
    vanilla_packager = PackageEngine(PackageChroma())

    bm25_preprocessor = TextPreprocessingEngine(BM25PreProcess())
    bm25_packager = PackageEngine(PackageSQLite())

    vanilla_store = DatabaseEngine(ChromaDBStore())
    vanilla_store.connect(chroma_path, collection)

    bm25_store = DatabaseEngine(SQLiteDBStore())
    bm25_store.connect(bm25_path, collection)

    strategies = [
                    {
                        'name':'vanilla',
                        'embed': vanilla_embedder,
                        'package': vanilla_packager,
                        'store': vanilla_store
                    },
                    
                    {
                        'name':'bm25',
                        'preprocess': bm25_preprocessor,
                        'package': bm25_packager,
                        'store': bm25_store
                    } 
                ]

    print("Embedding and loading abstracts...")

    loader = CompositeLoader(strategies=strategies)
    loader.prepare_load(abstracts, metadata)

    print("Abstracts embedded and loaded!")