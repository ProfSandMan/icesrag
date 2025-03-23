print("Importing Libraries...")

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

from sqlite3 import connect
import pandas as pd

print("Starting...")

# Variables:
collection = "ices"

chroma_path = "./chroma.db"
bm25_path = "./bm25.db"

# Retrieve abstracts and metadata
conn = connect("./ices.db")
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

# * could remove type (pretty sure it goes unused)
strategies = [
                {'name':'vanilla',
                 'embed': vanilla_embedder,
                 'package': vanilla_packager,
                 'store': vanilla_store
                },
                
                {'name':'bm25',
                 'preprocess': bm25_preprocessor,
                 'package': bm25_packager,
                 'store': bm25_store
                } 
            ]

print("Loading...")

loader = CompositeLoader(strategies=strategies)
loader.prepare_load(abstracts, metadata)

print("Done!")