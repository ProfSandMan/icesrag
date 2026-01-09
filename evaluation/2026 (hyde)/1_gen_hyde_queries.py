import concurrent.futures
import os
import threading
from pathlib import Path
from sqlite3 import connect

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

from icesrag.utils.hyde import HyDELLM
from icesrag.utils.llms import OpenAILLM

# * ==================================================================================
# * Define LLM - Thread-safe factory function
# * ==================================================================================
load_dotenv()

model = "gpt-4o-mini"
openai_api_key = os.getenv("OPENAI_API_KEY")

# Thread-local storage for HyDE LLM instances
# This ensures each thread gets its own instance, avoiding any potential
# thread-safety issues with shared state in the OpenAI client
_thread_local = threading.local()

def get_hyde_llm():
    """
    Get a thread-local HyDE LLM instance.
    
    Each thread gets its own instance to ensure thread safety.
    The OpenAI client is generally thread-safe, but creating separate
    instances per thread is the safest approach and has minimal overhead.
    """
    if not hasattr(_thread_local, 'hyde_llm'):
        llm = OpenAILLM(model=model, api_key=openai_api_key)
        _thread_local.hyde_llm = HyDELLM(llm=llm)
    return _thread_local.hyde_llm

# * ==================================================================================
# * Load existing queries
# * ==================================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

db_path = BASE_DIR / "data" / "ices.db"
conn = connect(db_path)

limit = 2 # Due to LLM intensity, only going to sample a subset 

sql = f"SELECT id, query, hyde_query FROM evaluation_queries WHERE query_type = 'long' ORDER BY RANDOM() LIMIT {int(limit/2)}"
data = pd.read_sql(sql, conn)

sql = f"SELECT id, query, hyde_query FROM evaluation_queries WHERE query_type = 'short' ORDER BY RANDOM() LIMIT {int(limit/2)}"
short_data = pd.read_sql(sql, conn)

data = pd.concat([data, short_data], axis=0, ignore_index=True)

# * ==================================================================================
# * Generate hyde queries in parallel
# * ==================================================================================

BATCH_SIZE = 25  # Commit every BATCH updates
updates = []

# Prepare list of tasks for parallel processing
tasks = [
    (row['id'], row['query'])
    for _, row in data.iterrows()
    if row['hyde_query'] is None
]

def process_hyde_query(args):
    """
    Process a single HyDE query generation task.
    
    This function is called by each thread and uses thread-local
    storage to get its own HyDE LLM instance, ensuring thread safety.
    """
    id, query = args
    try:
        # Get thread-local HyDE LLM instance
        hyde_llm = get_hyde_llm()
        hyde_query = hyde_llm.query(query)
        return (hyde_query, id, None)
    except Exception as e:
        return (None, id, str(e))

# Use ThreadPoolExecutor for IO-bound tasks (LLM API calls)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_hyde_query, t) for t in tasks]

    for fut in tqdm(concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Generating hyde queries"):
        hyde_query, id, error = fut.result()
        if hyde_query is not None:
            updates.append((hyde_query, id))
        elif error:
            print(f"Error processing query {id}: {error}")
        # Commit in batches to preserve progress
        if len(updates) >= BATCH_SIZE:
            conn.executemany(
                "UPDATE evaluation_queries SET hyde_query = ? WHERE id = ?",
                updates
            ) 
            conn.commit()
            updates = []

# Commit any remaining updates
if updates:
    conn.executemany(
        "UPDATE evaluation_queries SET hyde_query = ? WHERE id = ?",
        updates
    )
    conn.commit()

conn.close()
print("HyDE queries generated and saved!")