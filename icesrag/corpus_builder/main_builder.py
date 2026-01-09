"""
Main corpus builder script for the ICES RAG system.

This script orchestrates the complete corpus building pipeline:
1. Scrapes the ICES repository for papers from a given year
2. Extracts and normalizes authors and keywords from the scraped data
3. Generates embeddings for abstracts and stores them in vector databases

The script processes papers from the Texas Tech University Institutional Repository
(TTU IR) for the International Conference on Environmental Systems (ICES) collection.

Usage:
    Run this script directly to build the corpus for the configured year.
    Modify the YEAR constant to process different years.

Pipeline Steps:
    - scrape_repo(): Fetches paper metadata from TTU IR and stores in SQLite
    - load_authors_and_keywords(): Extracts normalized authors/keywords into separate tables
    - embed_abstracts(): Generates embeddings and stores in ChromaDB and BM25 index
"""

from pathlib import Path

from icesrag.corpus_builder.scrape_ices_repo import scrape_repo
from icesrag.corpus_builder.authors_and_key_words import load_authors_and_keywords
from icesrag.corpus_builder.embed_abstracts import embed_abstracts

# * ==================================================================================
# * Constants
# * ==================================================================================

# Get the project root (two levels up from this file)
SCRIPT_DIR = Path(__file__).parent.resolve()  # icesrag/corpus_builder/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()  # Root of icesrag project
DB_PATH = str(PROJECT_ROOT / "data")

YEAR = 2025

# * =========================================================== Scrape ICES Repository
# scrape_repo(YEAR, DB_PATH)

# * ========================================================= Load Authors and Keywords
load_authors_and_keywords(DB_PATH)

# * =================================================================== Embed Abstracts
embed_abstracts(DB_PATH)