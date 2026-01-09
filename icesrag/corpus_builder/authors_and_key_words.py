"""
Author and keyword extraction and normalization module.

This module extracts authors and keywords from the abstracts table in the database,
normalizes them, and stores them in separate normalized tables for efficient querying.

The module performs the following operations:
1. Extracts authors and keywords from JSON fields in the abstracts table
2. Normalizes author names (title case)
3. Cleans and normalizes keywords (removes extra whitespace, normalizes formatting)
4. Creates normalized relational tables (authors, keywords) with paper_id foreign keys

This should be run after scrape_ices_repo.py has populated the abstracts table,
and before embed_abstracts.py processes the data.
"""

import json

import pandas as pd
from sqlite3 import connect


def clean_keyword(keyword: str) -> str:
    """
    Clean and normalize a keyword string.
    
    Performs the following normalization steps:
    - Converts to title case
    - Replaces hyphens with spaces
    - Removes newline characters
    - Collapses multiple spaces into single spaces
    
    Parameters
    ----------
    keyword : str
        Raw keyword string to clean.
    
    Returns
    -------
    str
        Normalized keyword in title case with cleaned whitespace.
    
    Examples
    --------
    >>> clean_keyword("machine-learning\\n")
    'Machine Learning'
    >>> clean_keyword("  space   systems  ")
    'Space Systems'
    """
    keyword = keyword.title().replace("-", " ").replace("\n", " ")
    while "  " in keyword:
        keyword = keyword.replace("  ", " ")
    return keyword.title()


def load_authors_and_keywords(db_path: str) -> None:
    """
    Extract, normalize, and store authors and keywords in separate database tables.
    
    This function reads the abstracts table, extracts authors and keywords from
    JSON fields, normalizes them, and creates two new tables:
    - authors: (paper_id, name) - one row per author per paper
    - keywords: (paper_id, keyword) - one row per keyword per paper
    
    The function replaces existing authors and keywords tables if they exist,
    ensuring a clean rebuild of the normalized data.
    
    Parameters
    ----------
    db_path : str
        Path to the directory containing ices.db. The database file should be
        at {db_path}/ices.db.
    
    Returns
    -------
    None
        Data is written directly to the database tables.
    
    Raises
    ------
    FileNotFoundError
        If the database file is not found.
    sqlite3.OperationalError
        If there are issues accessing or writing to the database.
    """

    print(f"Loading authors and keywords...")

    conn = connect(db_path + '/ices.db')
    data = pd.read_sql(f"SELECT * FROM abstracts", conn)

    # Transform authors and keywords to lists
    data['authors'] = data['authors'].apply(lambda x: json.loads(x))
    data['keywords'] = data['keywords'].apply(lambda x: json.loads(x))

    # Replace "Not Found" with None
    data['authors'] = data['authors'].apply(lambda x: x if isinstance(x, str) == False else [])
    data['keywords'] = data['keywords'].apply(lambda x: x if isinstance(x, str) == False else [])

    # Build new dataframes
    authors_paper_id = []
    authors_name = []

    keywords_paper_id = []
    keywords_word = []

    # Need to replace \n with a space
    for index, row in data.iterrows():
        for author in row['authors']:
            authors_paper_id.append(row['id'])
            authors_name.append(author.title())
        for keyword in row['keywords']:
            keywords_paper_id.append(row['id'])
            keywords_word.append(clean_keyword(keyword))

    authors_df = pd.DataFrame({'paper_id': authors_paper_id, 'name': authors_name})
    keywords_df = pd.DataFrame({'paper_id': keywords_paper_id, 'keyword': keywords_word})

    # Push to database
    authors_df.to_sql("authors", conn, if_exists='replace', index=False)
    keywords_df.to_sql("keywords", conn, if_exists='replace', index=False)

    # Close connection
    conn.close()
    print("Authors and keywords loaded!")