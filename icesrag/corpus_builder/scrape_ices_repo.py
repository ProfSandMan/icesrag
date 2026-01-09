"""
ICES repository web scraping module.

This module scrapes paper metadata from the Texas Tech University Institutional
Repository (TTU IR) for the International Conference on Environmental Systems (ICES)
collection. It extracts paper information including titles, abstracts, authors,
keywords, publication dates, and PDF URLs.

The scraping process:
1. Fetches all paper URLs for a given year using the DSpace REST API
2. Visits each paper page and extracts metadata from HTML meta tags
3. Stores the extracted data in the abstracts table of the SQLite database
4. Handles rate limiting, retries, and error recovery

The module uses BeautifulSoup to parse HTML and extracts data from Dublin Core
and citation meta tags. It includes exponential backoff retry logic for robust
scraping in the face of network issues or rate limiting.

This is the first step in the corpus building pipeline and should be run before
authors_and_key_words.py and embed_abstracts.py.
"""

import json
import random
import time
from sqlite3 import connect

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from icesrag.corpus_builder.iceswebscraper import ICESScraper, fetch_url, scrape_ttu_ir_item_links_for_year


def scrape_repo(year: int, db_path: str) -> None:
    """
    Scrape ICES repository papers for a given year and store in database.
    
    This function orchestrates the scraping process:
    1. Retrieves all paper URLs for the specified year from TTU IR
    2. Checks which papers are already in the database to avoid duplicates
    3. Scrapes metadata from each new paper's webpage
    4. Stores the extracted data in the abstracts table
    
    The function includes rate limiting protection with random delays between
    requests and extended sleep periods for rate limit errors (429 status).
    
    Parameters
    ----------
    year : int
        The year to scrape papers from (e.g., 2025).
    db_path : str
        Path to the directory containing the database. The database file should
        be at {db_path}/ices.db. The database will be created if it doesn't exist.
    
    Returns
    -------
    None
        Data is written directly to the database. Prints summary of missed files
        if any scraping failures occurred.
    
    Notes
    -----
    - The function skips papers that are already in the database (based on URL)
    - Failed scrapes are tracked and reported at the end
    - Random delays (1-5 seconds) are added between successful requests
    - Rate limit errors trigger 30-second delays before continuing
    """
    # Generate already captured links
    conn = connect(db_path + '/ices.db')
    retrieved = pd.read_sql("SELECT DISTINCT url FROM abstracts", conn)['url'].to_list()

    # Find all urls for the given year
    urls = scrape_ttu_ir_item_links_for_year(year)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 ..."})

    # Loop through each URL and process it
    uncaptured = []
    for url in tqdm(urls, desc="Scraping Progress", unit="url"):
        if url not in retrieved:
            # Fetch the webpage content
            response = fetch_url(url, session)
            if response is None:
                print("None value retrieved from link!")
                time.sleep(30)  # Sleep longer if rate-limited
                uncaptured.append(url)
                continue  # Retry next URL
            elif response.status_code == 200:
                html_content = response.text
            elif response.status_code == 429:  # Too many requests
                print("Rate-limited! Sleeping longer...")
                time.sleep(30)  # Sleep longer if rate-limited
                uncaptured.append(url)
                continue  # Retry next URL

            # Parse the HTML content
            soup = BeautifulSoup(html_content, "html.parser")

            # Use the default scraping strategy
            scraper = ICESScraper()

            # Scrape data
            scraped_data = scraper.scrape(url, soup)

            # Intentional single loading (would do batch, but this will make sure we don't have to rerun if something fails)
            data = pd.DataFrame([scraped_data])
                # Convert list to json (when doing chroma, will retrieve, cast back to list, and then store in metadata tags)
            data['authors'] = data['authors'].apply(lambda x: json.dumps(x))
            data['keywords'] = data['keywords'].apply(lambda x: json.dumps(x))
                # Push to database
            try:
                data.to_sql("abstracts", conn, if_exists="append", index=False)
                retrieved.append(url)
            except:
                print(f"Duplicated URL or other upload problem: {url}")
                uncaptured.append(url)
            
            # Sleep for a random time between 1 and 5 seconds
            time.sleep(random.uniform(1, 5))

    # Push to database
    print("Database loaded!")
    print(f"Missed files {len(uncaptured)}:")
    for u in uncaptured:
        print(u)