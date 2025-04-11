import requests
from bs4 import BeautifulSoup
from icesrag.prototype_corpus_builder.iceswebscraper import ICESScraper
import time
import random
import pandas as pd
from sqlite3 import connect
from tqdm import tqdm
import json

def fetch_url(url, retries=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=10)  # Set a timeout
            response.raise_for_status()  # Raise error for 4xx/5xx responses
            return response
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error for {url}, retrying... ({attempt+1}/{retries})")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {url}: {e}")
            break  # Don't retry on HTTP errors
        except requests.exceptions.Timeout:
            print(f"Timeout for {url}, retrying... ({attempt+1}/{retries})")
        time.sleep(backoff_factor * (2 ** attempt) + random.uniform(0, 1))  # Exponential backoff

    print(f"Failed to fetch {url} after {retries} attempts")
    return None

# Generate already captured links
conn = connect("./data/ices.db")
retrieved = pd.read_sql("SELECT DISTINCT url FROM abstracts", conn)['url'].to_list()

# Open and read the file
with open("icesrag\corpus_builder\ICES Links.txt", "r") as file:
    urls = [line.strip() for line in file if line.strip()]  # Remove empty lines

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 ..."})

# Loop through each URL and process it
uncaptured = []
for url in tqdm(urls, desc="Scraping Progress", unit="url"):
    if url not in retrieved:
        # Fetch the webpage content
        response = fetch_url(url)
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