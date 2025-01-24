#Scraper for ICES Texas Tech Repo - Note: Only 10 Years of ICES Papers 2014-2024
import requests
import time
from bs4 import BeautifulSoup
from icesrag.load.scrape.strategy_pattern_scraper import WebScrapingStrategy
import pandas as pd

# Concrete Implementation
class ConcreteWebScraper(WebScrapingStrategy):
    """
    Concrete Implementation for scraping ICES Texas Tech Repo
    """
    def __init__(self):
        pass
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extracts the title of the paper from teh webpage
        """
        title_meta = soup.find("meta", {"name": "title"})
        return title_meta["content"].strip() if title_meta else "Not found"

    def extract_abstract(self, soup: BeautifulSoup) -> str:
        """
        Extracts the abstract of the paper from the webpage
        """
        abstract_meta = soup.find("meta", {"name": "description"})
        return abstract_meta["content"].strip() if abstract_meta else "Not found"

    def extract_authors(self, soup: BeautifulSoup) -> str:
        """
        Extracts the authors of the paper from the webpage
        """
        authors_meta = soup.find_all("meta", {"name": "citation_author"})
        return [author["content"].strip() for author in authors_meta] if authors_meta else "Not found"

    def extract_date(self, soup: BeautifulSoup) -> str:
        """
        Extracts the publication date of the paper from the webpage
        """
        date_meta = soup.find("meta", {"name": "citation_publication_date"})
        return date_meta["content"].strip() if date_meta else "Not found"

    def extract_keywords(self, soup: BeautifulSoup) -> str:
        """
        Extracts the keywords of the paper from the webpage
        """
        keywords_meta = soup.find("meta", {"name": "citation_keywords"})
        return [kw.strip() for kw in keywords_meta["content"].split(";")] if keywords_meta else "Not found"

    def extract_publisher(self, soup: BeautifulSoup) -> str:
        """
        Extracts the publisher of the paper from the webpage
        """
        publisher_meta = soup.find("meta", {"name": "citation_publisher"})
        return publisher_meta["content"].strip() if publisher_meta else "Not found"
    
    def extract_page_url(self, soup: BeautifulSoup):
        """
        Extracts the webpage link from the given HTML soup.
        """
        webpage_link = soup.find("link", {"rel": "cite-as"})
        return webpage_link["href"].strip() if webpage_link and "href" in webpage_link.attrs else "Not found"
        
    def extract_pdf_url(self, soup: BeautifulSoup):        
        """
        Extracts the downloadable PDF link from the given HTML soup.
        """
        download_link = soup.find("link", {"rel": "item", "type": "application/pdf"})
        return download_link["href"].strip() if download_link and "href" in download_link.attrs else "Not found"
    
    def scrape(self, soup: BeautifulSoup) -> dict:
        """Scrape data using the strategy - Returns dictionary of scraped data"""
        return {
            "Title": self.extract_title(soup),
            "Abstract": self.extract_abstract(soup),
            "Authors": self.extract_authors(soup),
            "Date": self.extract_date(soup),
            "Keywords": self.extract_keywords(soup),
            "Publisher": self.extract_publisher(soup),
            "Page URL": self.extract_page_url(soup),
            "PDF URL": self.extract_pdf_url(soup),
        }

# Base collection URL - Texas Tech University has a ton of different collections only scraping for ICES for now
base_collection_url = "https://ttu-ir.tdl.org/collections/ef7ac1dd-cfc8-4fb0-9bd9-81e30264df7f"

# Headers to mimic a real browser request
headers = {"User-Agent": "Mozilla/5.0"}

def get_all_item_links(total_pages=142):
    """Iterates through all pages of the collection and extracts item URLs."""
    item_links = set()  # Use a set to avoid duplicates

    for page in range(1, total_pages + 1):
        page_url = f"{base_collection_url}?cp.page={page}"
        print(f"Scraping page {page}...")
        
        try:
            response = requests.get(page_url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract links to items
            page_links = [
                a["href"] for a in soup.find_all("a", href=True)
                if "/items/" in a["href"]
            ]

            # Convert relative URLs to absolute URLs
            page_links = ["https://ttu-ir.tdl.org" + link if link.startswith("/") else link for link in page_links]

            if page_links:
                item_links.update(page_links)
            else:
                print(f"Page {page} contains no items.")

        except requests.exceptions.RequestException as e:
            print(f"Error on page {page}: {e}")
        
        time.sleep(1)  

    return list(item_links)

# Scrape all pages - uncomment the line below to get all item URLs
# all_item_urls = get_all_item_links(total_pages=142) #Total pages set to 142 for ICES collection because that is the number of pages in the ICES collection

# Save the extracted URLs
def save_urls(all_item_urls):
    with open("all_collection_item_urls.txt", "w") as file:
        for url in all_item_urls:
            file.write(url + "\n")

def scrape_meta(filename="all_collection_item_urls.txt"):
    scraped_data_list = []

    with open(filename, "r") as file:
        urls = [line.strip() for line in file.readlines()]

        for i in range(len(filename)):
            url = urls[i]
            response = requests.get(url)
            if response.status_code == 200:
                html_content = response.text
            else:
                raise Exception(f"Failed to fetch webpage content, status code: {response.status_code}")
            
            soup = BeautifulSoup(html_content, "html.parser")
            scraper = ConcreteWebScraper()
            
            scraped_data = scraper.scrape(soup)
            scraped_data_list.append(scraped_data)
            
            
            # Function returns a dictionary
            # concatonate these dictionaries into one pandas dataframe <- done up to this point

            # todo eventually - Put into SQL lite database
        df = pd.DataFrame(scraped_data_list)
        return df