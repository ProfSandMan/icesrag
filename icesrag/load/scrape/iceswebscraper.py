#Scraper for ICES Texas Tech Repo - Note: Only 10 Years of ICES Papers 2014-2024
#Initial webscraping design (12/1/24) Will back into a general strategy pattern later

import requests
from bs4 import BeautifulSoup
from icesrag.load.scrape.strategy_pattern_scraper import WebScrapingStrategy

# Concrete Implementation
class DefaultWebScrapingStrategy(WebScrapingStrategy):
    def extract_title(self, soup: BeautifulSoup) -> str:
        title_meta = soup.find("meta", {"name": "title"})
        return title_meta["content"].strip() if title_meta else "Not found"

    def extract_abstract(self, soup: BeautifulSoup) -> str:
        abstract_meta = soup.find("meta", {"name": "description"})
        return abstract_meta["content"].strip() if abstract_meta else "Not found"

    def extract_authors(self, soup: BeautifulSoup) -> str:
        authors_meta = soup.find_all("meta", {"name": "citation_author"})
        return [author["content"].strip() for author in authors_meta] if authors_meta else "Not found"

    def extract_date(self, soup: BeautifulSoup) -> str:
        date_meta = soup.find("meta", {"name": "citation_publication_date"})
        return date_meta["content"].strip() if date_meta else "Not found"

    def extract_keywords(self, soup: BeautifulSoup) -> str:
        keywords_meta = soup.find("meta", {"name": "citation_keywords"})
        return [kw.strip() for kw in keywords_meta["content"].split(";")] if keywords_meta else "Not found"

    def extract_publisher(self, soup: BeautifulSoup) -> str:
        publisher_meta = soup.find("meta", {"name": "citation_publisher"})
        return publisher_meta["content"].strip() if publisher_meta else "Not found"


# Context Class
class WebPageScraper:
    def __init__(self, strategy: WebScrapingStrategy):
        self.strategy = strategy

    def scrape(self, soup: BeautifulSoup) -> dict:
        """Scrape data using the strategy."""
        return {
            "Title": self.strategy.extract_title(soup),
            "Abstract": self.strategy.extract_abstract(soup),
            "Authors": self.strategy.extract_authors(soup),
            "Date": self.strategy.extract_date(soup),
            "Keywords": self.strategy.extract_keywords(soup),
            "Publisher": self.strategy.extract_publisher(soup),
        }

# Example Use
# Define the URL to scrape
url = "https://ttu-ir.tdl.org/items/12ce401a-cbe0-4e9e-a660-edaa39e6aaa1"

# Fetch the webpage content
response = requests.get(url)
if response.status_code == 200:
    html_content = response.text
else:
    raise Exception(f"Failed to fetch webpage content, status code: {response.status_code}")

# Parse the HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Use the default scraping strategy
strategy = DefaultWebScrapingStrategy()
scraper = WebPageScraper(strategy)

# Scrape data
scraped_data = scraper.scrape(soup)

# Print the results
for key, value in scraped_data.items():
    print(f"{key}: {value}")
