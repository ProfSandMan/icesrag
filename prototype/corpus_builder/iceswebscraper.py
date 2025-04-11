#Scraper for ICES Texas Tech Repo - Note: Only 10 Years of ICES Papers 2014-2024
#Initial webscraping design (12/1/24) Will back into a general strategy pattern later

from bs4 import BeautifulSoup

# Concrete Implementation
class ICESScraper():
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
    
    def extract_pdf_url(self, soup: BeautifulSoup) -> str:
        pdf_meta = soup.find("meta", {"name": "citation_pdf_url"})
        return pdf_meta["content"].strip() if pdf_meta else "Not found"    
    
    def scrape(self, root_url: str, soup: BeautifulSoup) -> dict:
        """Scrape data using the strategy."""
        return {
            "url": root_url,
            "title": self.extract_title(soup),
            "abstract": self.extract_abstract(soup),
            "authors": self.extract_authors(soup),
            "date": self.extract_date(soup),
            "keywords": self.extract_keywords(soup),
            "publisher": self.extract_publisher(soup),
            "paper_url": self.extract_pdf_url(soup)
        }