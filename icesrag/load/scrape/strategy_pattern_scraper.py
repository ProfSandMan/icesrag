#Scraper for ICES Texas Tech Repo - Note: Only 10 Years of ICES Papers 2014-2024
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup

# Overarching Abstract Strategy Class
class WebScrapingStrategy(ABC):
    @abstractmethod
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the title from the soup."""
        pass

    @abstractmethod
    def extract_abstract(self, soup: BeautifulSoup) -> str:
        """Extract the abstract from the soup."""
        pass

    @abstractmethod
    def extract_authors(self, soup: BeautifulSoup) -> str:
        """Extract the authors from the soup."""
        pass

    @abstractmethod
    def extract_date(self, soup: BeautifulSoup) -> str:
        """Extract the publication date from the soup."""
        pass

    @abstractmethod
    def extract_keywords(self, soup: BeautifulSoup) -> str:
        """Extract the keywords from the soup."""
        pass

    @abstractmethod
    def extract_publisher(self, soup: BeautifulSoup) -> str:
        """Extract the publisher from the soup."""
        pass
    @abstractmethod
    def extract_page_url(self, soup: BeautifulSoup) -> str:
        """ Extract the page url to texas tech"""
        pass
    @abstractmethod
    def extract_pdf_url(self, soup: BeautifulSoup) -> str:
        """ Extract the pdf download url to texas tech"""
        pass
    def scrape(self, soup: BeautifulSoup) -> dict:
        """Scrape using strategy and output in dictionary format."""
        pass