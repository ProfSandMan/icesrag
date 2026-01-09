"""
ICES web scraper utilities for TTU IR repository.

This module provides utilities for scraping the Texas Tech University Institutional
Repository (TTU IR) to extract ICES conference paper information. It includes:

1. DSpace API integration for discovering paper URLs by year
2. HTML parsing utilities for extracting paper metadata
3. Robust HTTP fetching with retry logic and exponential backoff

The module handles the JavaScript-rendered DSpace interface by directly calling
the REST API endpoints that the frontend uses, avoiding the need for headless browsers.

Key Components:
    - scrape_ttu_ir_item_links_for_year(): Discovers all paper URLs for a year
    - ICESScraper: Extracts metadata from individual paper pages
    - fetch_url(): Robust HTTP fetching with retry logic
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# * ==================================================================================
# * DSpace API Integration - Discover paper URLs by year
# * ==================================================================================

@dataclass(frozen=True)
class DSpaceQueryStyle:
    """
    Represents a parameterization style for the DSpace discover/search/objects endpoint.
    
    Different DSpace installations may use different query parameter formats.
    This class encapsulates a query style with a function to build the appropriate
    parameters for a given query.
    
    Attributes
    ----------
    name : str
        Human-readable name for this query style (for debugging/logging).
    build_params : callable
        Function that takes (year: int, page0: int, size: int, scope_uuid: str)
        and returns a dictionary of query parameters for the DSpace API.
    """
    name: str
    build_params: callable  # (year:int, page0:int, size:int, scope_uuid:str) -> dict

def _walk(obj: Any) -> Iterable[Any]:
    """
    Recursively yield all nested values in a dict/list tree (depth-first traversal).
    
    This is a utility function for traversing nested JSON structures to find
    specific values (like UUIDs) regardless of their location in the tree.
    
    Parameters
    ----------
    obj : Any
        The object to traverse (dict, list, or primitive value).
    
    Yields
    ------
    Any
        All values found in the tree, including nested structures.
    
    Examples
    --------
    >>> list(_walk({"a": {"b": [1, 2]}}))
    [{'b': [1, 2]}, [1, 2], 1, 2]
    """
    if isinstance(obj, dict):
        for v in obj.values():
            yield v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield v
            yield from _walk(v)

def _extract_item_uuids(payload: Any) -> List[str]:
    """
    Extract UUIDs that correspond to ITEMs from a DSpace HAL-ish JSON response.
    This is intentionally robust: it searches the JSON tree for dicts that look like
    {"uuid": "...", "type": "item"} or {"uuid": "...", "type": {"value":"item"}} etc.
    """
    uuids: List[str] = []
    seen = set()

    for node in _walk(payload):
        if not isinstance(node, dict):
            continue

        uuid = node.get("uuid")
        if not (isinstance(uuid, str) and len(uuid) >= 30):
            continue

        # Try to determine if this uuid is for an "item"
        t = node.get("type")
        is_item = False

        if isinstance(t, str):
            is_item = (t.lower() == "item")
        elif isinstance(t, dict):
            # Common shapes: {"value":"item"} or {"type":"item"} etc.
            for k in ("value", "type", "name"):
                tv = t.get(k)
                if isinstance(tv, str) and tv.lower() == "item":
                    is_item = True
                    break

        # Sometimes the "type" is adjacent elsewhere; accept UUIDs that appear under
        # an indexableObject with explicit item type.
        if not is_item and "indexableObject" in node and isinstance(node["indexableObject"], dict):
            t2 = node["indexableObject"].get("type")
            if isinstance(t2, str) and t2.lower() == "item":
                is_item = True

        if is_item and uuid not in seen:
            seen.add(uuid)
            uuids.append(uuid)

    return uuids

def scrape_ttu_ir_item_links_for_year(
    year: int,
    *,
    collection_uuid: str = "ef7ac1dd-cfc8-4fb0-9bd9-81e30264df7f",
    per_page: int = 100,
    base_site: str = "https://ttu-ir.tdl.org",
    timeout: int = 30,
    max_pages: int = 10_000,
    session: Optional[requests.Session] = None,
) -> List[str]:
    """
    Returns public item (paper) links like:
        https://ttu-ir.tdl.org/items/<uuid>
    for a given year, across all pages, without visiting each item page and without downloading PDFs.

    Why this works:
    - The UI page is rendered by JS and fetches results from:
      /server/api/discover/search/objects  (DSpace Discover API)
      so we call that API directly.

    Parameters
    ----------
    year : int
        Year to filter on (e.g., 2025).
    collection_uuid : str
        The collection scope UUID (default is ICES collection you provided).
    per_page : int
        Page size.
    base_site : str
        Site root.
    timeout : int
        Requests timeout seconds.
    max_pages : int
        Hard safety limit.
    session : Optional[requests.Session]
        Provide one if you want connection pooling / retries.

    Returns
    -------
    List[str]
        List of public /items/<uuid> URLs.
    """
    if year < 1000 or year > 3000:
        raise ValueError(f"year looks wrong: {year}")

    api_url = f"{base_site.rstrip('/')}/server/api/discover/search/objects"

    # A couple common query styles seen in DSpace 7/8 installations.
    styles: List[DSpaceQueryStyle] = [
        DSpaceQueryStyle(
            name="range-minmax-page-size",
            build_params=lambda y, p0, size, scope: {
                "scope": scope,
                "f.dateIssued.min": str(y),
                "f.dateIssued.max": str(y),
                "page": str(p0),   # 0-based
                "size": str(size),
            },
        ),
        DSpaceQueryStyle(
            name="range-minmax-spcpage-spcrpp",
            build_params=lambda y, p0, size, scope: {
                "scope": scope,
                "f.dateIssued.min": str(y),
                "f.dateIssued.max": str(y),
                "spc.page": str(p0 + 1),  # UI is often 1-based
                "spc.rpp": str(size),
            },
        ),
        DSpaceQueryStyle(
            name="equals-filter",
            build_params=lambda y, p0, size, scope: {
                "scope": scope,
                "f.dateIssued": f"{y},equals",
                "page": str(p0),
                "size": str(size),
            },
        ),
    ]

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; link-scraper/1.0)",
        # Some installs behave better if this header is present:
        "X-Requested-With": f"{base_site.rstrip('/')}/collections/{collection_uuid}/search",
    }

    sess = session or requests.Session()

    def try_fetch(style: DSpaceQueryStyle, page0: int) -> Tuple[Optional[dict], List[str]]:
        params = style.build_params(year, page0, per_page, collection_uuid)
        r = sess.get(api_url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None, []
        try:
            data = r.json()
        except Exception:
            return None, []
        uuids = _extract_item_uuids(data)
        return data, uuids

    # Pick a working style by probing page 0.
    chosen: Optional[DSpaceQueryStyle] = None
    first_payload: Optional[dict] = None
    first_uuids: List[str] = []

    for st in styles:
        payload, uuids = try_fetch(st, page0=0)
        if payload is not None:
            chosen = st
            first_payload = payload
            first_uuids = uuids
            break

    if chosen is None:
        raise RuntimeError(
            "Could not fetch search results from the DSpace API. "
            "The endpoint may require different params, auth, or is blocking requests."
        )

    # Now page through using the chosen style.
    all_uuids: List[str] = []
    seen = set()

    def add_many(xs: List[str]) -> None:
        for x in xs:
            if x not in seen:
                seen.add(x)
                all_uuids.append(x)

    add_many(first_uuids)

    page0 = 1
    while page0 < max_pages:
        _, uuids = try_fetch(chosen, page0=page0)
        if not uuids:
            break
        add_many(uuids)
        page0 += 1

    # Convert to public item URLs
    return [f"{base_site.rstrip('/')}/items/{u}" for u in all_uuids]

# * ==================================================================================
# * Scraper class to pull all information from a given paper URL
# * ==================================================================================

class ICESScraper:
    """
    Scraper for extracting metadata from ICES paper web pages.
    
    This class extracts paper metadata from HTML pages by parsing meta tags.
    It uses standard Dublin Core and citation meta tag conventions commonly
    used in academic repositories.
    
    The scraper extracts:
    - Title, abstract, publication date
    - Authors (multiple)
    - Keywords (semicolon-separated)
    - Publisher information
    - PDF download URL
    """
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract the paper title from HTML meta tags.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.
        
        Returns
        -------
        str
            Paper title, or "Not found" if not present.
        """
        title_meta = soup.find("meta", {"name": "title"})
        return title_meta["content"].strip() if title_meta else "Not found"

    def extract_abstract(self, soup: BeautifulSoup) -> str:
        """
        Extract the paper abstract from HTML meta tags.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.
        
        Returns
        -------
        str
            Paper abstract, or "Not found" if not present.
        """
        abstract_meta = soup.find("meta", {"name": "description"})
        return abstract_meta["content"].strip() if abstract_meta else "Not found"

    def extract_authors(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract all authors from HTML citation meta tags.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.
        
        Returns
        -------
        List[str]
            List of author names, or ["Not found"] if no authors found.
        """
        authors_meta = soup.find_all("meta", {"name": "citation_author"})
        return [author["content"].strip() for author in authors_meta] if authors_meta else ["Not found"]

    def extract_date(self, soup: BeautifulSoup) -> str:
        """
        Extract the publication date from HTML meta tags.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.
        
        Returns
        -------
        str
            Publication date, or "Not found" if not present.
        """
        date_meta = soup.find("meta", {"name": "citation_publication_date"})
        return date_meta["content"].strip() if date_meta else "Not found"

    def extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract keywords from HTML meta tags.
        
        Keywords are expected to be semicolon-separated in a single meta tag.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.
        
        Returns
        -------
        List[str]
            List of keywords (stripped of whitespace), or ["Not found"] if not present.
        """
        keywords_meta = soup.find("meta", {"name": "citation_keywords"})
        return [kw.strip() for kw in keywords_meta["content"].split(";")] if keywords_meta else ["Not found"]

    def extract_publisher(self, soup: BeautifulSoup) -> str:
        """
        Extract the publisher information from HTML meta tags.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.
        
        Returns
        -------
        str
            Publisher name, or "Not found" if not present.
        """
        publisher_meta = soup.find("meta", {"name": "citation_publisher"})
        return publisher_meta["content"].strip() if publisher_meta else "Not found"
    
    def extract_pdf_url(self, soup: BeautifulSoup) -> str:
        """
        Extract the PDF download URL from HTML meta tags.
        
        Parameters
        ----------
        soup : BeautifulSoup
            Parsed HTML document.
        
        Returns
        -------
        str
            PDF URL, or "Not found" if not present.
        """
        pdf_meta = soup.find("meta", {"name": "citation_pdf_url"})
        return pdf_meta["content"].strip() if pdf_meta else "Not found"    
    
    def scrape(self, root_url: str, soup: BeautifulSoup) -> dict:
        """
        Extract all metadata from a paper page.
        
        This is the main method that orchestrates extraction of all paper metadata
        fields and returns them as a dictionary.
        
        Parameters
        ----------
        root_url : str
            The URL of the paper page (stored in the output for reference).
        soup : BeautifulSoup
            Parsed HTML document of the paper page.
        
        Returns
        -------
        dict
            Dictionary containing all extracted metadata with keys:
            - url: Original paper URL
            - title: Paper title
            - abstract: Paper abstract
            - authors: List of author names
            - date: Publication date
            - keywords: List of keywords
            - publisher: Publisher name
            - paper_url: PDF download URL
        """
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

# * ==================================================================================
# * Function to fetch a URL with retries and exponential backoff
# * ==================================================================================

def fetch_url(url: str, session: requests.Session, retries: int = 5, backoff_factor: int = 2) -> Optional[requests.Response]:
    """
    Fetch a URL with automatic retry logic and exponential backoff.
    
    This function implements robust HTTP fetching with:
    - Automatic retries on connection errors and timeouts
    - Exponential backoff between retries
    - No retries on HTTP errors (4xx/5xx) - these are typically permanent
    - Random jitter to avoid thundering herd problems
    
    Parameters
    ----------
    url : str
        The URL to fetch.
    session : requests.Session
        A requests Session object to use for the HTTP request.
        Allows connection pooling and shared headers/cookies.
    retries : int, optional
        Maximum number of retry attempts (default: 5).
    backoff_factor : int, optional
        Base multiplier for exponential backoff delay (default: 2).
        Delay = backoff_factor * (2 ** attempt) + random(0, 1) seconds.
    
    Returns
    -------
    Optional[requests.Response]
        The HTTP response object if successful, None if all retries failed.
    
    Notes
    -----
    - Connection errors and timeouts trigger retries
    - HTTP errors (4xx/5xx) do not trigger retries as they're typically permanent
    - The function prints progress messages for debugging
    """
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
