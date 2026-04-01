import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

MEDIUM_RSS = "https://medium.com/feed/tag/{tag}"

def scrape_tag(tag: str, pages: int = 1) -> list:
    url = MEDIUM_RSS.format(tag=tag.strip().lower())
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            print(f"Failed to fetch {url} - Status: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.content, "xml")
        articles = []
        for item in soup.find_all("item"):
            creator = item.find("dc:creator")
            articles.append({
                "author":    creator.text if creator else "Unknown",
                "title":     item.title.text if item.title else "Untitled",
                "url":       item.link.text if item.link else url,
                "claps":     0,
                "responses": 0,
                "tag":       tag,
                "sentiment": None,
                "confidence":None,
                "reason":    None
            })
        return articles
    except Exception as e:
        print(f"Error scraping medium tag {tag}: {e}")
        return []

def scrape_tags_batch(tags: list, pages: int = 1) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda t: scrape_tag(t, pages), tags))
    flat = [a for tag_result in results for a in tag_result]
    if not flat:
        return pd.DataFrame()
    return pd.DataFrame(flat).drop_duplicates(subset=["url"])
