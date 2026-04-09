import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

MEDIUM_RSS = "https://medium.com/feed/tag/{tag}"

def extract_article_stats(link: str):
    """Scrape the actual article page for claps and responses since RSS doesn't include them."""
    try:
        r = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return 0, 0
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Claps
        claps = 0
        clap_span = soup.find('span', {'class': 'pw-multi-vote-count'})
        if clap_span:
            clap_text = clap_span.text.strip().replace('K', '000').replace('.', '')
            claps = int(clap_text) if clap_text.isdigit() else 0
            
        # Responses
        responses = 0
        responses_button = soup.find('button', {'aria-label': 'responses'})
        if responses_button:
            responses_text = responses_button.text.strip()
            # Often says '42 responses' 
            clean = responses_text.replace('responses', '').strip()
            responses = int(clean) if clean.isdigit() else 0
            
        return claps, responses
    except:
        return 0, 0

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
            link = item.link.text if item.link else url
            
            # Fetch deeper stats using the actual article link (Fixing the 0 claps issue)
            claps, responses = extract_article_stats(link)
            
            articles.append({
                "author":    creator.text if creator else "Unknown",
                "title":     item.title.text if item.title else "Untitled",
                "url":       link,
                "claps":     claps,
                "responses": responses,
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
