import requests
from bs4 import BeautifulSoup
import pandas as pd

SO_BASE = "https://stackoverflow.com/questions/tagged/{tag}?tab=votes&page={page}"

def scrape_tag(tag: str, pages: int = 2) -> list:
    questions = []
    for page in range(1, pages + 1):
        url = SO_BASE.format(tag=tag.strip(), page=page)
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
                
            soup = BeautifulSoup(r.content, "html.parser")
            for q in soup.select(".s-post-summary"):
                title_el  = q.select_one(".s-post-summary--content-title a")
                stats     = q.select(".s-post-summary--stats-item-number")
                views_str = q.select_one("[title*='views']")
                
                votes   = int(stats[0].text.replace(",","")) if stats else 0
                answers = int(stats[1].text.replace(",","")) if len(stats)>1 else 0
                views   = int(views_str["title"].split()[0].replace(",","")) if views_str else 0
                
                if title_el:
                    questions.append({
                        "title":             title_el.text.strip(),
                        "url":               "https://stackoverflow.com" + title_el["href"],
                        "votes":             votes,
                        "answers":           answers,
                        "views":             views,
                        "opportunity_score": round(views / (answers + 1), 1),
                        "tag":               tag
                    })
        except Exception as e:
            print(f"Error scraping SO tag {tag}: {e}")
    return questions

def scrape_tags_batch(tags: list, pages: int = 2) -> pd.DataFrame:
    all_questions = []
    for tag in tags:
        all_questions.extend(scrape_tag(tag, pages))
    if not all_questions:
        return pd.DataFrame()
    return pd.DataFrame(all_questions).drop_duplicates(subset=["url"]).sort_values("opportunity_score", ascending=False)
