import requests
from bs4 import BeautifulSoup
import pandas as pd
import time as _time
import re

SO_BASE = "https://stackoverflow.com/questions/tagged/{tag}?tab=votes&page={page}"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TopicPulse/3.0"}


def _fetch_question_details(url: str) -> dict:
    """
    Fetch an individual SO question page to extract body text,
    code blocks, and tags.  Returns defaults on failure.
    """
    defaults = {"body_text": "", "code_blocks": [], "tags": [], "has_code": False}
    try:
        r = requests.get(url, timeout=10, headers=HEADERS)
        if r.status_code != 200:
            return defaults

        soup = BeautifulSoup(r.content, "html.parser")

        # Extract tags
        tag_links = soup.select(".post-taglist .post-tag, .js-post-tag-list-item a")
        tags = [t.get_text(strip=True) for t in tag_links]

        # Extract question body (first post)
        body_el = soup.select_one(".s-prose, .postcell .post-text")
        if not body_el:
            return {**defaults, "tags": tags}

        # Extract code blocks before stripping HTML
        code_blocks = []
        for code_tag in body_el.find_all(["code", "pre"]):
            text = code_tag.get_text(strip=True)
            if text:
                code_blocks.append(text)

        # Strip HTML for body text (remove code tags first)
        body_copy = BeautifulSoup(str(body_el), "html.parser")
        for code_tag in body_copy.find_all(["code", "pre"]):
            code_tag.decompose()
        body_text = body_copy.get_text(separator=" ", strip=True)
        body_text = re.sub(r"\s+", " ", body_text).strip()

        return {
            "body_text": body_text,
            "code_blocks": code_blocks,
            "tags": tags,
            "has_code": len(code_blocks) > 0,
        }
    except Exception:
        return defaults


def scrape_tag(tag: str, pages: int = 2) -> list:
    questions = []
    for page in range(1, pages + 1):
        url = SO_BASE.format(tag=tag.strip(), page=page)
        try:
            r = requests.get(url, timeout=10, headers=HEADERS)
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
                    q_url = "https://stackoverflow.com" + title_el["href"]

                    # v3: Fetch individual question page for body/code/tags
                    details = _fetch_question_details(q_url)
                    _time.sleep(1)  # Rate limiting

                    questions.append({
                        "title":             title_el.text.strip(),
                        "url":               q_url,
                        "votes":             votes,
                        "answers":           answers,
                        "views":             views,
                        "opportunity_score": round(views / (answers + 1), 1),
                        "tag":               tag,
                        # v3 additions
                        "body_text":         details["body_text"],
                        "code_blocks":       details["code_blocks"],
                        "tags":              details["tags"] if details["tags"] else [tag],
                        "has_code":          details["has_code"],
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
