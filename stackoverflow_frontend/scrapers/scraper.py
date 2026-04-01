import os
from scrapers.users_scraper import scrape_users
from scrapers.tags_scraper import scrape_tags
from scrapers.questions_scraper import scrape_questions
from scrapers.specific_question_scraper import scrape_specific_question

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

def scrape_data():
    print("Starting scraping process...")
    scrape_users(os.path.join(DATA_DIR, "users_data.csv"))
    scrape_tags(os.path.join(DATA_DIR, "tags_data.csv"))
    scrape_questions(os.path.join(DATA_DIR, "newest_questions_data.csv"))
    scrape_specific_question(os.path.join(DATA_DIR, "specific_question_data.csv"))
    print("Scraping completed!")
