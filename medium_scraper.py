import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
from textblob import TextBlob
import os

def get_links(tag, pages):
    url = f'https://medium.com/tag/{tag}'
    links = []
    for page in range(1, pages + 1):
        print(f"Fetching page {page} for tag '{tag}'...")
        response = requests.get(f"{url}?page={page}")
        if response.status_code != 200:
            print(f"Failed to fetch {url}?page={page} - Status Code: {response.status_code}")
            continue
        soup = BeautifulSoup(response.content, 'html.parser')

        articles = soup.find_all('a', class_='af ag ah ai aj ak al am an ao ap aq ar as at')
        for article in articles:
            href = article.get('href')
            if href and href.startswith('/'):  # Ensure it's a valid Medium link
                links.append("https://medium.com" + href)
    return list(set(links))  # Remove duplicates

def get_article_data(links):
    articles = []
    for link in links:
        print(f"Fetching article: {link}")
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.content, 'html.parser')

            article = {
                'author': None,
                'claps': 0,
                'responses': 0,
                'title': None,
                'link': link,
                'sentiment': None
            }

            # Author
            author_meta = soup.find('meta', {'name': 'author'})
            if author_meta:
                article['author'] = author_meta.get('content')

            # Claps
            clap_span = soup.find('span', {'class': 'pw-multi-vote-count'})
            if clap_span:
                clap_text = clap_span.text.strip().replace('K', '000').replace('.', '')
                article['claps'] = int(clap_text) if clap_text.isdigit() else 0

            # Responses
            responses_button = soup.find('button', {'aria-label': 'responses'})
            if responses_button:
                responses_text = responses_button.text.strip()
                article['responses'] = int(responses_text.replace(' responses', '').strip()) if responses_text.isdigit() else 0

            # Title
            title = soup.find('h1', {'data-testid': 'storyTitle'})
            if title:
                article['title'] = title.text.strip()

                # Sentiment analysis on the title
                sentiment = TextBlob(article['title']).sentiment.polarity
                if sentiment > 0:
                    article['sentiment'] = 'Positive'
                elif sentiment < 0:
                    article['sentiment'] = 'Negative'
                else:
                    article['sentiment'] = 'Neutral'

            articles.append(article)

        except Exception as e:
            print(f"Error scraping {link}: {e}")

    return articles

def calculate_metrics(articles):
    # Convert to DataFrame for analysis
    df = pd.DataFrame(articles)

    # Total claps and responses for the topic
    total_claps = df['claps'].sum()
    total_responses = df['responses'].sum()

    # Sentiment summary
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    total_positive = sentiment_counts.get('Positive', 0)
    total_negative = sentiment_counts.get('Negative', 0)
    total_neutral = sentiment_counts.get('Neutral', 0)

    overall_sentiment = 'Neutral'
    if total_positive > total_negative and total_positive > total_neutral:
        overall_sentiment = 'Positive'
    elif total_negative > total_positive and total_negative > total_neutral:
        overall_sentiment = 'Negative'

    # Ranking authors by their total claps and responses
    author_metrics = df.groupby('author').agg(
        total_claps=('claps', 'sum'),
        total_responses=('responses', 'sum'),
        article_count=('link', 'count')
    ).reset_index()

    # Calculate influence score for authors
    author_metrics['influence_score'] = (
        author_metrics['total_claps'] * 2 + author_metrics['total_responses']
    )
    author_metrics = author_metrics.sort_values(by='influence_score', ascending=False)

    return total_claps, total_responses, author_metrics, total_positive, total_negative, total_neutral, overall_sentiment

def save_to_csv(articles, tag):
    output_dir = 'D:/medium_scraper/output'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    filename = os.path.join(output_dir, f"{tag}_articles.csv")
    keys = ['author', 'claps', 'responses', 'title', 'link', 'sentiment']
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(articles)
    print(f"Saved {len(articles)} articles to {filename}")

def main():
    tag = input("Enter the Medium tag to scrape: ")
    pages = int(input("Enter the number of pages to scrape: "))

    links = get_links(tag, pages)

    articles = get_article_data(links)

    save_to_csv(articles, tag)

    # Analyze metrics
    total_claps, total_responses, author_metrics, total_positive, total_negative, total_neutral, overall_sentiment = calculate_metrics(articles)

    print(f"Total Claps for '{tag}': {total_claps}")
    print(f"Total Responses for '{tag}': {total_responses}")
    print("Top Influential Authors:")
    print(author_metrics.head(10))

    print("\nSentiment Analysis:")
    print(f"Positive: {total_positive}, Negative: {total_negative}, Neutral: {total_neutral}")
    print(f"Overall Sentiment: {overall_sentiment}")

    # Save author metrics to a CSV
    metrics_filename = os.path.join('D:/medium_scraper/output', f"{tag}_author_metrics.csv")
    author_metrics.to_csv(metrics_filename, index=False)
    print(f"Saved author metrics to {metrics_filename}")

if __name__ == '__main__':
    main()
