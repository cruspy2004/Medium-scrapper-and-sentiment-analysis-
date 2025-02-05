from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import random
from medium_scraper import get_links, get_article_data, calculate_metrics, save_to_csv

app = Flask(__name__)

# Ensure upload and output folders exist
OUTPUT_FOLDER = 'D:/medium_scraper/output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Fun facts for loading screen
FUN_FACTS = [
    "Did you know? The first email was sent in 1971!",
    "Fun fact: Over 4 million blog posts are published daily.",
    "Medium hosts over 500,000 new articles every month!",
    "Web scraping is a technique to extract data from websites!",
    "The internet has over 1.5 billion websites today!"
]

# Store temporary scraping state
SCRAPING_RESULTS = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scrape', methods=['POST'])
def scrape():
    tag = request.form['tag']
    pages = int(request.form['pages'])
    fun_fact = random.choice(FUN_FACTS)

    # Show loading screen with a fun fact
    return render_template('loading.html', fun_fact=fun_fact, tag=tag, pages=pages)


@app.route('/process_scrape/<tag>/<int:pages>')
def process_scrape(tag, pages):
    try:
        # Scrape links and articles
        links = get_links(tag, pages)
        articles = get_article_data(links)

        if articles:
            csv_filename = f"{tag}_articles.csv"
            save_to_csv(articles, tag)

            # Calculate metrics
            _, total_responses, _, total_positive, total_negative, total_neutral, overall_sentiment = calculate_metrics(articles)

            # Store results for rendering
            SCRAPING_RESULTS[tag] = {
                'pages': pages,
                'total_responses': total_responses,
                'total_positive': total_positive,
                'total_negative': total_negative,
                'total_neutral': total_neutral,
                'overall_sentiment': overall_sentiment,
                'csv_filename': csv_filename
            }

            return redirect(url_for('results', tag=tag))
        else:
            return render_template('error.html', message="No articles found for the specified tag.")
    except Exception as e:
        return render_template('error.html', message=str(e))


@app.route('/results/<tag>')
def results(tag):
    if tag in SCRAPING_RESULTS:
        result = SCRAPING_RESULTS[tag]
        return render_template(
            'results.html',
            tag=tag,
            pages=result['pages'],
            total_responses=result['total_responses'],
            total_positive=result['total_positive'],
            total_negative=result['total_negative'],
            total_neutral=result['total_neutral'],
            overall_sentiment=result['overall_sentiment'],
            csv_filename=result['csv_filename']
        )
    else:
        return render_template('error.html', message="No results available for the specified tag.")


@app.route('/output/<path:filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
