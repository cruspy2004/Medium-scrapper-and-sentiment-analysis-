from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from scrapers.scraper import scrape_data  # Import the scraping logic

app = Flask(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data/<filename>")
def view_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return f"File '{filename}' not found!", 404
    data = pd.read_csv(file_path)
    return render_template("data_view.html", filename=filename, data=data.head(20).to_html(classes="data-table", index=False))

@app.route("/visualization")
def visualization():
    return render_template("visualization.html")

@app.route("/scraper", methods=["GET", "POST"])
def scraper():
    if request.method == "POST":
        scrape_data()  # Trigger the scraping process
        return redirect(url_for("index"))
    return render_template("scraper.html")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
