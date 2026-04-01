from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    # Landing page to choose between Medium and Stack Overflow scrapers
    return render_template('choose_scraper.html')

@app.route('/medium')
def medium_scraper():
    return redirect("http://127.0.0.1:5002/")

@app.route('/stackoverflow')
def stackoverflow_scraper():
    return redirect("http://127.0.0.1:5000/")


if __name__ == '__main__':
    app.run(port=4000, debug=True)  # Run the main router app on port 4000
