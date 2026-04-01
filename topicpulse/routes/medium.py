from flask import Blueprint, render_template, request, make_response
import time
import pandas as pd
import io
import os

from scrapers.medium import scrape_tags_batch
from analysis.sentiment import classify_sentiment_batch, generate_topic_summary
from analysis.influence import rank_authors

medium_bp = Blueprint('medium', __name__)

# Temporary in-memory storage, useful for small scale without DB
SCAN_RESULTS = {}

@medium_bp.route('/')
def index():
    return render_template('medium_form.html')

@medium_bp.route('/scan', methods=['POST'])
def scan():
    start_time = time.time()
    
    tags_input = request.form.get('tags', '')
    pages = int(request.form.get('pages', 1))
    
    tags_list = [t.strip() for t in tags_input.split(',') if t.strip()]
    
    articles_df = scrape_tags_batch(tags_list, pages)
    
    if articles_df.empty:
        return render_template('medium_results.html', error="No articles found.")
        
    titles = articles_df["title"].tolist()
    
    # 1. Sentiment Batch
    sentiments = classify_sentiment_batch(titles)
    
    # Optional logic if length mismatch from Gemini
    if len(sentiments) == len(articles_df):
        articles_df["sentiment"] = [s.get("sentiment", "NEUTRAL") for s in sentiments]
        articles_df["confidence"] = [s.get("confidence", 0.0) for s in sentiments]
        articles_df["reason"] = [s.get("reason", "") for s in sentiments]
    else:
        # fallback
        articles_df["sentiment"] = "NEUTRAL"
        articles_df["confidence"] = 0.0
        articles_df["reason"] = "Sentiment length mismatch"
        
    # 2. Influence Authors
    authors_df = rank_authors(articles_df)
    
    # 3. Topic Summary
    topic_str = ", ".join(tags_list)
    llm_summary = generate_topic_summary(topic_str, titles)
    
    # Provide simple sentiment breakdown
    sentiment_counts = articles_df["sentiment"].value_counts().to_dict()
    pos = sentiment_counts.get("POSITIVE", 0)
    neg = sentiment_counts.get("NEGATIVE", 0)
    neu = sentiment_counts.get("NEUTRAL", 0)
    
    total = pos + neg + neu
    overall = "NEUTRAL"
    if total > 0:
        if pos > neg and pos >= neu:
            overall = "POSITIVE"
        elif neg > pos and neg >= neu:
            overall = "NEGATIVE"
    
    scan_id = str(int(time.time()))
    SCAN_RESULTS[scan_id] = {
        "articles": articles_df,
        "authors": authors_df
    }
    
    t_elapsed = round(time.time() - start_time, 2)
    
    return render_template('medium_results.html', 
        topic=topic_str,
        articles=articles_df.to_dict('records')[:10], # Top 10 articles for display
        authors=authors_df.to_dict('records')[:10],   # Top 10 authors
        sentiment_summary={"positive": pos, "negative": neg, "neutral": neu, "overall": overall},
        llm_summary=llm_summary,
        scan_time_seconds=t_elapsed,
        scan_id=scan_id
    )

@medium_bp.route('/export/articles/<scan_id>')
def export_articles(scan_id):
    if scan_id not in SCAN_RESULTS:
        return "Scan not found", 404
        
    df = SCAN_RESULTS[scan_id]["articles"]
    out = io.StringIO()
    df.to_csv(out, index=False)
    
    response = make_response(out.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=medium_articles_{scan_id}.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@medium_bp.route('/export/authors/<scan_id>')
def export_authors(scan_id):
    if scan_id not in SCAN_RESULTS:
        return "Scan not found", 404
        
    df = SCAN_RESULTS[scan_id]["authors"]
    out = io.StringIO()
    df.to_csv(out, index=False)
    
    response = make_response(out.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=medium_authors_{scan_id}.csv"
    response.headers["Content-type"] = "text/csv"
    return response
