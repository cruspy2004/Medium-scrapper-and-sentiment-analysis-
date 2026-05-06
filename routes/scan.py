from flask import Blueprint, render_template, request, make_response, session, current_app
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
import io
import json

from scrapers.medium import scrape_tags_batch
from scrapers.stackoverflow import scrape_tags_batch as so_scrape_tags_batch
from analysis.sentiment import classify_sentiment_batch
from analysis.gap_score import compute_gap_scores

scan_bp = Blueprint('scan', __name__)

# In-memory storage for CSV export
SCAN_RESULTS = {}


def run_parallel_scrape(topics: list, pages: int):
    """Run Medium and Stack Overflow scrapers in parallel."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        medium_future = ex.submit(scrape_tags_batch, topics, pages)
        so_future = ex.submit(so_scrape_tags_batch, topics, pages)
        articles_df = medium_future.result()
        questions_df = so_future.result()
    return articles_df, questions_df


@scan_bp.route('/scan', methods=['POST'])
def scan():
    start_time = time.time()

    tags_input = request.form.get('tags', '')
    pages = int(request.form.get('pages', 1))

    tags_list = [t.strip() for t in tags_input.split(',') if t.strip()]
    topic_str = ", ".join(tags_list)

    if not tags_list:
        return render_template('results.html', error="Please enter at least one topic.")

    # 1. Parallel scrape — both platforms at once
    articles_df, questions_df = run_parallel_scrape(tags_list, pages)

    articles = []
    questions = []

    # 2. Process articles — sentiment classification
    if not articles_df.empty:
        titles = articles_df["title"].tolist()

        # Batch sentiment (up to 10 per call for speed)
        all_sentiments = []
        batch_size = 10
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            all_sentiments.extend(classify_sentiment_batch(batch))

        if len(all_sentiments) == len(articles_df):
            articles_df["sentiment"] = [s.get("sentiment", "NEUTRAL") for s in all_sentiments]
            articles_df["confidence"] = [s.get("confidence", 0.0) for s in all_sentiments]
            articles_df["reason"] = [s.get("reason", "") for s in all_sentiments]
        else:
            articles_df["sentiment"] = "NEUTRAL"
            articles_df["confidence"] = 0.0
            articles_df["reason"] = "Sentiment length mismatch"

        articles = articles_df.to_dict("records")

    # 3. Process questions
    if not questions_df.empty:
        questions = questions_df.to_dict("records")

    # 3.5 v3: Run quality prediction on SO questions (if model is loaded)
    predictor = current_app.config.get("PREDICTOR")
    if predictor and questions:
        try:
            questions = predictor.predict_batch(questions)
        except Exception as e:
            print(f"Quality prediction failed (continuing without): {e}")

    # 4. Compute content gap scores
    gaps = compute_gap_scores(articles, questions)

    # Attach the topic to each gap (needed for brief generation)
    for gap in gaps:
        gap["topic"] = topic_str

    # 5. Build sentiment summary
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
    if not articles_df.empty and "sentiment" in articles_df.columns:
        counts = articles_df["sentiment"].value_counts().to_dict()
        sentiment_summary = {
            "positive": int(counts.get("POSITIVE", 0)),
            "negative": int(counts.get("NEGATIVE", 0)),
            "neutral": int(counts.get("NEUTRAL", 0)),
        }

    # 5.5 v3: Build quality summary
    quality_summary = {"high": 0, "medium": 0, "low": 0}
    if questions:
        for q in questions:
            ql = q.get("quality_label", "")
            if ql == "HIGH":
                quality_summary["high"] += 1
            elif ql == "MEDIUM":
                quality_summary["medium"] += 1
            elif ql == "LOW":
                quality_summary["low"] += 1

    quality_passed = quality_summary["high"] + quality_summary["medium"]

    # 6. Store for CSV export (rebuild DF with quality columns)
    if questions:
        questions_export_df = pd.DataFrame(questions)
    else:
        questions_export_df = pd.DataFrame()

    scan_id = str(int(time.time()))
    SCAN_RESULTS[scan_id] = {
        "articles": articles_df if not articles_df.empty else pd.DataFrame(),
        "questions": questions_export_df,
    }

    t_elapsed = round(time.time() - start_time, 2)

    return render_template('results.html',
        topic=topic_str,
        gaps=gaps,
        gaps_json=json.dumps(gaps, default=str),
        all_articles=articles[:20],
        all_questions=questions[:10],
        sentiment_summary=sentiment_summary,
        quality_summary=quality_summary,
        quality_passed=quality_passed,
        scan_time_seconds=t_elapsed,
        scan_id=scan_id,
        total_articles=len(articles),
        total_questions=len(questions),
    )


@scan_bp.route('/export/articles/<scan_id>')
def export_articles(scan_id):
    if scan_id not in SCAN_RESULTS:
        return "Scan not found", 404

    df = SCAN_RESULTS[scan_id]["articles"]
    if df.empty:
        return "No article data available", 404

    out = io.StringIO()
    df.to_csv(out, index=False)

    response = make_response(out.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=articles_{scan_id}.csv"
    response.headers["Content-type"] = "text/csv"
    return response


@scan_bp.route('/export/questions/<scan_id>')
def export_questions(scan_id):
    if scan_id not in SCAN_RESULTS:
        return "Scan not found", 404

    df = SCAN_RESULTS[scan_id]["questions"]
    if df.empty:
        return "No question data available", 404

    out = io.StringIO()
    df.to_csv(out, index=False)

    response = make_response(out.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=questions_{scan_id}.csv"
    response.headers["Content-type"] = "text/csv"
    return response
