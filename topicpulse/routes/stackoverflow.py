from flask import Blueprint, render_template, request, make_response
import time
import pandas as pd
import io

from scrapers.stackoverflow import scrape_tags_batch
from analysis.opportunity import calculate_opportunities

so_bp = Blueprint('stackoverflow', __name__)

SCAN_RESULTS = {}

@so_bp.route('/')
def index():
    return render_template('so_form.html')

@so_bp.route('/scan', methods=['POST'])
def scan():
    start_time = time.time()
    
    tags_input = request.form.get('tags', '')
    pages = int(request.form.get('pages', 1))
    
    tags_list = [t.strip() for t in tags_input.split(',') if t.strip()]
    
    questions_df = scrape_tags_batch(tags_list, pages)
    
    if questions_df.empty:
        return render_template('so_results.html', error="No questions found.")
        
    questions_df = calculate_opportunities(questions_df)
    
    topic_str = ", ".join(tags_list)
    scan_id = str(int(time.time()))
    SCAN_RESULTS[scan_id] = questions_df
    
    t_elapsed = round(time.time() - start_time, 2)
    
    return render_template('so_results.html',
        topic=topic_str,
        questions=questions_df.to_dict('records')[:30],
        scan_time_seconds=t_elapsed,
        scan_id=scan_id
    )

@so_bp.route('/export/<scan_id>')
def export_csv(scan_id):
    if scan_id not in SCAN_RESULTS:
        return "Scan not found", 404
        
    df = SCAN_RESULTS[scan_id]
    out = io.StringIO()
    df.to_csv(out, index=False)
    
    response = make_response(out.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=so_questions_{scan_id}.csv"
    response.headers["Content-type"] = "text/csv"
    return response
