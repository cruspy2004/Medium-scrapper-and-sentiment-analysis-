import sys
import os

# Add the parent directory to sys.path so imports work on Vercel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medium_scraper_frontend import app

# Vercel expects the Flask app to be exposed as 'app'
