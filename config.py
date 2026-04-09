import os
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super-secret-key'
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
