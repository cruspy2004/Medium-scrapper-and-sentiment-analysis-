import os
import logging
from flask import Flask, render_template
from config import Config
from routes.medium import medium_bp
from routes.stackoverflow import so_bp
from routes.scan import scan_bp
from routes.brief import brief_bp

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# v2 blueprints — unified scan + brief API
app.register_blueprint(scan_bp)
app.register_blueprint(brief_bp)

# v1 blueprints — kept for backward compatibility
app.register_blueprint(medium_bp, url_prefix='/medium')
app.register_blueprint(so_bp, url_prefix='/stackoverflow')

# v3: Load question quality model at startup (if model files exist)
MODEL_PATH = os.path.join("models", "question_quality.pt")
VOCAB_TEXT_PATH = os.path.join("models", "vocab_text.json")
VOCAB_CODE_PATH = os.path.join("models", "vocab_code.json")

if all(os.path.exists(p) for p in [MODEL_PATH, VOCAB_TEXT_PATH, VOCAB_CODE_PATH]):
    try:
        from analysis.question_quality import QuestionQualityPredictor
        predictor = QuestionQualityPredictor(
            model_path=MODEL_PATH,
            vocab_text_path=VOCAB_TEXT_PATH,
            vocab_code_path=VOCAB_CODE_PATH,
        )
        app.config["PREDICTOR"] = predictor
        logger.info("QuestionQualityPredictor loaded successfully.")
    except Exception as e:
        logger.warning("Failed to load quality model: %s", e)
        app.config["PREDICTOR"] = None
else:
    logger.warning("Model files not found — quality prediction disabled. "
                   "Run 'python training/train.py' to generate model.")
    app.config["PREDICTOR"] = None


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Use port 5000 by default
    app.run(port=5000, debug=(app.config['FLASK_ENV'] == 'development'))

