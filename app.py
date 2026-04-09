from flask import Flask, render_template
from config import Config
from routes.medium import medium_bp
from routes.stackoverflow import so_bp
from routes.scan import scan_bp
from routes.brief import brief_bp

app = Flask(__name__)
app.config.from_object(Config)

# v2 blueprints — unified scan + brief API
app.register_blueprint(scan_bp)
app.register_blueprint(brief_bp)

# v1 blueprints — kept for backward compatibility
app.register_blueprint(medium_bp, url_prefix='/medium')
app.register_blueprint(so_bp, url_prefix='/stackoverflow')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Use port 5000 by default
    app.run(port=5000, debug=(app.config['FLASK_ENV'] == 'development'))
