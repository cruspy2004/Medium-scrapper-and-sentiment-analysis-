from flask import Flask, render_template
from config import Config
from routes.medium import medium_bp
from routes.stackoverflow import so_bp

app = Flask(__name__)
app.config.from_object(Config)

# Register blueprints
app.register_blueprint(medium_bp, url_prefix='/medium')
app.register_blueprint(so_bp, url_prefix='/stackoverflow')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Use port 5000 by default
    app.run(port=5000, debug=(app.config['FLASK_ENV'] == 'development'))
